import pandas as pd
import sys
import json
import re
from collections import defaultdict
from functools import lru_cache
import os
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")
DATA_CLUSTERS_DIR = os.path.join(ROOT_DIR, "data", "clusters")
DATA_KEYWORDS_DIR = os.path.join(ROOT_DIR, "data", "keywords")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

sys.path.insert(0, COMMON_DIR)
from abbreviations import expand_abbreviations
from category_utils import (
    kalshi_category_terms,
    kalshi_category_key,
    normalize_category_mapping,
    normalize_label,
    polymarket_category_keys,
    resolve_polymarket_keys_for_kalshi,
)
try:
    from pipeline.clustering.clustering_features import (
        CLUSTERING_POLICY_VERSION,
        annotate_pairs,
        build_keyword_statistics,
        connected_component_stats,
        prune_pairs,
    )
    from pipeline.clustering.event_profiles import (
        PROFILE_SCHEMA_VERSION,
        build_event_profiles,
        profiles_fingerprint,
    )
except ImportError:
    from clustering_features import (
        CLUSTERING_POLICY_VERSION,
        annotate_pairs,
        build_keyword_statistics,
        connected_component_stats,
        prune_pairs,
    )
    from event_profiles import (
        PROFILE_SCHEMA_VERSION,
        build_event_profiles,
        profiles_fingerprint,
    )
try:
    from keyword_config import (
        canonicalize_keyword,
        config_fingerprint,
        find_domain_entity_tokens,
        is_noise_keyword,
    )
except ImportError:
    canonicalize_keyword = lambda value: str(value).lower().strip()
    config_fingerprint = lambda: "legacy"
    find_domain_entity_tokens = lambda text: set()
    is_noise_keyword = lambda value: False

# Config
KALSHI_FILE = os.path.join(DATA_MARKETS_DIR, "markets_clean_kalshi.csv")
POLY_FILE = os.path.join(DATA_MARKETS_DIR, "markets_clean_polymarket.csv")
OUTPUT_JSON = os.path.join(DATA_CLUSTERS_DIR, "candidate_clusters.json")
SIGNATURES_CSV = os.path.join(DATA_CLUSTERS_DIR, "event_signatures.csv")
EVENT_KEYWORDS_INDEX_FILE = os.path.join(DATA_KEYWORDS_DIR, "event_keywords_index.json")
CATEGORY_MAPPING_FILE = os.path.join(CONFIG_DIR, "category_mapping.json")
EXCLUSIVE_CLUSTERS_FILE = os.path.join(DATA_CLUSTERS_DIR, "exclusive_clusters.json")
EVENT_PROFILES_FILE = os.path.join(DATA_CLUSTERS_DIR, "event_profiles.json")
CLUSTERING_RUN_REPORT_FILE = os.path.join(DATA_CLUSTERS_DIR, "clustering_run_report.json")

nlp = None
CONFIG_VERSION = config_fingerprint()


class SimpleToken:
    def __init__(self, text, index):
        self.text = text
        self.i = index
        self.is_punct = False
        self.is_space = False
        self.is_digit = text.isdigit()
        self.like_num = text.isdigit()
        self.lemma_ = canonicalize_keyword(text)
        self.is_stop = is_noise_keyword(text)
        self.pos_ = "PROPN" if text[:1].isupper() else "NOUN"


class SimpleDoc(list):
    ents = []
    noun_chunks = []


class RegexNLP:
    def __call__(self, text):
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", text or "")
        return SimpleDoc(SimpleToken(token, idx) for idx, token in enumerate(tokens))

def load_spacy():
    global nlp
    if nlp is None:
        print("Loading SpaCy model...")
        try:
            import spacy
        except ImportError:
            print("⚠️ SpaCy is not installed. Falling back to regex signature extraction.")
            nlp = RegexNLP()
            return
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading SpaCy model...")
            os.system(f'"{sys.executable}" -m spacy download en_core_web_sm')
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️ SpaCy model unavailable. Falling back to regex signature extraction.")
                nlp = RegexNLP()

def extract_signature_details(text, category):
    """
    Extracts signature tokens AND structured details for CSV export.
    Returns: (set_of_tokens, details_dict)
    """
    if not isinstance(text, str):
        text = ""
    
    # 1. Expand Abbreviations
    clean_text = expand_abbreviations(text)
    
    # 2. Extract Years (2020-2030)
    years = re.findall(r"\b(20[2-3][0-9])\b", clean_text)
    
    # 3. Extract Specific Entities & High Profile Words
    # CRITICAL: Use original 'text' to preserve Capitalization for NER!
    ner_text = text
    prefixes = ["Will ", "Who will ", "When will ", "Which ", "Where ", "Does ", "Is "]
    for p in prefixes:
        if ner_text.startswith(p):
            ner_text = ner_text[len(p):]
            break
            
    doc = nlp(ner_text)
    
    tokens = set()
    entities = []
    
    # Add Years to tokens
    for y in years:
        tokens.add(f"YEAR:{y}")
        
    # Add Category
    if category:
        try:
            cat_token = f"CAT:{category.lower()}"
            tokens.add(cat_token)
        except AttributeError as e:
            raise e

    high_profile_words = set()
    
    # 3a. Dictionary Lookup & Normalization (Advanced)
    # Check every token against abbreviations
    # flatten ABBREVIATIONS for fast lookup
    from abbreviations import ABBREVIATIONS
    
    # helper for fast normalized lookup
    def get_dict_matches(t_text):
        matches = []
        t_clean = t_text.lower().strip()
        # Direct key match
        if t_clean in ABBREVIATIONS:
            matches.append(ABBREVIATIONS[t_clean])
        # Value match (if the word itself is already a high profile concept)
        # (Optimisation: maybe too slow to check all values? Let's check keys and common sports prefixes)
        
        # Check sports prefixes e.g. "lakers" -> match "nba:lakers"
        for k, v in ABBREVIATIONS.items():
            if k.endswith(":" + t_clean):
                 matches.append(v)
        return matches

    # Process all tokens in the doc for dictionary matches + Lemmatization
    for token in doc:
        # Lemmatize all nouns/props/verbs
        if not token.is_stop and not token.is_punct:
             # Add Lemma
             high_profile_words.add(token.lemma_.lower())
             
             # Check Dictionary
             dict_hits = get_dict_matches(token.text)
             for dh in dict_hits:
                 high_profile_words.add(dh) # Add the expanded/normalized form

    # 3b. SpaCy Entities
    for ent in doc.ents:
        lbl = ent.label_
        val = ent.text.lower().strip()
        # Strip possessive 's
        if val.endswith("'s"):
            val = val[:-2]
        elif val.endswith("’s"): # Handle curly quote just in case
            val = val[:-2]
        
        if len(val) < 2: continue
        if val in ["the", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov"]: continue

        if lbl in ["PERSON", "ORG", "GPE", "NORP", "EVENT", "LAW", "LOC"]:
            tokens.add(f"{lbl}:{val}")
            entities.append(f"{lbl}:{val}")
            
            # Add entity main words to high profile
            high_profile_words.add(val)
            # Also add parts of the entity (e.g. "Donald Trump" -> "donald", "trump")
            for sub in val.split():
                 if len(sub) > 2:
                     high_profile_words.add(sub)
                     # Dict check sub-parts
                     for dh in get_dict_matches(sub):
                         high_profile_words.add(dh)

    # 3b-2. Noun Chunks (Phrases)
    # Extracts "Prison Break", "CPI Index" etc.
    for chunk in doc.noun_chunks:
        c_text = chunk.text.lower().strip()
        if len(c_text) > 3 and c_text not in ["the", "this", "that"]:
             high_profile_words.add(c_text)
             # Add individual words in chunk if they look important
             for w in c_text.split():
                 if len(w) > 3 and w not in ["the", "will", "year", "price"]:
                     high_profile_words.add(w)

    # 3b-3. Proper Nouns (PROPN) fallback
    # SpaCy NER often misses niche crypto/stocks/names, but tags them as PROPN
    for token in doc:
        if token.pos_ == "PROPN" and len(token.text) > 2:
            t_lower = token.text.lower()
            if t_lower not in high_profile_words: # Avoid dupes
                high_profile_words.add(t_lower)
                # Also Try to treat as a potential entity
                tokens.add(f"PROPN:{t_lower}")

    # 3b-4. Quoted Text
    # "Movie Title" or "Song Name"
    quoted = re.findall(r'"([^"]*)"', text) + re.findall(r"'([^']*)'", text)
    for q in quoted:
        q_clean = q.lower().strip()
        if len(q_clean) > 2:
            high_profile_words.add(q_clean)
            tokens.add(f"QUOTE:{q_clean}")

    # 3c. Fallback Keywords (Explicit)
    fallback_keywords = {
        "mars": "LOC:mars", "moon": "LOC:moon", "earth": "LOC:earth",
        "california": "GPE:california", "texas": "GPE:texas", "florida": "GPE:florida", "new york": "GPE:new york",
        "china": "GPE:china", "usa": "GPE:usa", "us": "GPE:us", "uk": "GPE:uk", "russia": "GPE:russia",
        "india": "GPE:india", "japan": "GPE:japan", "germany": "GPE:germany", "france": "GPE:france",
        "bitcoin": "ORG:bitcoin", "ethereum": "ORG:ethereum", "crypto": "ORG:crypto",
        "trump": "PERSON:trump", "biden": "PERSON:biden", "kamala": "PERSON:kamala", "harris": "PERSON:harris",
        "musk": "PERSON:musk", "putin": "PERSON:putin", "zelensky": "PERSON:zelensky",
        "openai": "ORG:openai", "anthropic": "ORG:anthropic", "google": "ORG:google", "microsoft": "ORG:microsoft"
    }
    
    clean_lower = clean_text.lower()
    for kw, token in fallback_keywords.items():
        if f" {kw} " in f" {clean_lower} ":
            if token not in tokens:
                tokens.add(token)
                entities.append(token)
                high_profile_words.add(kw)
                parts = token.split(":")
                if len(parts) > 1:
                    high_profile_words.add(parts[1])

    for token in find_domain_entity_tokens(clean_text):
        if token not in tokens:
            tokens.add(token)
            entities.append(token)
        parts = token.split(":", 1)
        if len(parts) > 1:
            high_profile_words.add(parts[1])

    # Regex for Percentages and Numbers
    pcts = re.findall(r"(\d+(?:\.\d+)?)%", clean_text)
    for p in pcts:
        tokens.add(f"PCT:{p}")
        
    # CRITICAL: Add High Profile Words to Tokens for Tier Logic
    # This allows "djt" (HPW:donald trump) to match "Trump" (HPW:trump) ?? No, "Donald Trump" (HPW:donald trump)
    for hpw in high_profile_words:
        canonical = canonicalize_keyword(hpw)
        if canonical and not is_noise_keyword(canonical):
            tokens.add(f"HPW:{canonical}")

    details = {
        "original_text": text,
        "clean_text": clean_text,
        "years": "|".join(years),
        "entities": "|".join(entities),
        "high_profile_words": sorted(list({e.split(":")[1] for e in entities})),
        "category": category,
        "keyword_config_version": CONFIG_VERSION,
        "signature_tokens": sorted(list(tokens))
    }
    
    return tokens, details

def check_incompatibility(sig1, sig2):
    """
    Hard incompatibility check (Tier 2/3 rules).
    """
    # Parse signatures
    years1 = {s.split(":")[1] for s in sig1 if s.startswith("YEAR:")}
    years2 = {s.split(":")[1] for s in sig2 if s.startswith("YEAR:")}
    
    persons1 = {s.split(":", 1)[1] for s in sig1 if s.startswith("PERSON:")}
    persons2 = {s.split(":", 1)[1] for s in sig2 if s.startswith("PERSON:")}
    
    # RULE 1: Year Mismatch
    # If both have years, they must overlap or be adjacent
    if years1 and years2:
        y1_ints = {int(y) for y in years1}
        y2_ints = {int(y) for y in years2}
        compatible = False
        for y1 in y1_ints:
            for y2 in y2_ints:
                if abs(y1 - y2) <= 1:
                    compatible = True
                    break
        if not compatible:
            return True

    # RULE 2: Person conflict.
    # SpaCy sometimes labels teams/titles as PERSON, so compare normalized tokens
    # instead of requiring exact raw entity equality.
    if persons1 and persons2 and not person_entities_compatible(persons1, persons2):
        return True

    return False

def normalize_entity_tokens(entity):
    text = re.sub(r"['’]s\b", "", str(entity).lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return {token for token in text.split() if len(token) > 1}

def person_entities_compatible(persons1, persons2):
    for person1 in persons1:
        tokens1 = normalize_entity_tokens(person1)
        if not tokens1:
            continue
        for person2 in persons2:
            tokens2 = normalize_entity_tokens(person2)
            if tokens1.intersection(tokens2):
                return True
    return False


def has_candidate_hard_conflict(candidate_a, candidate_b):
    """Structured conflict check used by candidate generation/debug tests."""
    a_years = {str(year) for year in candidate_a.get("years", []) if str(year)}
    b_years = {str(year) for year in candidate_b.get("years", []) if str(year)}
    if a_years and b_years:
        a_ints = {int(year) for year in a_years if year.isdigit()}
        b_ints = {int(year) for year in b_years if year.isdigit()}
        if a_ints and b_ints and all(abs(a - b) > 1 for a in a_ints for b in b_ints):
            return True

    a_entities = candidate_a.get("entities", {}) or {}
    b_entities = candidate_b.get("entities", {}) or {}
    a_gpes = {canonicalize_keyword(value) for value in a_entities.get("gpe", []) if value}
    b_gpes = {canonicalize_keyword(value) for value in b_entities.get("gpe", []) if value}
    if a_gpes and b_gpes and a_gpes.isdisjoint(b_gpes):
        return True

    return False

def extract_quarter_years(text):
    if not isinstance(text, str):
        return set()
    normalized = text.lower()
    return set(re.findall(r"\b(q[1-4])\s+(20[2-3][0-9])\b", normalized))

def extract_scope_terms(text):
    if not isinstance(text, str):
        return set()
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    terms = set()
    scope_groups = {
        "president": ("president", "presidential"),
        "senate": ("senate", "senator"),
        "house": ("house", "congressional district"),
        "governor": ("governor", "gubernatorial", "governorship"),
        "primary": ("primary", "nominee", "nomination"),
        "general": ("general election",),
        "approval": ("approval", "rating"),
        "winner": ("winner", "win", "champion"),
        "qualifier": ("qualifier", "qualify", "qualifies"),
        "margin": ("margin", "margin of victory"),
        "outright": ("outright", "first round"),
        "place": ("1st place", "2nd place", "3rd place", "place"),
        "artist": ("artist",),
        "album": ("album",),
        "song": ("song", "track"),
        "driver": ("driver", "drivers"),
        "constructor": ("constructor", "constructors"),
        "relegation": ("relegation", "relegated"),
    }
    for term, aliases in scope_groups.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            terms.add(term)
    if re.search(r"\btop\s+(?:\d+|four|five|ten)\b", normalized):
        terms.add("top_finish")
    return terms

def has_title_scope_conflict(k_title, p_title):
    k_scope = extract_scope_terms(k_title)
    p_scope = extract_scope_terms(p_title)

    political_offices = {"president", "senate", "house", "governor"}
    k_offices = k_scope.intersection(political_offices)
    p_offices = p_scope.intersection(political_offices)
    if k_offices and p_offices and not k_offices.intersection(p_offices):
        return True

    if ("winner" in k_scope and "qualifier" in p_scope) or ("qualifier" in k_scope and "winner" in p_scope):
        return True
    incompatible_dimensions = (
        {"artist", "album", "song"},
        {"driver", "constructor"},
        {"winner", "qualifier", "relegation", "top_finish"},
    )
    for dimensions in incompatible_dimensions:
        k_dimensions = k_scope.intersection(dimensions)
        p_dimensions = p_scope.intersection(dimensions)
        if k_dimensions and p_dimensions and k_dimensions.isdisjoint(p_dimensions):
            return True
    winner_conflicts = {"margin", "outright", "place"}
    for conflict_term in winner_conflicts:
        k_has_conflict = conflict_term in k_scope
        p_has_conflict = conflict_term in p_scope
        if k_has_conflict == p_has_conflict:
            continue
        if ("winner" in k_scope and p_has_conflict) or ("winner" in p_scope and k_has_conflict):
            return True

    return False

MONTH_ALIASES = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

DATE_MONTH_PATTERN = (
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20[2-3][0-9]))?\b"
)

DIRECTION_UP_TERMS = {
    "above", "at least", "exceed", "exceeds", "greater than", "high",
    "higher", "highest", "increase", "over", "rise", "up",
}

DIRECTION_DOWN_TERMS = {
    "at most", "below", "decrease", "drop", "fall", "fewer than",
    "less than", "low", "lower", "lowest", "under", "down",
}

THRESHOLD_CONTEXT_TERMS = {
    "%", "above", "approval", "at least", "at most", "below", "bps",
    "close", "fewer than", "greater than", "high", "higher", "low",
    "lower", "magnitude", "more than", "over", "percent", "price",
    "range", "score", "than", "under",
}

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
}

US_STATE_ABBREVIATIONS = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island",
    "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
}

COUNTRY_ALIASES = {
    "america": "united states",
    "argentina": "argentina",
    "argentine": "argentina",
    "argentinian": "argentina",
    "australia": "australia",
    "australian": "australia",
    "brazil": "brazil",
    "brazilian": "brazil",
    "britain": "united kingdom",
    "bulgaria": "bulgaria",
    "bulgarian": "bulgaria",
    "canada": "canada",
    "canadian": "canada",
    "china": "china",
    "chinese": "china",
    "colombia": "colombia",
    "colombian": "colombia",
    "czechia": "czechia",
    "denmark": "denmark",
    "danish": "denmark",
    "england": "england",
    "france": "france",
    "french": "france",
    "germany": "germany",
    "german": "germany",
    "guinea-bissau": "guinea-bissau",
    "great britain": "united kingdom",
    "india": "india",
    "indian": "india",
    "iran": "iran",
    "iraq": "iraq",
    "ireland": "ireland",
    "irish": "ireland",
    "israel": "israel",
    "italy": "italy",
    "italian": "italy",
    "japan": "japan",
    "mexico": "mexico",
    "mexican": "mexico",
    "nepal": "nepal",
    "palestine": "palestine",
    "peru": "peru",
    "peruvian": "peru",
    "poland": "poland",
    "polish": "poland",
    "romania": "romania",
    "romanian": "romania",
    "russia": "russia",
    "russian": "russia",
    "saudi arabia": "saudi arabia",
    "spain": "spain",
    "spanish": "spain",
    "slovenia": "slovenia",
    "slovenian": "slovenia",
    "syria": "syria",
    "ukraine": "ukraine",
    "ukrainian": "ukraine",
    "united kingdom": "united kingdom",
    "united states": "united states",
    "usa": "united states",
    "wales": "wales",
}

RELATION_SCOPE_TERMS = {
    "deal", "normalize", "relations", "sanction", "tariff", "treaty", "war",
}


def extract_month_day_markers(text):
    if not isinstance(text, str):
        return set()
    markers = set()
    normalized = text.lower()
    for match in re.finditer(DATE_MONTH_PATTERN, normalized):
        raw_month, day, year = match.groups()
        month = MONTH_ALIASES.get(raw_month[:3], raw_month)
        markers.add((month, int(day), year or ""))
    return markers


def date_markers_compatible(k_marker, p_marker):
    k_month, k_day, k_year = k_marker
    p_month, p_day, p_year = p_marker
    if k_month != p_month or k_day != p_day:
        return False
    return not k_year or not p_year or k_year == p_year


def has_date_scope_conflict(k_title, p_title):
    k_qy = extract_quarter_years(k_title)
    p_qy = extract_quarter_years(p_title)
    if k_qy and p_qy and k_qy.isdisjoint(p_qy):
        return True

    k_dates = extract_month_day_markers(k_title)
    p_dates = extract_month_day_markers(p_title)
    if k_dates and p_dates and not any(
        date_markers_compatible(k_marker, p_marker)
        for k_marker in k_dates
        for p_marker in p_dates
    ):
        return True
    return False


def text_has_any_term(text, terms):
    normalized = f" {re.sub(r'[^a-z0-9% ]+', ' ', str(text).lower())} "
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms if term != "%") or (
        "%" in terms and "%" in str(text)
    )


def extract_direction_terms(text):
    directions = set()
    if text_has_any_term(text, DIRECTION_UP_TERMS):
        directions.add("up")
    if text_has_any_term(text, DIRECTION_DOWN_TERMS):
        directions.add("down")
    return directions


def has_direction_conflict(k_title, p_title):
    k_dirs = extract_direction_terms(k_title)
    p_dirs = extract_direction_terms(p_title)
    return k_dirs == {"up"} and p_dirs == {"down"} or k_dirs == {"down"} and p_dirs == {"up"}


def normalize_number(raw_number):
    try:
        number = float(raw_number)
    except ValueError:
        return raw_number
    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def date_day_numbers(text):
    return {str(day) for _, day, _ in extract_month_day_markers(text)}


def significant_numbers(text):
    normalized = str(text)
    date_days = date_day_numbers(normalized)
    numbers = set()
    for raw in re.findall(r"\b\d+(?:\.\d+)?\b", normalized):
        value = normalize_number(raw)
        if re.fullmatch(r"20[2-3][0-9]", value):
            continue
        if value in date_days:
            continue
        numbers.add(value)
    return numbers


def has_numeric_threshold_context(text):
    return text_has_any_term(text, THRESHOLD_CONTEXT_TERMS)


def has_numeric_threshold_conflict(k_title, p_title):
    if not (has_numeric_threshold_context(k_title) and has_numeric_threshold_context(p_title)):
        return False
    k_numbers = significant_numbers(k_title)
    p_numbers = significant_numbers(p_title)
    return bool(k_numbers and p_numbers and k_numbers.isdisjoint(p_numbers))


def extract_named_terms(text, terms):
    normalized = f" {re.sub(r'[^a-z0-9 ]+', ' ', str(text).lower())} "
    found = set()
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            found.add(term)
    return found


@lru_cache(maxsize=None)
def extract_us_states(text):
    found = extract_named_terms(text, US_STATES)
    normalized = str(text).lower()
    for match in re.finditer(r"\b([a-z]{2})-(?:\d{1,2}|al)\b", normalized):
        state = US_STATE_ABBREVIATIONS.get(match.group(1))
        if state:
            found.add(state)
    return frozenset(found)


@lru_cache(maxsize=None)
def extract_us_districts(text):
    normalized = str(text).lower()
    districts = set()
    for match in re.finditer(r"\b([a-z]{2})-(\d{1,2}|al)\b", normalized):
        state = US_STATE_ABBREVIATIONS.get(match.group(1))
        if state:
            districts.add((state, match.group(2)))
    return frozenset(districts)


@lru_cache(maxsize=None)
def extract_countries(text):
    found = set()
    normalized = f" {re.sub(r'[^a-z0-9 ]+', ' ', str(text).lower())} "
    if re.search(r"\bu\s*s\b", normalized):
        found.add("united states")
    for alias, canonical in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            found.add(canonical)
    return frozenset(found)


def has_relation_scope(text):
    return text_has_any_term(text, RELATION_SCOPE_TERMS)


def has_location_conflict(k_title, p_title):
    k_districts = extract_us_districts(k_title)
    p_districts = extract_us_districts(p_title)
    if k_districts and p_districts and k_districts.isdisjoint(p_districts):
        return True

    k_states = extract_us_states(k_title)
    p_states = extract_us_states(p_title)
    if k_states and p_states and k_states.isdisjoint(p_states):
        return True
    shared_states = k_states.intersection(p_states)

    k_countries = extract_countries(k_title)
    p_countries = extract_countries(p_title)
    if k_states and not shared_states and p_countries.difference({"united states"}):
        return True
    if p_states and not shared_states and k_countries.difference({"united states"}):
        return True
    if k_countries and p_countries and k_countries.isdisjoint(p_countries):
        return True
    if (
        len(k_countries) >= 2
        and len(p_countries) >= 2
        and k_countries != p_countries
        and has_relation_scope(k_title)
        and has_relation_scope(p_title)
    ):
        return True
    return False


def normalize_scope_text(value):
    return re.sub(r"[^a-z0-9 ]+", " ", str(value).lower()).strip()


@lru_cache(maxsize=None)
def scope_granularity_markers(event_id, title, category, sub_category):
    scope = " ".join(str(value) for value in (event_id, title, category, sub_category))
    normalized = f" {normalize_scope_text(scope)} "
    match_level = (" match ", " game ", " map ", " maps ", " bo1 ", " bo3 ", " bo5 ", " vs ")
    event_level = (" tournament ", " championship ", " champion ", " major ", " cup ", " open ", " season winner ", " season ")
    is_match = any(term in normalized for term in match_level)
    is_event = any(term in normalized for term in event_level)
    compact = re.sub(r"[^a-z0-9]+", "", scope.lower())
    return is_match or "map" in compact, is_event


def has_scope_granularity_conflict(k_id, p_id, k_data, p_data):
    k_match, k_event = scope_granularity_markers(
        str(k_id),
        str(k_data.get("title", "")),
        str(k_data.get("category", "")),
        str(k_data.get("sub_category", "")),
    )
    p_match, p_event = scope_granularity_markers(
        str(p_id),
        str(p_data.get("title", "")),
        str(p_data.get("category", "")),
        str(p_data.get("sub_category", "")),
    )

    return (k_match and p_event and not k_event) or (p_match and k_event and not p_event)


def hard_reject_reason(k_id, p_id, k_data, p_data, k_sigs, p_sigs, common_keywords=None):
    k_sig = k_sigs.get(k_id, set())
    p_sig = p_sigs.get(p_id, set())
    if k_sig and p_sig and check_incompatibility(k_sig, p_sig):
        return "signature_incompatibility"

    k_title = k_data.get("title", "")
    p_title = p_data.get("title", "")
    if common_keywords is None:
        common_keywords = set(k_data.get("keywords", [])).intersection(p_data.get("keywords", []))
    if is_broad_specific_single_keyword(k_title, p_title, common_keywords):
        return "broad_specific_scope"
    if has_scope_granularity_conflict(k_id, p_id, k_data, p_data):
        return "scope_granularity_conflict"
    if has_sports_domain_conflict(k_data, p_data):
        return "sports_domain_conflict"
    if has_date_scope_conflict(k_title, p_title):
        return "date_scope_conflict"
    if has_title_scope_conflict(k_title, p_title):
        return "title_scope_conflict"
    if has_direction_conflict(k_title, p_title):
        return "direction_conflict"
    if has_numeric_threshold_conflict(k_title, p_title):
        return "numeric_threshold_conflict"
    if has_location_conflict(k_title, p_title):
        return "location_conflict"
    return ""


def is_hard_rejected_pair(k_id, p_id, k_data, p_data, k_sigs, p_sigs):
    return bool(hard_reject_reason(k_id, p_id, k_data, p_data, k_sigs, p_sigs))

def load_category_mapping():
    try:
        with open(CATEGORY_MAPPING_FILE, 'r') as f:
            return normalize_category_mapping(json.load(f))
    except FileNotFoundError:
        print("⚠️ category_mapping.json not found. Clustering will use category fallbacks only.")
        return {}

def build_category_groups(event_keywords_index):
    kalshi_events_by_cat = defaultdict(list)
    poly_events_by_cat = defaultdict(list)
    poly_category_keys_by_event = {}

    for event_id, event_data in event_keywords_index.items():
        platform = event_data.get('platform', '')
        category = event_data.get('category', '')
        sub_category = event_data.get('sub_category', '')

        if platform == 'kalshi':
            cat_key = kalshi_category_key(category, sub_category)
            kalshi_events_by_cat[cat_key].append(event_id)
        elif platform == 'polymarket':
            category_keys = polymarket_category_keys(category, sub_category)
            poly_category_keys_by_event[event_id] = category_keys
            for cat_key in category_keys:
                poly_events_by_cat[cat_key].append(event_id)

    return kalshi_events_by_cat, poly_events_by_cat, poly_category_keys_by_event

def common_keywords_for_pair(event_keywords_index, k_id, p_id):
    k_keywords = set(event_keywords_index.get(k_id, {}).get('keywords', []))
    p_keywords = set(event_keywords_index.get(p_id, {}).get('keywords', []))
    return k_keywords, p_keywords, k_keywords.intersection(p_keywords)

def build_keyword_sets(event_keywords_index):
    return {
        event_id: set(event_data.get('keywords', []))
        for event_id, event_data in event_keywords_index.items()
    }

def build_polymarket_keyword_index(event_keywords_index, keyword_sets):
    poly_events_by_keyword = defaultdict(set)
    for event_id, event_data in event_keywords_index.items():
        if event_data.get('platform', '') != 'polymarket':
            continue
        for keyword in keyword_sets.get(event_id, set()):
            poly_events_by_keyword[keyword].add(event_id)
    return poly_events_by_keyword

TITLE_STOP_WORDS = {
    "a", "an", "and", "any", "are", "at", "be", "before", "by", "does",
    "end", "for", "from", "have", "has", "how", "if", "in", "is", "many",
    "much", "next", "of", "on", "or", "the", "this", "to", "what", "when",
    "where", "which", "who", "will", "with",
}

GENERIC_SINGLE_KEYWORDS = {
    "announced", "arrested", "attend", "before", "confirmed", "countries",
    "country", "election", "file", "files", "game", "law", "market",
    "house", "order", "orders", "president", "presidential", "price",
    "primary", "seat", "seats", "season", "senate", "winner", "year",
}

SPORT_DOMAIN_FAMILIES = {
    "baseball": {"baseball", "mlb", "wbc", "world baseball classic", "world series"},
    "basketball": {"basketball", "nba", "ncaa basketball", "ncaa cbb", "cbb", "college basketball"},
    "boxing": {"boxing"},
    "chess": {"chess"},
    "cricket": {"cricket", "ipl"},
    "darts": {"darts"},
    "esports": {"esports", "video games", "valorant", "league of legends", "counter-strike", "cs2", "dota"},
    "football": {"football", "nfl", "ncaa football", "cfb", "super bowl"},
    "golf": {"golf", "pga", "pga tour", "the masters"},
    "hockey": {"hockey", "nhl", "khl", "stanley cup"},
    "lacrosse": {"lacrosse"},
    "mma": {"mma", "ufc"},
    "motorsport": {"motorsport", "formula 1", "f1", "nascar", "indycar"},
    "rugby": {"rugby", "rugby union", "rugby league"},
    "soccer": {"soccer", "fifa", "fifa world cup", "epl", "mls", "uefa", "champions league"},
    "tennis": {"tennis", "atp", "wta", "roland garros", "wimbledon"},
}

def normalized_title_tokens(title):
    text = re.sub(r"[^a-z0-9-]+", " ", str(title).lower())
    tokens = set()
    for token in text.split():
        if not token or token in TITLE_STOP_WORDS:
            continue
        ordinal_match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", token)
        normalized = ordinal_match.group(1) if ordinal_match else token
        if len(normalized) > 1 or normalized.isdigit():
            tokens.add(normalized)
    return tokens

def title_overlap_metrics(k_title, p_title):
    k_tokens = normalized_title_tokens(k_title)
    p_tokens = normalized_title_tokens(p_title)
    if not k_tokens or not p_tokens:
        return 0.0, 0.0, k_tokens, p_tokens
    common = k_tokens.intersection(p_tokens)
    min_overlap = len(common) / min(len(k_tokens), len(p_tokens))
    jaccard = len(common) / len(k_tokens.union(p_tokens))
    return min_overlap, jaccard, k_tokens, p_tokens

def is_distinctive_keyword(keyword):
    text = str(keyword).strip().lower()
    if text.startswith("year:"):
        return False
    if text in GENERIC_SINGLE_KEYWORDS:
        return False
    return " " in text or "-" in text or any(char.isdigit() for char in text) or len(text) >= 9


def evidence_keywords(common_keywords):
    evidence = set()
    for keyword in common_keywords:
        text = str(keyword).strip().lower()
        if text.startswith("year:"):
            continue
        if is_distinctive_keyword(text) or text not in GENERIC_SINGLE_KEYWORDS:
            evidence.add(keyword)
    return evidence


def evidence_keyword_key(keyword):
    text = str(keyword).strip().lower()
    if text.endswith(("ss", "us", "is", "as")):
        return text
    if text.endswith("ies") and len(text) > 5:
        return text[:-3] + "y"
    if text.endswith("s") and len(text) > 4:
        return text[:-1]
    return text


def single_keyword_has_category_support(keyword, k_data, p_data):
    text = str(keyword).strip().lower()
    if not text:
        return False
    k_terms = kalshi_category_terms(k_data.get("category", ""), k_data.get("sub_category", ""))
    p_terms = polymarket_category_keys(p_data.get("category", ""), p_data.get("sub_category", ""))
    if text in k_terms or text in p_terms:
        return True
    category_text = " ".join(
        str(value).lower()
        for value in (
            k_data.get("category", ""),
            k_data.get("sub_category", ""),
            p_data.get("category", ""),
            p_data.get("sub_category", ""),
        )
    )
    return text in category_text


QUALITY_TIER_ORDER = {"weak": 0, "medium": 1, "strong": 2, "exact": 3}


def candidate_quality(
    common_keywords,
    k_data,
    p_data,
    is_category_candidate,
    is_threshold_candidate,
    is_textual_fallback,
    category_resolution_mode="legacy",
):
    min_overlap, jaccard, k_tokens, p_tokens = title_overlap_metrics(
        k_data.get("title", ""),
        p_data.get("title", ""),
    )
    evidence = evidence_keywords(common_keywords)
    canonical_evidence = {evidence_keyword_key(keyword) for keyword in evidence}
    distinctive_count = sum(1 for keyword in canonical_evidence if is_distinctive_keyword(keyword))

    score = 0
    score += len(canonical_evidence) * 70
    score += distinctive_count * 35
    score += int(min_overlap * 100)
    score += int(jaccard * 160)
    if is_category_candidate:
        score += 15 if category_resolution_mode == "broad_fallback" else 45
    if is_threshold_candidate:
        score += 35
    if is_textual_fallback:
        score += 55
    if k_tokens and k_tokens == p_tokens:
        score += 140
    if len(canonical_evidence) <= 1 and distinctive_count == 0:
        score -= 80
    if jaccard < 0.22 and min_overlap < 0.50:
        score -= 60

    if k_tokens and k_tokens == p_tokens:
        tier = "exact"
    elif jaccard >= 0.75 and min_overlap >= 0.85:
        tier = "exact"
    elif score >= 430:
        tier = "strong"
    elif score >= 270:
        tier = "medium"
    else:
        tier = "weak"

    if (
        tier == "weak"
        and len(canonical_evidence) == 1
        and is_category_candidate
        and category_resolution_mode != "broad_fallback"
    ):
        keyword = next(iter(canonical_evidence))
        category_supported = single_keyword_has_category_support(keyword, k_data, p_data)
        tight_titles = min_overlap >= 0.65 and jaccard >= 0.33
        if min_overlap >= 0.50 and jaccard >= 0.25 and (category_supported or tight_titles):
            score = max(score, 270)
            tier = "medium"

    return {
        "quality_score": score,
        "quality_tier": tier,
        "evidence_keywords": sorted(canonical_evidence),
        "evidence_keyword_count": len(canonical_evidence),
        "distinctive_keyword_count": distinctive_count,
    }

@lru_cache(maxsize=None)
def sports_domain_families(platform, category, sub_category, title=""):
    if platform == "kalshi":
        terms = kalshi_category_terms(category, sub_category)
    else:
        terms = polymarket_category_keys(category, sub_category)
    normalized_title = f" {normalize_scope_text(title)} "
    return frozenset({
        family for family, aliases in SPORT_DOMAIN_FAMILIES.items()
        if terms.intersection(aliases) or any(
            re.search(rf"\b{re.escape(alias)}\b", normalized_title)
            for alias in aliases
        )
    })


def has_sports_domain_conflict(k_data, p_data):
    k_domains = sports_domain_families(
        "kalshi",
        str(k_data.get("category", "")),
        str(k_data.get("sub_category", "")),
        str(k_data.get("title", "")),
    )
    p_domains = sports_domain_families(
        "polymarket",
        str(p_data.get("category", "")),
        str(p_data.get("sub_category", "")),
        str(p_data.get("title", "")),
    )
    return bool(k_domains and p_domains and k_domains.isdisjoint(p_domains))


def effective_scope_keywords(common_keywords):
    return {
        evidence_keyword_key(keyword)
        for keyword in common_keywords
        if str(keyword).strip() and not str(keyword).strip().lower().startswith("year:")
    }


def is_broad_specific_single_keyword(k_title, p_title, common_keywords):
    effective_keywords = effective_scope_keywords(common_keywords)
    if len(effective_keywords) != 1:
        return False
    keyword = next(iter(effective_keywords))
    if is_distinctive_keyword(keyword):
        return False
    k_lower = str(k_title).lower().strip()
    p_lower = str(p_title).lower().strip()

    def is_broad_question(title):
        if title.startswith("who will"):
            return True
        return bool(re.match(
            r"^which\s+(?:people|persons?|candidates?|countries|companies|artists?|teams?|parties|states)\b",
            title,
        ))

    return is_broad_question(k_lower) != is_broad_question(p_lower)

def has_strong_textual_match(k_data, p_data, k_keywords, p_keywords, common_keywords):
    if not common_keywords:
        return False
    if has_sports_domain_conflict(k_data, p_data):
        return False

    k_title = k_data.get("title", "")
    p_title = p_data.get("title", "")
    min_overlap, jaccard, k_tokens, p_tokens = title_overlap_metrics(k_title, p_title)
    if not k_tokens or not p_tokens:
        return False
    if k_tokens == p_tokens:
        return True
    if min_overlap >= 0.85 and jaccard >= 0.75:
        return True
    if is_broad_specific_single_keyword(k_title, p_title, common_keywords):
        return False

    canonical_evidence = {
        evidence_keyword_key(keyword)
        for keyword in evidence_keywords(common_keywords)
    }
    common_count = len(canonical_evidence)
    has_distinctive = any(is_distinctive_keyword(keyword) for keyword in canonical_evidence)

    if common_count >= 4 and min_overlap >= 0.45:
        return True
    if common_count >= 3 and min_overlap >= 0.55:
        return True
    if common_count >= 2 and min_overlap >= 0.70:
        return True
    if common_count >= 2 and has_distinctive and min_overlap >= 0.55:
        return True
    if common_count == 1 and has_distinctive and min_overlap >= 0.75 and jaccard >= 0.50:
        return True
    return False

def should_keep_candidate_pair(k_keywords, p_keywords, common_keywords, k_data=None, p_data=None):
    min_len = min(len(k_keywords), len(p_keywords))
    if min_len == 0:
        return False
    meaningful_common = evidence_keywords(common_keywords)
    canonical_evidence = {evidence_keyword_key(keyword) for keyword in meaningful_common}
    common_count = len(canonical_evidence)
    overlap_ratio = common_count / min_len
    if common_count >= 3:
        if k_data and p_data:
            min_overlap, jaccard, _, _ = title_overlap_metrics(
                k_data.get("title", ""),
                p_data.get("title", ""),
            )
            has_distinctive = any(is_distinctive_keyword(keyword) for keyword in canonical_evidence)
            if has_distinctive:
                return min_overlap >= 0.35 or jaccard >= 0.22 or overlap_ratio >= 0.45
            return min_overlap >= 0.60 or jaccard >= 0.30 or overlap_ratio >= 0.55
        return True
    if common_count >= 2:
        if overlap_ratio >= 0.45 or any(is_distinctive_keyword(keyword) for keyword in canonical_evidence):
            return True
        if k_data and p_data:
            min_overlap, jaccard, _, _ = title_overlap_metrics(
                k_data.get("title", ""),
                p_data.get("title", ""),
            )
            return min_overlap >= 0.55 and jaccard >= 0.30
        return False
    if common_count == 1:
        keyword = next(iter(canonical_evidence))
        if not is_distinctive_keyword(keyword):
            if not k_data or not p_data:
                return False
        if k_data and p_data:
            min_overlap, jaccard, _, _ = title_overlap_metrics(
                k_data.get("title", ""),
                p_data.get("title", ""),
            )
            if is_distinctive_keyword(keyword) and min_overlap >= 0.50 and jaccard >= 0.25:
                return True
            if min_overlap >= 0.65 and jaccard >= 0.30:
                return True
            return (
                min_overlap >= 0.50
                and jaccard >= 0.25
                and single_keyword_has_category_support(keyword, k_data, p_data)
            )
        return overlap_ratio >= 0.8
    return False


def has_broad_fallback_evidence(common_keywords, k_data, p_data):
    evidence = {evidence_keyword_key(keyword) for keyword in evidence_keywords(common_keywords)}
    if len(evidence) < 2:
        return False

    distinctive_count = sum(1 for keyword in evidence if is_distinctive_keyword(keyword))
    min_overlap, jaccard, _, _ = title_overlap_metrics(
        k_data.get("title", ""),
        p_data.get("title", ""),
    )
    if len(evidence) >= 3:
        return min_overlap >= 0.35 or jaccard >= 0.22
    if distinctive_count:
        return min_overlap >= 0.45 or jaccard >= 0.25
    return min_overlap >= 0.60 or jaccard >= 0.30


def candidate_reason(
    is_category_candidate,
    is_threshold_candidate,
    is_textual_fallback,
    common_keywords,
    k_data,
    p_data,
    category_resolution_mode="legacy",
    category_match_keys=None,
):
    if is_category_candidate and is_threshold_candidate:
        path = (
            "broad_category_keywords"
            if category_resolution_mode == "broad_fallback"
            else "category_keywords"
        )
    elif is_textual_fallback:
        path = "strong_textual_fallback"
    else:
        path = "not_kept"

    min_overlap, jaccard, _, _ = title_overlap_metrics(k_data.get("title", ""), p_data.get("title", ""))
    quality = candidate_quality(
        common_keywords,
        k_data,
        p_data,
        is_category_candidate,
        is_threshold_candidate,
        is_textual_fallback,
        category_resolution_mode,
    )
    return {
        "path": path,
        "category_match": bool(is_category_candidate),
        "category_resolution_mode": category_resolution_mode,
        "category_match_keys": sorted(category_match_keys or []),
        "keyword_threshold": bool(is_threshold_candidate),
        "textual_fallback": bool(is_textual_fallback),
        "common_keyword_count": len(common_keywords),
        "title_min_overlap": round(min_overlap, 3),
        "title_jaccard": round(jaccard, 3),
        **quality,
    }

def build_pairs_to_compare(event_keywords_index, category_mapping, k_sigs=None, p_sigs=None):
    k_sigs = k_sigs or {}
    p_sigs = p_sigs or {}
    kalshi_events_by_cat, poly_events_by_cat, poly_category_keys_by_event = build_category_groups(
        event_keywords_index
    )
    keyword_sets = build_keyword_sets(event_keywords_index)
    poly_events_by_keyword = build_polymarket_keyword_index(event_keywords_index, keyword_sets)

    pairs_to_compare = []
    seen_pairs = set()

    for kalshi_cat, kalshi_events in kalshi_events_by_cat.items():
        category_resolution = resolve_polymarket_keys_for_kalshi(
            kalshi_cat,
            category_mapping,
            available_poly_keys=poly_events_by_cat,
        )
        polymarket_keys = set(category_resolution.keys)
        pm_events = set()
        for pm_key in polymarket_keys:
            pm_events.update(poly_events_by_cat.get(pm_key, []))

        for k_id in kalshi_events:
            k_data = event_keywords_index.get(k_id, {})
            k_keywords = keyword_sets.get(k_id, set())
            if not k_keywords:
                continue

            keyword_matched_pm_events = set()
            for keyword in k_keywords:
                keyword_matched_pm_events.update(poly_events_by_keyword.get(keyword, set()))

            for p_id in keyword_matched_pm_events:
                pair_key = (k_id, p_id)
                if pair_key in seen_pairs:
                    continue

                p_data = event_keywords_index.get(p_id, {})
                p_keywords = keyword_sets.get(p_id, set())
                common_keywords = k_keywords.intersection(p_keywords)
                p_category_keys = poly_category_keys_by_event.get(p_id, set())
                category_match_keys = polymarket_keys.intersection(p_category_keys)
                is_category_candidate = bool(category_match_keys) and p_id in pm_events
                passes_keyword_threshold = should_keep_candidate_pair(
                    k_keywords,
                    p_keywords,
                    common_keywords,
                    k_data,
                    p_data,
                )
                is_threshold_candidate = passes_keyword_threshold
                if category_resolution.is_broad_fallback and is_category_candidate:
                    is_threshold_candidate = passes_keyword_threshold and has_broad_fallback_evidence(
                        common_keywords,
                        k_data,
                        p_data,
                    )
                is_textual_fallback = has_strong_textual_match(
                    k_data,
                    p_data,
                    k_keywords,
                    p_keywords,
                    common_keywords,
                )

                if not (is_category_candidate and is_threshold_candidate) and not is_textual_fallback:
                    continue
                seen_pairs.add(pair_key)

                reject_reason = hard_reject_reason(
                    k_id,
                    p_id,
                    k_data,
                    p_data,
                    k_sigs,
                    p_sigs,
                    common_keywords,
                )
                if reject_reason:
                    continue

                pairs_to_compare.append({
                    'kalshi_id': k_id,
                    'poly_id': p_id,
                    'common_keywords': sorted(common_keywords),
                    'score': len({
                        evidence_keyword_key(keyword)
                        for keyword in evidence_keywords(common_keywords)
                    }),
                    'candidate_reason': candidate_reason(
                        is_category_candidate,
                        is_threshold_candidate,
                        is_textual_fallback,
                        common_keywords,
                        k_data,
                        p_data,
                        category_resolution.mode,
                        category_match_keys,
                    )
                })

    return pairs_to_compare

def parse_signature_tokens(raw_tokens):
    try:
        import ast
        return set(ast.literal_eval(raw_tokens))
    except (ValueError, SyntaxError):
        return set()

def load_existing_signatures(force_full=False):
    k_sigs = {}
    p_sigs = {}
    existing_rows = []
    existing_event_ids = set()

    if not force_full and os.path.exists(SIGNATURES_CSV):
        print(f"Loading cached signatures from {SIGNATURES_CSV}...")
        df_sigs = pd.read_csv(SIGNATURES_CSV, dtype=str)

        for _, row in df_sigs.iterrows():
            tokens = parse_signature_tokens(row.get("signature_tokens", "[]"))
            eid = str(row["event_id"])
            platform = row["platform"]
            existing_event_ids.add(eid)

            if platform == "Kalshi":
                k_sigs[eid] = tokens
            else:
                p_sigs[eid] = tokens

            existing_rows.append(row.to_dict())

        print(f"Loaded {len(k_sigs)} Kalshi and {len(p_sigs)} Polymarket signatures from cache.")

    return k_sigs, p_sigs, existing_rows, existing_event_ids

def source_new_event_ids(events_k, events_p):
    source_new_ids = set()
    for _, row in events_k.iterrows():
        if row.get("SyncStatus") == "New": source_new_ids.add(str(row["Event ID"]))
    for _, row in events_p.iterrows():
        if row.get("SyncStatus") == "New": source_new_ids.add(str(row["Event ID"]))
    return source_new_ids

def extract_missing_signatures(events, platform, existing_event_ids, signatures):
    new_rows = []
    platform_label = "Kalshi" if platform == "kalshi" else "Polymarket"

    for _, row in events.iterrows():
        eid = str(row["Event ID"])
        if eid not in existing_event_ids:
            title = row["Market Title"]
            cat = normalize_label(row["Category"])

            load_spacy()
            sig, details = extract_signature_details(title, cat)
            signatures[eid] = sig

            details["event_id"] = eid
            details["platform"] = platform_label
            details["SyncStatus"] = "New"
            new_rows.append(details)

    return new_rows

def refresh_signature_statuses(existing_rows, source_new_ids):
    updated_any = False
    for r in existing_rows:
        eid = str(r["event_id"])
        if eid in source_new_ids:
            if r.get("SyncStatus") != "New":
                r["SyncStatus"] = "New"
                updated_any = True
        else:
            if r.get("SyncStatus") != "Existing":
                r["SyncStatus"] = "Existing"
                updated_any = True
    return updated_any

def save_signatures_if_needed(existing_rows, new_rows, updated_any):
    if new_rows or updated_any:
        print(f"Extracted {len(new_rows)} new signatures and updated {updated_any} statuses.")
        all_rows = existing_rows + new_rows
        print(f"Exporting {len(all_rows)} total signature details to {SIGNATURES_CSV}...")
        pd.DataFrame(all_rows).to_csv(SIGNATURES_CSV, index=False)
    else:
        print("No new lines or status changes to process.")

def load_event_keywords_index():
    if os.path.exists(EVENT_KEYWORDS_INDEX_FILE):
        with open(EVENT_KEYWORDS_INDEX_FILE, 'r') as f:
            event_keywords_index = json.load(f)
        print(f"Loaded {len(event_keywords_index)} events from keyword index.")
        return event_keywords_index
    else:
        raise FileNotFoundError(f"{EVENT_KEYWORDS_INDEX_FILE} not found. Run event_keyword_indexer.py first.")

def build_candidate_clusters(pairs_to_compare, event_keywords_index):
    candidate_clusters = []
    for idx, pair in enumerate(sorted(pairs_to_compare, key=candidate_order_key), start=1):
        k_id = pair['kalshi_id']
        p_id = pair['poly_id']

        k_data = event_keywords_index.get(k_id, {})
        p_data = event_keywords_index.get(p_id, {})

        candidate_clusters.append({
            'cluster_id': idx,
            'kalshi': {
                'event_id': k_id,
                'title': k_data.get('title', ''),
                'category': k_data.get('category', ''),
                'sub_category': k_data.get('sub_category', ''),
                'keywords': k_data.get('keywords', [])
            },
            'polymarket': {
                'event_id': p_id,
                'title': p_data.get('title', ''),
                'category': p_data.get('category', ''),
                'sub_category': p_data.get('sub_category', ''),
                'keywords': p_data.get('keywords', [])
            },
            'common_keywords': pair['common_keywords'],
            'match_score': pair['score'],
            'candidate_reason': pair.get('candidate_reason', {}),
            'llm_verified': None
        })
    return candidate_clusters


def optimize_candidate_pairs(pairs_to_compare, event_keywords_index, max_candidate_rank=2):
    profiles = build_event_profiles(event_keywords_index)
    statistics = build_keyword_statistics(event_keywords_index)
    annotated = annotate_pairs(pairs_to_compare, profiles, statistics)
    kept, rejected = prune_pairs(annotated, max_reciprocal_rank=max_candidate_rank)
    report = {
        "policy_version": CLUSTERING_POLICY_VERSION,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_fingerprint": profiles_fingerprint(profiles),
        "input_pairs": len(pairs_to_compare),
        "output_pairs": len(kept),
        "reduction_count": len(pairs_to_compare) - len(kept),
        "reduction_ratio": round(
            (len(pairs_to_compare) - len(kept)) / max(1, len(pairs_to_compare)),
            6,
        ),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "components_before": connected_component_stats(annotated)[:20],
        "components_after": connected_component_stats(kept)[:20],
        "max_candidate_rank": max_candidate_rank,
    }
    return kept, profiles, report


def write_structured_clustering_artifacts(profiles, report):
    with open(EVENT_PROFILES_FILE, "w") as handle:
        json.dump(
            {event_id: profile.to_dict() for event_id, profile in sorted(profiles.items())},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with open(CLUSTERING_RUN_REPORT_FILE, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"Saved {len(profiles)} structured profiles to {EVENT_PROFILES_FILE}")
    print(f"Saved clustering diagnostics to {CLUSTERING_RUN_REPORT_FILE}")

def write_candidate_clusters(candidate_clusters):
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(candidate_clusters, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(candidate_clusters)} candidate clusters to {OUTPUT_JSON}")

def build_exclusive_clusters(pairs_to_compare, event_keywords_index):
    used_kalshi = set()
    used_poly = set()
    exclusive_clusters = []

    for pair in sorted(pairs_to_compare, key=candidate_order_key):
        k_id = pair['kalshi_id']
        p_id = pair['poly_id']

        if k_id in used_kalshi or p_id in used_poly:
            continue

        k_data = event_keywords_index.get(k_id, {})
        p_data = event_keywords_index.get(p_id, {})

        cluster = {
            'cluster_id': len(exclusive_clusters) + 1,
            'kalshi': {
                'event_id': k_id,
                'title': k_data.get('title', ''),
                'category': k_data.get('category', ''),
                'sub_category': k_data.get('sub_category', ''),
                'keywords': k_data.get('keywords', [])
            },
            'polymarket': {
                'event_id': p_id,
                'title': p_data.get('title', ''),
                'category': p_data.get('category', ''),
                'sub_category': p_data.get('sub_category', ''),
                'keywords': p_data.get('keywords', [])
            },
            'common_keywords': pair['common_keywords'],
            'match_score': pair['score'],
            'candidate_reason': pair.get('candidate_reason', {})
        }

        exclusive_clusters.append(cluster)
        used_kalshi.add(k_id)
        used_poly.add(p_id)

    return exclusive_clusters, used_kalshi, used_poly


def candidate_order_key(pair):
    reason = pair.get("candidate_reason", {})
    semantic = reason.get("semantic", {})
    return (
        -float(semantic.get("semantic_score", 0)),
        -int(reason.get("quality_score", 0)),
        -int(pair.get("score", 0)),
        -float(reason.get("title_min_overlap", 0)),
        -float(reason.get("title_jaccard", 0)),
        str(pair.get("kalshi_id", "")),
        str(pair.get("poly_id", "")),
    )

def write_exclusive_clusters(exclusive_clusters, used_kalshi, used_poly):
    print(f"Created {len(exclusive_clusters)} exclusive clusters.")
    print(f"  Kalshi events used: {len(used_kalshi)}")
    print(f"  Polymarket events used: {len(used_poly)}")

    with open(EXCLUSIVE_CLUSTERS_FILE, 'w') as f:
        json.dump(exclusive_clusters, f, indent=2, ensure_ascii=False)
    print(f"Saved exclusive clusters to {EXCLUSIVE_CLUSTERS_FILE}")

def build_id_to_details(df_k, df_p):
    id_to_details = {}

    for _, row in df_k.iterrows():
        eid = str(row["Event ID"])
        id_to_details[eid] = {
            "event_id": eid,
            "original_text": row["Market Title"],
            "category": f"{normalize_label(row['Category'])} / {normalize_label(row.get('Sub-Category', ''))}",
            "platform": "kalshi",
            "description": str(row.get("Description", "")),
            "bet_rules": str(row.get("Bet_Rules", "")),
            "event_description": str(row.get("Event_Description", ""))
        }

    for _, row in df_p.iterrows():
        eid = str(row["Event ID"])
        id_to_details[eid] = {
            "event_id": eid,
            "original_text": row["Market Title"],
            "category": normalize_label(row["Category"]),
            "platform": "polymarket",
            "description": str(row.get("Description", "")),
            "bet_rules": str(row.get("Bet_Rules", "")),
            "event_description": str(row.get("Event_Description", ""))
        }
    return id_to_details

def build_graph_data(pairs_to_compare, id_to_details):
    graph_data = {"nodes": {}, "edges": []}

    for pair in pairs_to_compare:
        k_id = pair["kalshi_id"]
        p_id = pair["poly_id"]
        if k_id in id_to_details:
            graph_data["nodes"][k_id] = id_to_details[k_id]
        if p_id in id_to_details:
            graph_data["nodes"][p_id] = id_to_details[p_id]
        graph_data["edges"].append({
            "source": k_id,
            "target": p_id,
            "weight": pair["score"]
        })
    return graph_data

def write_graph_data(graph_data):
    GRAPH_FILE = os.path.join(DATA_CLUSTERS_DIR, "graph_data.json")
    print(f"Saving {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges to {GRAPH_FILE}...")
    with open(GRAPH_FILE, "w") as f:
        json.dump(graph_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Force re-extraction of all signatures")
    parser.add_argument("--legacy-clustering", action="store_true", help="Write legacy candidates without v2 pruning")
    parser.add_argument("--max-candidate-rank", type=int, default=2)
    args = parser.parse_args()

    print("Loading Data...")
    df_k = pd.read_csv(KALSHI_FILE)
    df_p = pd.read_csv(POLY_FILE)
    events_k = df_k.drop_duplicates(subset=["Event ID"])
    events_p = df_p.drop_duplicates(subset=["Event ID"])
    print(f"Unique Events: Kalshi={len(events_k)} | Polymarket={len(events_p)}")

    print("Step 1/5: Load or refresh event signatures...")
    k_sigs, p_sigs, existing_rows, existing_event_ids = load_existing_signatures(force_full=args.full)
    source_new_ids = source_new_event_ids(events_k, events_p)
    new_rows = []
    new_rows.extend(extract_missing_signatures(events_k, "kalshi", existing_event_ids, k_sigs))
    new_rows.extend(extract_missing_signatures(events_p, "polymarket", existing_event_ids, p_sigs))
    updated_any = refresh_signature_statuses(existing_rows, source_new_ids)
    save_signatures_if_needed(existing_rows, new_rows, updated_any)

    print("Step 2/5: Load event keyword index...")
    event_keywords_index = load_event_keywords_index()
    category_mapping = load_category_mapping()

    print("Step 3/6: Generate candidate pairs...")
    pairs_to_compare = build_pairs_to_compare(event_keywords_index, category_mapping, k_sigs, p_sigs)
    legacy_pair_count = len(pairs_to_compare)
    if args.legacy_clustering:
        profiles = build_event_profiles(event_keywords_index)
        report = {
            "policy_version": "legacy",
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "input_pairs": legacy_pair_count,
            "output_pairs": legacy_pair_count,
            "reduction_count": 0,
            "reduction_ratio": 0.0,
        }
    else:
        pairs_to_compare, profiles, report = optimize_candidate_pairs(
            pairs_to_compare,
            event_keywords_index,
            max_candidate_rank=args.max_candidate_rank,
        )
    pairs_to_compare.sort(key=candidate_order_key)
    print(
        f"Clustering candidates: {legacy_pair_count} legacy -> {len(pairs_to_compare)} v2 "
        f"({report.get('reduction_ratio', 0):.1%} reduction)."
    )

    print("Step 4/6: Write structured profiles and diagnostics...")
    write_structured_clustering_artifacts(profiles, report)

    print("Step 5/6: Write candidate outputs...")
    candidate_clusters = build_candidate_clusters(pairs_to_compare, event_keywords_index)
    write_candidate_clusters(candidate_clusters)

    exclusive_clusters, used_kalshi, used_poly = build_exclusive_clusters(pairs_to_compare, event_keywords_index)
    write_exclusive_clusters(exclusive_clusters, used_kalshi, used_poly)

    print("Step 6/6: Write graph debug output...")
    graph_data = build_graph_data(pairs_to_compare, build_id_to_details(df_k, df_p))
    write_graph_data(graph_data)

    print("Done. Candidate outputs are ready for LLM verification.")

if __name__ == "__main__":
    main()
