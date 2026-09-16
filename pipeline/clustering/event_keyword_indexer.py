"""
Event Keyword Indexer - Associates normalized keywords to each event.

Simple approach:
1. SpaCy extracts named entities (PERSON, ORG, GPE, etc.)
2. Entities become keywords directly (with simple normalization)
3. Remaining tokens: keep only non-stop words
4. Simple alias dictionary for known abbreviations
"""

import pandas as pd
import json
import os
import re
import argparse
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")
DATA_KEYWORDS_DIR = os.path.join(ROOT_DIR, "data", "keywords")
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from category_utils import normalize_label
from keyword_config import (
    add_keyword,
    canonicalize_keyword,
    find_alias_matches,
    find_domain_entity_tokens,
    is_noise_keyword,
    normalize_phrase,
)

# Config
KALSHI_FILE = os.path.join(DATA_MARKETS_DIR, "markets_clean_kalshi.csv")
POLY_FILE = os.path.join(DATA_MARKETS_DIR, "markets_clean_polymarket.csv")
EVENT_KEYWORDS_INDEX_FILE = os.path.join(DATA_KEYWORDS_DIR, "event_keywords_index.json")

# Common words to ignore
STOP_WORDS = {
    'will', 'the', 'a', 'an', 'be', 'is', 'are', 'was', 'were', 'been',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'or', 'and',
    'that', 'this', 'it', 'as', 'if', 'when', 'than', 'but', 'its', 'our',
    'who', 'what', 'where', 'which', 'how', 'why', 'before', 'after',
    'more', 'most', 'some', 'any', 'no', 'not', 'only', 'over', 'under',
    'again', 'there', 'vs', 'vs.', 'versus', 'win', 'winner', 'won', 'wins',
    'next', 'first', 'last', 'new', 'old', 'high', 'low', 'up', 'down',
    'yes', 'hit', 'reach', 'pass', 'end', 'start', 'become', 'visit',
    'leave', 'out', 'get', 'make', 'take', 'have', 'has', 'had', 'do',
    'does', 'did', 'being', 'his', 'her', 'their', 'my', 'your',
    'lifetime', 'invade', 'election', 'president', 'announce', 'release',
    'lose', 'loser', 'lost', 'further', 'then', 'once', 'here', 'all',
    'each', 'few', 'other', 'such', 'into', 'through', 'during', 'above',
    'below', 'between', 'while', 'about', 'against', 'would', 'could',
    'should', 'having', 'doing', 'gets', 'got', 'makes', 'made', 'takes',
    'took', 'begin', 'finish', 'launch', 'hold', 'keep', 'set', 'run',
    'play', 'move', 'live', 'work', 'part', 'come', 'back', 'much', 'many'
}

# Simple aliases for known abbreviations that are safe in every domain.
GLOBAL_ALIASES = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'donald trump': 'trump',
    'donald j trump': 'trump',
    'donald j. trump': 'trump',
    'elon': 'elon musk',
    'musk': 'elon musk',
    'usa': 'united states',
    'u.s.': 'united states',
    'u.s.a.': 'united states',
}

# Sports/team aliases are intentionally context-gated. Tokens like "city",
# "real", "paris", or "sporting" are common English/location words outside
# sports and should not create football keywords in politics/culture markets.
SPORTS_ALIASES = {
    # UCL club normalization
    'psg': 'paris saint-germain',
    'paris sg': 'paris saint-germain',
    'paris saint germain': 'paris saint-germain',
    'paris': 'paris saint-germain',
    'om': 'olympique de marseille',
    'marseille': 'olympique de marseille',
    'olympique marseille': 'olympique de marseille',
    'monaco': 'as monaco',
    'as monaco': 'as monaco',
    'bayern': 'bayern munich',
    'bayern munich': 'bayern munich',
    'fc bayern': 'bayern munich',
    'dortmund': 'borussia dortmund',
    'borussia dortmund': 'borussia dortmund',
    'leverkusen': 'bayer leverkusen',
    'bayer leverkusen': 'bayer leverkusen',
    'eintracht frankfurt': 'eintracht frankfurt',
    'francfort': 'eintracht frankfurt',
    'real madrid': 'real madrid',
    'real': 'real madrid',
    'barcelona': 'fc barcelona',
    'fc barcelona': 'fc barcelona',
    'barca': 'fc barcelona',
    'atletico madrid': 'atletico madrid',
    'atlético madrid': 'atletico madrid',
    'athletic bilbao': 'athletic bilbao',
    'juventus': 'juventus',
    'juventus turin': 'juventus',
    'inter': 'inter milan',
    'inter milan': 'inter milan',
    'napoli': 'naples',
    'naples': 'naples',
    'atalanta': 'atalanta',
    'man city': 'manchester city',
    'manchester city': 'manchester city',
    'city': 'manchester city',
    'man utd': 'manchester united',
    'manchester united': 'manchester united',
    'liverpool': 'liverpool',
    'arsenal': 'arsenal',
    'chelsea': 'chelsea',
    'tottenham': 'tottenham hotspur',
    'spurs': 'tottenham hotspur',
    'newcastle': 'newcastle united',
    'sporting': 'sporting cp',
    'sporting lisbon': 'sporting cp',
    'sporting portugal': 'sporting cp',
    'benfica': 'benfica',
    'ajax': 'ajax',
    'ajax amsterdam': 'ajax',
    'psv': 'psv eindhoven',
    'psv eindhoven': 'psv eindhoven',
    'copenhagen': 'fc copenhagen',
    'kopenghagen': 'fc copenhagen',
    'kopengja': 'fc copenhagen',
    'fc copenhagen': 'fc copenhagen',
    'fc kopenhagen': 'fc copenhagen',
    'kobenhavn': 'fc copenhagen',
    'københavn': 'fc copenhagen',
    'fc københavn': 'fc copenhagen',
    'club bruges': 'club bruges',
    'club brugge': 'club bruges',
    'bruges': 'club bruges',
    'union sg': 'union saint-gilloise',
    'union saint gilloise': 'union saint-gilloise',
    'olympiacos': 'olympiakos',
    'olympiakos': 'olympiakos',
    'galatasaray': 'galatasaray',
    'slavia prague': 'slavia prague',
    'slavia praha': 'slavia prague',
    'bodo glimt': 'bodø/glimt',
    'bodo/glimt': 'bodø/glimt',
    'qarabag': 'qarabağ',
    'qarabag fk': 'qarabağ',
    'kairat': 'kairat almaty',
    'kairat almaty': 'kairat almaty',
    'pafos': 'páfos',
    'paphos': 'páfos',
}

ALIASES = {**GLOBAL_ALIASES, **SPORTS_ALIASES}

SPORTS_CONTEXT_TERMS = {
    "africa cup of nations",
    "basketball",
    "bundesliga",
    "cfb",
    "champions league",
    "club world cup",
    "copa",
    "epl",
    "esports",
    "fifa",
    "football",
    "game",
    "games",
    "hockey",
    "la liga",
    "liga mx",
    "ligue 1",
    "match",
    "mls",
    "nba",
    "ncaa",
    "nfl",
    "nhl",
    "premier league",
    "serie a",
    "soccer",
    "sports",
    "tournament",
    "ucl",
    "ufc",
    "vs",
    "vs.",
    "world cup",
}

# SpaCy model
nlp = None


class SimpleToken:
    def __init__(self, text, index):
        self.text = text
        self.i = index
        self.is_punct = False
        self.is_space = False
        self.is_digit = text.isdigit()
        self.like_num = text.isdigit()
        self.lemma_ = normalize_phrase(text)
        self.is_stop = is_noise_keyword(text)
        self.pos_ = "PROPN" if text[:1].isupper() else "NOUN"


class SimpleDoc(list):
    ents = []
    noun_chunks = []


class RegexNLP:
    """Small fallback tokenizer used when SpaCy is unavailable."""

    def __call__(self, text):
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", text or "")
        return SimpleDoc(SimpleToken(token, idx) for idx, token in enumerate(tokens))


def load_spacy():
    """Load SpaCy model."""
    global nlp
    if nlp is None:
        print("Loading SpaCy model...")
        try:
            import spacy
        except ImportError:
            print("⚠️ SpaCy is not installed. Falling back to regex keyword extraction.")
            nlp = RegexNLP()
            return nlp

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading SpaCy model...")
            import sys
            os.system(f'"{sys.executable}" -m spacy download en_core_web_sm')
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️ SpaCy model unavailable. Falling back to regex keyword extraction.")
                nlp = RegexNLP()
    return nlp


def normalize(text):
    """Simple normalization: lowercase, uniform spaces."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[’']s$", "", text)  # Remove possessives
    text = text.strip("'’")
    text = re.sub(r'\s+', ' ', text)
    return text


def is_sports_context(title, category="", sub_category=""):
    """Return True only when sports/team aliases are safe to apply."""
    text = f" {normalize(title)} "
    if re.search(r"\b(?:vs|vs\.|versus)\b", text):
        return True

    category_text = f"{normalize_label(category)} {normalize_label(sub_category)}".lower()
    if any(term in category_text for term in SPORTS_CONTEXT_TERMS):
        return True

    combined = f"{text} {category_text} "
    return any(f" {term} " in combined for term in SPORTS_CONTEXT_TERMS)


def apply_alias(term, sports_context=False):
    normalized = normalize(term)
    if not normalized:
        return normalized
    configured = canonicalize_keyword(normalized)
    if configured != normalized:
        return configured
    if normalized in GLOBAL_ALIASES:
        return GLOBAL_ALIASES[normalized]
    if sports_context and normalized in SPORTS_ALIASES:
        return SPORTS_ALIASES[normalized]
    return normalized


def clean_entity(ent_text):
    """Clean entity: remove 'Will' at start if present."""
    ent_text = ent_text.strip()
    if ent_text.lower().startswith('will '):
        ent_text = ent_text[5:]
    return ent_text


def add_year_keywords(keywords, title):
    for year in re.findall(r"\b(20[2-3][0-9])\b", title or ""):
        keywords.add(f"year:{year}")


def add_structured_keywords(keywords, title):
    for state, district in re.findall(r"\b([A-Za-z]{2})-(\d{1,2}|AL)\b", title or "", re.I):
        keywords.add(f"district:{state.upper()}-{district.upper()}")
    for group_id in re.findall(r"\bgroup\s+([A-Za-z0-9]+)\b", title or "", re.I):
        if len(group_id) == 1 or group_id.isdigit():
            keywords.add(f"group:{group_id.upper()}")


def add_keyword_with_variants(keywords, value):
    if not value or is_noise_keyword(value):
        return
    keywords.add(value)
    if " " not in value and "-" not in value and value.endswith("s") and len(value) > 4:
        if value.endswith(("ss", "us", "is", "as")):
            return
        if value.endswith("ies") and len(value) > 5:
            singular = value[:-3] + "y"
        else:
            singular = value[:-1]
        if singular and not is_noise_keyword(singular):
            keywords.add(singular)


def extract_keywords_simple(title, nlp_model, category="", sub_category=""):
    """
    Extract keywords from title using simple approach:
    1. SpaCy entities -> keywords
    2. Remaining non-stop tokens -> keywords
    """
    if not title or not isinstance(title, str):
        return []
    
    doc = nlp_model(title)
    sports_context = is_sports_context(title, category, sub_category)
    keywords = set()
    entity_tokens = set()  # Track tokens already in an entity
    add_year_keywords(keywords, title)
    add_structured_keywords(keywords, title)
    keywords.update(find_alias_matches(title))
    
    # 1. Extract named entities
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'LOC', 'NORP']:
            ent_clean = clean_entity(ent.text)
            ent_norm = normalize(ent_clean)
            if len(ent_norm) > 2:
                final = apply_alias(ent_norm, sports_context=sports_context)
                add_keyword_with_variants(keywords, final)
                # Mark entity tokens
                for token in ent:
                    entity_tokens.add(token.i)
                    token_text = normalize(token.text)
                    if token_text in STOP_WORDS or is_noise_keyword(token_text) or len(token_text) < 3:
                        continue
                    add_keyword_with_variants(
                        keywords,
                        apply_alias(token_text, sports_context=sports_context),
                    )
    
    # 2. Individual tokens (not in entities, not stop words, not numbers)
    for token in doc:
        if token.i in entity_tokens:
            continue
        if token.is_punct or token.is_space or token.is_digit:
            continue
        if token.like_num:
            continue
        
        token_text = normalize(token.text)
        
        if token_text in STOP_WORDS or is_noise_keyword(token_text) or len(token_text) < 3:
            continue
        
        final = apply_alias(token_text, sports_context=sports_context)
        add_keyword_with_variants(keywords, final)
    
    return sorted(list(keywords))


def extract_keywords_and_entities(title, nlp_model, category="", sub_category=""):
    keywords = set(extract_keywords_simple(title, nlp_model, category, sub_category))
    entities = {"person": set(), "org": set(), "gpe": set(), "loc": set(), "topic": set()}

    for token in find_domain_entity_tokens(title):
        label, value = token.split(":", 1)
        entities.setdefault(label.lower(), set()).add(value)
        add_keyword(keywords, value)

    doc = nlp_model(title or "")
    label_map = {
        "PERSON": "person",
        "ORG": "org",
        "GPE": "gpe",
        "LOC": "loc",
        "NORP": "gpe",
    }
    for ent in getattr(doc, "ents", []):
        key = label_map.get(ent.label_)
        if not key:
            continue
        value = canonicalize_keyword(clean_entity(ent.text))
        if value and not is_noise_keyword(value):
            entities.setdefault(key, set()).add(value)
            keywords.add(value)

    return {
        "keywords": sorted(keywords),
        "entities": {key: sorted(values) for key, values in entities.items()},
    }


def load_existing_index():
    """Load existing event keywords index."""
    if os.path.exists(EVENT_KEYWORDS_INDEX_FILE):
        with open(EVENT_KEYWORDS_INDEX_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_index(index):
    """Save the event keywords index."""
    with open(EVENT_KEYWORDS_INDEX_FILE, 'w') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def index_events(df, platform, nlp_model, existing_index, incremental=False):
    """Index all events from a dataframe."""
    stats = {'processed': 0, 'skipped': 0, 'keywords_found': 0}

    for event_id, event_rows in df.groupby(df['Event ID'].astype(str), sort=True):
        row = event_rows.iloc[0]

        # Skip if already indexed in incremental mode
        if incremental and event_id in existing_index:
            has_new_row = (
                'SyncStatus' in event_rows.columns
                and (event_rows['SyncStatus'] == 'New').any()
            )
            if not has_new_row:
                stats['skipped'] += 1
                continue

        title = row.get('Market Title', '')
        category = normalize_label(row.get('Category', ''))
        sub_category = normalize_label(row.get('Sub-Category', ''))
        keywords = extract_keywords_simple(title, nlp_model, category, sub_category)
        bet_titles = sorted({
            str(value).strip()
            for value in event_rows.get('Bet Title', pd.Series(dtype=str)).tolist()
            if str(value).strip() and str(value).strip().lower() not in {'nan', 'n/a'}
        })

        existing_index[event_id] = {
            'platform': platform,
            'title': title,
            'category': category,
            'sub_category': sub_category,
            'keywords': keywords,
            'bet_titles': bet_titles,
            'bet_title_count': len(bet_titles),
        }

        stats['processed'] += 1
        stats['keywords_found'] += len(keywords)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Index event keywords')
    parser.add_argument('--full', action='store_true', help='Re-index all events')
    args = parser.parse_args()
    
    print("🔄 Starting Event Keyword Indexing...")
    
    nlp_model = load_spacy()
    
    # Load existing index
    existing_index = {} if args.full else load_existing_index()
    print(f"📂 Existing index has {len(existing_index)} events")
    
    # Load event data
    print("📂 Loading event data...")
    df_kalshi = pd.read_csv(KALSHI_FILE, dtype=str)
    df_poly = pd.read_csv(POLY_FILE, dtype=str)
    print(f"   Kalshi: {len(df_kalshi)} events, Polymarket: {len(df_poly)} events")
    
    # Index Kalshi events
    print("\n📊 Indexing Kalshi events...")
    stats_k = index_events(df_kalshi, 'kalshi', nlp_model, existing_index, not args.full)
    print(f"   Processed: {stats_k['processed']}, Skipped: {stats_k['skipped']}")
    print(f"   Keywords found: {stats_k['keywords_found']}")
    
    # Index Polymarket events
    print("\n📊 Indexing Polymarket events...")
    stats_p = index_events(df_poly, 'polymarket', nlp_model, existing_index, not args.full)
    print(f"   Processed: {stats_p['processed']}, Skipped: {stats_p['skipped']}")
    print(f"   Keywords found: {stats_p['keywords_found']}")
    
    # Save index
    print(f"\n💾 Saving index to {EVENT_KEYWORDS_INDEX_FILE}...")
    save_index(existing_index)
    
    # Summary
    total_events = len(existing_index)
    total_keywords = sum(len(e['keywords']) for e in existing_index.values())
    avg_keywords = total_keywords / total_events if total_events > 0 else 0
    
    print("\n" + "=" * 50)
    print("✅ Indexing Complete!")
    print("=" * 50)
    print(f"Total events indexed: {total_events}")
    print(f"Average keywords per event: {avg_keywords:.2f}")
    
    # Show sample
    print("\n📋 Sample entries:\n")
    samples = list(existing_index.items())[:3]
    for event_id, data in samples:
        print(f"  Event: {event_id} ({data['platform']})")
        print(f"  Title: {data['title'][:60]}...")
        print(f"  Keywords: {data['keywords'][:5]}...")
        print()


if __name__ == "__main__":
    main()
