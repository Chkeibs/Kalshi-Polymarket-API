import pandas as pd
import os
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher

# --- CONFIGURATION ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")
POSTPROCESS_DIR = os.path.join(ROOT_DIR, "pipeline", "postprocess")
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")
DATA_MATCHES_DIR = os.path.join(ROOT_DIR, "data", "matches")

sys.path.insert(0, COMMON_DIR)
sys.path.insert(0, POSTPROCESS_DIR)

CONFIRMED_MATCHES = os.path.join(DATA_MATCHES_DIR, "confirmed_matches.csv")
KALSHI_FILE = os.path.join(DATA_MARKETS_DIR, "markets_clean_kalshi.csv")
POLY_FILE = os.path.join(DATA_MARKETS_DIR, "markets_clean_polymarket.csv")
OUTPUT_FILE = os.path.join(DATA_MATCHES_DIR, "confirmed_matches_bets.csv")
BATCH_SIZE = 30
EVENT_BATCH_SIZE = 12
BET_RULES_MAX_CHARS = 260
EVENT_RULES_MAX_CHARS = 180
DATE_TOLERANCE_DAYS = 1


try:
    from pipeline.common.llm_config import safe_call_llm
    from pipeline.postprocess.clean_confirmed_matches import clean_files
except ImportError:
    from llm_config import safe_call_llm
    from clean_confirmed_matches import clean_files

# Load .env manually since python-dotenv isn't installed
def load_env():
    import os
    env_path = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v
load_env()

def load_data():
    print("Loading data...")
    matches = pd.read_csv(CONFIRMED_MATCHES, dtype={"Kalshi_ID": str, "Polymarket_ID": str})
    
    # Use robust cleaning
    matches["Kalshi_ID"] = matches["Kalshi_ID"].apply(clean_id)
    matches["Polymarket_ID"] = matches["Polymarket_ID"].apply(clean_id)
    
    df_k = pd.read_csv(KALSHI_FILE, dtype={"Event ID": str, "Bet ID": str})
    df_p = pd.read_csv(POLY_FILE, dtype={"Event ID": str, "Bet ID": str})
    
    # Ensure dataframe IDs are also clean
    df_k["Event ID"] = df_k["Event ID"].apply(clean_id)
    df_p["Event ID"] = df_p["Event ID"].apply(clean_id)
    df_k["Bet ID"] = df_k["Bet ID"].apply(clean_id)
    df_p["Bet ID"] = df_p["Bet ID"].apply(clean_id)
    
    print(f"Loaded {len(matches)} event matches. {len(df_k)} Kalshi rows, {len(df_p)} Poly rows.")
    k_map = df_k.set_index("Event ID")["Market Title"].to_dict()
    p_map = df_p.set_index("Event ID")["Market Title"].to_dict()
    
    def fill_k(row):
        if pd.isna(row.get("Kalshi_Title")) or str(row.get("Kalshi_Title")).lower() in ["nan", "none", ""]:
            cid = clean_id(row["Kalshi_ID"])
            if ";" in cid: cid = cid.split(";")[0] # Just take first for title lookup
            return k_map.get(cid, "Unknown")
        return row["Kalshi_Title"]
        
    def fill_p(row):
        if pd.isna(row.get("Polymarket_Title")) or str(row.get("Polymarket_Title")).lower() in ["nan", "none", ""]:
            cid = clean_id(row["Polymarket_ID"])
            if ";" in cid: cid = cid.split(";")[0]
            return p_map.get(cid, "Unknown")
        return row["Polymarket_Title"]

    if "Kalshi_Title" not in matches.columns: matches["Kalshi_Title"] = "Unknown"
    if "Polymarket_Title" not in matches.columns: matches["Polymarket_Title"] = "Unknown"

    matches["Kalshi_Title"] = matches.apply(fill_k, axis=1)
    matches["Polymarket_Title"] = matches.apply(fill_p, axis=1)
    
    # Pre-load all titles for fast lookup
    k_event_map = df_k.set_index("Event ID")["Market Title"].to_dict()
    k_bet_map = df_k.set_index("Bet ID")["Bet Title"].to_dict()
    p_event_map = df_p.set_index("Event ID")["Market Title"].to_dict()
    p_bet_map = df_p.set_index("Bet ID")["Bet Title"].to_dict()

    return matches, df_k, df_p, k_event_map, k_bet_map, p_event_map, p_bet_map

def clean_id(val):
    if pd.isna(val): return ""
    s = str(val).strip().replace("*", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s

def get_bets_for_event(df, event_id):
    # event_id might be a semicolon-separated string from a cluster
    if ";" in str(event_id):
        raw_ids = str(event_id).split(";")
        ids = [clean_id(i) for i in raw_ids if i.strip()]
        return df[df["Event ID"].isin(ids)].to_dict(orient="records")
    
    cid = clean_id(event_id)
    return df[df["Event ID"] == cid].to_dict(orient="records")

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def safe_text(value):
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def compact_text(value, max_chars):
    txt = re.sub(r"\s+", " ", safe_text(value))
    if len(txt) <= max_chars:
        return txt
    return txt[: max_chars - 3].rstrip() + "..."


def normalize_text(value):
    return re.sub(r"[^a-z0-9 ]+", " ", safe_text(value).lower()).strip()


def title_similarity(a, b):
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def parse_month_day_year(date_str):
    date_str = safe_text(date_str).replace(".", "")
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def extract_year(text):
    m = re.search(r"\b(20\d{2})\b", safe_text(text))
    if not m:
        return None
    return int(m.group(1))


def extract_day_threshold(text):
    t = safe_text(text).lower()
    for pattern in (
        r"at least\s+(\d+)\s+days?\b",
        r"\b(\d+)\s*\+\s*days?\b",
        r"\b(\d+)\s+days?\s*\+\b",
    ):
        m = re.search(pattern, t)
        if m:
            return int(m.group(1))
    return None


def extract_percent_threshold(text):
    t = safe_text(text).lower()
    for pattern in (
        r"at least\s+(\d+(?:\.\d+)?)\s*%",
        r"\b(\d+(?:\.\d+)?)\s*%\s*\+",
        r"\b(\d+(?:\.\d+)?)\s*\+\s*%",
    ):
        m = re.search(pattern, t)
        if m:
            return float(m.group(1))
    return None


def is_numeric_bucket_title(title):
    """
    Detect compact numeric labels like:
    - "10+ days [Yes]"
    - "30"
    - "14%"
    """
    t = safe_text(title).lower()
    t = re.sub(r"\[(yes|no)\]\s*$", "", t).strip()
    return bool(
        re.match(
            r"^\d+(?:\.\d+)?(?:\+)?(?:\s*(?:day|days|%|pts?|points?|runs?|goals?|votes?))?$",
            t
        )
    )


def extract_deadline_date_for_bet(bet):
    title = safe_text(bet.get("Bet Title"))
    rules = safe_text(bet.get("Bet_Rules")) or safe_text(bet.get("Description"))
    ev_desc = safe_text(bet.get("Event_Description"))
    combined = f"{title} {rules} {ev_desc}"
    month_pattern = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

    m = re.search(
        rf"\b(?:before|by|until|on or before|no later than)\s+({month_pattern}\s+\d{{1,2}},?\s+\d{{4}})\b",
        combined,
        flags=re.IGNORECASE,
    )
    if m:
        parsed = parse_month_day_year(m.group(1))
        if parsed:
            return parsed

    m = re.search(rf"\b({month_pattern}\s+\d{{1,2}},?\s+\d{{4}})\b", title, flags=re.IGNORECASE)
    if m:
        parsed = parse_month_day_year(m.group(1))
        if parsed:
            return parsed

    m = re.search(rf"\b({month_pattern}\s+\d{{1,2}})\b", title, flags=re.IGNORECASE)
    if m:
        yr = extract_year(combined)
        if yr:
            parsed = parse_month_day_year(f"{m.group(1)}, {yr}")
            if parsed:
                return parsed

    return None


def build_bet_text(bet):
    return " ".join([
        safe_text(bet.get("Bet Title")),
        safe_text(bet.get("Bet_Rules")) or safe_text(bet.get("Description")),
        safe_text(bet.get("Event_Description")),
    ]).strip()


def is_hard_conflict(k_bet, p_bet):
    k_text = build_bet_text(k_bet)
    p_text = build_bet_text(p_bet)

    k_day = extract_day_threshold(k_text)
    p_day = extract_day_threshold(p_text)
    if k_day is not None and p_day is not None and k_day != p_day:
        return True

    k_pct = extract_percent_threshold(k_text)
    p_pct = extract_percent_threshold(p_text)
    if k_pct is not None and p_pct is not None and abs(k_pct - p_pct) > 1e-6:
        return True

    k_deadline = extract_deadline_date_for_bet(k_bet)
    p_deadline = extract_deadline_date_for_bet(p_bet)
    if k_deadline and p_deadline and abs((k_deadline - p_deadline).days) > DATE_TOLERANCE_DAYS:
        return True

    return False


def needs_description_double_check(k_bet, p_bet):
    k_text = build_bet_text(k_bet)
    p_text = build_bet_text(p_bet)
    sim = title_similarity(k_bet.get("Bet Title", ""), p_bet.get("Bet Title", ""))

    k_day = extract_day_threshold(k_text)
    p_day = extract_day_threshold(p_text)
    k_pct = extract_percent_threshold(k_text)
    p_pct = extract_percent_threshold(p_text)
    k_deadline = extract_deadline_date_for_bet(k_bet)
    p_deadline = extract_deadline_date_for_bet(p_bet)

    # User request: day-threshold / numeric bucket style should always be double-checked with descriptions.
    if k_day is not None and p_day is not None:
        return True
    if is_numeric_bucket_title(k_bet.get("Bet Title")) or is_numeric_bucket_title(p_bet.get("Bet Title")):
        return True

    # If one side has explicit numeric/date constraints and the other doesn't, verify with detail prompt.
    if (k_day is None) != (p_day is None):
        return True
    if (k_pct is None) != (p_pct is None):
        return True
    if (k_deadline is None) != (p_deadline is None):
        return True

    # If explicit constraints already align, no need to pay for a second call.
    day_aligned = (k_day is not None and p_day is not None and k_day == p_day)
    pct_aligned = (k_pct is not None and p_pct is not None and abs(k_pct - p_pct) <= 1e-6)
    deadline_aligned = (
        k_deadline is not None and
        p_deadline is not None and
        abs((k_deadline - p_deadline).days) <= DATE_TOLERANCE_DAYS
    )
    if day_aligned or pct_aligned or deadline_aligned:
        return False

    # Borderline semantic similarity should be validated with descriptions.
    if sim < 0.76:
        return True

    # Temporal/comparator language is usually where costly false positives happen.
    risk_terms = ("before", "by", "until", "at least", "+", "over", "under", "first", "after")
    combined = f"{k_text.lower()} {p_text.lower()}"
    if any(term in combined for term in risk_terms):
        return True

    return False


def format_bet_detail_for_prompt(bet):
    return (
        f"title: {safe_text(bet.get('Bet Title'))}\n"
        f"bet_rules: {compact_text(bet.get('Bet_Rules') or bet.get('Description'), BET_RULES_MAX_CHARS)}\n"
        f"event_description: {compact_text(bet.get('Event_Description'), EVENT_RULES_MAX_CHARS)}\n"
        f"expiration: {safe_text(bet.get('Expiration Date'))}"
    )


def call_llm_bet_detail_check(row, k_bet, p_bet):
    prompt = f"""
Second-pass strict bet equivalence check for one pair.
Decide if the two bets are truly equivalent in resolution criteria (thresholds, dates, scope, and outcome meaning).
If uncertain, reply no.

Event context:
Kalshi Event: {row.get('Kalshi_Title', 'Unknown')}
Polymarket Event: {row.get('Polymarket_Title', 'Unknown')}

Kalshi Bet:
{format_bet_detail_for_prompt(k_bet)}

Polymarket Bet:
{format_bet_detail_for_prompt(p_bet)}

Reply with exactly one word: yes or no.
"""
    response_text = safe_call_llm(
        prompt,
        system_message="Strict verifier. Reply with exactly yes or no.",
        context="Bet Detail Check"
    )
    if not response_text:
        return False
    return response_text.strip().lower().startswith("yes")


def call_llm_bets_event(row, k_bets_subset, p_bets):
    k_event_title = row.get('Kalshi_Title', 'Unknown')
    p_event_title = row.get('Polymarket_Title', 'Unknown')

    prompt = f"""
We already know these two events are likely identical.
For each Kalshi bet, choose the single matching Polymarket bet ID or "none".
If you are not sure, choose none (avoid false positives).
Important: thresholds and date windows must be equivalent.
Reply with lines: "K_ID -> P_ID yes" or "K_ID -> none no".

Events:
K: {k_event_title}
P: {p_event_title}

Kalshi bets:
"""
    for kb in k_bets_subset:
        prompt += f"- {kb.get('Bet ID')}: {kb.get('Bet Title', '')}\n"

    prompt += "\nPolymarket bets:\n"
    for pb in p_bets:
        prompt += f"- {pb.get('Bet ID')}: {pb.get('Bet Title', '')}\n"

    system_msg = "Reply only with the required K_ID -> P_ID yes/no lines. No explanations."
    response_text = safe_call_llm(prompt, system_message=system_msg, context="Bet Matching")
    if not response_text:
        return []

    # Parse mapping lines
    decisions = []
    lines = [ln.strip() for ln in response_text.strip().split('\n') if ln.strip()]
    for line in lines:
        m = re.match(r"^(\S+)\s*->\s*(\S+)\s+(yes|no)\b", line, flags=re.IGNORECASE)
        if not m:
            continue
        k_id = clean_id(m.group(1))
        p_id = clean_id(m.group(2))
        decision = m.group(3).lower()
        decisions.append((k_id, p_id, decision))

    return decisions

def build_match_rows(
    row,
    decisions,
    k_event_id,
    p_event_id,
    k_event_map,
    k_bet_map,
    p_event_map,
    p_bet_map,
    k_bets_by_id,
    p_bets_by_id,
    detail_check_cache
):
    pairs = []
    seen_k = set()
    seen_p = set()
    for k_id, p_id, decision in decisions:
        if decision != "yes":
            continue
        if not k_id or not p_id or p_id.lower() == "none":
            continue
        if k_id in seen_k or p_id in seen_p:
            continue

        k_bet = k_bets_by_id.get(clean_id(k_id))
        p_bet = p_bets_by_id.get(clean_id(p_id))
        if not k_bet or not p_bet:
            continue

        if is_hard_conflict(k_bet, p_bet):
            continue

        reasoning = "LLM yes"
        if needs_description_double_check(k_bet, p_bet):
            cache_key = (clean_id(k_id), clean_id(p_id))
            if cache_key not in detail_check_cache:
                detail_check_cache[cache_key] = call_llm_bet_detail_check(row, k_bet, p_bet)
            if not detail_check_cache[cache_key]:
                continue
            reasoning = "LLM yes + desc double-check"

        k_ev_title = k_event_map.get(k_event_id, "Unknown")
        k_bet_title = k_bet_map.get(k_id, "Unknown")
        p_ev_title = p_event_map.get(p_event_id, "Unknown")
        p_bet_title = p_bet_map.get(p_id, "Unknown")

        pairs.append({
            "Kalshi_Event_ID": k_event_id,
            "Kalshi_Event_Title": k_ev_title,
            "Kalshi_Bet_ID": k_id,
            "Kalshi_Bet_Title": k_bet_title,
            "Polymarket_Event_ID": p_event_id,
            "Polymarket_Event_Title": p_ev_title,
            "Polymarket_Bet_ID": p_id,
            "Polymarket_Bet_Title": p_bet_title,
            "Polymarket_Outcome": p_bet_title,
            "Reasoning": reasoning,
            "MatchType": "Identical",
            "Mapping": "Same"
        })
        seen_k.add(k_id)
        seen_p.add(p_id)

    return pairs

def main():
    # Clean expired bets/events before processing
    clean_files()

    matches, df_k, df_p, k_event_map, k_bet_map, p_event_map, p_bet_map = load_data()
    
    # Filter out already processed events to avoid re-running expensive LLM bet matching
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing_bets = pd.read_csv(OUTPUT_FILE, dtype={"Kalshi_Event_ID": str, "Polymarket_Event_ID": str})
            # Clean IDs
            df_existing_bets["Kalshi_Event_ID"] = df_existing_bets["Kalshi_Event_ID"].apply(clean_id)
            df_existing_bets["Polymarket_Event_ID"] = df_existing_bets["Polymarket_Event_ID"].apply(clean_id)
            
            # Create set of processed pairs
            processed_pairs = set(zip(df_existing_bets["Kalshi_Event_ID"], df_existing_bets["Polymarket_Event_ID"]))
            
            print(f"Found {len(processed_pairs)} previously processed event pairs in {OUTPUT_FILE}. Filtering...")
            
            # Filter matches dataframe
            initial_len = len(matches)
            
            # Vectorized check is tricky with zip/set, use apply
            def is_processed(row):
                return (clean_id(row["Kalshi_ID"]), clean_id(row["Polymarket_ID"])) in processed_pairs

            matches["processed"] = matches.apply(is_processed, axis=1)
            
            # Keep only unprocessed
            matches = matches[~matches["processed"]].copy()
            matches.drop(columns=["processed"], inplace=True)
            
            print(f"Skipping already matched bets. Rows to process: {len(matches)} (was {initial_len})")
        except Exception as e:
            print(f"Warning: Could not filter existing bets: {e}")

    print(f"🔄 Matching bets for {len(matches)} confirmed events...")
    
    all_bet_matches = []

    # Re-doing loop structure for cleanliness
    # Define worker function
    def process_row(row):
        k_ev_id = str(row["Kalshi_ID"])
        p_ev_id = str(row["Polymarket_ID"])
        # Carry over SyncStatus
        status = row.get("SyncStatus", "Existing")
        
        k_bets = get_bets_for_event(df_k, k_ev_id)
        p_bets = get_bets_for_event(df_p, p_ev_id)
        
        if not k_bets:
            print(f"DEBUG: No Kalshi bets for {k_ev_id}")
            return []
        if not p_bets:
            print(f"DEBUG: No Polymarket bets for {p_ev_id}")
            return []
        
        print(f"DEBUG: {k_ev_id} ({len(k_bets)} bets) <-> {p_ev_id} ({len(p_bets)} bets)")
        k_bets_by_id = {clean_id(b.get("Bet ID")): b for b in k_bets}
        p_bets_by_id = {clean_id(b.get("Bet ID")): b for b in p_bets}
        detail_check_cache = {}
        pairs = []
        for k_batch in chunked(k_bets, BATCH_SIZE):
            decisions = call_llm_bets_event(row, k_batch, p_bets)
            if decisions:
                pairs.extend(
                    build_match_rows(
                        row,
                        decisions,
                        k_ev_id,
                        p_ev_id,
                        k_event_map,
                        k_bet_map,
                        p_event_map,
                        p_bet_map,
                        k_bets_by_id,
                        p_bets_by_id,
                        detail_check_cache
                    )
                )
        
        # Add SyncStatus to each pair
        for p in pairs:
            p["SyncStatus"] = status
        return pairs

    final_results = []
    rows = list(matches.iterrows())

    for batch_idx, batch in enumerate(chunked(rows, EVENT_BATCH_SIZE), start=1):
        print(f"\nProcessing event batch {batch_idx} ({len(batch)} events)...")
        for _, row in batch:
            try:
                res = process_row(row)
                if res:
                    final_results.extend(res)
            except Exception as e:
                print(f"Error processing row: {e}")

    if final_results:
        df_new = pd.DataFrame(final_results)
        print(f"Found {len(df_new)} total bet matches across all events.")
        
        if os.path.exists(OUTPUT_FILE):
            print(f"Merging with existing {OUTPUT_FILE}...")
            df_existing = pd.read_csv(OUTPUT_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Ensure IDs are strings for duplication check
            df_combined["Kalshi_Bet_ID"] = df_combined["Kalshi_Bet_ID"].astype(str)
            df_combined["Polymarket_Bet_ID"] = df_combined["Polymarket_Bet_ID"].astype(str)
            df_combined.drop_duplicates(subset=["Kalshi_Bet_ID", "Polymarket_Bet_ID"], keep='last', inplace=True)
            df_out = df_combined
        else:
            df_out = df_new
            
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"SUCCESS: Saved {len(df_out)} total matched bets to {OUTPUT_FILE}")
    else:
        print("No matches were generated by the LLM for these events.")

if __name__ == "__main__":
    main()
