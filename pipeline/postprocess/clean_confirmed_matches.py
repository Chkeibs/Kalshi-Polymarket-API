#!/usr/bin/env python3
"""
CLEAN CONFIRMED MATCHES
Prunes expired events and bets from the confirmed files based on currently active markets.
This ensures the app doesn't try to track closed/expired positions.
"""

import pandas as pd
import os
import json
import re
import sys
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")
DATA_MATCHES_DIR = os.path.join(ROOT_DIR, "data", "matches")
DATA_CLUSTERS_DIR = os.path.join(ROOT_DIR, "data", "clusters")
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from pipeline.common.match_cache import MatchCache
from pipeline.llm_events.incremental_matcher import deterministic_reject_reason

# Files
KALSHI_MARKETS_CSV = os.path.join(DATA_MARKETS_DIR, "markets_clean_kalshi.csv")
POLY_MARKETS_CSV = os.path.join(DATA_MARKETS_DIR, "markets_clean_polymarket.csv")
CONFIRMED_EVENTS_CSV = os.path.join(DATA_MATCHES_DIR, "confirmed_matches.csv")
CONFIRMED_BETS_CSV = os.path.join(DATA_MATCHES_DIR, "confirmed_matches_bets.csv")
CANDIDATE_CLUSTERS_JSON = os.path.join(DATA_CLUSTERS_DIR, "candidate_clusters.json")
MATCH_CACHE_FILE = os.path.join(ROOT_DIR, "data", "cache", "match_cache.jsonl")

TITLE_STOP_WORDS = {
    "a", "an", "and", "any", "are", "at", "be", "before", "by", "does",
    "end", "for", "from", "have", "has", "how", "if", "in", "is", "many",
    "much", "next", "of", "on", "or", "the", "this", "to", "what", "when",
    "where", "which", "who", "will", "with",
}

def safe_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()

def normalize_title_tokens(text):
    normalized = re.sub(r"[^a-z0-9-]+", " ", safe_text(text).lower())
    return {
        token
        for token in normalized.split()
        if token and token not in TITLE_STOP_WORDS and not token.isdigit() and len(token) > 1
    }

def parse_common_keywords(value):
    text = safe_text(value)
    if not text:
        return []
    try:
        import ast
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [safe_text(item).lower() for item in parsed if safe_text(item)]
    except (ValueError, SyntaxError):
        pass
    return [part.strip().lower() for part in text.split("|") if part.strip()]

def derived_common_keywords(k_title, p_title):
    return sorted(normalize_title_tokens(k_title).intersection(normalize_title_tokens(p_title)))

def load_candidate_lookup():
    if not os.path.exists(CANDIDATE_CLUSTERS_JSON):
        return {}
    with open(CANDIDATE_CLUSTERS_JSON, "r") as handle:
        clusters = json.load(handle)
    return {
        (safe_text(cluster.get("kalshi", {}).get("event_id")), safe_text(cluster.get("polymarket", {}).get("event_id"))): cluster
        for cluster in clusters
    }

def build_market_event_lookup(df, platform):
    lookup = {}
    for _, row in df.iterrows():
        event_id = safe_text(row.get("Event ID"))
        if not event_id or event_id in lookup:
            continue
        lookup[event_id] = {
            "event_id": event_id,
            "title": safe_text(row.get("Market Title")),
            "category": safe_text(row.get("Category")),
            "sub_category": safe_text(row.get("Sub-Category")),
            "platform": platform,
        }
    return lookup

def cluster_from_confirmed_row(row, candidate_lookup, kalshi_events, poly_events):
    k_id = safe_text(row.get("Kalshi_ID"))
    p_id = safe_text(row.get("Polymarket_ID"))
    if not k_id or not p_id:
        return None

    candidate = candidate_lookup.get((k_id, p_id))
    if candidate:
        return candidate

    k_data = kalshi_events.get(k_id, {})
    p_data = poly_events.get(p_id, {})
    k_title = safe_text(row.get("Kalshi_Title")) or k_data.get("title", "")
    p_title = safe_text(row.get("Polymarket_Title")) or p_data.get("title", "")
    common_keywords = parse_common_keywords(row.get("Common_Keywords"))
    if not common_keywords:
        common_keywords = derived_common_keywords(k_title, p_title)

    return {
        "kalshi": {
            "event_id": k_id,
            "title": k_title,
            "category": k_data.get("category", ""),
            "sub_category": k_data.get("sub_category", ""),
        },
        "polymarket": {
            "event_id": p_id,
            "title": p_title,
            "category": p_data.get("category", ""),
            "sub_category": p_data.get("sub_category", ""),
        },
        "common_keywords": common_keywords,
        "match_score": safe_text(row.get("Match_Score")) or len(common_keywords),
    }

def deterministic_confirmed_reject_reason(row, candidate_lookup, kalshi_events, poly_events):
    cluster = cluster_from_confirmed_row(row, candidate_lookup, kalshi_events, poly_events)
    if not cluster:
        return "invalid_missing_event_ids"
    return deterministic_reject_reason(cluster)

def expire_cache_pairs(pair_ids, reason):
    if not pair_ids:
        return
    cache = MatchCache(cache_file=MATCH_CACHE_FILE, confirmed_csv=CONFIRMED_EVENTS_CSV)
    changed = False
    for pair in pair_ids:
        if pair in cache.matches:
            cache.matches[pair]["is_expired"] = True
            cache.matches[pair]["reasoning"] = f"expired_by_cleanup:{reason}"
            changed = True
    if changed:
        cache.save()

def clean_files():
    print("🧹 Starting Cleanup of Expired Matches...")
    stats = {
        "events_expired_removed": 0,
        "events_invalid_removed": 0,
        "bets_removed": 0,
    }
    
    # 1. Load Active Market IDs
    if not os.path.exists(KALSHI_MARKETS_CSV) or not os.path.exists(POLY_MARKETS_CSV):
        print("⚠️ Market files missing. Skipping cleanup.")
        return

    try:
        df_k = pd.read_csv(KALSHI_MARKETS_CSV, dtype=str)
        df_p = pd.read_csv(POLY_MARKETS_CSV, dtype=str)
        
        active_k_events = set(df_k["Event ID"].unique())
        active_p_events = set(df_p["Event ID"].unique())
        
        active_k_bets = set(df_k["Bet ID"].unique())
        active_p_bets = set(df_p["Bet ID"].unique())
        kalshi_events = build_market_event_lookup(df_k, "kalshi")
        poly_events = build_market_event_lookup(df_p, "polymarket")
        candidate_lookup = load_candidate_lookup()
        
        print(f"Active Context: {len(active_k_events)} K-Events, {len(active_p_events)} P-Events")
        print(f"Active Context: {len(active_k_bets)} K-Bets, {len(active_p_bets)} P-Bets")
        
    except Exception as e:
        print(f"❌ Error loading market files: {e}")
        return stats

    removed_event_pairs = set()
    remaining_event_pairs = set()

    # 2. Clean Events File (confirmed_matches.csv)
    if os.path.exists(CONFIRMED_EVENTS_CSV):
        try:
            df_ev = pd.read_csv(CONFIRMED_EVENTS_CSV, dtype=str)
            initial_len = len(df_ev)
            
            # Keep row only if BOTH Event IDs are still active
            # (Note: sometimes IDs in matches have * or other artifacts, usually cleaned, but we check roughly)
            def is_event_active(row):
                kid = str(row.get("Kalshi_ID", "")).strip().replace("*", "")
                pid = str(row.get("Polymarket_ID", "")).strip().replace("*", "")
                return (kid in active_k_events) and (pid in active_p_events)

            active_mask = df_ev.apply(is_event_active, axis=1)
            expired_pairs = {
                (safe_text(row.get("Kalshi_ID")).replace("*", ""), safe_text(row.get("Polymarket_ID")).replace("*", ""))
                for _, row in df_ev[~active_mask].iterrows()
            }
            df_ev_active = df_ev[active_mask].copy()

            invalid_reasons = []
            keep_rows = []
            for _, row in df_ev_active.iterrows():
                reason = deterministic_confirmed_reject_reason(row, candidate_lookup, kalshi_events, poly_events)
                if reason:
                    invalid_reasons.append(reason)
                    removed_event_pairs.add((safe_text(row.get("Kalshi_ID")), safe_text(row.get("Polymarket_ID"))))
                    continue
                keep_rows.append(row)

            df_ev_clean = pd.DataFrame(keep_rows, columns=df_ev.columns)
            remaining_event_pairs = {
                (safe_text(row.get("Kalshi_ID")), safe_text(row.get("Polymarket_ID")))
                for _, row in df_ev_clean.iterrows()
            }
            removed_event_pairs.update(expired_pairs)
            stats["events_expired_removed"] = len(expired_pairs)
            stats["events_invalid_removed"] = len(invalid_reasons)
            if invalid_reasons:
                print(f"  Deterministic confirmed rejects: {dict(Counter(invalid_reasons))}")
                expire_cache_pairs(removed_event_pairs, "deterministic_reject")
            
            if len(df_ev_clean) < initial_len:
                df_ev_clean.to_csv(CONFIRMED_EVENTS_CSV, index=False)
                print(f"✅ Pruned {initial_len - len(df_ev_clean)} event matches from {CONFIRMED_EVENTS_CSV}")
            else:
                print(f"Values in {CONFIRMED_EVENTS_CSV} are all active.")
                
        except Exception as e:
            print(f"❌ Error cleaning events file: {e}")

    # 3. Clean Bets File (confirmed_matches_bets.csv)
    if os.path.exists(CONFIRMED_BETS_CSV):
        try:
            df_bets = pd.read_csv(CONFIRMED_BETS_CSV, dtype=str)
            initial_len = len(df_bets)
            
            def is_bet_active(row):
                k_bid = str(row.get("Kalshi_Bet_ID", "")).strip()
                p_bid = str(row.get("Polymarket_Bet_ID", "")).strip()
                return (k_bid in active_k_bets) and (p_bid in active_p_bets)

            def event_pair_still_confirmed(row):
                pair = (safe_text(row.get("Kalshi_Event_ID")), safe_text(row.get("Polymarket_Event_ID")))
                if remaining_event_pairs:
                    return pair in remaining_event_pairs
                return pair not in removed_event_pairs

            df_bets_clean = df_bets[df_bets.apply(is_bet_active, axis=1) & df_bets.apply(event_pair_still_confirmed, axis=1)]
            stats["bets_removed"] = initial_len - len(df_bets_clean)
            
            if len(df_bets_clean) < initial_len:
                df_bets_clean.to_csv(CONFIRMED_BETS_CSV, index=False)
                print(f"✅ Pruned {initial_len - len(df_bets_clean)} expired bets from {CONFIRMED_BETS_CSV}")
            else:
                print(f"Values in {CONFIRMED_BETS_CSV} are all active.")
                
        except Exception as e:
            print(f"❌ Error cleaning bets file: {e}")

    return stats

if __name__ == "__main__":
    clean_files()
