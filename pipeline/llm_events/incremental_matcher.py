#!/usr/bin/env python3
"""
Incremental Matching Pipeline
Processes clusters from candidate_clusters.json with LLM verification.
When a match is confirmed, removes those events from other clusters.
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")
DATA_CLUSTERS_DIR = os.path.join(ROOT_DIR, "data", "clusters")
DATA_MATCHES_DIR = os.path.join(ROOT_DIR, "data", "matches")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

# --- CONFIGURATION ---
KALSHI_CSV = os.path.join(ROOT_DIR, "data", "markets", "markets_clean_kalshi.csv")
POLY_CSV = os.path.join(ROOT_DIR, "data", "markets", "markets_clean_polymarket.csv")
CANDIDATE_CLUSTERS_JSON = os.path.join(DATA_CLUSTERS_DIR, "candidate_clusters.json")
CONFIRMED_EVENTS_CSV = os.path.join(DATA_MATCHES_DIR, "confirmed_matches.csv")
PROMPT_LOG = os.path.join(ROOT_DIR, "logs", "llm_prompts.log")
CONFIRMED_BETS_CSV = os.path.join(DATA_MATCHES_DIR, "confirmed_matches_bets.csv")
EVENT_BATCH_SIZE_DEFAULT = 40
EVENT_DESC_MAX_CHARS = 320
EVENT_RULES_MAX_CHARS = 220
TITLE_SIM_UNCERTAIN_MAX = 0.82
LOW_CONFIDENCE_MATCH_SCORE = 3
HIGH_PRIORITY_SCORE = 360
MEDIUM_PRIORITY_SCORE = 250
QUALITY_TIER_ORDER = {"weak": 0, "medium": 1, "strong": 2, "exact": 3}

try:
    from pipeline.common.llm_config import safe_call_llm, API_KEY
    from pipeline.common.match_cache import MatchCache
    from pipeline.clustering.group_events_cluster import (
        has_date_scope_conflict as has_cluster_date_scope_conflict,
        has_direction_conflict,
        has_location_conflict,
        has_numeric_threshold_conflict,
        has_scope_granularity_conflict as has_cluster_scope_granularity_conflict,
        has_sports_domain_conflict as has_cluster_sports_domain_conflict,
        has_title_scope_conflict as has_cluster_title_scope_conflict,
        is_broad_specific_single_keyword as is_cluster_broad_specific_single_keyword,
    )
except ImportError:
    from llm_config import safe_call_llm, API_KEY
    from match_cache import MatchCache
    from group_events_cluster import (
        has_date_scope_conflict as has_cluster_date_scope_conflict,
        has_direction_conflict,
        has_location_conflict,
        has_numeric_threshold_conflict,
        has_scope_granularity_conflict as has_cluster_scope_granularity_conflict,
        has_sports_domain_conflict as has_cluster_sports_domain_conflict,
        has_title_scope_conflict as has_cluster_title_scope_conflict,
        is_broad_specific_single_keyword as is_cluster_broad_specific_single_keyword,
    )


def load_candidate_clusters():
    """Load candidate clusters from JSON file."""
    if not os.path.exists(CANDIDATE_CLUSTERS_JSON):
        print(f"ERROR: {CANDIDATE_CLUSTERS_JSON} not found. Run group_events_cluster.py first.")
        return []
    
    with open(CANDIDATE_CLUSTERS_JSON, 'r') as f:
        clusters = json.load(f)
    
    return clusters


def get_already_matched_event_ids():
    """Get Event IDs that are already in confirmed_matches.csv or cache."""
    matched_ids = set()
    
    # From CSV
    if os.path.exists(CONFIRMED_EVENTS_CSV):
        df = pd.read_csv(CONFIRMED_EVENTS_CSV, dtype={"Kalshi_ID": str, "Polymarket_ID": str})
        if "Kalshi_ID" in df.columns:
            matched_ids.update(df["Kalshi_ID"].dropna().unique())
        if "Polymarket_ID" in df.columns:
            matched_ids.update(df["Polymarket_ID"].dropna().unique())
    
    # From cache
    cache = MatchCache()
    for m in cache.get_active_matches():
        matched_ids.add(m.get('kalshi_id', ''))
        matched_ids.add(m.get('polymarket_id', ''))
    
    return matched_ids


def filter_clusters_by_matched(clusters, matched_ids):
    """
    Strict filter: skip clusters when EITHER event is already matched.
    This avoids re-checking previously matched events.
    """
    filtered = []
    for cluster in clusters:
        k_id = cluster['kalshi']['event_id']
        p_id = cluster['polymarket']['event_id']
        
        # Skip if either side is already matched
        if k_id in matched_ids or p_id in matched_ids:
            continue
        
        filtered.append(cluster)
    
    return filtered

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

def extract_numbers(text):
    return set(re.findall(r"\b\d+\b", safe_text(text)))

def has_granularity_risk(k_title, p_title):
    return has_cluster_scope_granularity_conflict(
        "",
        "",
        {"title": k_title},
        {"title": p_title},
    )

def has_granularity_risk_for_cluster(cluster):
    k_data = cluster.get("kalshi", {})
    p_data = cluster.get("polymarket", {})
    return has_cluster_scope_granularity_conflict(
        k_data.get("event_id", ""),
        p_data.get("event_id", ""),
        k_data,
        p_data,
    )

GENERIC_SINGLE_KEYWORDS = {
    "announced", "arrested", "attend", "before", "confirmed", "countries",
    "country", "election", "file", "files", "game", "law", "market",
    "house", "order", "orders", "president", "presidential", "price",
    "primary", "seat", "seats", "season", "senate", "winner", "year",
}

def has_sports_domain_conflict(cluster):
    k_data = cluster.get("kalshi", {})
    p_data = cluster.get("polymarket", {})
    return has_cluster_sports_domain_conflict(k_data, p_data)


def is_broad_specific_single_keyword(cluster):
    return is_cluster_broad_specific_single_keyword(
        cluster.get("kalshi", {}).get("title", ""),
        cluster.get("polymarket", {}).get("title", ""),
        cluster.get("common_keywords", []),
    )

def distinctive_keyword_count(cluster):
    count = 0
    for keyword in cluster.get("common_keywords", []) or []:
        keyword = safe_text(keyword).lower()
        if keyword and keyword not in GENERIC_SINGLE_KEYWORDS:
            count += 1
    return count


def candidate_quality_reason(cluster):
    reason = cluster.get("candidate_reason", {}) or {}
    if not isinstance(reason, dict):
        return {}
    return reason


def candidate_quality_score(cluster):
    reason = candidate_quality_reason(cluster)
    try:
        return int(reason.get("quality_score", 0) or 0)
    except (TypeError, ValueError):
        return 0


def candidate_quality_tier(cluster):
    tier = safe_text(candidate_quality_reason(cluster).get("quality_tier")).lower()
    if tier in QUALITY_TIER_ORDER:
        return tier
    return "weak"


def filter_clusters_by_min_quality(clusters, min_quality=None):
    if not min_quality:
        return clusters, []
    min_quality = safe_text(min_quality).lower()
    if min_quality not in QUALITY_TIER_ORDER:
        raise ValueError(f"Unknown min_quality '{min_quality}'. Use one of {sorted(QUALITY_TIER_ORDER)}")
    min_rank = QUALITY_TIER_ORDER[min_quality]
    kept = []
    deferred = []
    for cluster in clusters:
        if QUALITY_TIER_ORDER[candidate_quality_tier(cluster)] >= min_rank:
            kept.append(cluster)
        else:
            deferred.append(cluster)
    return kept, deferred


def llm_priority_score(cluster):
    """
    Higher score means cheaper/safer to verify first.
    This does not decide the match, it only orders LLM work.
    """
    score = int(cluster.get("match_score", 0) or 0)
    sim = title_similarity(
        cluster.get("kalshi", {}).get("title", ""),
        cluster.get("polymarket", {}).get("title", "")
    )
    distinctives = distinctive_keyword_count(cluster)
    exact_title_bonus = 80 if sim >= 0.98 else 0
    quality = candidate_quality_score(cluster)
    tier_bonus = QUALITY_TIER_ORDER[candidate_quality_tier(cluster)] * 80
    return quality + tier_bonus + (score * 60) + int(sim * 100) + (distinctives * 12) + exact_title_bonus

def priority_bucket(cluster):
    tier = candidate_quality_tier(cluster)
    if tier in {"exact", "strong"}:
        return "high"
    if tier == "medium":
        return "medium"
    if candidate_quality_score(cluster) > 0:
        return "low"

    score = llm_priority_score(cluster)
    if score >= HIGH_PRIORITY_SCORE:
        return "high"
    if score >= MEDIUM_PRIORITY_SCORE:
        return "medium"
    return "low"

def rank_clusters_for_llm(clusters):
    return sorted(
        clusters,
        key=lambda cluster: (
            llm_priority_score(cluster),
            int(cluster.get("match_score", 0) or 0),
            title_similarity(cluster.get("kalshi", {}).get("title", ""), cluster.get("polymarket", {}).get("title", "")),
            cluster.get("cluster_id", 0) or 0,
        ),
        reverse=True,
    )

def priority_counts(clusters):
    counts = {"high": 0, "medium": 0, "low": 0}
    for cluster in clusters:
        counts[priority_bucket(cluster)] += 1
    return counts


def quality_counts(clusters):
    counts = {"exact": 0, "strong": 0, "medium": 0, "weak": 0}
    for cluster in clusters:
        counts[candidate_quality_tier(cluster)] += 1
    return counts

def apply_llm_budget(clusters, max_llm_clusters=None):
    if max_llm_clusters is None or max_llm_clusters <= 0:
        return clusters, []
    selected = clusters[:max_llm_clusters]
    deferred = clusters[max_llm_clusters:]
    return selected, deferred

def deterministic_reject_reason(cluster):
    """
    Cheap pre-LLM rejects for cases that should not spend tokens.
    These rules are intentionally conservative: they only reject obvious scope/domain mismatches.
    """
    if has_granularity_risk_for_cluster(cluster):
        return "pre_reject_scope_granularity"
    if has_sports_domain_conflict(cluster):
        return "pre_reject_sports_domain_conflict"
    if is_broad_specific_single_keyword(cluster):
        return "pre_reject_broad_specific_single_keyword"
    k_title = cluster.get("kalshi", {}).get("title", "")
    p_title = cluster.get("polymarket", {}).get("title", "")
    if has_cluster_date_scope_conflict(k_title, p_title):
        return "pre_reject_date_scope_conflict"
    if has_cluster_title_scope_conflict(k_title, p_title):
        return "pre_reject_title_scope_conflict"
    if has_direction_conflict(k_title, p_title):
        return "pre_reject_direction_conflict"
    if has_numeric_threshold_conflict(k_title, p_title):
        return "pre_reject_numeric_threshold_conflict"
    if has_location_conflict(k_title, p_title):
        return "pre_reject_location_conflict"
    return ""

def split_preverified_clusters(clusters):
    llm_clusters = []
    rejected = []
    for cluster in clusters:
        reason = deterministic_reject_reason(cluster)
        if reason:
            rejected.append((cluster, reason))
        else:
            llm_clusters.append(cluster)
    return llm_clusters, rejected

def is_uncertain_cluster(cluster):
    k_title = cluster.get("kalshi", {}).get("title", "")
    p_title = cluster.get("polymarket", {}).get("title", "")
    common_keywords = cluster.get("common_keywords", []) or []
    score = int(cluster.get("match_score", 0) or 0)
    sim = title_similarity(k_title, p_title)

    if score <= LOW_CONFIDENCE_MATCH_SCORE:
        return True
    if len(common_keywords) <= 3:
        return True
    if sim <= TITLE_SIM_UNCERTAIN_MAX:
        return True

    k_nums = extract_numbers(k_title)
    p_nums = extract_numbers(p_title)
    if k_nums and p_nums and k_nums != p_nums:
        return True

    if has_granularity_risk(k_title, p_title):
        return True

    return False

def build_event_context_maps():
    """
    Build per-event context (descriptions/rules/sample bet titles) for cheap targeted double-checks.
    """
    def build_map(filepath):
        event_map = {}
        if not os.path.exists(filepath):
            return event_map

        df = pd.read_csv(filepath, dtype=str)
        for _, row in df.iterrows():
            event_id = safe_text(row.get("Event ID"))
            if not event_id:
                continue

            ctx = event_map.setdefault(
                event_id,
                {
                    "event_description": "",
                    "description": "",
                    "bet_rules": "",
                    "bet_titles": []
                }
            )

            for field in ("event_description", "description", "bet_rules"):
                csv_col = {
                    "event_description": "Event_Description",
                    "description": "Description",
                    "bet_rules": "Bet_Rules"
                }[field]
                val = safe_text(row.get(csv_col))
                if val and len(val) > len(ctx[field]):
                    ctx[field] = val

            bet_title = safe_text(row.get("Bet Title"))
            if bet_title and bet_title not in ctx["bet_titles"] and len(ctx["bet_titles"]) < 4:
                ctx["bet_titles"].append(bet_title)

        return event_map

    return build_map(KALSHI_CSV), build_map(POLY_CSV)

def call_llm_event_detail_check(cluster, k_ctx_map, p_ctx_map, context="Event Matching Detail Check"):
    """
    Second-pass check only for uncertain pairs using richer descriptions/rules.
    """
    k_data = cluster["kalshi"]
    p_data = cluster["polymarket"]
    k_id = k_data.get("event_id", "")
    p_id = p_data.get("event_id", "")
    k_ctx = k_ctx_map.get(k_id, {})
    p_ctx = p_ctx_map.get(p_id, {})

    k_desc = compact_text(k_ctx.get("event_description") or k_ctx.get("description"), EVENT_DESC_MAX_CHARS)
    p_desc = compact_text(p_ctx.get("event_description") or p_ctx.get("description"), EVENT_DESC_MAX_CHARS)
    k_rules = compact_text(k_ctx.get("bet_rules"), EVENT_RULES_MAX_CHARS)
    p_rules = compact_text(p_ctx.get("bet_rules"), EVENT_RULES_MAX_CHARS)
    k_examples = "; ".join(k_ctx.get("bet_titles", [])[:3])
    p_examples = "; ".join(p_ctx.get("bet_titles", [])[:3])

    prompt = f"""
You are doing a strict second-pass verification for one candidate event pair.
Decide if these are the SAME event market intent (not just similar names/teams).
If one is about a match and the other about tournament winner (or any different scope), answer no.

Kalshi:
- title: {k_data.get('title', '')}
- category: {k_data.get('category', '')} / {k_data.get('sub_category', '')}
- event_description: {k_desc}
- rules: {k_rules}
- sample_bets: {k_examples}

Polymarket:
- title: {p_data.get('title', '')}
- category: {p_data.get('category', '')}
- event_description: {p_desc}
- rules: {p_rules}
- sample_bets: {p_examples}

Reply with exactly one word: yes or no.
"""
    response_text = safe_call_llm(
        prompt,
        system_message="Strict verifier. Reply with exactly yes or no.",
        context=context
    )
    if not response_text:
        return False
    return response_text.strip().lower().startswith("yes")

def call_llm_for_cluster_batch(clusters, context="Event Matching"):
    """
    Call LLM to verify a BATCH of cluster pairs at once.
    Returns yes/no only to reduce token usage and cost.
    """
    prompt = """
You are verifying candidate pairs of betting events.
For EACH pair, answer only "yes" or "no" if they refer to the exact same real-world event.
Return ONE line per pair in the format:
<N>. yes
<N>. no

PAIRS:
"""
    for i, cluster in enumerate(clusters, 1):
        k_data = cluster['kalshi']
        p_data = cluster['polymarket']
        k_cat = f"{k_data.get('category', '')} / {k_data.get('sub_category', '')}".strip(" /")
        p_cat = p_data.get('category', '')
        prompt += (
            f"{i}. Kalshi: \"{k_data.get('title', '')}\" (Cat: {k_cat}) | "
            f"Polymarket: \"{p_data.get('title', '')}\" (Cat: {p_cat})\n"
        )

    system_msg = "Reply only with numbered yes/no lines. No explanations."

    response_text = safe_call_llm(prompt, system_message=system_msg, context=context)
    if not response_text:
        return {}

    # Parse response
    results = {}
    lines = [ln.strip() for ln in response_text.strip().split('\n') if ln.strip()]
    assigned = set()

    def add_match(idx):
        if 0 <= idx < len(clusters):
            cluster = clusters[idx]
            results[idx] = {
                "Kalshi_ID": cluster['kalshi']['event_id'],
                "Polymarket_ID": cluster['polymarket']['event_id'],
                "Kalshi_Title": cluster['kalshi']['title'],
                "Polymarket_Title": cluster['polymarket']['title'],
                "Reasoning": "LLM yes",
                "Common_Keywords": cluster.get('common_keywords', []),
                "Match_Score": cluster.get('match_score', 0)
            }

    next_idx = 0
    for line in lines:
        m = re.match(r"^(?:pair\s*)?(\d+)\s*[\.:\)-]?\s*(yes|no)\b", line, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            decision = m.group(2).lower()
            if decision == "yes":
                add_match(idx)
            assigned.add(idx)
            continue

        # Fallback: line is just yes/no without numbering
        if line.lower() in {"yes", "no"}:
            while next_idx in assigned and next_idx < len(clusters):
                next_idx += 1
            if next_idx < len(clusters):
                if line.lower() == "yes":
                    add_match(next_idx)
                assigned.add(next_idx)
                next_idx += 1

    return results



def parse_single_match(response_text, cluster):
    """Parse LLM response for a single cluster verification."""
    if not response_text:
        return None
    
    response_lower = response_text.lower().strip()
    
    # Check if it's a match
    if response_lower.startswith('match') and 'no match' not in response_lower:
        # Extract reason
        parts = response_text.split('|')
        reason = parts[1].strip() if len(parts) > 1 else "Confirmed match"
        
        return {
            "Kalshi_ID": cluster['kalshi']['event_id'],
            "Polymarket_ID": cluster['polymarket']['event_id'],
            "Kalshi_Title": cluster['kalshi']['title'],
            "Polymarket_Title": cluster['polymarket']['title'],
            "Reasoning": reason,
            "Common_Keywords": cluster.get('common_keywords', []),
            "Match_Score": cluster.get('match_score', 0)
        }
    
    return None


def remove_matched_events_from_clusters(clusters, confirmed_k_ids, confirmed_p_ids):
    """
    Remove clusters that contain already-confirmed events.
    Returns filtered list of clusters still to process.
    """
    filtered = []
    for cluster in clusters:
        k_id = cluster['kalshi']['event_id']
        p_id = cluster['polymarket']['event_id']
        
        # Skip if either event is already confirmed
        if k_id in confirmed_k_ids or p_id in confirmed_p_ids:
            continue
        
        filtered.append(cluster)
    
    return filtered


def run_legacy_incremental_pipeline(api_key=None, full_retry=False, batch_size=EVENT_BATCH_SIZE_DEFAULT, dry_run=False, max_llm_clusters=None, min_quality=None):
    """
    Main pipeline function.
    Processes clusters in batches, removing confirmed events as we go.
    """
    if not api_key:
        api_key = API_KEY
        
    print("=" * 60)
    print("🔄 CLUSTER VERIFICATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load candidate clusters
    print("\n📂 Step 1: Loading candidate clusters...")
    clusters = load_candidate_clusters()
    print(f"  Loaded {len(clusters)} clusters from {CANDIDATE_CLUSTERS_JSON}")
    
    if not clusters:
        print("❌ No clusters to process. Run group_events_cluster.py first.")
        return {"new_event_matches": 0}
    
    # Step 2: Get already matched events
    print("\n📋 Step 2: Loading already matched events...")
    matched_ids = set() if full_retry else get_already_matched_event_ids()
    print(f"  Found {len(matched_ids)} already matched event IDs")
    
    # Step 3: Filter out already-matched clusters
    print("\n🔍 Step 3: Filtering clusters...")
    clusters = filter_clusters_by_matched(clusters, matched_ids)
    print(f"  {len(clusters)} clusters remaining after filtering")
    
    if not clusters:
        print("✅ All clusters already processed. Nothing to do.")
        return {"new_event_matches": 0}

    print("\n🧹 Step 4: Cheap pre-verification rejects...")
    clusters, pre_rejected = split_preverified_clusters(clusters)
    print(f"  {len(pre_rejected)} clusters rejected before LLM")
    print(f"  {len(clusters)} clusters remaining for LLM")

    if not clusters:
        update_cluster_verification_status([], set(), set(), pre_rejected)
        return {"new_event_matches": 0, "pre_rejected": len(pre_rejected), "clusters_processed": 0}

    quality_before = quality_counts(clusters)
    clusters, deferred_by_quality = filter_clusters_by_min_quality(clusters, min_quality=min_quality)
    if deferred_by_quality:
        print(f"  Quality filter >= {min_quality}: kept {len(clusters)}, deferred {len(deferred_by_quality)}")

    clusters = rank_clusters_for_llm(clusters)
    counts = priority_counts(clusters)
    clusters, deferred_clusters = apply_llm_budget(clusters, max_llm_clusters=max_llm_clusters)
    print(f"  Priority buckets before budget: {counts}")
    print(f"  Quality tiers before filter: {quality_before}")
    if deferred_clusters:
        print(f"  Budget selected {len(clusters)} clusters and deferred {len(deferred_clusters)}")

    if dry_run:
        print("🧪 Dry run enabled: stopping before LLM calls.")
        return {
            "new_event_matches": 0,
            "clusters_processed": len(clusters),
            "pre_rejected": len(pre_rejected),
            "deferred_by_quality": len(deferred_by_quality),
            "deferred_by_budget": len(deferred_clusters),
            "priority_counts": counts,
            "quality_counts": quality_before,
            "dry_run": True
        }
    
    # Step 5: Process clusters with LLM
    print(f"\n🤖 Step 5: Processing {len(clusters)} clusters with LLM...")
    
    all_matches = []
    confirmed_k_ids = set()
    confirmed_p_ids = set()
    k_ctx_map, p_ctx_map = build_event_context_maps()
    # Process in batches
    for i in tqdm(range(0, len(clusters), batch_size), desc="Batches"):
        batch = clusters[i:i+batch_size]
        
        # Filter out events confirmed in previous batches
        batch = remove_matched_events_from_clusters(batch, confirmed_k_ids, confirmed_p_ids)
        
        if not batch:
            continue
        
        # Process batch together
        try:
            print(f"\nProcessing batch of {len(batch)} pairs...")
            batch_results = call_llm_for_cluster_batch(batch, context=f"Event Matching Batch {i // batch_size + 1}")
            
            for idx in sorted(batch_results):
                match = batch_results[idx]
                if not match:
                    continue
                cluster = batch[idx]
                if match['Kalshi_ID'] in confirmed_k_ids or match['Polymarket_ID'] in confirmed_p_ids:
                    continue

                if is_uncertain_cluster(cluster):
                    detail_ok = call_llm_event_detail_check(
                        cluster,
                        k_ctx_map,
                        p_ctx_map,
                        context=f"Event Detail Check Batch {i // batch_size + 1}"
                    )
                    if not detail_ok:
                        continue
                    match["Reasoning"] = "LLM yes + detail yes"
                else:
                    match["Reasoning"] = "LLM yes"

                all_matches.append(match)
                # Add to confirmed sets so we skip these in future batches
                confirmed_k_ids.add(match['Kalshi_ID'])
                confirmed_p_ids.add(match['Polymarket_ID'])
        
        except Exception as e:
            print(f"  Error processing batch: {e}")
    
    print(f"\n✅ Found {len(all_matches)} confirmed matches")
    
    # Step 5: Save matches
    if all_matches:
        print(f"\n💾 Step 6: Saving {len(all_matches)} matches...")
        
        # Save to cache
        cache = MatchCache()
        cache.add_matches_from_list(all_matches)
        
        # Save to CSV
        df_new = pd.DataFrame(all_matches)
        df_new["SyncStatus"] = "New"
        
        if os.path.exists(CONFIRMED_EVENTS_CSV):
            df_existing = pd.read_csv(CONFIRMED_EVENTS_CSV)
            if "SyncStatus" in df_existing.columns:
                df_existing["SyncStatus"] = "Existing"
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=["Kalshi_ID", "Polymarket_ID"], inplace=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(CONFIRMED_EVENTS_CSV, index=False)
        print(f"  Saved to {CONFIRMED_EVENTS_CSV}")
    
    # Step 7: Update candidate_clusters.json with llm_verified status
    print("\n📝 Step 7: Updating cluster verification status...")
    update_cluster_verification_status(all_matches, confirmed_k_ids, confirmed_p_ids, pre_rejected)
    
    return {
        "new_event_matches": len(all_matches),
        "clusters_processed": len(clusters),
        "pre_rejected": len(pre_rejected),
        "deferred_by_quality": len(deferred_by_quality),
        "deferred_by_budget": len(deferred_clusters),
        "priority_counts": counts,
        "quality_counts": quality_before,
    }


def run_incremental_pipeline(
    api_key=None,
    full_retry=False,
    batch_size=EVENT_BATCH_SIZE_DEFAULT,
    dry_run=False,
    max_llm_clusters=None,
    min_quality=None,
    engine=None,
    apply=None,
    include_outcomes=True,
):
    """Run the versioned verifier by default; retain the legacy engine for rollback/tests."""
    engine = (engine or os.environ.get("LLM_MATCHER_ENGINE", "v2")).lower()
    if engine == "legacy":
        return run_legacy_incremental_pipeline(
            api_key=api_key,
            full_retry=full_retry,
            batch_size=batch_size,
            dry_run=dry_run,
            max_llm_clusters=max_llm_clusters,
            min_quality=min_quality,
        )
    if engine != "v2":
        raise ValueError(f"Unknown matcher engine: {engine}")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if apply is None:
        apply = os.environ.get("LLM_APPLY_RESULTS", "false").lower() in {"1", "true", "yes"}

    from pipeline.llm_events.verifier import VerifierConfig, run_v2_verifier

    config = VerifierConfig(
        cheap_batch_size=min(40, max(1, batch_size)),
        include_outcomes=include_outcomes,
    )
    return run_v2_verifier(
        dry_run=dry_run,
        apply=bool(apply),
        include_outcomes=include_outcomes,
        max_candidates=max_llm_clusters,
        config=config,
    )


def update_cluster_verification_status(matches, confirmed_k_ids, confirmed_p_ids, pre_rejected=None):
    """Update candidate_clusters.json with LLM verification results."""
    if not os.path.exists(CANDIDATE_CLUSTERS_JSON):
        return
    
    with open(CANDIDATE_CLUSTERS_JSON, 'r') as f:
        clusters = json.load(f)
    
    # Build lookup for confirmed matches
    confirmed_pairs = {(m['Kalshi_ID'], m['Polymarket_ID']) for m in matches}
    rejected_pairs = {
        (cluster['kalshi']['event_id'], cluster['polymarket']['event_id']): reason
        for cluster, reason in (pre_rejected or [])
    }
    
    for cluster in clusters:
        k_id = cluster['kalshi']['event_id']
        p_id = cluster['polymarket']['event_id']
        
        if (k_id, p_id) in confirmed_pairs:
            cluster['llm_verified'] = True
        elif k_id in confirmed_k_ids or p_id in confirmed_p_ids:
            # Event was matched with a different pair
            cluster['llm_verified'] = 'skipped'
        elif (k_id, p_id) in rejected_pairs:
            cluster['llm_verified'] = rejected_pairs[(k_id, p_id)]
    
    with open(CANDIDATE_CLUSTERS_JSON, 'w') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    
    print(f"  Updated verification status in {CANDIDATE_CLUSTERS_JSON}")


# Removed unused functions: parse_single_match



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify cluster matches with LLM")
    parser.add_argument("--batch-size", type=int, default=EVENT_BATCH_SIZE_DEFAULT, help="Batch size for processing")
    parser.add_argument("--full", action="store_true", help="Process all clusters (ignore already matched)")
    parser.add_argument("--dry-run", action="store_true", help="Run filtering/pre-verification without LLM calls")
    parser.add_argument("--max-llm-clusters", type=int, default=None, help="Optional budget: only send top N ranked clusters to LLM")
    parser.add_argument("--min-quality", choices=sorted(QUALITY_TIER_ORDER), default=None, help="Optional quality floor before LLM: weak/medium/strong/exact")
    parser.add_argument("--engine", choices=("v2", "legacy"), default="v2", help="Verifier engine")
    parser.add_argument("--apply", action="store_true", help="Write validated v2 matches to confirmed_matches.csv")
    parser.add_argument("--events-only", action="store_true", help="Do not verify outcome-aware candidates")
    args = parser.parse_args()
    
    result = run_incremental_pipeline(
        batch_size=args.batch_size,
        full_retry=args.full,
        dry_run=args.dry_run,
        max_llm_clusters=args.max_llm_clusters,
        min_quality=args.min_quality,
        engine=args.engine,
        apply=args.apply,
        include_outcomes=not args.events_only,
    )
    print(f"\n🎯 RESULT: {result}")
