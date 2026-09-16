#!/usr/bin/env python3
"""
Fetch CLEAN market data (IDs, Title, Bet Title, Category, Expiration) from Kalshi and Polymarket.
Focuses ONLY on open/active markets.
"""

import os
import sys
import time
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

try:
    from pipeline.common.category_mapping import CATEGORY_MAPPING
    from pipeline.common.schemas import (
        KALSHI_MARKET_REQUIRED_COLUMNS,
        POLYMARKET_MARKET_REQUIRED_COLUMNS,
        align_columns,
    )
except ImportError:
    from category_mapping import CATEGORY_MAPPING
    from schemas import KALSHI_MARKET_REQUIRED_COLUMNS, POLYMARKET_MARKET_REQUIRED_COLUMNS, align_columns

# API Base URLs. Both can be overridden for demo/testing environments.
KALSHI_API_BASE = os.getenv("KALSHI_API_BASE", "https://external-api.kalshi.com/trade-api/v2").rstrip("/")
POLYMARKET_API_BASE = os.getenv("POLYMARKET_API_BASE", "https://gamma-api.polymarket.com").rstrip("/")

# Output files
KALSHI_OUTPUT = os.path.join(DATA_MARKETS_DIR, "markets_clean_kalshi.csv")
POLYMARKET_OUTPUT = os.path.join(DATA_MARKETS_DIR, "markets_clean_polymarket.csv")

DEFAULT_MAX_PAGES = 50
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
_thread_local = threading.local()


def get_http_session():
    """Return a per-thread session so concurrent calls reuse connections safely."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = session
    return session


def request_json(url, params=None, timeout=10, retries=4, backoff=0.5):
    """GET JSON with bounded retries for rate limits and transient failures."""
    for attempt in range(retries):
        try:
            response = get_http_session().get(url, params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
                continue
    return None


def finalize_market_dataframe(df, required_columns):
    """Ensure fetch output keeps the shared schema contract."""
    if df.empty:
        return pd.DataFrame(columns=required_columns)
    df = align_columns(df, required_columns)
    if "Bet ID" in df.columns:
        df["Bet ID"] = df["Bet ID"].astype(str)
        df.drop_duplicates(subset=["Bet ID"], inplace=True)
    return df

def normalize_category(raw_cat):
    """Map various API categories to a standard set."""
    if not raw_cat:
        return "Other"
    
    clean = raw_cat.lower().strip()
    return CATEGORY_MAPPING.get(clean, clean.title())

def normalize_date(date_str):
    """Normalize date string to YYYY-MM-DD HH:MM:SS format (UTC)."""
    if not date_str:
        return ""
    try:
        # Handle ISO strings like "2025-12-31T12:00:00Z" or "2025-12-31T12:00:00"
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
             # Try parsing other common formats if needed
             dt = datetime.fromisoformat(date_str)
        
        # Convert to UTC naive string for consistency
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return date_str

def is_expired(date_str):
    """Check if the normalized date string is in the past."""
    if not date_str:
        return False
        
    try:
        dt = datetime.fromisoformat(date_str)
        now = datetime.utcnow()
        return dt < now
    except:
        return False


def fetch_kalshi_series_tags(series_tickers):
    """Fetch tags for a list of series tickers."""
    unique_tickers = sorted({ticker for ticker in series_tickers if ticker})
    print(f"Fetching tags for {len(unique_tickers)} unique series...")
    series_tags = {}
    
    def get_series_tags(ticker):
        data = request_json(f"{KALSHI_API_BASE}/series/{ticker}", timeout=5, retries=3)
        if not data:
            return None
        tags = data.get("series", {}).get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return ticker, ", ".join(tags)

    # Be gentle with concurrency
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_series_tags, t) for t in unique_tickers]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                series_tags[result[0]] = result[1]
            if i % 20 == 0 and i > 0:
                time.sleep(0.5) # Rate limit protection
                
    return series_tags


def fetch_kalshi_markets(max_pages=DEFAULT_MAX_PAGES):
    """Fetch active Kalshi markets through nested, cursor-paginated events."""
    print("Fetching Kalshi markets...")
    all_events = []
    cursor = None
    page = 1
    
    # 1. Fetch Events
    while True:
        if max_pages and page > max_pages:
            break
        try:
            params = {"limit": 200, "status": "open", "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor

            data = request_json(f"{KALSHI_API_BASE}/events", params=params, timeout=15, retries=4)
            if not data:
                print("  [Error] Failed to fetch events after 3 retries. Skipping page.")
                break
            events = data.get("events", [])
            if not events:
                break
            
            all_events.extend(events)
            cursor = data.get("cursor")
            if not cursor:
                break
            page += 1
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching Kalshi events: {e}")
            break

    print(f"  Found {len(all_events)} active events with nested markets.")

    # 1.5 Fetch Series Tags (for Sub-Category)
    series_tickers = [e.get("series_ticker") for e in all_events if e.get("series_ticker")]
    series_tags_map = fetch_kalshi_series_tags(series_tickers)
    print(f"  Mapped {len(series_tags_map)} series tags.")

    # The nested endpoint avoids the previous N+1 event-detail fetch. Retain a
    # bounded fallback for responses/environments that omit the markets field.
    missing_nested = [event for event in all_events if "markets" not in event]
    fallback_details = {}

    def get_event_detail(event):
        ticker = event.get("event_ticker")
        if not ticker:
            return ticker, None
        return ticker, request_json(f"{KALSHI_API_BASE}/events/{ticker}", timeout=10, retries=3)

    if missing_nested:
        print(f"  Nested markets missing for {len(missing_nested)} events; using detail fallback.")
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(get_event_detail, event) for event in missing_nested]
            for future in as_completed(futures):
                ticker, detail = future.result()
                if ticker and detail:
                    fallback_details[ticker] = detail

    clean_data = []

    for event_summary in all_events:
        ticker = event_summary.get("event_ticker", "")
        if "markets" in event_summary:
            event_detail = event_summary
            markets = event_summary.get("markets") or []
        else:
            detail_package = fallback_details.get(ticker) or {}
            event_detail = detail_package.get("event") or event_summary
            markets = detail_package.get("markets") or event_detail.get("markets") or []

        for m in markets:
                if m.get("status") != "active":
                    continue
                
                # IDs
                event_id = event_summary.get("event_ticker", "")
                bet_id = m.get("ticker", "")

                # Normalize Title (Prefer detail title if available)
                raw_market_title = (event_detail.get("title") or event_summary.get("title", "")).strip()
                
                # Split "fused" titles (e.g., "Event Name: Suffix")
                market_title = raw_market_title
                suffix = ""
                if ": " in raw_market_title:
                    parts = raw_market_title.split(": ", 1)
                    market_title = parts[0].strip()
                    suffix = parts[1].strip()
                
                # Normalize Bet Title
                # Kalshi Specifics: 
                # binary markets often have generic titles but specific yes_sub_title
                # multi-outcome markets often use subtitles
                raw_bet_title = m.get("title", "").strip()
                subtitle = m.get("subtitle", "").strip()
                yes_sub = m.get("yes_sub_title", "").strip()
                
                # Clean artifacts like "::" often found in Kalshi subtitles
                if subtitle == "::":
                    subtitle = ""
                
                # Logic to find the best descriptive bet title
                bet_title_base = raw_bet_title
                if yes_sub and yes_sub.lower() not in ["yes", "no", ""]:
                    bet_title_base = yes_sub
                elif subtitle and subtitle != raw_bet_title:
                    bet_title_base = f"{raw_bet_title} ({subtitle})"
                elif raw_bet_title == raw_market_title or raw_bet_title == market_title:
                    bet_title_base = f"{raw_bet_title} (Yes)"
                elif not bet_title_base and subtitle:
                    bet_title_base = subtitle

                # Combine with suffix if it exists
                if suffix:
                    bet_title = f"{suffix}: {bet_title_base}"
                else:
                    bet_title = bet_title_base
                
                # Normalize Category
                raw_cat = event_detail.get("category") or event_summary.get("category", "") or m.get("category", "")
                cat = normalize_category(raw_cat)
                
                # Extract Sub-Category (try multiple sources)
                series_ticker = event_detail.get("series_ticker") or event_summary.get("series_ticker", "")
                
                sub_cat = (
                    series_tags_map.get(series_ticker) or
                    event_detail.get("sub_category") or 
                    event_summary.get("sub_category") or 
                    m.get("sub_category") or
                    ""
                ).strip()
                
                # Description (Rules) - Highly specific to each bet (market level)
                # Prioritize market-level rules_summary, then primary, then secondary
                rules_list = []
                for rf in ["rules_summary", "rules_primary", "rules_secondary"]:
                    val = m.get(rf, "").strip()
                    if val:
                        rules_list.append(val)
                
                desc = " ".join(rules_list)
                
                # Global Event Context
                # Prioritize event-level rules_summary if available
                event_desc = (event_detail.get("rules_summary") or 
                              event_detail.get("rules_primary") or 
                              event_detail.get("settlement_criteria") or
                              event_detail.get("rules") or 
                              event_summary.get("description") or 
                              event_summary.get("sub_title") or "")

                if not desc:
                    # Fallback to event details
                    desc = event_desc

                # Status
                status = m.get("status", "")

                # Normalize Expiration
                expiration = normalize_date(m.get("expiration_time"))

                clean_data.append({
                    "Platform": "Kalshi",
                    "Event ID": event_id,
                    "Bet ID": bet_id,
                    "Market Title": market_title,
                    "Bet Title": bet_title,
                    "Category": cat,
                    "Sub-Category": sub_cat,
                    "Series Ticker": series_ticker,
                    "Description": desc, # This is the bet-level rule (kept for compatibility)
                    "Bet_Rules": desc,
                    "Event_Description": event_desc,
                    "Status": status,
                    "Expiration Date": expiration
                })

    return finalize_market_dataframe(pd.DataFrame(clean_data), KALSHI_MARKET_REQUIRED_COLUMNS)

def fetch_polymarket_markets(max_pages=DEFAULT_MAX_PAGES):
    """Fetch active Polymarket events through stable keyset pagination."""
    print("Fetching Polymarket markets...")
    all_clean_data = []
    after_cursor = None
    seen_cursors = set()
    limit = 500
    page = 1
    
    while True:
        if max_pages and page > max_pages:
            break
        try:
            params = {
                "limit": limit,
                "active": "true",
                "closed": "false",
                "archived": "false",
            }
            if after_cursor:
                params["after_cursor"] = after_cursor

            payload = request_json(f"{POLYMARKET_API_BASE}/events/keyset", params=params, timeout=15, retries=4)
            if not payload:
                break
            events = payload.get("events", [])
            
            if not events:
                break
                
            for event in events:
                if not event.get("active") or event.get("closed"):
                    continue

                event_id = event.get("id", "")
                market_title = event.get("title", "").strip()
                market_desc = event.get("description", "")
                
                # Category parsing
                tags = event.get("tags", [])
                raw_cat = tags[0].get("label", "") if tags else ""
                cat = normalize_category(raw_cat)
                
                # Sub-Category from remaining tags
                sub_cat = ""
                if len(tags) > 1:
                    # Join remaining tags, excluding "Polymarket" or redundant ones if any
                    sub_tags = [t.get("label", "") for t in tags[1:] if t.get("label")]
                    sub_cat = ", ".join(sub_tags)
                
                markets = event.get("markets", [])
                for m in markets:
                    if not m.get("active") or m.get("closed"):
                        continue
                    
                    market_bet_id = m.get("id", "")
                    
                    # Polymarket markets often have multiple outcomes (e.g., Team A, Team B, Draw)
                    # We explode these into separate rows to allow 1-to-1 matching with Kalshi.
                    outcomes_raw = m.get("outcomes")
                    clob_token_ids_raw = m.get("clobTokenIds")
                    
                    # Try to parse strings if they are not already lists
                    outcomes_list = []
                    if isinstance(outcomes_raw, str):
                        try: outcomes_list = json.loads(outcomes_raw)
                        except: outcomes_list = [outcomes_raw]
                    elif isinstance(outcomes_raw, list):
                        outcomes_list = outcomes_raw
                        
                    token_ids_list = []
                    if isinstance(clob_token_ids_raw, str):
                        try: token_ids_list = json.loads(clob_token_ids_raw)
                        except: token_ids_list = [clob_token_ids_raw]
                    elif isinstance(clob_token_ids_raw, list):
                        token_ids_list = clob_token_ids_raw

                    # If no specific outcomes/tokens, handle as a generic "Yes/No" market (usual for binary)
                    if not outcomes_list or not token_ids_list:
                        # Fallback to single row
                        bet_title = m.get("groupItemTitle") or m.get("question", "")
                        if "\n" in bet_title: bet_title = bet_title.split("\n")[0].strip()
                        if bet_title == market_title: bet_title = f"{bet_title} (Yes)"
                        
                        all_clean_data.append({
                            "Platform": "Polymarket",
                            "Event ID": event_id,
                            "Bet ID": market_bet_id,
                            "Market Title": market_title,
                            "Bet Title": bet_title.strip(),
                            "Category": cat,
                            "Sub-Category": sub_cat,
                            "Description": market_desc,
                            "Bet_Rules": m.get("question", ""),
                            "Event_Description": market_desc,
                            "Status": "active" if m.get("active") else "closed",
                            "Expiration Date": normalize_date(m.get("endDate") or event.get("endDate")),
                            "Outcomes": str(outcomes_list),
                            "Token_IDs": str(token_ids_list),
                            "Original_Market_ID": market_bet_id,
                            "Outcome_Name": "Yes"
                        })
                        continue

                    # Explode! Iterate through outcomes and create a row for each.
                    # Use the clobTokenId as the unique Bet ID.
                    for i, outcome_name in enumerate(outcomes_list):
                        # Ensure we have a matching token ID
                        token_id = token_ids_list[i] if i < len(token_ids_list) else f"{market_bet_id}_{i}"
                        
                        bet_title_base = m.get("groupItemTitle") or m.get("question", "")
                        if "\n" in bet_title_base: bet_title_base = bet_title_base.split("\n")[0].strip()
                        
                        # Format title to include the outcome for clarity
                        bet_title = f"{bet_title_base} [{outcome_name}]"
                        if bet_title_base == market_title:
                            bet_title = f"{market_title} [{outcome_name}]"

                        all_clean_data.append({
                            "Platform": "Polymarket",
                            "Event ID": event_id,
                            "Bet ID": str(token_id), # Unique per outcome
                            "Market Title": market_title,
                            "Bet Title": bet_title,
                            "Category": cat,
                            "Sub-Category": sub_cat,
                            "Description": market_desc,
                            "Bet_Rules": m.get("question", ""),
                            "Event_Description": market_desc,
                            "Status": "active" if m.get("active") else "closed",
                            "Expiration Date": normalize_date(m.get("endDate") or event.get("endDate")),
                            "Outcomes": str(outcomes_list),
                            "Token_IDs": str(token_ids_list),
                            "Original_Market_ID": market_bet_id, # Keep for debugging
                            "Outcome_Name": outcome_name
                        })
            
            next_cursor = payload.get("next_cursor")
            if not next_cursor:
                break
            if next_cursor in seen_cursors:
                print("  [Warn] Repeated Polymarket keyset cursor. Stopping fetch.")
                break
            seen_cursors.add(next_cursor)
            after_cursor = next_cursor
            page += 1
            print(f"  Polymarket Page {page-1} done ({len(all_clean_data)} markets found so far)")
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching Polymarket: {e}")
            break
            
    return finalize_market_dataframe(pd.DataFrame(all_clean_data), POLYMARKET_MARKET_REQUIRED_COLUMNS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    args = parser.parse_args()
    
    # 1. Kalshi
    df_kalshi = fetch_kalshi_markets(max_pages=args.max_pages)
    if not df_kalshi.empty:
        os.makedirs(DATA_MARKETS_DIR, exist_ok=True)
        df_kalshi.to_csv(KALSHI_OUTPUT, index=False)
        print(f"Saved {len(df_kalshi)} Kalshi markets to {KALSHI_OUTPUT}")
    else:
        print("No Kalshi markets found.")

    # 2. Polymarket
    df_poly = fetch_polymarket_markets(max_pages=args.max_pages)
    if not df_poly.empty:
        os.makedirs(DATA_MARKETS_DIR, exist_ok=True)
        df_poly.to_csv(POLYMARKET_OUTPUT, index=False)
        print(f"Saved {len(df_poly)} Polymarket markets to {POLYMARKET_OUTPUT}")
    else:
        print("No Polymarket markets found.")

    # Validation stats
    print("\n=== Validation Stats ===")
    if not df_kalshi.empty:
        print(f"Kalshi: {len(df_kalshi)} rows")
        print("Model:")
        print(df_kalshi[["Event ID", "Bet ID", "Market Title", "Category"]].head(2))
    
    if not df_poly.empty:
        print(f"Polymarket: {len(df_poly)} rows")
        print("Model:")
        print(df_poly[["Event ID", "Bet ID", "Market Title", "Category"]].head(2))

if __name__ == "__main__":
    main()
