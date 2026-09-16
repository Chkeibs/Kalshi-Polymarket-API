import os
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON_DIR = os.path.join(ROOT_DIR, "pipeline", "common")
DATA_MARKETS_DIR = os.path.join(ROOT_DIR, "data", "markets")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

try:
    from pipeline.fetch.fetch_clean_data import fetch_kalshi_markets, fetch_polymarket_markets
    from pipeline.common.schemas import (
        KALSHI_MARKET_REQUIRED_COLUMNS,
        POLYMARKET_MARKET_REQUIRED_COLUMNS,
        align_columns,
    )
except ImportError:
    from fetch_clean_data import fetch_kalshi_markets, fetch_polymarket_markets
    from schemas import KALSHI_MARKET_REQUIRED_COLUMNS, POLYMARKET_MARKET_REQUIRED_COLUMNS, align_columns

KALSHI_CSV = os.path.join(DATA_MARKETS_DIR, "markets_clean_kalshi.csv")
POLY_CSV = os.path.join(DATA_MARKETS_DIR, "markets_clean_polymarket.csv")


PLATFORM_CONFIG = {
    "kalshi": {
        "filepath": KALSHI_CSV,
        "required_columns": KALSHI_MARKET_REQUIRED_COLUMNS,
        "fetch": fetch_kalshi_markets,
    },
    "polymarket": {
        "filepath": POLY_CSV,
        "required_columns": POLYMARKET_MARKET_REQUIRED_COLUMNS,
        "fetch": fetch_polymarket_markets,
    },
}


class MarketDataManager:
    def __init__(self):
        os.makedirs(DATA_MARKETS_DIR, exist_ok=True)

    def _get_config(self, platform):
        key = str(platform).lower()
        if key not in PLATFORM_CONFIG:
            raise ValueError(f"Unknown platform '{platform}'. Expected one of: {', '.join(PLATFORM_CONFIG)}")
        return PLATFORM_CONFIG[key]

    def _align_market_df(self, df, platform):
        config = self._get_config(platform)
        if df.empty:
            return pd.DataFrame(columns=config["required_columns"])
        aligned = align_columns(df, config["required_columns"])
        aligned["Bet ID"] = aligned["Bet ID"].astype(str)
        return aligned

    def load_data(self, platform):
        config = self._get_config(platform)
        filepath = config["filepath"]
        if os.path.exists(filepath):
            try:
                # Check if it's an LFS pointer
                with open(filepath, 'r') as f:
                    head = f.read(50)
                if head.startswith("version https://git-lfs.github.com/spec/v1"):
                    print(f"Warning: {filepath} is a Git LFS pointer. Treating as empty.")
                    return pd.DataFrame()

                df = pd.read_csv(filepath)
                if "Bet ID" not in df.columns:
                    print(f"Warning: {filepath} missing 'Bet ID' column. Treating as corrupted/empty.")
                    return pd.DataFrame(columns=config["required_columns"])
                return self._align_market_df(df, platform)
            except Exception as e:
                print(f"Error loading {filepath}: {e}. Returning empty DataFrame.")
                return pd.DataFrame(columns=config["required_columns"])
        return pd.DataFrame(columns=config["required_columns"])

    def sync_data(self, platform):
        """
        Fetch fresh data, compare with existing CSV, update statuses.
        Returns tuple: (updated_df, stats_dict)
        """
        platform = platform.lower()
        config = self._get_config(platform)
        filepath = config["filepath"]
        
        # 1. Fetch Fresh Data
        print(f"Fetching fresh data for {platform}...")
        df_new = self._align_market_df(config["fetch"](max_pages=50), platform)
            
        if df_new.empty:
            print("Warning: No new data fetched.")
            return self.load_data(platform), {"new": 0, "existing": 0, "removed": 0}

        # 2. Load Existing Data
        df_old = self.load_data(platform)
        
        # 3. Diff Logic
        # Key is 'Bet ID'
        if df_old.empty:
            # All are new
            df_new["SyncStatus"] = "New"
            df_new = self._align_market_df(df_new, platform)
            df_new.to_csv(filepath, index=False)
            return df_new, {"new": len(df_new), "existing": 0, "removed": 0}
        
        # Ensure Bet ID is string
        df_new["Bet ID"] = df_new["Bet ID"].astype(str)
        df_old["Bet ID"] = df_old["Bet ID"].astype(str)
        
        new_ids = set(df_new["Bet ID"])
        old_ids = set(df_old["Bet ID"])
        
        # Calculate sets
        added_ids = new_ids - old_ids
        maintained_ids = new_ids & old_ids
        removed_ids = old_ids - new_ids
        
        # SAFETY CHECK: If we are losing > 50% of data and have > 1000 items, it's likely a fetch error.
        if not df_old.empty and len(df_old) > 1000:
             if len(df_new) < (0.5 * len(df_old)):
                 print(f"CRITICAL WARNING: Attempting to replace {len(df_old)} items with {len(df_new)} items.")
                 print("This looks like a partial fetch failure. Aborting sync to protect data.")
                 return df_old, {"new": 0, "existing": len(df_old), "removed": 0, "error": "Safety Abort"}

        print(f"Stats: {len(added_ids)} New, {len(maintained_ids)} Existing, {len(removed_ids)} Removed")
        
        # Construct Final DataFrame
        
        # for Maintained: Take the NEW data (to update prices/status) 
        # BUT preserve "New" status if it was already "New" (so we don't lose it before matching)
        df_maintained = df_new[df_new["Bet ID"].isin(maintained_ids)].copy()
        
        # Build lookup for old statuses
        if "SyncStatus" in df_old.columns:
            old_status_map = df_old.set_index("Bet ID")["SyncStatus"].to_dict()
        else:
            old_status_map = {}
        
        def resolve_status(row):
            bid = str(row["Bet ID"])
            prev = old_status_map.get(bid, "Existing")
            # If it was New, keep it New. If it was Existing, keep Existing.
            return prev
            
        df_maintained["SyncStatus"] = df_maintained.apply(resolve_status, axis=1)
        
        # for Added: Take the NEW data, flag as New
        df_added = df_new[df_new["Bet ID"].isin(added_ids)].copy()
        df_added["SyncStatus"] = "New"
        
        # for Removed: User said "ones not active anymore get removed". 
        # So we do NOT include them in the final DF.
        
        # Concatenate
        df_final = pd.concat([df_maintained, df_added], ignore_index=True)
        df_final = self._align_market_df(df_final, platform)
        
        # Save
        df_final.to_csv(filepath, index=False)
        
        stats = {
            "new": len(added_ids),
            "existing": len(maintained_ids),
            "removed": len(removed_ids)
        }
        
        return df_final, stats

    def reset_pipeline_data(self):
        """Delete all intermediate and result files for a clean start."""
        files_to_delete = [
            KALSHI_CSV,
            POLY_CSV,
            os.path.join(ROOT_DIR, "data", "clusters", "event_signatures.csv"),
            os.path.join(ROOT_DIR, "data", "clusters", "graph_data.json"),
            os.path.join(ROOT_DIR, "data", "clusters", "candidate_clusters.json"),
            os.path.join(ROOT_DIR, "data", "clusters", "candidate_outcome_pairs.json"),
            os.path.join(ROOT_DIR, "data", "matches", "confirmed_matches.csv"),
            os.path.join(ROOT_DIR, "data", "matches", "confirmed_matches_bets.csv"),
            os.path.join(ROOT_DIR, "logs", "llm_prompts.log"),
            os.path.join(ROOT_DIR, "config", "poly_id_map.json"),
            "matches_hybrid.csv",
            "matches_llm.csv",
            "pruned_inversions.csv",
            "markets_clean_kalshi_test.csv",
            "markets_clean_polymarket_test.csv"
        ]
        deleted_count = 0
        for f in files_to_delete:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"Deleted: {f}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
        return deleted_count

if __name__ == "__main__":
    # Test run
    dm = MarketDataManager()
    # df, stats = dm.sync_data("kalshi")
    # print(stats)
