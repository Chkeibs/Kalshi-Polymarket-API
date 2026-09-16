import json
import os
import shutil
import pandas as pd
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_FILE = os.path.join(ROOT_DIR, "data", "cache", "match_cache.jsonl")
CONFIRMED_CSV = os.path.join(ROOT_DIR, "data", "matches", "confirmed_matches.csv")

class MatchCache:
    def __init__(self, cache_file=CACHE_FILE, confirmed_csv=CONFIRMED_CSV):
        self.cache_file = cache_file
        self.confirmed_csv = confirmed_csv
        self.matches = {}  # Key: (kalshi_id, polymarket_id) -> Match Dict
        self.load()

    def load(self):
        """Load matches from JSONL file."""
        self.matches = {}
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        k_id = str(data.get('kalshi_id', ''))
                        p_id = str(data.get('polymarket_id', ''))
                        if k_id and p_id:
                            self.matches[(k_id, p_id)] = data
                    except json.JSONDecodeError:
                        continue
        
        # Backfill from CSV if cache is empty but CSV exists
        # This handles the migration case
        if not self.matches and self.confirmed_csv and os.path.exists(self.confirmed_csv):
            self._backfill_from_csv()

    def _backfill_from_csv(self):
        """One-time migration from legacy CSV to JSONL cache."""
        print(f"Backfilling match cache from {self.confirmed_csv}...")
        try:
            df = pd.read_csv(self.confirmed_csv)
            now_iso = datetime.now(timezone.utc).isoformat()
            count = 0
            for _, row in df.iterrows():
                k_id = str(row.get("Kalshi_ID", ""))
                p_id = str(row.get("Polymarket_ID", ""))
                
                # Basic validation
                if k_id and p_id and k_id.lower() != "nan" and p_id.lower() != "nan":
                    key = (k_id, p_id)
                    self.matches[key] = {
                        "kalshi_id": k_id,
                        "polymarket_id": p_id,
                        "match_score": 1.0,
                        "match_method": "legacy_csv",
                        "created_at": now_iso,
                        "last_verified_at": now_iso,
                        "is_expired": False,
                        "reasoning": row.get("Reasoning", "")
                    }
                    count += 1
            
            print(f"Backfilled {count} matches from legacy CSV.")
            self.save()
        except Exception as e:
            print(f"Error backfilling from CSV: {e}")

    def save(self):
        """Atomic save to JSONL."""
        temp_file = self.cache_file + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                for m in self.matches.values():
                    f.write(json.dumps(m) + "\n")
            shutil.move(temp_file, self.cache_file)
        except Exception as e:
            print(f"Error saving match cache: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def prune(self, active_kalshi_ids, active_poly_ids):
        """
        Mark matches as expired if they are not in the active sets.
        active_kalshi_ids: set of active Kalshi Event IDs (str)
        active_poly_ids: set of active Polymarket Event IDs (str)
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        stats = {"total": len(self.matches), "kept": 0, "expired": 0, "removed": 0}
        
        for key, m in self.matches.items():
            k_id = m['kalshi_id']
            p_id = m['polymarket_id']
            
            # Check presence in active sets
            is_active_k = k_id in active_kalshi_ids
            is_active_p = p_id in active_poly_ids
            
            if is_active_k and is_active_p:
                m['is_expired'] = False
                m['last_verified_at'] = now_iso
                stats['kept'] += 1
            else:
                m['is_expired'] = True
                stats['expired'] += 1
                
        self.save()
        return stats

    def get_active_matches(self):
        """Return list of match dicts that are NOT expired."""
        return [m for m in self.matches.values() if not m.get('is_expired', False)]

    def get_known_pair_ids(self):
        """Return set of (kalshi_id, polymarket_id) tuples for ALL matches (active or not)."""
        return set(self.matches.keys())
    
    def get_active_pair_ids(self):
         """Return set of (kalshi_id, polymarket_id) tuples for ACTIVE matches only."""
         return {(m['kalshi_id'], m['polymarket_id']) for m in self.get_active_matches()}

    def add_match(self, k_id, p_id, reason="", method="llm", score=1.0):
        """Add or update a single match."""
        now_iso = datetime.now(timezone.utc).isoformat()
        k_id = str(k_id)
        p_id = str(p_id)
        key = (k_id, p_id)
        
        if key in self.matches:
            # Update existing
            m = self.matches[key]
            m['last_verified_at'] = now_iso
            m['is_expired'] = False
            # Update reasoning/score if provided? Usually we trust the latest or keep original.
            # Let's keep original created_at but update reasoning if new
            if reason:
                m['reasoning'] = reason
        else:
            # Create new
            self.matches[key] = {
                "kalshi_id": k_id,
                "polymarket_id": p_id,
                "match_score": score,
                "match_method": method,
                "created_at": now_iso,
                "last_verified_at": now_iso,
                "is_expired": False,
                "reasoning": reason
            }

    def add_matches_from_list(self, match_list):
        """
        Add multiple matches from the standard list format used in the pipeline.
        match_list: list of dicts with keys 'Kalshi_ID', 'Polymarket_ID', 'Reasoning'
        """
        for item in match_list:
            k_id = item.get("Kalshi_ID", item.get("kalshi_id"))
            p_id = item.get("Polymarket_ID", item.get("polymarket_id"))
            if not k_id or not p_id:
                continue
            self.add_match(
                k_id,
                p_id,
                item.get("Reasoning", ""),
                method="llm"
            )
        self.save()
