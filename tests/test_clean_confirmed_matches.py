import json
import os
import sys
import tempfile

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pipeline.postprocess.clean_confirmed_matches as cleaner  # noqa: E402


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cleaner_removes_deterministically_invalid_event_bets_and_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        kalshi_path = os.path.join(tmpdir, "kalshi.csv")
        poly_path = os.path.join(tmpdir, "poly.csv")
        events_path = os.path.join(tmpdir, "confirmed_matches.csv")
        bets_path = os.path.join(tmpdir, "confirmed_matches_bets.csv")
        candidates_path = os.path.join(tmpdir, "candidate_clusters.json")
        cache_path = os.path.join(tmpdir, "match_cache.jsonl")

        originals = {
            "KALSHI_MARKETS_CSV": cleaner.KALSHI_MARKETS_CSV,
            "POLY_MARKETS_CSV": cleaner.POLY_MARKETS_CSV,
            "CONFIRMED_EVENTS_CSV": cleaner.CONFIRMED_EVENTS_CSV,
            "CONFIRMED_BETS_CSV": cleaner.CONFIRMED_BETS_CSV,
            "CANDIDATE_CLUSTERS_JSON": cleaner.CANDIDATE_CLUSTERS_JSON,
            "MATCH_CACHE_FILE": cleaner.MATCH_CACHE_FILE,
        }
        cleaner.KALSHI_MARKETS_CSV = kalshi_path
        cleaner.POLY_MARKETS_CSV = poly_path
        cleaner.CONFIRMED_EVENTS_CSV = events_path
        cleaner.CONFIRMED_BETS_CSV = bets_path
        cleaner.CANDIDATE_CLUSTERS_JSON = candidates_path
        cleaner.MATCH_CACHE_FILE = cache_path

        try:
            write_csv(
                kalshi_path,
                [
                    {
                        "Event ID": "KGOOD",
                        "Bet ID": "KGOOD-BET",
                        "Market Title": "Which bills will become law in 2026?",
                        "Bet Title": "Bill A",
                        "Category": "Politics",
                        "Sub-Category": "Congress",
                    },
                    {
                        "Event ID": "KXLOLMAP-26MAR03IGAL-1",
                        "Bet ID": "KXLOLMAP-26MAR03IGAL-1-BET",
                        "Market Title": "LPL 2026",
                        "Bet Title": "Map 1",
                        "Category": "Sports",
                        "Sub-Category": "Esports",
                    },
                ],
            )
            write_csv(
                poly_path,
                [
                    {
                        "Event ID": "PGOOD",
                        "Bet ID": "PGOOD-BET",
                        "Market Title": "Which bills will become law in 2026?",
                        "Bet Title": "Bill A",
                        "Category": "Politics",
                        "Sub-Category": "Politics",
                    },
                    {
                        "Event ID": "PBAD",
                        "Bet ID": "PBAD-BET",
                        "Market Title": "LoL: LPL 2026 Season Winner",
                        "Bet Title": "Team",
                        "Category": "Sports",
                        "Sub-Category": "Esports",
                    },
                ],
            )
            write_csv(
                events_path,
                [
                    {
                        "Kalshi_ID": "KGOOD",
                        "Polymarket_ID": "PGOOD",
                        "Kalshi_Title": "Which bills will become law in 2026?",
                        "Polymarket_Title": "Which bills will become law in 2026?",
                        "Common_Keywords": "['bills', 'law']",
                        "Match_Score": "2",
                        "Reasoning": "ok",
                    },
                    {
                        "Kalshi_ID": "KXLOLMAP-26MAR03IGAL-1",
                        "Polymarket_ID": "PBAD",
                        "Kalshi_Title": "LPL 2026",
                        "Polymarket_Title": "LoL: LPL 2026 Season Winner",
                        "Common_Keywords": "['lpl']",
                        "Match_Score": "1",
                        "Reasoning": "bad",
                    },
                ],
            )
            write_csv(
                bets_path,
                [
                    {
                        "Kalshi_Event_ID": "KGOOD",
                        "Kalshi_Bet_ID": "KGOOD-BET",
                        "Polymarket_Event_ID": "PGOOD",
                        "Polymarket_Bet_ID": "PGOOD-BET",
                    },
                    {
                        "Kalshi_Event_ID": "KXLOLMAP-26MAR03IGAL-1",
                        "Kalshi_Bet_ID": "KXLOLMAP-26MAR03IGAL-1-BET",
                        "Polymarket_Event_ID": "PBAD",
                        "Polymarket_Bet_ID": "PBAD-BET",
                    },
                ],
            )
            with open(candidates_path, "w") as handle:
                json.dump(
                    [
                        {
                            "kalshi": {
                                "event_id": "KXLOLMAP-26MAR03IGAL-1",
                                "title": "LPL 2026",
                                "category": "Sports",
                                "sub_category": "Esports",
                            },
                            "polymarket": {
                                "event_id": "PBAD",
                                "title": "LoL: LPL 2026 Season Winner",
                                "category": "Sports",
                                "sub_category": "Esports",
                            },
                            "common_keywords": ["lpl"],
                            "match_score": 1,
                        }
                    ],
                    handle,
                )
            with open(cache_path, "w") as handle:
                handle.write(json.dumps({"kalshi_id": "KGOOD", "polymarket_id": "PGOOD", "is_expired": False}) + "\n")
                handle.write(json.dumps({"kalshi_id": "KXLOLMAP-26MAR03IGAL-1", "polymarket_id": "PBAD", "is_expired": False}) + "\n")

            stats = cleaner.clean_files()

            events = pd.read_csv(events_path, dtype=str)
            bets = pd.read_csv(bets_path, dtype=str)
            cache_rows = [json.loads(line) for line in open(cache_path) if line.strip()]
            bad_cache = next(row for row in cache_rows if row["kalshi_id"] == "KXLOLMAP-26MAR03IGAL-1")

            assert stats["events_invalid_removed"] == 1
            assert set(events["Kalshi_ID"]) == {"KGOOD"}
            assert set(bets["Kalshi_Event_ID"]) == {"KGOOD"}
            assert bad_cache["is_expired"] is True
            assert bad_cache["reasoning"] == "expired_by_cleanup:deterministic_reject"
        finally:
            for name, value in originals.items():
                setattr(cleaner, name, value)


def run() -> None:
    test_cleaner_removes_deterministically_invalid_event_bets_and_cache()


if __name__ == "__main__":
    run()
    print("ok")
