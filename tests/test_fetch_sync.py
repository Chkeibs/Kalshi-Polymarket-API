import os
import sys
import tempfile

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.common.schemas import KALSHI_MARKET_REQUIRED_COLUMNS, missing_columns
from pipeline.fetch import data_manager
from pipeline.fetch.data_manager import MarketDataManager
from pipeline.fetch.fetch_clean_data import finalize_market_dataframe


def sample_kalshi_df(*bet_ids):
    rows = []
    for bet_id in bet_ids:
        rows.append(
            {
                "Platform": "Kalshi",
                "Event ID": f"EVENT-{bet_id}",
                "Bet ID": bet_id,
                "Market Title": f"Market {bet_id}",
                "Bet Title": f"Bet {bet_id}",
                "Category": "Politics",
                "Sub-Category": "",
                "Series Ticker": "",
                "Description": "",
                "Bet_Rules": "",
                "Event_Description": "",
                "Status": "active",
                "Expiration Date": "2026-01-01 00:00:00",
            }
        )
    return pd.DataFrame(rows)


def test_finalize_market_dataframe_adds_required_columns():
    df = pd.DataFrame([{"Bet ID": "B1", "Market Title": "A"}])
    out = finalize_market_dataframe(df, KALSHI_MARKET_REQUIRED_COLUMNS)
    assert not missing_columns(out.columns, KALSHI_MARKET_REQUIRED_COLUMNS)
    assert list(out.columns[: len(KALSHI_MARKET_REQUIRED_COLUMNS)]) == list(KALSHI_MARKET_REQUIRED_COLUMNS)


def test_market_data_manager_sync_preserves_status_and_removes_inactive():
    original = data_manager.PLATFORM_CONFIG["kalshi"].copy()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_fetch(max_pages=50):
                calls.append(max_pages)
                if len(calls) == 1:
                    return sample_kalshi_df("B1", "B2")
                return sample_kalshi_df("B1", "B3")

            data_manager.PLATFORM_CONFIG["kalshi"] = {
                **original,
                "filepath": os.path.join(tmp, "markets_clean_kalshi.csv"),
                "fetch": fake_fetch,
            }

            manager = MarketDataManager()
            first_df, first_stats = manager.sync_data("kalshi")
            assert first_stats == {"new": 2, "existing": 0, "removed": 0}
            assert set(first_df["SyncStatus"]) == {"New"}

            second_df, second_stats = manager.sync_data("kalshi")
            assert second_stats == {"new": 1, "existing": 1, "removed": 1}
            assert set(second_df["Bet ID"]) == {"B1", "B3"}
            assert second_df.set_index("Bet ID").loc["B1", "SyncStatus"] == "New"
            assert second_df.set_index("Bet ID").loc["B3", "SyncStatus"] == "New"
            assert not missing_columns(second_df.columns, KALSHI_MARKET_REQUIRED_COLUMNS)
    finally:
        data_manager.PLATFORM_CONFIG["kalshi"] = original


def test_market_data_manager_rejects_unknown_platform():
    manager = MarketDataManager()
    try:
        manager.load_data("unknown")
    except ValueError as exc:
        assert "Unknown platform" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown platform")


def run() -> None:
    test_finalize_market_dataframe_adds_required_columns()
    test_market_data_manager_sync_preserves_status_and_removes_inactive()
    test_market_data_manager_rejects_unknown_platform()


if __name__ == "__main__":
    run()
    print("ok")
