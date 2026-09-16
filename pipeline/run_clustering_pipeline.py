#!/usr/bin/env python3
"""Run the deterministic clustering pipeline without any LLM stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
CLUSTERING_DIR = ROOT_DIR / "pipeline" / "clustering"
RUN_REPORT = ROOT_DIR / "data" / "clusters" / "clustering_run_report.json"
AUDIT_REPORT = ROOT_DIR / "data" / "reports" / "clustering" / "clustering_audit.json"


def run_script(name: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(CLUSTERING_DIR / name), *args],
        check=True,
        cwd=ROOT_DIR,
    )


def run(fetch: bool = False, full_index: bool = True) -> dict[str, object]:
    fetch_stats = None
    if fetch:
        from pipeline.fetch.data_manager import MarketDataManager

        manager = MarketDataManager()
        _, kalshi_stats = manager.sync_data("kalshi")
        _, polymarket_stats = manager.sync_data("polymarket")
        fetch_stats = {"kalshi": kalshi_stats, "polymarket": polymarket_stats}

    index_args = ("--full",) if full_index else ()
    run_script("event_keyword_indexer.py", *index_args)
    run_script("group_events_cluster.py")
    run_script("outcome_candidate_generator.py")
    run_script("clustering_audit.py")

    result = {
        "mode": "clustering_only",
        "llm_calls": 0,
        "clustering": json.loads(RUN_REPORT.read_text()),
        "audit": json.loads(AUDIT_REPORT.read_text()),
    }
    if fetch_stats is not None:
        result["fetch"] = fetch_stats
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parsing and clustering without LLM verification.")
    parser.add_argument("--fetch", action="store_true", help="Refresh Kalshi and Polymarket CSVs first")
    parser.add_argument("--incremental-index", action="store_true", help="Reuse unchanged keyword index entries")
    args = parser.parse_args()
    print(json.dumps(run(fetch=args.fetch, full_index=not args.incremental_index), indent=2))


if __name__ == "__main__":
    main()
