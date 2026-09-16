from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.telemetry import write_json_atomic  # noqa: E402
from pipeline.llm_events.verifier import load_envelopes  # noqa: E402


REPORT_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_calibration")
OUTPUT_PATH = os.path.join(REPORT_DIR, "consensus_cost_projection.json")


def _load(split: str) -> dict:
    path = Path(REPORT_DIR) / f"{split}_cascade_report.json"
    if not path.exists():
        raise FileNotFoundError(f"run cascade_evaluation.py --split {split} first")
    return json.loads(path.read_text(encoding="utf-8"))


def project(incremental_fraction: float = 0.05) -> dict:
    reports = [_load("development"), _load("holdout")]
    evaluation_pairs = sum(report["examples"] for report in reports)
    standard_cost = sum(report["measured_standard_cost_usd"] for report in reports)
    exact_matches = sum(report["metrics"]["confusion"].get("MATCH->MATCH", 0) for report in reports)
    standard_per_pair = standard_cost / evaluation_pairs
    batch_per_pair = standard_per_pair / 2

    all_candidates = load_envelopes(include_outcomes=True)
    event_candidates = [candidate for candidate in all_candidates if candidate.candidate_type == "event_event"]

    def scenario(count: int) -> dict[str, float | int]:
        batch_cost = count * batch_per_pair
        return {
            "candidates": count,
            "first_run_standard_usd": round(count * standard_per_pair, 2),
            "first_run_batch_usd": round(batch_cost, 2),
            "incremental_batch_usd_at_fraction": round(batch_cost * incremental_fraction, 2),
        }

    output = {
        "basis": {
            "evaluation_pairs": evaluation_pairs,
            "measured_standard_cost_usd": round(standard_cost, 8),
            "standard_cost_per_pair_usd": round(standard_per_pair, 8),
            "batch_cost_per_pair_usd": round(batch_per_pair, 8),
            "batch_cost_per_confirmed_match_usd": round((standard_cost / 2) / exact_matches, 8),
            "incremental_fraction": incremental_fraction,
        },
        "scenarios": {
            "events_only": scenario(len(event_candidates)),
            "events_and_outcomes": scenario(len(all_candidates)),
            "unchanged_cache_hit": {
                "new_or_changed_candidates": 0,
                "llm_cost_usd": 0.0,
            },
        },
        "readme_cost_gate": {
            "recommended_first_run_usd": 0.30,
            "conservative_first_run_usd": 0.55,
            "typical_incremental_usd": 0.03,
            "passed": False,
        },
        "limitations": [
            "Projection assumes the 300-example evaluation routing mix represents production candidates.",
            "The evaluation set is intentionally difficult, so the full-run estimate may be conservative.",
            "OpenAI Batch pricing is modeled as 50% of measured standard usage.",
            "No production activation is allowed while the documented cost gate is missed.",
        ],
    }
    write_json_atomic(OUTPUT_PATH, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project consensus verifier cost from measured evaluation usage")
    parser.add_argument("--incremental-fraction", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(project(args.incremental_fraction), indent=2, sort_keys=True))
