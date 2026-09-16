from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache  # noqa: E402
from pipeline.llm_events.decision_schema import Decision, VerificationStage  # noqa: E402
from pipeline.llm_events.evaluation import load_silver_set  # noqa: E402
from pipeline.llm_events.provider import OpenAIProvider  # noqa: E402
from pipeline.llm_events.telemetry import RunTelemetry, write_json_atomic  # noqa: E402
from pipeline.llm_events.verifier import HierarchicalVerifier, VerifierConfig, verifiable_explicit_no  # noqa: E402


CALIBRATION_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_calibration")


def load_low_predictions(split: str) -> dict[str, dict]:
    path = os.path.join(CALIBRATION_DIR, f"{split}_predictions.jsonl")
    return {raw["pair_id"]: raw for raw in (json.loads(line) for line in Path(path).read_text().splitlines())}


def json_eligible(raw: dict, threshold: float) -> bool:
    return (
        raw["actual"] == Decision.NO_MATCH.value
        and float(raw["confidence"]) >= threshold
        and bool(raw["conflicting_fields"])
        and raw["conflicting_fields"] != ["category"]
        and len(raw["evidence"]) >= 2
    )


def run(
    split: str = "development",
    fixed_low_threshold: float | None = None,
    fixed_none_threshold: float | None = None,
) -> dict:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    examples = load_silver_set(split=split)
    envelopes = [example.envelope for example in examples]
    expected = {example.envelope.pair_id: example.expected_decision for example in examples}
    low = load_low_predictions(split)
    config = VerifierConfig(
        cheap_model="gpt-5.4-nano-2026-03-17",
        cheap_reasoning="none",
        max_cost_usd=2.0,
    )
    verifier = HierarchicalVerifier(
        OpenAIProvider(),
        config=config,
        cache=DecisionCache(os.path.join(CALIBRATION_DIR, f"{split}_none_cache.jsonl")),
        telemetry=RunTelemetry(os.path.join(CALIBRATION_DIR, f"{split}_none_telemetry.jsonl")),
    )
    none_records = verifier.process_stage(envelopes, VerificationStage.CHEAP)

    if fixed_low_threshold is not None and fixed_none_threshold is not None:
        thresholds = [(fixed_low_threshold, fixed_none_threshold)]
    else:
        thresholds = [
            (low_value / 100, none_value / 100)
            for low_value in range(75, 100)
            for none_value in range(0, 100)
        ]
    negative_total = sum(value == Decision.NO_MATCH for value in expected.values())
    candidates = []
    for low_threshold, none_threshold in thresholds:
        selected = []
        for envelope in envelopes:
            pair_id = envelope.pair_id
            if not json_eligible(low[pair_id], low_threshold):
                continue
            if not verifiable_explicit_no(none_records[pair_id], none_threshold):
                continue
            selected.append(pair_id)
        true_rejects = sum(expected[pair_id] == Decision.NO_MATCH for pair_id in selected)
        candidates.append(
            {
                "low_threshold": low_threshold,
                "none_threshold": none_threshold,
                "auto_rejects": len(selected),
                "true_rejects": true_rejects,
                "unsafe_rejects": len(selected) - true_rejects,
                "precision": round(true_rejects / len(selected), 6) if selected else 1.0,
                "negative_recall": round(true_rejects / negative_total, 6) if negative_total else 1.0,
            }
        )
    fixed = fixed_low_threshold is not None
    if fixed:
        selected = candidates[0]
    else:
        safe = [candidate for candidate in candidates if candidate["unsafe_rejects"] == 0]
        selected = max(safe, key=lambda candidate: (candidate["true_rejects"], -candidate["low_threshold"], -candidate["none_threshold"]))
    report = {
        "split": split,
        "examples": len(examples),
        "models": ["gpt-5.4-nano/low", "gpt-5.4-nano/none"],
        "fixed_validation": fixed,
        "selected": selected,
        "new_none_cost_usd": round(verifier.spent, 8),
    }
    write_json_atomic(os.path.join(CALIBRATION_DIR, f"{split}_dual_report.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate two-pass cheap rejection consensus")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--fixed-low-threshold", type=float, default=None)
    parser.add_argument("--fixed-none-threshold", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.split, args.fixed_low_threshold, args.fixed_none_threshold), indent=2, sort_keys=True))
