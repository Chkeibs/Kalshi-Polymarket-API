from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache  # noqa: E402
from pipeline.llm_events.decision_schema import Decision, RelationType, VerificationStage  # noqa: E402
from pipeline.llm_events.evaluation import load_silver_set, metrics  # noqa: E402
from pipeline.llm_events.provider import OpenAIProvider  # noqa: E402
from pipeline.llm_events.telemetry import RunTelemetry, write_json_atomic  # noqa: E402
from pipeline.llm_events.verifier import (  # noqa: E402
    KALSHI_CSV,
    POLYMARKET_CSV,
    HierarchicalVerifier,
    VerifierConfig,
    build_context_maps,
    targeted_contexts,
)


CALIBRATION_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_calibration")


def run(
    split: str = "development",
    fixed_match_threshold: float | None = None,
    fixed_no_threshold: float | None = None,
) -> dict:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    examples = load_silver_set(split=split)
    envelopes = [example.envelope for example in examples]
    expected = {example.envelope.pair_id: example for example in examples}
    contexts = targeted_contexts(envelopes, build_context_maps(KALSHI_CSV), build_context_maps(POLYMARKET_CSV))
    config = VerifierConfig(
        detail_model="gpt-5.4-nano-2026-03-17",
        detail_reasoning="low",
        detail_batch_size=8,
        max_cost_usd=0.10,
    )
    verifier = HierarchicalVerifier(
        OpenAIProvider(),
        config=config,
        cache=DecisionCache(os.path.join(CALIBRATION_DIR, f"{split}_rich_cache.jsonl")),
        telemetry=RunTelemetry(os.path.join(CALIBRATION_DIR, f"{split}_rich_telemetry.jsonl")),
    )
    records = verifier.process_stage(envelopes, VerificationStage.DETAIL, contexts)
    base_metrics = metrics(examples, records)

    thresholds = (
        [(fixed_match_threshold, fixed_no_threshold)]
        if fixed_match_threshold is not None and fixed_no_threshold is not None
        else ([] if split == "holdout" else [(match / 100, no / 100) for match in range(50, 100) for no in range(50, 100)])
    )
    candidates = []
    for match_threshold, no_threshold in thresholds:
        accepted_match = []
        accepted_no = []
        for pair_id, record in records.items():
            if record.status != "valid" or len(record.evidence) < 2:
                continue
            if record.decision == Decision.MATCH and record.confidence >= match_threshold:
                accepted_match.append(pair_id)
            elif record.decision == Decision.NO_MATCH and record.confidence >= no_threshold and record.conflicting_fields:
                accepted_no.append(pair_id)
        match_errors = sum(
            expected[pair_id].expected_decision != Decision.MATCH
            or expected[pair_id].expected_relation != records[pair_id].relation_type
            for pair_id in accepted_match
        )
        no_errors = sum(expected[pair_id].expected_decision != Decision.NO_MATCH for pair_id in accepted_no)
        candidates.append(
            {
                "match_threshold": match_threshold,
                "no_threshold": no_threshold,
                "accepted_match": len(accepted_match),
                "accepted_no": len(accepted_no),
                "match_errors": match_errors,
                "no_errors": no_errors,
                "auto_coverage": round((len(accepted_match) + len(accepted_no)) / len(examples), 6),
            }
        )
    fixed = fixed_match_threshold is not None
    if split == "holdout" and not fixed:
        selected = {
            "status": "holdout_not_used_for_threshold_selection",
            "match_threshold": None,
            "no_threshold": None,
            "accepted_match": 0,
            "accepted_no": 0,
            "match_errors": 0,
            "no_errors": 0,
            "auto_coverage": 0.0,
        }
    elif fixed:
        selected = candidates[0]
    else:
        safe = [candidate for candidate in candidates if candidate["match_errors"] == candidate["no_errors"] == 0]
        selected = (
            max(
                safe,
                key=lambda candidate: (
                    candidate["accepted_match"] + candidate["accepted_no"],
                    -candidate["match_threshold"],
                    -candidate["no_threshold"],
                ),
            )
            if safe
            else {
                "match_threshold": None,
                "no_threshold": None,
                "accepted_match": 0,
                "accepted_no": 0,
                "match_errors": 0,
                "no_errors": 0,
                "auto_coverage": 0.0,
                "status": "no_safe_threshold",
            }
        )
    report = {
        "split": split,
        "examples": len(examples),
        "model": config.detail_model,
        "reasoning": config.detail_reasoning,
        "fixed_validation": fixed,
        "selected": selected,
        "metrics": base_metrics,
        "new_cost_usd": round(verifier.spent, 8),
        "cached_total_cost_usd": round(sum(record.cost_usd for record in records.values()), 8),
    }
    write_json_atomic(os.path.join(CALIBRATION_DIR, f"{split}_rich_report.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate rich-context nano decisions")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--fixed-match-threshold", type=float, default=None)
    parser.add_argument("--fixed-no-threshold", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.split, args.fixed_match_threshold, args.fixed_no_threshold), indent=2, sort_keys=True))
