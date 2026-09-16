from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache  # noqa: E402
from pipeline.llm_events.decision_schema import Decision, ReasonCode, VerificationStage  # noqa: E402
from pipeline.llm_events.evaluation import load_silver_set  # noqa: E402
from pipeline.llm_events.provider import OpenAIProvider  # noqa: E402
from pipeline.llm_events.telemetry import RunTelemetry, write_json_atomic  # noqa: E402
from pipeline.llm_events.verifier import HierarchicalVerifier, VerifierConfig, verifiable_explicit_no  # noqa: E402


CALIBRATION_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_calibration")


def eligible(record, envelope, threshold: float, strategy: str) -> bool:
    if not verifiable_explicit_no(record, threshold):
        return False
    if strategy == "model_only":
        return True
    parser_conflicts = set(envelope.shared.get("explicit_conflict_fields", []))
    model_conflicts = set(record.conflicting_fields)
    if parser_conflicts.intersection(model_conflicts):
        return True
    if strategy == "parser_backed":
        return False
    return record.reason_code in {ReasonCode.DIFFERENT_SUBJECT, ReasonCode.DIFFERENT_SCOPE} and record.confidence >= threshold


def run(split: str = "development", fixed_threshold: float | None = None, fixed_strategy: str = "model_only") -> dict:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    examples = load_silver_set(split=split)
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    config = VerifierConfig(
        cheap_model="gpt-5.4-nano-2026-03-17",
        cheap_reasoning="low",
        max_cost_usd=2.0,
        include_outcomes=True,
    )
    verifier = HierarchicalVerifier(
        OpenAIProvider(),
        config=config,
        cache=DecisionCache(os.path.join(CALIBRATION_DIR, f"{split}_cache.jsonl")),
        telemetry=RunTelemetry(os.path.join(CALIBRATION_DIR, f"{split}_telemetry.jsonl")),
    )
    envelopes = [example.envelope for example in examples]
    records = verifier.process_stage(envelopes, VerificationStage.CHEAP)
    expected = {example.envelope.pair_id: example.expected_decision for example in examples}
    envelope_map = {envelope.pair_id: envelope for envelope in envelopes}

    predictions_path = os.path.join(CALIBRATION_DIR, f"{split}_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for pair_id in sorted(records):
            record = records[pair_id]
            handle.write(
                json.dumps(
                    {
                        "pair_id": pair_id,
                        "expected": expected[pair_id].value,
                        "actual": record.decision.value,
                        "reason": record.reason_code.value,
                        "confidence": record.confidence,
                        "conflicting_fields": record.conflicting_fields,
                        "missing_fields": record.missing_fields,
                        "evidence": record.evidence,
                        "parser_conflicts": envelope_map[pair_id].shared.get("explicit_conflict_fields", []),
                        "one_sided_fields": envelope_map[pair_id].shared.get("one_sided_fields", []),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )

    candidates = []
    negative_total = sum(value == Decision.NO_MATCH for value in expected.values())
    strategies = (fixed_strategy,) if fixed_threshold is not None else ("model_only", "parser_backed", "parser_or_semantic")
    thresholds = (fixed_threshold,) if fixed_threshold is not None else tuple(value / 100 for value in range(50, 100))
    for strategy in strategies:
        for threshold in thresholds:
            selected = [
                pair_id for pair_id, record in records.items()
                if eligible(record, envelope_map[pair_id], threshold, strategy)
            ]
            true_rejects = sum(expected[pair_id] == Decision.NO_MATCH for pair_id in selected)
            unsafe = len(selected) - true_rejects
            candidates.append(
                {
                    "strategy": strategy,
                    "threshold": threshold,
                    "auto_rejects": len(selected),
                    "true_rejects": true_rejects,
                    "unsafe_rejects": unsafe,
                    "precision": round(true_rejects / len(selected), 6) if selected else 1.0,
                    "negative_recall": round(true_rejects / negative_total, 6) if negative_total else 1.0,
                }
            )
    safe = [candidate for candidate in candidates if candidate["unsafe_rejects"] == 0]
    selected = candidates[0] if fixed_threshold is not None else max(
        safe,
        key=lambda candidate: (candidate["true_rejects"], -candidate["threshold"]),
    )
    report = {
        "split": split,
        "examples": len(examples),
        "model": config.cheap_model,
        "reasoning": config.cheap_reasoning,
        "new_cost_usd": round(verifier.spent, 8),
        "selected": selected,
        "fixed_validation": fixed_threshold is not None,
        "safe_frontier": sorted(
            (candidate for candidate in safe if candidate["auto_rejects"] > 0),
            key=lambda candidate: (-candidate["true_rejects"], candidate["threshold"], candidate["strategy"]),
        )[:20],
        "decision_counts": dict(Counter(record.decision.value for record in records.values())),
        "predictions_path": predictions_path,
    }
    write_json_atomic(os.path.join(CALIBRATION_DIR, f"{split}_report.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate safe cheap-model auto-reject routing")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--fixed-strategy", choices=("model_only", "parser_backed", "parser_or_semantic"), default="model_only")
    args = parser.parse_args()
    print(json.dumps(run(args.split, args.fixed_threshold, args.fixed_strategy), indent=2, sort_keys=True))
