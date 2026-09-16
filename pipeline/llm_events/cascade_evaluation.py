from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache  # noqa: E402
from pipeline.llm_events.consensus import needs_frontier, needs_tiebreak, route_counts, select_consensus  # noqa: E402
from pipeline.llm_events.decision_schema import VerificationStage  # noqa: E402
from pipeline.llm_events.evaluation import load_silver_set, metrics  # noqa: E402
from pipeline.llm_events.prompt_builder import prompt_version  # noqa: E402
from pipeline.llm_events.telemetry import write_json_atomic  # noqa: E402
from pipeline.llm_events.verifier import (  # noqa: E402
    KALSHI_CSV,
    POLYMARKET_CSV,
    build_context_maps,
    targeted_contexts,
)


CALIBRATION_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_calibration")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_annotations")

MODELS = {
    "nano": "gpt-5.4-nano-2026-03-17",
    "mini": "gpt-5.4-mini-2026-03-17",
    "frontier": "gpt-5.4-2026-03-05",
    "tiebreak": "gpt-5.5-2026-04-23",
}


def load_records(cache: DecisionCache, envelopes, contexts, model: str):
    records = {}
    for envelope in envelopes:
        record = cache.get(
            envelope.pair_id,
            envelope.content_hash(VerificationStage.DETAIL, contexts[envelope.pair_id]),
            VerificationStage.DETAIL,
            model,
            prompt_version(VerificationStage.DETAIL),
        )
        if record:
            records[envelope.pair_id] = record
    return records


def run(split: str = "development") -> dict:
    examples = load_silver_set(split=split)
    envelopes = [example.envelope for example in examples]
    contexts = targeted_contexts(envelopes, build_context_maps(KALSHI_CSV), build_context_maps(POLYMARKET_CSV))
    nano = load_records(
        DecisionCache(os.path.join(CALIBRATION_DIR, f"{split}_rich_cache.jsonl")),
        envelopes,
        contexts,
        MODELS["nano"],
    )
    mini = load_records(
        DecisionCache(os.path.join(ANNOTATION_DIR, "mini_low_cache.jsonl")),
        envelopes,
        contexts,
        MODELS["mini"],
    )
    frontier = load_records(
        DecisionCache(os.path.join(ANNOTATION_DIR, "frontier_none_cache.jsonl")),
        envelopes,
        contexts,
        MODELS["frontier"],
    )
    tiebreak = load_records(
        DecisionCache(os.path.join(ANNOTATION_DIR, "frontier_tiebreak_cache.jsonl")),
        envelopes,
        contexts,
        MODELS["tiebreak"],
    )
    missing = {
        "nano": len(envelopes) - len(nano),
        "mini": len(envelopes) - len(mini),
        "frontier": len(envelopes) - len(frontier),
    }
    if any(missing.values()):
        raise RuntimeError(f"cascade caches incomplete: {missing}")

    final = {}
    selected = {}
    frontier_used = set()
    tiebreak_used = set()
    tiebreak_required_missing = []
    for envelope in envelopes:
        pair_id = envelope.pair_id
        requires_frontier = needs_frontier(nano[pair_id], mini[pair_id])
        if requires_frontier:
            frontier_used.add(pair_id)
        requires_tiebreak = requires_frontier and needs_tiebreak(mini[pair_id], frontier[pair_id])
        if requires_tiebreak and pair_id in tiebreak:
            tiebreak_used.add(pair_id)
        elif requires_tiebreak:
            tiebreak_required_missing.append(pair_id)
        result = select_consensus(nano[pair_id], mini[pair_id], frontier[pair_id], tiebreak.get(pair_id))
        selected[pair_id] = result
        final[pair_id] = result.record

    result_metrics = metrics(examples, final)
    standard_cost = (
        sum(record.cost_usd for record in nano.values())
        + sum(record.cost_usd for record in mini.values())
        + sum(frontier[pair_id].cost_usd for pair_id in frontier_used)
        + sum(tiebreak[pair_id].cost_usd for pair_id in tiebreak_used)
    )
    report = {
        "split": split,
        "examples": len(examples),
        "models": MODELS,
        "routes": route_counts(selected),
        "frontier_calls": len(frontier_used),
        "tiebreak_calls": len(tiebreak_used),
        "tiebreak_required_missing": len(tiebreak_required_missing),
        "tiebreak_required_pair_ids": sorted(tiebreak_required_missing),
        "metrics": result_metrics,
        "measured_standard_cost_usd": round(standard_cost, 8),
        "estimated_batch_cost_usd": round(standard_cost / 2, 8),
        "cost_per_pair_standard": round(standard_cost / len(examples), 8),
    }
    write_json_atomic(os.path.join(CALIBRATION_DIR, f"{split}_cascade_report.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the nano-mini-frontier cascade")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    args = parser.parse_args()
    print(json.dumps(run(args.split), indent=2, sort_keys=True))
