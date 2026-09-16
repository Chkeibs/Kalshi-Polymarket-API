from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Sequence

from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache
from pipeline.llm_events.decision_schema import Decision, DecisionRecord, VerificationStage
from pipeline.llm_events.gold_set import GoldExample, load_gold_set
from pipeline.llm_events.decision_schema import ReasonCode, RelationType
from pipeline.llm_events.prompt_builder import from_event_cluster, from_outcome_candidate
from pipeline.llm_events.silver_set import SILVER_PATH
from pipeline.llm_events.provider import LLMProvider, OpenAIProvider
from pipeline.llm_events.telemetry import RunTelemetry, write_json_atomic
from pipeline.llm_events.verifier import HierarchicalVerifier, VerifierConfig


BENCHMARK_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_benchmarks")


@dataclass(frozen=True)
class ModelCandidate:
    model: str
    reasoning_effort: str | None


DEFAULT_MODELS = (
    ModelCandidate("gpt-5-nano-2025-08-07", "minimal"),
    ModelCandidate("gpt-5.4-nano-2026-03-17", "none"),
    ModelCandidate("gpt-5.4-nano-2026-03-17", "low"),
    ModelCandidate("gpt-5.4-mini-2026-03-17", "none"),
)


def load_silver_set(split: str | None = None) -> list[GoldExample]:
    result = []
    with open(SILVER_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if split and raw["split"] != split:
                continue
            if raw["source_type"] == "event":
                envelope = from_event_cluster(raw["candidate"])
            else:
                envelope = from_outcome_candidate(raw["candidate"])
            envelope = replace(envelope, pair_id=raw["example_id"])
            result.append(
                GoldExample(
                    example_id=raw["example_id"],
                    base_pair_id=raw["example_id"],
                    split=raw["split"],
                    provenance=raw["provenance"],
                    expected_decision=Decision(raw["expected_decision"]),
                    expected_relation=RelationType(raw["expected_relation"]),
                    expected_reason=ReasonCode.OTHER,
                    envelope=envelope,
                )
            )
    return result


def metrics(examples: Sequence[GoldExample], records: dict[str, DecisionRecord]) -> dict[str, Any]:
    expected_by_id = {example.envelope.pair_id: example for example in examples}
    confusion: Counter[tuple[str, str]] = Counter()
    provenance: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    invalid = 0
    false_no_from_missing = 0
    relation_mismatch = 0
    grouped: dict[str, dict[str, list[tuple[GoldExample, DecisionRecord | None, Decision]]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, example in expected_by_id.items():
        record = records.get(pair_id)
        actual = record.decision if record and record.status == "valid" else Decision.AMBIGUOUS
        if record is None or record.status != "valid":
            invalid += 1
        confusion[(example.expected_decision.value, actual.value)] += 1
        provenance[example.provenance][(example.expected_decision.value, actual.value)] += 1
        if example.expected_decision == Decision.AMBIGUOUS and actual == Decision.NO_MATCH:
            false_no_from_missing += 1
        if (
            example.expected_decision == Decision.MATCH
            and actual == Decision.MATCH
            and record is not None
            and record.relation_type != example.expected_relation
        ):
            relation_mismatch += 1
        candidate_reason = example.envelope.shared.get("candidate_reason", {})
        quality_tier = example.envelope.shared.get("quality_tier")
        if not quality_tier and isinstance(candidate_reason, dict):
            quality_tier = candidate_reason.get("quality_tier")
        quality_tier = quality_tier or "unknown"
        categories = sorted(
            {
                str(example.envelope.kalshi.get("category", "unknown") or "unknown"),
                str(example.envelope.polymarket.get("category", "unknown") or "unknown"),
            }
        )
        groups = {
            "candidate_type": example.envelope.candidate_type,
            "quality_tier": str(quality_tier),
            "expected_relation": example.expected_relation.value,
            "category_pair": " | ".join(categories),
            "information_status": "missing_critical_info" if example.expected_decision == Decision.AMBIGUOUS else "explicit",
        }
        for group_name, group_value in groups.items():
            grouped[group_name][group_value].append((example, record, actual))

    raw_match_to_match = confusion[(Decision.MATCH.value, Decision.MATCH.value)]
    true_match = raw_match_to_match - relation_mismatch
    false_match = relation_mismatch + sum(
        count
        for (expected, actual), count in confusion.items()
        if actual == Decision.MATCH.value and expected != Decision.MATCH.value
    )
    missed_match = relation_mismatch + sum(
        count
        for (expected, actual), count in confusion.items()
        if expected == Decision.MATCH.value and actual != Decision.MATCH.value
    )
    precision = true_match / (true_match + false_match) if true_match + false_match else 1.0
    recall = true_match / (true_match + missed_match) if true_match + missed_match else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    resolved = sum(count for (_, actual), count in confusion.items() if actual != Decision.AMBIGUOUS.value)
    unsafe_auto_reject = confusion[(Decision.MATCH.value, Decision.NO_MATCH.value)] + false_no_from_missing
    safe_rejects = confusion[(Decision.NO_MATCH.value, Decision.NO_MATCH.value)]
    all_rejects = sum(count for (_, actual), count in confusion.items() if actual == Decision.NO_MATCH.value)
    safe_reject_precision = safe_rejects / all_rejects if all_rejects else 1.0
    negative_total = sum(count for (expected, _), count in confusion.items() if expected == Decision.NO_MATCH.value)
    non_match_total = len(examples) - sum(count for (expected, _), count in confusion.items() if expected == Decision.MATCH.value)
    false_positive_rate = false_match / non_match_total if non_match_total else 0.0
    false_negative_rate = missed_match / (true_match + missed_match) if true_match + missed_match else 0.0
    safe_reject_recall = safe_rejects / negative_total if negative_total else 1.0

    def summarize_group(items: list[tuple[GoldExample, DecisionRecord | None, Decision]]) -> dict[str, Any]:
        tp = fp = fn = resolved_group = unsafe = 0
        for example, record, actual in items:
            exact_match = (
                actual == Decision.MATCH
                and example.expected_decision == Decision.MATCH
                and record is not None
                and record.relation_type == example.expected_relation
            )
            tp += int(exact_match)
            fp += int(actual == Decision.MATCH and not exact_match)
            fn += int(example.expected_decision == Decision.MATCH and not exact_match)
            resolved_group += int(actual != Decision.AMBIGUOUS)
            unsafe += int(
                actual == Decision.NO_MATCH
                and example.expected_decision in {Decision.MATCH, Decision.AMBIGUOUS}
            )
        group_precision = tp / (tp + fp) if tp + fp else 1.0
        group_recall = tp / (tp + fn) if tp + fn else 1.0
        group_f1 = 2 * group_precision * group_recall / (group_precision + group_recall) if group_precision + group_recall else 0.0
        return {
            "examples": len(items),
            "precision_match": round(group_precision, 6),
            "recall_match": round(group_recall, 6),
            "f1_match": round(group_f1, 6),
            "coverage": round(resolved_group / len(items), 6) if items else 0.0,
            "false_match": fp,
            "unsafe_auto_reject": unsafe,
        }

    return {
        "examples": len(examples),
        "precision_match": round(precision, 6),
        "recall_match": round(recall, 6),
        "f1_match": round(f1, 6),
        "false_positive_rate": round(false_positive_rate, 6),
        "false_negative_rate": round(false_negative_rate, 6),
        "coverage": round(resolved / len(examples), 6) if examples else 0.0,
        "ambiguous_rate": round(1 - resolved / len(examples), 6) if examples else 0.0,
        "false_match": false_match,
        "relation_mismatch": relation_mismatch,
        "match_to_no_match": confusion[(Decision.MATCH.value, Decision.NO_MATCH.value)],
        "false_no_from_missing": false_no_from_missing,
        "unsafe_auto_reject": unsafe_auto_reject,
        "provisional_false_match": false_match,
        "safe_reject_precision": round(safe_reject_precision, 6),
        "safe_reject_recall": round(safe_reject_recall, 6),
        "invalid": invalid,
        "confusion": {f"{expected}->{actual}": count for (expected, actual), count in sorted(confusion.items())},
        "by_provenance": {
            name: {f"{expected}->{actual}": count for (expected, actual), count in sorted(values.items())}
            for name, values in sorted(provenance.items())
        },
        "segments": {
            group_name: {
                value: summarize_group(items)
                for value, items in sorted(values.items())
            }
            for group_name, values in sorted(grouped.items())
        },
    }


def benchmark_model(
    candidate: ModelCandidate,
    examples: Sequence[GoldExample],
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    provider = provider or OpenAIProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        config = VerifierConfig(
            cheap_model=candidate.model,
            cheap_reasoning=candidate.reasoning_effort,
            detail_model=candidate.model,
            review_model=candidate.model,
            max_cost_usd=5.0,
            include_outcomes=False,
        )
        verifier = HierarchicalVerifier(
            provider,
            config=config,
            cache=DecisionCache(os.path.join(tmpdir, "cache.jsonl")),
            telemetry=RunTelemetry(os.path.join(tmpdir, "telemetry.jsonl")),
        )
        envelopes = [example.envelope for example in examples]
        records = verifier.process_stage(envelopes, VerificationStage.CHEAP)
        result = metrics(examples, records)
        result.update(
            {
                "model": candidate.model,
                "reasoning_effort": candidate.reasoning_effort,
                "cost_usd": round(verifier.spent, 8),
                "input_tokens": sum(record.usage.input_tokens for record in verifier.records),
                "output_tokens": sum(record.usage.output_tokens for record in verifier.records),
                "reasoning_tokens": sum(record.usage.reasoning_tokens for record in verifier.records),
                "latency_ms": sum(record.latency_ms for record in verifier.records),
            }
        )
        return result


def choose_best(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        result for result in results
        if result["invalid"] == 0 and result["unsafe_auto_reject"] == 0
    ]
    pool = eligible or list(results)
    return min(
        pool,
        key=lambda result: (
            result["unsafe_auto_reject"],
            result["invalid"],
            -result["safe_reject_precision"],
            -result["safe_reject_recall"],
            result["cost_usd"],
        ),
    )


def run_benchmark(split: str = "development", limit: int | None = None, dataset: str = "contract") -> dict[str, Any]:
    examples = load_gold_set(split=split) if dataset == "contract" else load_silver_set(split=split)
    if limit:
        buckets: dict[str, list[GoldExample]] = defaultdict(list)
        for example in examples:
            buckets[example.expected_decision.value].append(example)
        sampled = []
        keys = sorted(buckets)
        while len(sampled) < limit and keys:
            next_keys = []
            for key in keys:
                if buckets[key]:
                    sampled.append(buckets[key].pop(0))
                    next_keys.append(key)
                    if len(sampled) >= limit:
                        break
            keys = next_keys
        examples = sampled
    results = []
    for candidate in DEFAULT_MODELS:
        print(f"Benchmarking {candidate.model} ({candidate.reasoning_effort}) on {len(examples)} examples...")
        result = benchmark_model(candidate, examples)
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    report = {"dataset": dataset, "split": split, "examples": len(examples), "results": results, "selected": choose_best(results)}
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    write_json_atomic(os.path.join(BENCHMARK_DIR, f"benchmark_{dataset}_{split}_{len(examples)}.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark cheap LLM classifiers")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", choices=("contract", "silver"), default="contract")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.split, args.limit, args.dataset), indent=2, sort_keys=True))
