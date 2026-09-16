from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache  # noqa: E402
from pipeline.llm_events.decision_schema import Decision, DecisionRecord, RelationType, VerificationStage  # noqa: E402
from pipeline.llm_events.gold_set import REVIEW_CANDIDATES_PATH  # noqa: E402
from pipeline.llm_events.prompt_builder import CandidateEnvelope, from_event_cluster, from_outcome_candidate  # noqa: E402
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


ANNOTATION_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_annotations")
SILVER_PATH = os.path.join(ROOT_DIR, "data", "evaluation", "silver_consensus_v1.jsonl")
DISAGREEMENTS_PATH = os.path.join(ROOT_DIR, "data", "evaluation", "annotation_disagreements.csv")
SILVER_MANIFEST_PATH = os.path.join(ROOT_DIR, "data", "evaluation", "silver_consensus_v1_manifest.json")
HUMAN_RESOLUTIONS_PATH = os.path.join(ROOT_DIR, "data", "evaluation", "human_resolutions_v1.json")

ANNOTATORS = (
    ("mini_low", "gpt-5.4-mini-2026-03-17", "low"),
    ("frontier_none", "gpt-5.4-2026-03-05", "none"),
)
TIEBREAKER = ("frontier_tiebreak", "gpt-5.5-2026-04-23", "none")


def load_review_candidates(limit: int | None = None) -> tuple[list[CandidateEnvelope], dict[str, dict[str, Any]]]:
    envelopes = []
    metadata = {}
    with open(REVIEW_CANDIDATES_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if raw["source_type"] == "event":
                envelope = from_event_cluster(raw["candidate"])
            else:
                envelope = from_outcome_candidate(raw["candidate"])
            envelope = replace(envelope, pair_id=raw["pair_id"])
            envelopes.append(envelope)
            metadata[envelope.pair_id] = raw
            if limit and len(envelopes) >= limit:
                break
    return envelopes, metadata


def annotate(
    name: str,
    model: str,
    reasoning: str | None,
    envelopes: Sequence[CandidateEnvelope],
    contexts: dict[str, dict[str, Any]],
) -> dict[str, DecisionRecord]:
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    config = VerifierConfig(
        detail_model=model,
        detail_reasoning=reasoning,
        detail_batch_size=8,
        max_cost_usd=5.0,
        pricing_mode="standard",
    )
    verifier = HierarchicalVerifier(
        OpenAIProvider(),
        config=config,
        cache=DecisionCache(os.path.join(ANNOTATION_DIR, f"{name}_cache.jsonl")),
        telemetry=RunTelemetry(os.path.join(ANNOTATION_DIR, f"{name}_telemetry.jsonl")),
    )
    records = verifier.process_stage(list(envelopes), VerificationStage.DETAIL, contexts)
    report = {
        "annotator": name,
        "model": model,
        "reasoning": reasoning,
        "records": len(records),
        "cost_usd": round(sum(record.cost_usd for record in records.values()), 8),
        "new_cost_usd": round(verifier.spent, 8),
        "input_tokens": sum(record.usage.input_tokens for record in records.values()),
        "output_tokens": sum(record.usage.output_tokens for record in records.values()),
        "invalid": sum(record.status != "valid" for record in records.values()),
    }
    write_json_atomic(os.path.join(ANNOTATION_DIR, f"{name}_report.json"), report)
    return records


def consensus(
    envelopes: Sequence[CandidateEnvelope],
    metadata: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, DecisionRecord]],
) -> dict[str, Any]:
    names = sorted(annotations)
    if len(names) < 2:
        raise ValueError("at least two annotators are required")
    accepted = []
    disagreements = []
    counts = Counter()
    human_payload = json.loads(Path(HUMAN_RESOLUTIONS_PATH).read_text(encoding="utf-8"))
    human_resolutions = human_payload.get("resolutions", {})
    for envelope in envelopes:
        records = [annotations[name].get(envelope.pair_id) for name in names]
        valid = [record for record in records if record is not None and record.status == "valid"]

        def vote_key(record: DecisionRecord) -> tuple[str, str]:
            relation = record.relation_type.value if record.decision == Decision.MATCH else RelationType.NONE.value
            return record.decision.value, relation

        votes = Counter(vote_key(record) for record in valid)
        winning_key, winning_votes = votes.most_common(1)[0] if votes else (("", ""), 0)
        winners = [record for record in valid if vote_key(record) == winning_key]
        human = human_resolutions.get(envelope.pair_id)
        if winning_votes >= 2 or human:
            first = winners[0] if winners else valid[0]
            final_decision = Decision(human["decision"]) if human else first.decision
            final_relation = RelationType(human["relation_type"]) if human else first.relation_type
            raw = metadata[envelope.pair_id]
            accepted.append(
                {
                    "example_id": envelope.pair_id,
                    "split": "holdout" if int(envelope.pair_id[-2:], 16) % 5 == 0 else "development",
                    "provenance": "human_resolved_model_disagreement" if human else "independent_model_consensus_pending_human_audit",
                    "known_status": raw["known_status"],
                    "expected_decision": final_decision.value,
                    "expected_relation": final_relation.value,
                    "annotators": names + ([human_payload.get("reviewer", "human")] if human else []),
                    "confidence": 1.0 if human else min(record.confidence for record in winners),
                    "candidate_type": envelope.candidate_type,
                    "source_type": raw["source_type"],
                    "candidate": raw["candidate"],
                }
            )
            counts[f"accepted:{final_decision.value}"] += 1
            counts[f"accepted_type:{envelope.candidate_type}"] += 1
            if human:
                counts["human_resolved"] += 1
        else:
            raw = metadata[envelope.pair_id]
            row = {
                "Pair_ID": envelope.pair_id,
                "Candidate_Type": envelope.candidate_type,
                "Known_Status": raw["known_status"],
                "Kalshi_Title": envelope.kalshi.get("title") or envelope.kalshi.get("market_title", ""),
                "Polymarket_Title": envelope.polymarket.get("title") or envelope.polymarket.get("market_title", ""),
                "Human_Final_Label": "",
                "Human_Notes": "",
            }
            for name, record in zip(names, records):
                row[f"{name}_Decision"] = record.decision.value if record else "NOT_RUN"
                row[f"{name}_Reason"] = record.reason_code.value if record else ""
            disagreements.append(row)
            counts["disagreement"] += 1

    with open(SILVER_PATH, "w", encoding="utf-8") as handle:
        for item in sorted(accepted, key=lambda value: value["example_id"]):
            handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
    import pandas as pd

    pd.DataFrame(disagreements).to_csv(DISAGREEMENTS_PATH, index=False)
    manifest = {
        "version": "silver-consensus-v1",
        "annotators": names,
        "reviewed_candidates": len(envelopes),
        "consensus": len(accepted),
        "disagreements": len(disagreements),
        "counts": dict(sorted(counts.items())),
        "limitation": "Model consensus is a silver set. Human review is required before calling it gold.",
    }
    write_json_atomic(SILVER_MANIFEST_PATH, manifest)
    return manifest


def run(limit: int | None = None) -> dict[str, Any]:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    envelopes, metadata = load_review_candidates(limit)
    contexts = targeted_contexts(envelopes, build_context_maps(KALSHI_CSV), build_context_maps(POLYMARKET_CSV))
    annotations = {}
    for name, model, reasoning in ANNOTATORS:
        print(f"Annotating {len(envelopes)} candidates with {name}: {model}/{reasoning}")
        annotations[name] = annotate(name, model, reasoning, envelopes, contexts)
    first_name, second_name = ANNOTATORS[0][0], ANNOTATORS[1][0]
    disagreements = []
    for envelope in envelopes:
        first = annotations[first_name].get(envelope.pair_id)
        second = annotations[second_name].get(envelope.pair_id)
        if not first or not second or first.status != "valid" or second.status != "valid":
            disagreements.append(envelope)
            continue
        first_key = (first.decision, first.relation_type if first.decision == Decision.MATCH else RelationType.NONE)
        second_key = (second.decision, second.relation_type if second.decision == Decision.MATCH else RelationType.NONE)
        if first_key != second_key:
            disagreements.append(envelope)
    tie_name, tie_model, tie_reasoning = TIEBREAKER
    print(f"Annotating {len(disagreements)} disagreements with {tie_name}: {tie_model}/{tie_reasoning}")
    annotations[tie_name] = annotate(tie_name, tie_model, tie_reasoning, disagreements, contexts) if disagreements else {}
    return consensus(envelopes, metadata, annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an independently annotated silver evaluation set")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.limit), indent=2, sort_keys=True))
