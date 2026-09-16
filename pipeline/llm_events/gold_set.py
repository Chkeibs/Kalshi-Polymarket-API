from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_schema import Decision, ReasonCode, RelationType, stable_hash
from pipeline.llm_events.prompt_builder import CandidateEnvelope, from_event_cluster, from_outcome_candidate


CANDIDATES_PATH = os.path.join(ROOT_DIR, "data", "clusters", "candidate_clusters.json")
CONFIRMED_PATH = os.path.join(ROOT_DIR, "data", "matches", "confirmed_matches.csv")
GOLD_PATH = os.path.join(ROOT_DIR, "data", "evaluation", "llm_gold_v1.jsonl")
MANIFEST_PATH = os.path.join(ROOT_DIR, "data", "evaluation", "llm_gold_v1_manifest.json")
REVIEW_QUEUE_PATH = os.path.join(ROOT_DIR, "data", "evaluation", "manual_review_queue.csv")
REVIEW_CANDIDATES_PATH = os.path.join(ROOT_DIR, "data", "evaluation", "review_candidates_v1.jsonl")
OUTCOME_CANDIDATES_PATH = os.path.join(ROOT_DIR, "data", "clusters", "candidate_outcome_pairs.json")

GENERIC_TERMS = {
    "before", "election", "event", "game", "law", "market", "primary", "season",
    "winner", "will", "year", "yes", "world", "price", "president", "senate", "house",
}


@dataclass(frozen=True)
class GoldExample:
    example_id: str
    base_pair_id: str
    split: str
    provenance: str
    expected_decision: Decision
    expected_relation: RelationType
    expected_reason: ReasonCode
    envelope: CandidateEnvelope

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "base_pair_id": self.base_pair_id,
            "split": self.split,
            "provenance": self.provenance,
            "expected_decision": self.expected_decision.value,
            "expected_relation": self.expected_relation.value,
            "expected_reason": self.expected_reason.value,
            "candidate": self.envelope.source,
        }


def _split(base_pair_id: str) -> str:
    bucket = int(hashlib.sha256(base_pair_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "holdout" if bucket < 2 else "development"


def _example(
    base: CandidateEnvelope,
    candidate: dict[str, Any],
    provenance: str,
    decision: Decision,
    reason: ReasonCode,
) -> GoldExample:
    example_id = f"gold-{stable_hash({'base': base.pair_id, 'provenance': provenance, 'candidate': candidate})[:20]}"
    envelope = replace(from_event_cluster(candidate), pair_id=example_id)
    return GoldExample(
        example_id=example_id,
        base_pair_id=base.pair_id,
        split=_split(base.pair_id),
        provenance=provenance,
        expected_decision=decision,
        expected_relation=RelationType.EVENT_EVENT if decision == Decision.MATCH else RelationType.NONE,
        expected_reason=reason,
        envelope=envelope,
    )


def _replace_title(candidate: dict[str, Any], side: str, title: str) -> dict[str, Any]:
    cloned = json.loads(json.dumps(candidate))
    cloned[side]["title"] = title
    return cloned


def _sync_keywords(candidate: dict[str, Any]) -> dict[str, Any]:
    for side in ("kalshi", "polymarket"):
        title = str(candidate[side].get("title", "")).lower()
        keywords = []
        for keyword in candidate[side].get("keywords", []):
            value = str(keyword)
            if value.startswith("year:"):
                if value.split(":", 1)[1] in title:
                    keywords.append(value)
            elif value.lower() in title:
                keywords.append(value)
        for year in re.findall(r"\b20[2-3]\d\b", title):
            token = f"year:{year}"
            if token not in keywords:
                keywords.append(token)
        candidate[side]["keywords"] = sorted(set(keywords))
    candidate["common_keywords"] = sorted(
        set(candidate["kalshi"].get("keywords", [])).intersection(candidate["polymarket"].get("keywords", []))
    )
    candidate["match_score"] = max(0, len(candidate["common_keywords"]) - 1)
    return candidate


def _subject_conflict(base: CandidateEnvelope) -> GoldExample | None:
    candidate = json.loads(json.dumps(base.source))
    title = str(candidate["polymarket"].get("title", ""))
    common = sorted(
        (
            str(term) for term in candidate.get("common_keywords", [])
            if len(str(term)) >= 4 and str(term).lower() not in GENERIC_TERMS and not str(term).startswith("year:")
        ),
        key=lambda value: (-len(value), value),
    )
    if not common:
        return None
    term = common[0]
    mutated, count = re.subn(rf"\b{re.escape(term)}\b", "UnrelatedSubject", title, count=1, flags=re.IGNORECASE)
    if not count:
        mutated = f"{title} [explicitly a different subject: UnrelatedSubject]"
    return _example(
        base,
        _sync_keywords(_replace_title(candidate, "polymarket", mutated)),
        "synthetic_explicit_subject_conflict",
        Decision.NO_MATCH,
        ReasonCode.DIFFERENT_SUBJECT,
    )


def _year_conflict(base: CandidateEnvelope) -> GoldExample | None:
    candidate = json.loads(json.dumps(base.source))
    title = str(candidate["polymarket"].get("title", ""))
    match = re.search(r"\b(20[2-3]\d)\b", title)
    if not match:
        return None
    original = int(match.group(1))
    replacement = original + 4 if original <= 2035 else original - 4
    mutated = title[: match.start()] + str(replacement) + title[match.end() :]
    return _example(
        base,
        _sync_keywords(_replace_title(candidate, "polymarket", mutated)),
        "synthetic_explicit_time_conflict",
        Decision.NO_MATCH,
        ReasonCode.DIFFERENT_TIME,
    )


def _missing_year(base: CandidateEnvelope) -> GoldExample | None:
    candidate = json.loads(json.dumps(base.source))
    k_title = str(candidate["kalshi"].get("title", ""))
    p_title = str(candidate["polymarket"].get("title", ""))
    if not re.search(r"\b20[2-3]\d\b", k_title) or not re.search(r"\b20[2-3]\d\b", p_title):
        return None
    mutated = re.sub(r"\b20[2-3]\d\b", "", p_title)
    mutated = re.sub(r"\s+", " ", mutated).strip()
    return _example(
        base,
        _sync_keywords(_replace_title(candidate, "polymarket", mutated)),
        "synthetic_missing_time",
        Decision.AMBIGUOUS,
        ReasonCode.MISSING_CRITICAL_INFO,
    )


def _threshold_conflict(base: CandidateEnvelope) -> GoldExample | None:
    candidate = json.loads(json.dumps(base.source))
    title = str(candidate["polymarket"].get("title", ""))
    numbers = list(re.finditer(r"\b(\d+(?:\.\d+)?)\b", title))
    for match in numbers:
        value = match.group(1)
        if re.fullmatch(r"20[2-3]\d", value):
            continue
        replacement = str(int(float(value)) + 7)
        mutated = title[: match.start()] + replacement + title[match.end() :]
        return _example(
            base,
            _sync_keywords(_replace_title(candidate, "polymarket", mutated)),
            "synthetic_explicit_threshold_conflict",
            Decision.NO_MATCH,
            ReasonCode.DIFFERENT_THRESHOLD,
        )
    return None


def load_legacy_positive_envelopes() -> list[CandidateEnvelope]:
    candidates = json.load(open(CANDIDATES_PATH, "r", encoding="utf-8"))
    by_pair = {
        (str(candidate["kalshi"]["event_id"]), str(candidate["polymarket"]["event_id"])): candidate
        for candidate in candidates
    }
    confirmed = pd.read_csv(CONFIRMED_PATH, dtype=str).fillna("")
    envelopes = []
    for row in confirmed.to_dict("records"):
        candidate = by_pair.get((str(row.get("Kalshi_ID", "")), str(row.get("Polymarket_ID", ""))))
        if candidate:
            envelopes.append(from_event_cluster(candidate))
    return sorted(envelopes, key=lambda envelope: envelope.pair_id)


def build_gold_examples(target_size: int = 1000) -> list[GoldExample]:
    positives = load_legacy_positive_envelopes()
    examples: list[GoldExample] = []
    for base in positives:
        examples.append(
            _example(
                base,
                json.loads(json.dumps(base.source)),
                "legacy_confirmed_positive_pending_reaudit",
                Decision.MATCH,
                ReasonCode.SAME_PREDICATE,
            )
        )

    mutations = (_subject_conflict, _year_conflict, _missing_year, _threshold_conflict)
    for mutation in mutations:
        for base in positives:
            item = mutation(base)
            if item:
                examples.append(item)
            if len(examples) >= target_size:
                break
        if len(examples) >= target_size:
            break

    deduplicated = {example.example_id: example for example in examples}
    result = sorted(deduplicated.values(), key=lambda example: example.example_id)
    if len(result) < min(target_size, 800):
        raise RuntimeError(f"Only {len(result)} gold examples generated; expected at least 800")
    return result[:target_size]


def build_manual_review_queue(sample_size: int = 300) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidates = json.load(open(CANDIDATES_PATH, "r", encoding="utf-8"))
    outcome_candidates = json.load(open(OUTCOME_CANDIDATES_PATH, "r", encoding="utf-8"))
    confirmed = pd.read_csv(CONFIRMED_PATH, dtype=str).fillna("")
    confirmed_pairs = set(zip(confirmed["Kalshi_ID"], confirmed["Polymarket_ID"]))
    confirmed_source = []
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        pair = (str(candidate["kalshi"]["event_id"]), str(candidate["polymarket"]["event_id"]))
        if pair in confirmed_pairs:
            confirmed_source.append(candidate)
            continue
        reason = candidate.get("candidate_reason", {}) or {}
        key = (str(reason.get("quality_tier", "weak")), str(reason.get("category_resolution_mode", "unknown")))
        strata.setdefault(key, []).append(candidate)

    selected_events = sorted(confirmed_source, key=stable_hash)[:60]
    unmatched_target = 140
    keys = sorted(strata)
    cursor = {key: 0 for key in keys}
    while len(selected_events) < 60 + unmatched_target and keys:
        next_keys = []
        for key in keys:
            values = sorted(strata[key], key=lambda candidate: stable_hash(candidate))
            index = cursor[key]
            if index < len(values):
                selected_events.append(values[index])
                cursor[key] += 1
                next_keys.append(key)
                if len(selected_events) >= 60 + unmatched_target:
                    break
        keys = next_keys

    selected_outcomes = []
    outcome_strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in outcome_candidates:
        score = int(candidate.get("match_score", 0) or 0)
        score_band = "high" if score >= 20 else "medium" if score >= 12 else "low"
        key = (str(candidate.get("candidate_type", "")), score_band)
        outcome_strata.setdefault(key, []).append(candidate)
    keys = sorted(outcome_strata)
    cursor = {key: 0 for key in keys}
    outcome_target = max(0, sample_size - len(selected_events))
    while len(selected_outcomes) < outcome_target and keys:
        next_keys = []
        for key in keys:
            values = sorted(outcome_strata[key], key=stable_hash)
            index = cursor[key]
            if index < len(values):
                selected_outcomes.append(values[index])
                cursor[key] += 1
                next_keys.append(key)
                if len(selected_outcomes) >= outcome_target:
                    break
        keys = next_keys

    selected: list[tuple[str, dict[str, Any]]] = [("event", candidate) for candidate in selected_events]
    selected.extend(("outcome", candidate) for candidate in selected_outcomes)
    rows = []
    raw_rows = []
    for source_type, candidate in selected:
        envelope = from_event_cluster(candidate) if source_type == "event" else from_outcome_candidate(candidate)
        reason = candidate.get("candidate_reason", {}) or {}
        k_title = envelope.kalshi.get("title") or envelope.kalshi.get("market_title", "")
        p_title = envelope.polymarket.get("title") or envelope.polymarket.get("market_title", "")
        known_status = "legacy_confirmed" if source_type == "event" and (
            str(envelope.kalshi.get("event_id", "")), str(envelope.polymarket.get("event_id", ""))
        ) in confirmed_pairs else "unlabeled"
        rows.append(
            {
                "Pair_ID": envelope.pair_id,
                "Kalshi_ID": envelope.kalshi["event_id"],
                "Kalshi_Title": k_title,
                "Polymarket_ID": envelope.polymarket["event_id"],
                "Polymarket_Title": p_title,
                "Candidate_Type": envelope.candidate_type,
                "Known_Status": known_status,
                "Quality_Tier": reason.get("quality_tier", ""),
                "Category_Mode": reason.get("category_resolution_mode", ""),
                "Annotator_1": "",
                "Annotator_2": "",
                "Final_Label": "",
                "Relation_Type": "",
                "Conflict_Or_Missing_Field": "",
                "Notes": "",
            }
        )
        raw_rows.append(
            {
                "pair_id": envelope.pair_id,
                "candidate_type": envelope.candidate_type,
                "known_status": known_status,
                "source_type": source_type,
                "candidate": candidate,
            }
        )
    return pd.DataFrame(rows), raw_rows


def write_gold_set(target_size: int = 1000) -> dict[str, Any]:
    examples = build_gold_examples(target_size)
    Path(GOLD_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD_PATH, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
    review, review_candidates = build_manual_review_queue()
    review.to_csv(REVIEW_QUEUE_PATH, index=False)
    with open(REVIEW_CANDIDATES_PATH, "w", encoding="utf-8") as handle:
        for candidate in review_candidates:
            handle.write(json.dumps(candidate, ensure_ascii=True, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for example in examples:
        for key in (example.split, example.expected_decision.value, example.provenance):
            counts[key] = counts.get(key, 0) + 1
    manifest = {
        "version": "llm-gold-v1",
        "examples": len(examples),
        "sha256": hashlib.sha256(Path(GOLD_PATH).read_bytes()).hexdigest(),
        "counts": dict(sorted(counts.items())),
        "manual_review_queue": len(review),
        "manual_review_candidates_sha256": hashlib.sha256(Path(REVIEW_CANDIDATES_PATH).read_bytes()).hexdigest(),
        "limitations": [
            "Legacy confirmed positives require independent human re-audit.",
            "Synthetic mutations test the decision contract but do not replace real negative labels.",
            "Production activation requires two annotations on the manual review queue.",
        ],
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_gold_set(path: str = GOLD_PATH, split: str | None = None) -> list[GoldExample]:
    result = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if split and raw["split"] != split:
                continue
            envelope = replace(from_event_cluster(raw["candidate"]), pair_id=raw["example_id"])
            result.append(
                GoldExample(
                    example_id=raw["example_id"],
                    base_pair_id=raw["base_pair_id"],
                    split=raw["split"],
                    provenance=raw["provenance"],
                    expected_decision=Decision(raw["expected_decision"]),
                    expected_relation=RelationType(raw["expected_relation"]),
                    expected_reason=ReasonCode(raw["expected_reason"]),
                    envelope=envelope,
                )
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the versioned LLM evaluation set")
    parser.add_argument("--size", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(write_gold_set(args.size), indent=2, sort_keys=True))
