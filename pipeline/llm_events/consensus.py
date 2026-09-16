from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

from pipeline.llm_events.decision_schema import (
    Decision,
    DecisionRecord,
    ReasonCode,
    RelationType,
)


@dataclass(frozen=True)
class ConsensusResult:
    record: DecisionRecord
    route: str
    needs_frontier: bool = False
    needs_tiebreak: bool = False


def vote_key(record: DecisionRecord) -> tuple[Decision, RelationType]:
    relation = record.relation_type if record.decision == Decision.MATCH else RelationType.NONE
    return record.decision, relation


def _valid(record: DecisionRecord | None) -> bool:
    return record is not None and record.status == "valid"


def _ambiguous(base: DecisionRecord, route: str) -> ConsensusResult:
    record = copy.deepcopy(base)
    record.decision = Decision.AMBIGUOUS
    record.relation_type = RelationType.NONE
    record.reason_code = ReasonCode.INSUFFICIENT_EVIDENCE
    record.conflicting_fields = []
    record.source = f"consensus:{route}"
    return ConsensusResult(record=record, route=route)


def _accepted(base: DecisionRecord, route: str) -> ConsensusResult:
    record = copy.deepcopy(base)
    record.source = f"consensus:{route}"
    return ConsensusResult(record=record, route=route)


def needs_frontier(primary: DecisionRecord, secondary: DecisionRecord) -> bool:
    return not (
        _valid(primary)
        and _valid(secondary)
        and primary.decision != Decision.AMBIGUOUS
        and vote_key(primary) == vote_key(secondary)
    )


def needs_tiebreak(secondary: DecisionRecord, frontier: DecisionRecord) -> bool:
    if not _valid(secondary) or not _valid(frontier):
        return True
    if vote_key(secondary) == vote_key(frontier):
        return secondary.decision == Decision.MATCH
    return not (secondary.decision == frontier.decision == Decision.AMBIGUOUS)


def select_consensus(
    primary: DecisionRecord,
    secondary: DecisionRecord,
    frontier: DecisionRecord | None = None,
    tiebreak: DecisionRecord | None = None,
) -> ConsensusResult:
    """Select a conservative decision without mutating cached model records.

    A cheap-model agreement may finalize a non-ambiguous decision. Otherwise a
    frontier result is required. MATCH after escalation requires the frontier
    and tiebreak models to agree on the exact relation; NO_MATCH requires two
    agreeing strong models. Every other route abstains.
    """
    fallback = secondary if _valid(secondary) else primary
    if not _valid(primary) or not _valid(secondary):
        result = _ambiguous(fallback, "invalid_initial")
        return ConsensusResult(result.record, result.route, needs_frontier=True)

    if not needs_frontier(primary, secondary):
        return _accepted(secondary, "primary_secondary_agreement")

    if not _valid(frontier):
        result = _ambiguous(fallback, "frontier_required")
        return ConsensusResult(result.record, result.route, needs_frontier=True)

    if vote_key(secondary) == vote_key(frontier):
        if secondary.decision == Decision.NO_MATCH:
            return _accepted(frontier, "secondary_frontier_no_match")
        if secondary.decision == Decision.AMBIGUOUS:
            return _ambiguous(frontier, "secondary_frontier_ambiguous")

    if not needs_tiebreak(secondary, frontier):
        return _ambiguous(frontier, "strong_models_ambiguous")
    if not _valid(tiebreak):
        result = _ambiguous(frontier, "tiebreak_required")
        return ConsensusResult(result.record, result.route, needs_tiebreak=True)

    strong = (secondary, frontier, tiebreak)
    no_votes: dict[tuple[Decision, RelationType], list[DecisionRecord]] = {}
    for record in strong:
        no_votes.setdefault(vote_key(record), []).append(record)
    no_key = (Decision.NO_MATCH, RelationType.NONE)
    if len(no_votes.get(no_key, [])) >= 2:
        return _accepted(no_votes[no_key][-1], "strong_no_match_consensus")

    if frontier.decision == Decision.MATCH and vote_key(frontier) == vote_key(tiebreak):
        return _accepted(tiebreak, "frontier_tiebreak_match")

    return _ambiguous(tiebreak, "unresolved_strong_disagreement")


def route_counts(results: Mapping[str, ConsensusResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results.values():
        counts[result.route] = counts.get(result.route, 0) + 1
    return dict(sorted(counts.items()))
