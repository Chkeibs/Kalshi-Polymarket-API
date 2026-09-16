import copy
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.consensus import needs_frontier, needs_tiebreak, select_consensus
from pipeline.llm_events.decision_schema import (
    Decision,
    DecisionRecord,
    ReasonCode,
    RelationType,
    VerificationStage,
)


def record(decision, model, relation=RelationType.NONE, status="valid"):
    return DecisionRecord(
        pair_id="pair-1",
        content_hash="hash",
        decision=decision,
        relation_type=relation,
        reason_code=ReasonCode.SAME_PREDICATE if decision == Decision.MATCH else ReasonCode.EXPLICIT_CONFLICT,
        stage=VerificationStage.DETAIL,
        conflicting_fields=[] if decision == Decision.MATCH else ["time"],
        evidence=["left", "right"],
        confidence=0.95,
        model=model,
        prompt_version="detail-v4",
        status=status,
    )


def test_primary_secondary_exact_agreement_finalizes_without_frontier():
    primary = record(Decision.NO_MATCH, "nano")
    secondary = record(Decision.NO_MATCH, "mini")

    result = select_consensus(primary, secondary)

    assert not needs_frontier(primary, secondary)
    assert result.record.decision == Decision.NO_MATCH
    assert result.route == "primary_secondary_agreement"


def test_frontier_and_tiebreak_must_agree_before_escalated_match():
    primary = record(Decision.AMBIGUOUS, "nano")
    secondary = record(Decision.AMBIGUOUS, "mini")
    frontier = record(Decision.MATCH, "frontier", RelationType.EVENT_EVENT)

    missing = select_consensus(primary, secondary, frontier)
    accepted = select_consensus(
        primary,
        secondary,
        frontier,
        record(Decision.MATCH, "tiebreak", RelationType.EVENT_EVENT),
    )

    assert needs_tiebreak(secondary, frontier)
    assert missing.record.decision == Decision.AMBIGUOUS
    assert missing.needs_tiebreak
    assert accepted.record.decision == Decision.MATCH
    assert accepted.route == "frontier_tiebreak_match"


def test_tiebreak_blocks_unsupported_match_and_cache_records_are_not_mutated():
    primary = record(Decision.MATCH, "nano", RelationType.EVENT_EVENT)
    secondary = record(Decision.MATCH, "mini", RelationType.EVENT_EVENT)
    frontier = record(Decision.NO_MATCH, "frontier")
    tiebreak = record(Decision.AMBIGUOUS, "tiebreak")
    originals = copy.deepcopy((primary, secondary, frontier, tiebreak))

    result = select_consensus(primary, secondary, frontier, tiebreak)

    # Primary+secondary agreement is intentionally accepted before escalation.
    assert result.record.decision == Decision.MATCH
    assert (primary, secondary, frontier, tiebreak) == originals


def test_two_strong_no_match_votes_reject_and_disagreement_otherwise_abstains():
    primary = record(Decision.AMBIGUOUS, "nano")
    secondary = record(Decision.NO_MATCH, "mini")
    frontier = record(Decision.MATCH, "frontier", RelationType.EVENT_EVENT)
    reject = select_consensus(primary, secondary, frontier, record(Decision.NO_MATCH, "tiebreak"))
    abstain = select_consensus(primary, secondary, frontier, record(Decision.AMBIGUOUS, "tiebreak"))

    assert reject.record.decision == Decision.NO_MATCH
    assert reject.route == "strong_no_match_consensus"
    assert abstain.record.decision == Decision.AMBIGUOUS


def test_invalid_initial_record_requires_frontier_instead_of_becoming_no_match():
    primary = record(Decision.NO_MATCH, "nano", status="error")
    secondary = record(Decision.NO_MATCH, "mini")

    result = select_consensus(primary, secondary)

    assert result.record.decision == Decision.AMBIGUOUS
    assert result.needs_frontier
