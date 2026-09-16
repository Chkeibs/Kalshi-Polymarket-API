import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.consolidation import consolidate_matches  # noqa: E402
from pipeline.llm_events.decision_schema import (  # noqa: E402
    Decision,
    DecisionRecord,
    ReasonCode,
    RelationType,
    VerificationStage,
)
from pipeline.llm_events.prompt_builder import from_event_cluster  # noqa: E402


def envelope(kid, pid, score):
    return from_event_cluster(
        {
            "kalshi": {"event_id": kid, "title": kid, "category": "Test", "sub_category": "", "keywords": [kid]},
            "polymarket": {"event_id": pid, "title": pid, "category": "Test", "sub_category": "", "keywords": [pid]},
            "common_keywords": ["shared"],
            "match_score": score,
            "candidate_reason": {"quality_score": 0, "quality_tier": "medium"},
        }
    )


def record(env, confidence):
    return DecisionRecord(
        pair_id=env.pair_id,
        content_hash="h",
        decision=Decision.MATCH,
        relation_type=RelationType.EVENT_EVENT,
        reason_code=ReasonCode.SAME_PREDICATE,
        stage=VerificationStage.DETAIL,
        confidence=confidence,
    )


def test_global_matching_beats_greedy_first_yes():
    k1p1 = envelope("K1", "P1", 0)
    k1p2 = envelope("K1", "P2", 0)
    k2p1 = envelope("K2", "P1", 0)
    envelopes = {env.pair_id: env for env in (k1p1, k1p2, k2p1)}
    selected, rejected = consolidate_matches(
        [record(k1p1, 0.90), record(k1p2, 0.80), record(k2p1, 0.70)],
        envelopes,
    )
    selected_ids = {item.pair_id for item in selected}
    assert selected_ids == {k1p2.pair_id, k2p1.pair_id}
    assert {item.pair_id for item in rejected} == {k1p1.pair_id}


def run():
    test_global_matching_beats_greedy_first_yes()


if __name__ == "__main__":
    run()
    print("ok")
