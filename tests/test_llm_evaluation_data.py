import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.gold_set import build_gold_examples  # noqa: E402
from pipeline.llm_events.prompt_builder import build_detail_context, from_event_cluster, from_outcome_candidate  # noqa: E402


def test_gold_set_has_required_size_and_no_family_leakage():
    examples = build_gold_examples(1000)
    assert 800 <= len(examples) <= 1200
    splits_by_base = {}
    for example in examples:
        splits_by_base.setdefault(example.base_pair_id, set()).add(example.split)
    assert all(len(splits) == 1 for splits in splits_by_base.values())


def test_event_context_never_uses_an_arbitrary_contract_rule():
    envelope = from_event_cluster(
        {
            "kalshi": {"event_id": "K", "title": "Who wins?", "category": "Politics", "sub_category": "", "keywords": ["wins"]},
            "polymarket": {"event_id": "P", "title": "Who wins?", "category": "Politics", "sub_category": "", "keywords": ["wins"]},
            "common_keywords": ["wins"],
            "match_score": 1,
        }
    )
    context = {
        "event_description": "A broad event.",
        "contracts": [
            {"bet_id": "A", "bet_title": "Alice", "description": "", "bet_rules": "Alice-only rule", "expiration": ""},
            {"bet_id": "B", "bet_title": "Bob", "description": "", "bet_rules": "Bob-only rule", "expiration": ""},
        ],
    }
    detail = build_detail_context(envelope, context, context)
    assert detail["kalshi_context"]["rules"] == ""
    assert "Alice-only rule" not in str(detail)


def test_outcome_context_selects_the_exact_contract_rule():
    envelope = from_outcome_candidate(
        {
            "candidate_type": "outcome_outcome",
            "kalshi": {
                "event_id": "K", "bet_id": "B", "market_title": "Who wins?", "bet_title": "Bob",
                "outcome_subject": "Bob", "outcome_key": "bob", "market_intent": "wins", "category": "Politics", "sub_category": "",
            },
            "polymarket": {
                "event_id": "P", "bet_id": "PB", "original_market_id": "M", "market_title": "Who wins?", "bet_title": "Bob [Yes]",
                "outcome_subject": "Bob", "outcome_key": "bob", "outcome_name": "Yes", "market_intent": "wins", "category": "Politics", "sub_category": "",
            },
            "shared_market_keywords": ["wins"],
            "shared_outcome_keywords": ["bob"],
            "match_score": 10,
        }
    )
    k_context = {
        "event_description": "A broad event.",
        "contracts": [
            {"bet_id": "A", "bet_title": "Alice", "description": "", "bet_rules": "Alice-only rule", "expiration": ""},
            {"bet_id": "B", "bet_title": "Bob", "description": "Bob description", "bet_rules": "Bob exact rule", "expiration": "2028-01-01"},
        ],
    }
    p_context = {
        "event_description": "A broad event.",
        "contracts": [
            {"bet_id": "PB", "bet_title": "Bob [Yes]", "description": "", "bet_rules": "Bob poly rule", "expiration": "2028-01-01"},
        ],
    }
    detail = build_detail_context(envelope, k_context, p_context)
    assert "Bob exact rule" in detail["kalshi_context"]["rules"]
    assert "Alice-only rule" not in str(detail)


def run():
    test_gold_set_has_required_size_and_no_family_leakage()
    test_event_context_never_uses_an_arbitrary_contract_rule()
    test_outcome_context_selects_the_exact_contract_rule()


if __name__ == "__main__":
    run()
    print("ok")
