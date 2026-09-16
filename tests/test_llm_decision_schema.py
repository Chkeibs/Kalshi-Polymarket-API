import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_schema import (  # noqa: E402
    Decision,
    RelationType,
    validate_decision_payload,
)


def wire(pair_id="p1", decision="M", relation="EE", reason="SP", conflicts=None, missing=None):
    return {
        "id": pair_id,
        "d": decision,
        "r": relation,
        "c": reason,
        "x": conflicts or [],
        "m": missing or [],
        "e": ["left evidence", "right evidence"],
        "q": 93,
    }


def test_compact_wire_schema_maps_to_internal_contract():
    result = validate_decision_payload({"decisions": [wire()]}, {"p1"})[0]
    assert result["decision"] == Decision.MATCH
    assert result["relation_type"] == RelationType.EVENT_EVENT
    assert result["confidence"] == 0.93


def test_no_match_without_explicit_conflict_becomes_ambiguous():
    result = validate_decision_payload(
        {"decisions": [wire(decision="N", relation="N", reason="IE")]},
        {"p1"},
    )[0]
    assert result["decision"] == Decision.AMBIGUOUS
    assert result["relation_type"] == RelationType.NONE


def test_missing_information_is_not_silently_converted_to_no_match():
    result = validate_decision_payload(
        {"decisions": [wire(decision="A", relation="N", reason="MI", missing=["time"])]},
        {"p1"},
    )[0]
    assert result["decision"] == Decision.AMBIGUOUS
    assert result["missing_fields"] == ["time"]


def test_match_with_missing_or_conflicting_fields_is_downgraded():
    result = validate_decision_payload(
        {"decisions": [wire(decision="M", relation="EE", reason="MI", missing=["time"])]},
        {"p1"},
    )[0]
    assert result["decision"] == Decision.AMBIGUOUS
    assert result["relation_type"] == RelationType.NONE


def test_unknown_and_duplicate_pair_ids_are_rejected():
    try:
        validate_decision_payload({"decisions": [wire("wrong")]}, {"p1"})
        raise AssertionError("unknown ID should fail")
    except ValueError:
        pass

    try:
        validate_decision_payload({"decisions": [wire(), wire()]}, {"p1"})
        raise AssertionError("duplicate ID should fail")
    except ValueError:
        pass


def run():
    test_compact_wire_schema_maps_to_internal_contract()
    test_no_match_without_explicit_conflict_becomes_ambiguous()
    test_missing_information_is_not_silently_converted_to_no_match()
    test_match_with_missing_or_conflicting_fields_is_downgraded()
    test_unknown_and_duplicate_pair_ids_are_rejected()


if __name__ == "__main__":
    run()
    print("ok")
