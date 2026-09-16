import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.canary import (  # noqa: E402
    DIMENSIONS,
    TREATMENTS,
    _validate_canary_decisions,
    build_canary_examples,
    build_canary_requests,
    canary_json_schema,
    run_canary,
)
from pipeline.llm_events.decision_schema import Decision  # noqa: E402


def test_canary_sample_is_event_only_and_has_multiple_label_sources():
    examples = build_canary_examples(40)
    assert len(examples) == 40
    assert all(example.envelope.candidate_type == "event_event" for example in examples)
    assert len({example.label_source for example in examples}) >= 2


def test_canary_schema_requires_all_equivalence_dimensions():
    fields = canary_json_schema()["properties"]["decisions"]["items"]["properties"]["f"]
    assert fields["required"] == list(DIMENSIONS)


def test_canary_match_with_unknown_dimension_is_downgraded_to_ambiguous():
    raw = {
        "id": "example-1",
        "d": "M",
        "r": "EE",
        "c": "SP",
        "x": [],
        "m": [],
        "e": ["Kalshi says X", "Polymarket says X"],
        "q": 99,
        "f": {dimension: "same" for dimension in DIMENSIONS},
    }
    raw["f"]["time"] = "unknown"
    validated, missing = _validate_canary_decisions({"decisions": [raw]}, {"example-1"})
    assert not missing
    assert validated[0][0]["decision"] == Decision.AMBIGUOUS
    assert "time" in validated[0][0]["missing_fields"]


def test_canary_dry_run_has_four_treatments_and_never_writes_production_matches():
    report = run_canary(limit=20, max_cost_usd=1.0, execute=False)
    assert report["mode"] == "dry_run"
    assert report["treatments"] == [treatment.name for treatment in TREATMENTS]
    assert report["estimate"]["requests"] > 0
    assert report["production_effect"].startswith("none")


def test_canary_requests_include_descriptions_only_for_description_treatments():
    examples = build_canary_examples(12)
    contexts = {example.example_id: {"kalshi_context": {"event_description": "left"}, "polymarket_context": {"event_description": "right"}} for example in examples}
    requests = build_canary_requests(examples, contexts, batch_size=4)
    by_treatment = {request.treatment.name: request for request in requests}
    compact_payload = json.loads(by_treatment["compact_matrix_none"].request.user_message.split("INPUT_PAIRS_JSON:\n", 1)[1])
    rich_payload = json.loads(by_treatment["descriptions_matrix_none"].request.user_message.split("INPUT_PAIRS_JSON:\n", 1)[1])
    assert "targeted_context" not in compact_payload["pairs"][0]
    assert rich_payload["pairs"][0]["targeted_context"]["kalshi_context"]["event_description"] == "left"


def test_canary_defaults_to_one_pair_per_request():
    examples = build_canary_examples(12)
    requests = build_canary_requests(examples, {})
    assert len(requests) == len(examples) * len(TREATMENTS)
    assert all(len(request.examples) == 1 for request in requests)


def test_conflict_precedence_treatment_requires_explicit_conflicts_to_be_preserved():
    treatment = next(item for item in TREATMENTS if item.name == "descriptions_conflict_precedence_none")
    from pipeline.llm_events.canary import _system_prompt

    prompt = _system_prompt(treatment)
    assert "never override it" in prompt
    assert "return AMBIGUOUS rather than MATCH" in prompt


def run():
    test_canary_sample_is_event_only_and_has_multiple_label_sources()
    test_canary_schema_requires_all_equivalence_dimensions()
    test_canary_match_with_unknown_dimension_is_downgraded_to_ambiguous()
    test_canary_dry_run_has_four_treatments_and_never_writes_production_matches()
    test_canary_requests_include_descriptions_only_for_description_treatments()
    test_canary_defaults_to_one_pair_per_request()
    test_conflict_precedence_treatment_requires_explicit_conflicts_to_be_preserved()


if __name__ == "__main__":
    run()
    print("ok")
