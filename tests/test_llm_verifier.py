import json
import os
import sys
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache  # noqa: E402
from pipeline.llm_events.decision_schema import Decision, VerificationStage  # noqa: E402
from pipeline.llm_events.prompt_builder import from_event_cluster  # noqa: E402
from pipeline.llm_events.provider import FakeProvider  # noqa: E402
from pipeline.llm_events.telemetry import RunTelemetry  # noqa: E402
from pipeline.llm_events.verifier import HierarchicalVerifier, VerifierConfig  # noqa: E402


def cluster(kid, pid, kt, pt):
    return from_event_cluster(
        {
            "kalshi": {"event_id": kid, "title": kt, "category": "Politics", "sub_category": "Elections", "keywords": ["election"]},
            "polymarket": {"event_id": pid, "title": pt, "category": "Politics", "sub_category": "Elections", "keywords": ["election"]},
            "common_keywords": ["election"],
            "match_score": 3,
            "candidate_reason": {"quality_tier": "medium", "quality_score": 300},
        }
    )


def wire(envelope, decision, relation, reason, conflicts=None, missing=None, confidence=95):
    return {
        "id": envelope.pair_id,
        "d": decision,
        "r": relation,
        "c": reason,
        "x": conflicts or [],
        "m": missing or [],
        "e": ["left title", "right title"],
        "q": confidence,
    }


def test_hierarchical_routing_skips_explicit_negative_and_verifies_match():
    negative = cluster("K1", "P1", "Texas Senate election", "Iowa Senate election")
    positive = cluster("K2", "P2", "France presidential election 2027", "2027 French presidential election")

    def responses(request):
        payload = json.loads(request.user_message.split("INPUT_PAIRS_JSON:\n", 1)[1])
        decisions = []
        for raw in payload["pairs"]:
            envelope = negative if raw["id"] == negative.pair_id else positive
            if "cheap-" in request.custom_id and envelope is negative:
                decisions.append(wire(envelope, "N", "N", "DL", ["location_or_jurisdiction"]))
            else:
                decisions.append(wire(envelope, "M", "EE", "SP"))
        return {"decisions": decisions}

    with tempfile.TemporaryDirectory() as tmpdir:
        provider = FakeProvider(responses)
        verifier = HierarchicalVerifier(
            provider,
            config=VerifierConfig(cheap_model="gpt-5.4-nano", detail_model="gpt-5.4-nano", review_model="gpt-5.4-mini"),
            cache=DecisionCache(os.path.join(tmpdir, "cache.jsonl")),
            telemetry=RunTelemetry(os.path.join(tmpdir, "telemetry.jsonl")),
        )
        final = verifier.verify([negative, positive], {})
        by_id = {record.pair_id: record for record in final}
        assert by_id[negative.pair_id].decision == Decision.NO_MATCH
        assert by_id[positive.pair_id].decision == Decision.MATCH
        assert len(provider.calls) == 2


def test_ambiguous_detail_is_escalated_to_reviewer():
    candidate = cluster("K3", "P3", "Candidate wins election", "Candidate wins election in 2028")

    def responses(request):
        if "cheap-" in request.custom_id or "detail-" in request.custom_id:
            result = wire(candidate, "A", "N", "MI", missing=["time"], confidence=60)
        else:
            result = wire(candidate, "M", "EE", "SP", confidence=91)
        return {"decisions": [result]}

    with tempfile.TemporaryDirectory() as tmpdir:
        provider = FakeProvider(responses)
        verifier = HierarchicalVerifier(
            provider,
            config=VerifierConfig(cheap_model="gpt-5.4-nano", detail_model="gpt-5.4-nano", review_model="gpt-5.4-mini"),
            cache=DecisionCache(os.path.join(tmpdir, "cache.jsonl")),
            telemetry=RunTelemetry(os.path.join(tmpdir, "telemetry.jsonl")),
        )
        final = verifier.verify([candidate], {})
        assert final[0].decision == Decision.MATCH
        assert len(provider.calls) == 3


def test_partial_batch_retries_only_missing_pair_id():
    first = cluster("K4", "P4", "Alpha election", "Alpha election")
    second = cluster("K5", "P5", "Beta election", "Beta election")

    def responses(request):
        payload = json.loads(request.user_message.split("INPUT_PAIRS_JSON:\n", 1)[1])
        ids = [item["id"] for item in payload["pairs"]]
        selected = ids[:1] if len(ids) > 1 else ids
        envelopes = {first.pair_id: first, second.pair_id: second}
        return {"decisions": [wire(envelopes[pair_id], "M", "EE", "SP") for pair_id in selected]}

    with tempfile.TemporaryDirectory() as tmpdir:
        provider = FakeProvider(responses)
        verifier = HierarchicalVerifier(
            provider,
            config=VerifierConfig(cheap_model="gpt-5.4-nano", detail_model="gpt-5.4-nano", review_model="gpt-5.4-mini"),
            cache=DecisionCache(os.path.join(tmpdir, "cache.jsonl")),
            telemetry=RunTelemetry(os.path.join(tmpdir, "telemetry.jsonl")),
        )
        records = verifier.process_stage([first, second], VerificationStage.CHEAP)
        assert set(records) == {first.pair_id, second.pair_id}
        assert len(provider.calls) == 2
        initial_payload = json.loads(provider.calls[0].user_message.split("INPUT_PAIRS_JSON:\n", 1)[1])
        retry_payload = json.loads(provider.calls[1].user_message.split("INPUT_PAIRS_JSON:\n", 1)[1])
        initial_ids = [item["id"] for item in initial_payload["pairs"]]
        assert [item["id"] for item in retry_payload["pairs"]] == [initial_ids[1]]


def test_identical_second_run_uses_cache_without_provider_calls():
    candidate = cluster("K6", "P6", "Gamma election", "Gamma election")

    def response(request):
        return {"decisions": [wire(candidate, "M", "EE", "SP")]}

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache.jsonl")
        config = VerifierConfig(cheap_model="gpt-5.4-nano", detail_model="gpt-5.4-nano", review_model="gpt-5.4-mini")
        first_provider = FakeProvider(response)
        first = HierarchicalVerifier(
            first_provider,
            config=config,
            cache=DecisionCache(cache_path),
            telemetry=RunTelemetry(os.path.join(tmpdir, "telemetry1.jsonl")),
        )
        first.verify([candidate], {})
        assert len(first_provider.calls) == 2

        second_provider = FakeProvider(lambda request: RuntimeError("provider must not be called"))
        second = HierarchicalVerifier(
            second_provider,
            config=config,
            cache=DecisionCache(cache_path),
            telemetry=RunTelemetry(os.path.join(tmpdir, "telemetry2.jsonl")),
        )
        final = second.verify([candidate], {})
        assert final[0].decision == Decision.MATCH
        assert second_provider.calls == []


def test_consensus_verifier_calls_frontier_and_tiebreak_only_for_unresolved_pair():
    negative = cluster("K7", "P7", "Texas election", "Iowa election")
    positive = cluster("K8", "P8", "France election 2027", "2027 France election")

    def responses(request):
        payload = json.loads(request.user_message.split("INPUT_PAIRS_JSON:\n", 1)[1])
        decisions = []
        for raw in payload["pairs"]:
            envelope = negative if raw["id"] == negative.pair_id else positive
            if envelope is negative:
                decisions.append(wire(envelope, "N", "N", "DL", ["location_or_jurisdiction"]))
            elif request.model in {"gpt-5.4-nano-2026-03-17", "gpt-5.4-mini-2026-03-17"}:
                decisions.append(wire(envelope, "A", "N", "MI", missing=["time"], confidence=65))
            else:
                decisions.append(wire(envelope, "M", "EE", "SP"))
        return {"decisions": decisions}

    with tempfile.TemporaryDirectory() as tmpdir:
        provider = FakeProvider(responses)
        verifier = HierarchicalVerifier(
            provider,
            config=VerifierConfig(max_cost_usd=1.0),
            cache=DecisionCache(os.path.join(tmpdir, "cache.jsonl")),
            telemetry=RunTelemetry(os.path.join(tmpdir, "telemetry.jsonl")),
        )
        final = verifier.verify_consensus([negative, positive], {})
        by_id = {record.pair_id: record for record in final}

        assert by_id[negative.pair_id].decision == Decision.NO_MATCH
        assert by_id[positive.pair_id].decision == Decision.MATCH
        assert [request.model for request in provider.calls] == [
            "gpt-5.4-nano-2026-03-17",
            "gpt-5.4-mini-2026-03-17",
            "gpt-5.4-2026-03-05",
            "gpt-5.5-2026-04-23",
        ]
        assert verifier.consensus_routes == {
            "frontier_tiebreak_match": 1,
            "primary_secondary_agreement": 1,
        }


def run():
    test_hierarchical_routing_skips_explicit_negative_and_verifies_match()
    test_ambiguous_detail_is_escalated_to_reviewer()
    test_partial_batch_retries_only_missing_pair_id()
    test_identical_second_run_uses_cache_without_provider_calls()
    test_consensus_verifier_calls_frontier_and_tiebreak_only_for_unresolved_pair()


if __name__ == "__main__":
    run()
    print("ok")
