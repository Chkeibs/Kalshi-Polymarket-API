import os
import sys
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache  # noqa: E402
from pipeline.llm_events.decision_schema import (  # noqa: E402
    Decision,
    DecisionRecord,
    ReasonCode,
    RelationType,
    Usage,
    VerificationStage,
)
from pipeline.llm_events.pricing import PricingCatalog  # noqa: E402
from pipeline.llm_events.telemetry import RunTelemetry  # noqa: E402


def record(content_hash="h1", status="valid"):
    return DecisionRecord(
        pair_id="pair-1",
        content_hash=content_hash,
        decision=Decision.MATCH,
        relation_type=RelationType.EVENT_EVENT,
        reason_code=ReasonCode.SAME_PREDICATE,
        stage=VerificationStage.CHEAP,
        model="gpt-5.4-nano-2026-03-17",
        prompt_version="cheap-v2",
        status=status,
    )


def test_cache_roundtrip_and_content_invalidation():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "decisions.jsonl")
        cache = DecisionCache(path)
        cache.put(record())
        loaded = DecisionCache(path)
        assert loaded.get("pair-1", "h1", VerificationStage.CHEAP, "gpt-5.4-nano-2026-03-17", "cheap-v2")
        assert loaded.get("pair-1", "changed", VerificationStage.CHEAP, "gpt-5.4-nano-2026-03-17", "cheap-v2") is None


def test_error_records_are_never_cache_hits():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DecisionCache(os.path.join(tmpdir, "decisions.jsonl"))
        cache.put(record(status="error"))
        assert cache.get("pair-1", "h1", VerificationStage.CHEAP, "gpt-5.4-nano-2026-03-17", "cheap-v2") is None


def test_pricing_uses_uncached_and_cached_tokens_once():
    catalog = PricingCatalog()
    usage = Usage(input_tokens=1000, cached_input_tokens=400, output_tokens=100)
    cost = catalog.cost("gpt-5.4-nano-2026-03-17", usage, "standard")
    expected = (600 * 0.2 + 400 * 0.02 + 100 * 1.25) / 1_000_000
    assert abs(cost - expected) < 1e-12


def test_monthly_telemetry_cost_counts_logged_project_usage():
    with tempfile.TemporaryDirectory() as tmpdir:
        telemetry = RunTelemetry(os.path.join(tmpdir, "telemetry.jsonl"))
        item = record()
        item.cost_usd = 0.125
        telemetry.append(item, "run-1")
        assert abs(telemetry.monthly_cost() - 0.125) < 1e-12


def run():
    test_cache_roundtrip_and_content_invalidation()
    test_error_records_are_never_cache_hits()
    test_pricing_uses_uncached_and_cached_tokens_once()
    test_monthly_telemetry_cost_counts_logged_project_usage()


if __name__ == "__main__":
    run()
    print("ok")
