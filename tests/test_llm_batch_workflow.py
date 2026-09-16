import json
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events import batch_workflow
from pipeline.llm_events.decision_cache import DecisionCache
from pipeline.llm_events.decision_schema import VerificationStage
from pipeline.llm_events.prompt_builder import from_event_cluster, prompt_version


def candidate():
    return from_event_cluster(
        {
            "kalshi": {"event_id": "K-BATCH", "title": "France election 2027", "category": "Politics"},
            "polymarket": {"event_id": "P-BATCH", "title": "2027 France election", "category": "Politics"},
            "common_keywords": ["france", "election", "2027"],
            "match_score": 3,
        }
    )


def test_submit_refuses_manifest_above_explicit_cost_cap(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "request_count": 1,
                "estimate": {"cost_usd_high": 0.5},
                "input_path": str(tmp_path / "input.jsonl"),
                "stage": "detail",
                "prompt_version": "detail-v4",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_MAX_BATCH_COST_USD", "0.01")

    with pytest.raises(RuntimeError, match="exceeds LLM_MAX_BATCH_COST_USD"):
        batch_workflow.submit_manifest(str(manifest))


def test_collect_batch_result_validates_and_caches_decision(tmp_path, monkeypatch):
    envelope = candidate()
    cache = DecisionCache(str(tmp_path / "cache.jsonl"))
    monkeypatch.setattr(batch_workflow, "BATCH_DIR", str(tmp_path))
    monkeypatch.setattr(batch_workflow, "load_envelopes", lambda include_outcomes: [envelope])
    monkeypatch.setattr(batch_workflow, "_contexts", lambda envelopes, stage: {})
    monkeypatch.setattr(batch_workflow, "DecisionCache", lambda: cache)

    manifest = {
        "stage": "detail",
        "pass_name": "primary",
        "model": "gpt-5.4-nano-2026-03-17",
        "include_outcomes": False,
        "requests": [{"custom_id": "batch-1", "pair_ids": [envelope.pair_id]}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    content = {
        "decisions": [
            {
                "id": envelope.pair_id,
                "d": "M",
                "r": "EE",
                "c": "SP",
                "x": [],
                "m": [],
                "e": ["France election 2027", "2027 France election"],
                "q": 96,
            }
        ]
    }
    result = {
        "custom_id": "batch-1",
        "response": {
            "status_code": 200,
            "body": {
                "model": manifest["model"],
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 60},
            },
        },
    }
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    report = batch_workflow.collect_manifest(str(manifest_path), str(result_path))
    record = cache.get(
        envelope.pair_id,
        envelope.content_hash(VerificationStage.DETAIL, {}),
        VerificationStage.DETAIL,
        manifest["model"],
        prompt_version(VerificationStage.DETAIL),
    )

    assert report["records_cached"] == 1
    assert report["missing_pair_ids"] == []
    assert report["cost_usd"] > 0
    assert record is not None
    assert record.decision.value == "MATCH"
