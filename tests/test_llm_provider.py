import json
import os
import sys
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_schema import decision_json_schema  # noqa: E402
from pipeline.llm_events.provider import LLMRequest, OpenAIProvider, chat_completion_body  # noqa: E402


def request(custom_id="r1", model="gpt-5.4-nano-2026-03-17"):
    return LLMRequest(custom_id, model, "system", "user", decision_json_schema(), reasoning_effort="none")


def test_chat_body_uses_structured_outputs_without_unsupported_temperature():
    body = chat_completion_body(request())
    assert body["response_format"]["type"] == "json_schema"
    assert body["reasoning_effort"] == "none"
    assert "temperature" not in body


def test_batch_file_has_stable_custom_ids_and_one_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = OpenAIProvider.write_batch_file([request("one"), request("two")], os.path.join(tmpdir, "batch.jsonl"))
        lines = [json.loads(line) for line in open(path)]
        assert [line["custom_id"] for line in lines] == ["one", "two"]
        assert all(line["url"] == "/v1/chat/completions" for line in lines)

        try:
            OpenAIProvider.write_batch_file([request("one"), request("two", "gpt-5-mini")], os.path.join(tmpdir, "bad.jsonl"))
            raise AssertionError("mixed model batch should fail")
        except ValueError:
            pass


def run():
    test_chat_body_uses_structured_outputs_without_unsupported_temperature()
    test_batch_file_has_stable_custom_ids_and_one_model()


if __name__ == "__main__":
    run()
    print("ok")
