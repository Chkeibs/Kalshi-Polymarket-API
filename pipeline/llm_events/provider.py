from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from pipeline.llm_events.decision_schema import Usage


@dataclass(frozen=True)
class LLMRequest:
    custom_id: str
    model: str
    system_message: str
    user_message: str
    json_schema: dict[str, Any]
    max_output_tokens: int = 1600
    reasoning_effort: str | None = None


@dataclass
class ProviderResponse:
    custom_id: str
    payload: dict[str, Any] | None
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: int = 0
    retries: int = 0
    status: str = "valid"
    error: str = ""


class LLMProvider(Protocol):
    name: str

    def complete_json(self, request: LLMRequest) -> ProviderResponse:
        ...


def chat_completion_body(request: LLMRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": request.system_message},
            {"role": "user", "content": request.user_message},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "event_match_decisions",
                "strict": True,
                "schema": request.json_schema,
            },
        },
        "max_completion_tokens": request.max_output_tokens,
    }
    if request.reasoning_effort:
        body["reasoning_effort"] = request.reasoning_effort
    return body


def _usage_from_completion(completion: Any) -> Usage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return Usage()
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return Usage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        cached_input_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        reasoning_tokens=int(getattr(completion_details, "reasoning_tokens", 0) or 0),
    )


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, max_retries: int = 2, timeout: float = 90.0):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use OpenAIProvider") from exc
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), max_retries=max_retries, timeout=timeout)
        self.max_retries = max_retries

    def complete_json(self, request: LLMRequest) -> ProviderResponse:
        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(**chat_completion_body(request))
            content = completion.choices[0].message.content or ""
            payload = json.loads(content)
            return ProviderResponse(
                custom_id=request.custom_id,
                payload=payload,
                usage=_usage_from_completion(completion),
                model=str(getattr(completion, "model", request.model)),
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            return ProviderResponse(
                custom_id=request.custom_id,
                payload=None,
                model=request.model,
                latency_ms=round((time.perf_counter() - started) * 1000),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def write_batch_file(requests: Sequence[LLMRequest], path: str) -> str:
        if not requests:
            raise ValueError("cannot create an empty batch")
        models = {request.model for request in requests}
        if len(models) != 1:
            raise ValueError("OpenAI batch files must use one model")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for request in requests:
                line = {
                    "custom_id": request.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": chat_completion_body(request),
                }
                handle.write(json.dumps(line, ensure_ascii=True, separators=(",", ":")) + "\n")
        return str(output)

    def submit_batch_file(self, path: str, metadata: dict[str, str] | None = None) -> dict[str, str]:
        with open(path, "rb") as handle:
            uploaded = self.client.files.create(file=handle, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata or {},
        )
        return {"batch_id": batch.id, "input_file_id": uploaded.id, "status": str(batch.status)}

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        batch = self.client.batches.retrieve(batch_id)
        return batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)

    def download_batch_results(self, batch_id: str, output_path: str) -> str:
        batch = self.client.batches.retrieve(batch_id)
        if not getattr(batch, "output_file_id", None):
            raise RuntimeError(f"batch {batch_id} has no output file (status={batch.status})")
        response = self.client.files.content(batch.output_file_id)
        content = response.text if hasattr(response, "text") else str(response)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        return str(output)


class FakeProvider:
    name = "fake"

    def __init__(self, responses: dict[str, dict[str, Any] | Exception] | Callable[[LLMRequest], dict[str, Any] | Exception]):
        self.responses = responses
        self.calls: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> ProviderResponse:
        self.calls.append(request)
        value = self.responses(request) if callable(self.responses) else self.responses.get(request.custom_id)
        if isinstance(value, Exception):
            return ProviderResponse(request.custom_id, None, model=request.model, status="error", error=str(value))
        if value is None:
            return ProviderResponse(request.custom_id, None, model=request.model, status="error", error="missing fake response")
        return ProviderResponse(
            custom_id=request.custom_id,
            payload=value,
            usage=Usage(input_tokens=100, output_tokens=20),
            model=request.model,
            latency_ms=1,
        )
