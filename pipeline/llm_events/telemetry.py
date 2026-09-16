from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Iterable

from pipeline.llm_events.decision_schema import DecisionRecord


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LOG_PATH = os.path.join(ROOT_DIR, "logs", "llm_runs_v2.jsonl")


class RunTelemetry:
    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self.log_path = log_path

    def append(self, record: DecisionRecord, run_id: str) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        event = {
            "run_id": run_id,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            **record.to_dict(),
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")

    def monthly_cost(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        if not os.path.exists(self.log_path):
            return 0.0
        total = 0.0
        with open(self.log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                    logged_at = datetime.fromisoformat(str(event.get("logged_at", "")))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if logged_at.tzinfo is None:
                    logged_at = logged_at.replace(tzinfo=timezone.utc)
                if (logged_at.year, logged_at.month) == (now.year, now.month):
                    total += float(event.get("cost_usd", 0.0) or 0.0)
        return total


def build_run_summary(run_id: str, records: Iterable[DecisionRecord], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    records = list(records)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "decisions": len(records),
        "input_tokens": sum(record.usage.input_tokens for record in records),
        "cached_input_tokens": sum(record.usage.cached_input_tokens for record in records),
        "output_tokens": sum(record.usage.output_tokens for record in records),
        "reasoning_tokens": sum(record.usage.reasoning_tokens for record in records),
        "cost_usd": round(sum(record.cost_usd for record in records), 8),
        "latency_ms": sum(record.latency_ms for record in records),
        "errors": sum(record.status != "valid" for record in records),
    }
    for record in records:
        key = f"{record.stage.value}:{record.decision.value}"
        summary[key] = summary.get(key, 0) + 1
    if extra:
        summary.update(extra)
    return summary


def write_json_atomic(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
    shutil.move(temp_path, path)
