from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Iterable

from pipeline.llm_events.decision_schema import DecisionRecord, VerificationStage


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CACHE_PATH = os.path.join(ROOT_DIR, "data", "cache", "llm_decisions_v2.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecisionCache:
    def __init__(self, path: str = DEFAULT_CACHE_PATH):
        self.path = path
        self.records: dict[tuple[str, str, str, str, str], DecisionRecord] = {}
        self.load()

    @staticmethod
    def key(record: DecisionRecord) -> tuple[str, str, str, str, str]:
        return (
            record.pair_id,
            record.content_hash,
            record.stage.value,
            record.model,
            record.prompt_version,
        )

    def load(self) -> None:
        self.records = {}
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = DecisionRecord.from_dict(json.loads(line))
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
                self.records[self.key(record)] = record

    def get(
        self,
        pair_id: str,
        content_hash: str,
        stage: VerificationStage,
        model: str,
        prompt_version: str,
    ) -> DecisionRecord | None:
        record = self.records.get((pair_id, content_hash, stage.value, model, prompt_version))
        if record is None or record.status != "valid":
            return None
        return record

    def put(self, record: DecisionRecord, save: bool = True) -> None:
        if not record.created_at:
            record.created_at = utc_now()
        self.records[self.key(record)] = record
        if save:
            self.save()

    def put_many(self, records: Iterable[DecisionRecord]) -> None:
        for record in records:
            self.put(record, save=False)
        self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            for key in sorted(self.records):
                handle.write(json.dumps(self.records[key].to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
        shutil.move(temp_path, self.path)

    def by_pair(self, pair_id: str) -> list[DecisionRecord]:
        return sorted(
            (record for record in self.records.values() if record.pair_id == pair_id),
            key=lambda record: (record.created_at, record.stage.value, record.model),
        )

    def stats(self) -> dict[str, int]:
        stats: dict[str, int] = {"total": len(self.records)}
        for record in self.records.values():
            key = f"{record.stage.value}:{record.decision.value}"
            stats[key] = stats.get(key, 0) + 1
        return stats

    def import_legacy_matches(self, records: Iterable[DecisionRecord]) -> int:
        count = 0
        for record in records:
            record.stage = VerificationStage.LEGACY
            self.put(record, save=False)
            count += 1
        if count:
            self.save()
        return count
