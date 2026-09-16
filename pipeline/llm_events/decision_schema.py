from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


PARSER_SCHEMA_VERSION = "event-parser-v2"
DECISION_POLICY_VERSION = "event-equivalence-v4"
CHEAP_PROMPT_VERSION = "cheap-v4"
DETAIL_PROMPT_VERSION = "detail-v4"
REVIEW_PROMPT_VERSION = "review-v4"


class Decision(str, Enum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    AMBIGUOUS = "AMBIGUOUS"


class RelationType(str, Enum):
    EVENT_EVENT = "EVENT_EVENT"
    EVENT_OUTCOME_BRIDGE = "EVENT_OUTCOME_BRIDGE"
    NONE = "NONE"


class ReasonCode(str, Enum):
    SAME_PREDICATE = "SAME_PREDICATE"
    EXPLICIT_CONFLICT = "EXPLICIT_CONFLICT"
    MISSING_CRITICAL_INFO = "MISSING_CRITICAL_INFO"
    DIFFERENT_SCOPE = "DIFFERENT_SCOPE"
    DIFFERENT_SUBJECT = "DIFFERENT_SUBJECT"
    DIFFERENT_TIME = "DIFFERENT_TIME"
    DIFFERENT_LOCATION = "DIFFERENT_LOCATION"
    DIFFERENT_THRESHOLD = "DIFFERENT_THRESHOLD"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    OTHER = "OTHER"


class VerificationStage(str, Enum):
    DETERMINISTIC = "deterministic"
    CHEAP = "cheap"
    DETAIL = "detail"
    REVIEW = "review"
    LEGACY = "legacy"


ALLOWED_FIELDS = {
    "subject",
    "action_or_metric",
    "event_type",
    "scope",
    "time",
    "location_or_jurisdiction",
    "threshold_and_direction",
    "resolution_predicate",
    "outcome_subject",
    "category",
}

DECISION_WIRE = {"M": Decision.MATCH, "N": Decision.NO_MATCH, "A": Decision.AMBIGUOUS}
RELATION_WIRE = {"EE": RelationType.EVENT_EVENT, "EO": RelationType.EVENT_OUTCOME_BRIDGE, "N": RelationType.NONE}
REASON_WIRE = {
    "SP": ReasonCode.SAME_PREDICATE,
    "XC": ReasonCode.EXPLICIT_CONFLICT,
    "MI": ReasonCode.MISSING_CRITICAL_INFO,
    "DS": ReasonCode.DIFFERENT_SCOPE,
    "DU": ReasonCode.DIFFERENT_SUBJECT,
    "DT": ReasonCode.DIFFERENT_TIME,
    "DL": ReasonCode.DIFFERENT_LOCATION,
    "DH": ReasonCode.DIFFERENT_THRESHOLD,
    "IE": ReasonCode.INSUFFICIENT_EVIDENCE,
    "O": ReasonCode.OTHER,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_pair_id(candidate_type: str, kalshi: Mapping[str, Any], polymarket: Mapping[str, Any]) -> str:
    identity = {
        "candidate_type": candidate_type,
        "kalshi_event_id": str(kalshi.get("event_id", "")),
        "kalshi_bet_id": str(kalshi.get("bet_id", "")),
        "polymarket_event_id": str(polymarket.get("event_id", "")),
        "polymarket_bet_id": str(polymarket.get("bet_id", "")),
        "polymarket_market_id": str(polymarket.get("original_market_id", "")),
    }
    prefix = "outcome" if candidate_type != "event_event" else "event"
    return f"{prefix}-{stable_hash(identity)[:24]}"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)


@dataclass
class DecisionRecord:
    pair_id: str
    content_hash: str
    decision: Decision
    relation_type: RelationType
    reason_code: ReasonCode
    stage: VerificationStage
    conflicting_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    parser_schema_version: str = PARSER_SCHEMA_VERSION
    decision_policy_version: str = DECISION_POLICY_VERSION
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    retries: int = 0
    status: str = "valid"
    created_at: str = ""
    source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["decision"] = self.decision.value
        result["relation_type"] = self.relation_type.value
        result["reason_code"] = self.reason_code.value
        result["stage"] = self.stage.value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionRecord":
        payload = dict(data)
        payload["decision"] = Decision(payload["decision"])
        payload["relation_type"] = RelationType(payload["relation_type"])
        payload["reason_code"] = ReasonCode(payload["reason_code"])
        payload["stage"] = VerificationStage(payload["stage"])
        payload["usage"] = Usage(**dict(payload.get("usage", {})))
        return cls(**payload)


def decision_json_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "d": {"type": "string", "enum": sorted(DECISION_WIRE)},
            "r": {"type": "string", "enum": sorted(RELATION_WIRE)},
            "c": {"type": "string", "enum": sorted(REASON_WIRE)},
            "x": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_FIELDS)}, "maxItems": 4},
            "m": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_FIELDS)}, "maxItems": 4},
            "e": {"type": "array", "items": {"type": "string", "maxLength": 140}, "maxItems": 2},
            "q": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["id", "d", "r", "c", "x", "m", "e", "q"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"decisions": {"type": "array", "items": item}},
        "required": ["decisions"],
    }


def validate_decision_payload(payload: Mapping[str, Any], expected_pair_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("decisions"), list):
        raise ValueError("response must contain a decisions list")

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload["decisions"]:
        if not isinstance(raw, Mapping):
            raise ValueError("each decision must be an object")
        compact = "id" in raw
        pair_id = str(raw.get("id" if compact else "pair_id", ""))
        if pair_id not in expected_pair_ids:
            raise ValueError(f"unknown pair_id: {pair_id}")
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen.add(pair_id)

        decision = DECISION_WIRE[str(raw.get("d", ""))] if compact else Decision(str(raw.get("decision", "")))
        relation = RELATION_WIRE[str(raw.get("r", ""))] if compact else RelationType(str(raw.get("relation_type", "")))
        reason = REASON_WIRE[str(raw.get("c", ""))] if compact else ReasonCode(str(raw.get("reason_code", "")))
        conflicting = [str(value) for value in raw.get("x" if compact else "conflicting_fields", [])]
        missing = [str(value) for value in raw.get("m" if compact else "missing_fields", [])]
        if any(value not in ALLOWED_FIELDS for value in conflicting + missing):
            raise ValueError(f"invalid dimension for {pair_id}")
        if decision == Decision.NO_MATCH and not conflicting:
            decision = Decision.AMBIGUOUS
            reason = ReasonCode.INSUFFICIENT_EVIDENCE
        if decision == Decision.NO_MATCH and reason == ReasonCode.SAME_PREDICATE:
            decision = Decision.AMBIGUOUS
            reason = ReasonCode.INSUFFICIENT_EVIDENCE
        if decision == Decision.AMBIGUOUS and reason == ReasonCode.SAME_PREDICATE and not missing:
            reason = ReasonCode.INSUFFICIENT_EVIDENCE
        if decision == Decision.MATCH and (conflicting or missing or reason != ReasonCode.SAME_PREDICATE):
            decision = Decision.AMBIGUOUS
            relation = RelationType.NONE
            reason = ReasonCode.MISSING_CRITICAL_INFO if missing else ReasonCode.INSUFFICIENT_EVIDENCE
        if decision == Decision.MATCH and relation == RelationType.NONE:
            raise ValueError(f"MATCH requires a relation_type for {pair_id}")
        if decision != Decision.MATCH:
            relation = RelationType.NONE

        validated.append(
            {
                "pair_id": pair_id,
                "decision": decision,
                "relation_type": relation,
                "reason_code": reason,
                "conflicting_fields": conflicting[:5],
                "missing_fields": missing[:5],
                "evidence": [str(value)[:180] for value in raw.get("e" if compact else "evidence", [])[:4]],
                "confidence": min(1.0, max(0.0, float(raw.get("q", 0) / 100 if compact else raw.get("confidence", 0.0)))),
            }
        )
    return validated
