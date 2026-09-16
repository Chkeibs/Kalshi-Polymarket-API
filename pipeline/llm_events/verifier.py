from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from pipeline.llm_events.consolidation import confirmed_row, consolidate_matches
from pipeline.llm_events.consensus import needs_frontier, needs_tiebreak, route_counts, select_consensus
from pipeline.llm_events.decision_cache import DecisionCache
from pipeline.llm_events.decision_schema import (
    CHEAP_PROMPT_VERSION,
    DECISION_POLICY_VERSION,
    DETAIL_PROMPT_VERSION,
    PARSER_SCHEMA_VERSION,
    REVIEW_PROMPT_VERSION,
    Decision,
    DecisionRecord,
    ReasonCode,
    RelationType,
    Usage,
    VerificationStage,
    decision_json_schema,
    validate_decision_payload,
)
from pipeline.llm_events.pricing import PricingCatalog, estimate_text_tokens
from pipeline.llm_events.prompt_builder import (
    CandidateEnvelope,
    build_detail_context,
    build_prompt,
    from_event_cluster,
    from_outcome_candidate,
    prompt_version,
)
from pipeline.llm_events.provider import LLMProvider, LLMRequest, OpenAIProvider, ProviderResponse
from pipeline.llm_events.telemetry import RunTelemetry, build_run_summary, write_json_atomic


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVENT_CANDIDATES = os.path.join(ROOT_DIR, "data", "clusters", "candidate_clusters.json")
OUTCOME_CANDIDATES = os.path.join(ROOT_DIR, "data", "clusters", "candidate_outcome_pairs.json")
KALSHI_CSV = os.path.join(ROOT_DIR, "data", "markets", "markets_clean_kalshi.csv")
POLYMARKET_CSV = os.path.join(ROOT_DIR, "data", "markets", "markets_clean_polymarket.csv")
CONFIRMED_CSV = os.path.join(ROOT_DIR, "data", "matches", "confirmed_matches.csv")
AMBIGUOUS_CSV = os.path.join(ROOT_DIR, "data", "matches", "ambiguous_matches.csv")
DECISIONS_JSONL = os.path.join(ROOT_DIR, "data", "matches", "llm_decisions_v2.jsonl")
SHADOW_REPORT = os.path.join(ROOT_DIR, "data", "reports", "llm_shadow_report.json")
DRY_RUN_REPORT = os.path.join(ROOT_DIR, "data", "reports", "llm_dry_run.json")


@dataclass
class VerifierConfig:
    cheap_model: str = field(default_factory=lambda: os.environ.get("LLM_CHEAP_MODEL", "gpt-5.4-nano-2026-03-17"))
    detail_model: str = field(default_factory=lambda: os.environ.get("LLM_DETAIL_MODEL", "gpt-5.4-nano-2026-03-17"))
    review_model: str = field(default_factory=lambda: os.environ.get("LLM_REVIEW_MODEL", "gpt-5.4-mini-2026-03-17"))
    cheap_reasoning: str | None = field(default_factory=lambda: os.environ.get("LLM_CHEAP_REASONING", "none"))
    detail_reasoning: str | None = field(default_factory=lambda: os.environ.get("LLM_DETAIL_REASONING", "low"))
    review_reasoning: str | None = field(default_factory=lambda: os.environ.get("LLM_REVIEW_REASONING", "low"))
    primary_model: str = field(default_factory=lambda: os.environ.get("LLM_PRIMARY_MODEL", "gpt-5.4-nano-2026-03-17"))
    secondary_model: str = field(default_factory=lambda: os.environ.get("LLM_SECONDARY_MODEL", "gpt-5.4-mini-2026-03-17"))
    frontier_model: str = field(default_factory=lambda: os.environ.get("LLM_FRONTIER_MODEL", "gpt-5.4-2026-03-05"))
    tiebreak_model: str = field(default_factory=lambda: os.environ.get("LLM_TIEBREAK_MODEL", "gpt-5.5-2026-04-23"))
    primary_reasoning: str | None = field(default_factory=lambda: os.environ.get("LLM_PRIMARY_REASONING", "low"))
    secondary_reasoning: str | None = field(default_factory=lambda: os.environ.get("LLM_SECONDARY_REASONING", "low"))
    frontier_reasoning: str | None = field(default_factory=lambda: os.environ.get("LLM_FRONTIER_REASONING", "none"))
    tiebreak_reasoning: str | None = field(default_factory=lambda: os.environ.get("LLM_TIEBREAK_REASONING", "none"))
    cheap_batch_size: int = 20
    detail_batch_size: int = 8
    review_batch_size: int = 4
    max_batch_chars: int = 55_000
    cheap_no_match_confidence: float = 0.86
    final_confidence: float = 0.80
    consensus_batch_size: int = 8
    max_cost_usd: float = field(default_factory=lambda: float(os.environ.get("LLM_MAX_RUN_COST_USD", "0.25")))
    max_monthly_cost_usd: float = field(default_factory=lambda: float(os.environ.get("LLM_MAX_MONTHLY_COST_USD", "5.00")))
    retry_missing_once: bool = True
    include_outcomes: bool = True
    pricing_mode: str = "standard"


def _load_json_list(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def load_envelopes(include_outcomes: bool = True) -> list[CandidateEnvelope]:
    envelopes = [from_event_cluster(candidate) for candidate in _load_json_list(EVENT_CANDIDATES)]
    if include_outcomes:
        envelopes.extend(from_outcome_candidate(candidate) for candidate in _load_json_list(OUTCOME_CANDIDATES))
    deduplicated: dict[str, CandidateEnvelope] = {}
    for envelope in envelopes:
        previous = deduplicated.get(envelope.pair_id)
        if previous is None or int(envelope.shared.get("match_score", 0)) > int(previous.shared.get("match_score", 0)):
            deduplicated[envelope.pair_id] = envelope
    return [deduplicated[pair_id] for pair_id in sorted(deduplicated)]


def build_context_maps(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    result: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        event_id = str(row.get("Event ID", "")).strip()
        if not event_id:
            continue
        context = result.setdefault(
            event_id,
            {"event_description": "", "contracts": []},
        )
        event_description = str(row.get("Event_Description", "")).strip()
        if len(event_description) > len(context["event_description"]):
            context["event_description"] = event_description
        bet_id = str(row.get("Bet ID", "")).strip()
        title = str(row.get("Bet Title", "")).strip()
        contract = {
            "bet_id": bet_id,
            "bet_title": title,
            "description": str(row.get("Description", "")).strip(),
            "bet_rules": str(row.get("Bet_Rules", "")).strip(),
            "expiration": str(row.get("Expiration Date", "")).strip(),
        }
        if bet_id and not any(item["bet_id"] == bet_id for item in context["contracts"]):
            context["contracts"].append(contract)
    return result


def targeted_contexts(
    envelopes: Iterable[CandidateEnvelope],
    kalshi_contexts: Mapping[str, Mapping[str, Any]],
    polymarket_contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for envelope in envelopes:
        result[envelope.pair_id] = build_detail_context(
            envelope,
            kalshi_contexts.get(str(envelope.kalshi.get("event_id", "")), {}),
            polymarket_contexts.get(str(envelope.polymarket.get("event_id", "")), {}),
        )
    return result


def chunk_envelopes(
    envelopes: Sequence[CandidateEnvelope],
    stage: VerificationStage,
    contexts: Mapping[str, Mapping[str, Any]],
    max_items: int,
    max_chars: int,
) -> list[list[CandidateEnvelope]]:
    sized = []
    for envelope in envelopes:
        payload = envelope.compact_payload()
        if stage != VerificationStage.CHEAP:
            payload["targeted_context"] = contexts.get(envelope.pair_id, {})
        sized.append((len(json.dumps(payload, ensure_ascii=True)), envelope.pair_id, envelope))
    sized.sort(key=lambda item: (item[0], item[1]))

    batches: list[list[CandidateEnvelope]] = []
    current: list[CandidateEnvelope] = []
    current_chars = 0
    for size, _, envelope in sized:
        if current and (len(current) >= max_items or current_chars + size > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(envelope)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _record_from_payload(
    payload: Mapping[str, Any],
    envelope: CandidateEnvelope,
    stage: VerificationStage,
    response: ProviderResponse,
    usage: Usage,
    latency_ms: int,
    content_hash: str,
    pricing: PricingCatalog,
    pricing_mode: str,
) -> DecisionRecord:
    model = response.model or ""
    cost = pricing.cost(model, usage, pricing_mode) if model else 0.0
    return DecisionRecord(
        pair_id=envelope.pair_id,
        content_hash=content_hash,
        decision=payload["decision"],
        relation_type=payload["relation_type"],
        reason_code=payload["reason_code"],
        stage=stage,
        conflicting_fields=list(payload["conflicting_fields"]),
        missing_fields=list(payload["missing_fields"]),
        evidence=list(payload["evidence"]),
        confidence=float(payload["confidence"]),
        provider="openai" if model else "unknown",
        model=model,
        prompt_version=prompt_version(stage),
        usage=usage,
        cost_usd=cost,
        latency_ms=latency_ms,
        retries=response.retries,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _error_record(
    envelope: CandidateEnvelope,
    stage: VerificationStage,
    model: str,
    content_hash: str,
    response: ProviderResponse,
    usage: Usage | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        pair_id=envelope.pair_id,
        content_hash=content_hash,
        decision=Decision.AMBIGUOUS,
        relation_type=RelationType.NONE,
        reason_code=ReasonCode.PROVIDER_ERROR,
        stage=stage,
        missing_fields=["resolution_predicate"],
        provider="openai",
        model=model,
        prompt_version=prompt_version(stage),
        usage=usage or response.usage,
        latency_ms=response.latency_ms,
        status="error",
        created_at=datetime.now(timezone.utc).isoformat(),
        evidence=[response.error[:180]] if response.error else [],
    )


def verifiable_explicit_no(record: DecisionRecord, min_confidence: float = 0.86) -> bool:
    return (
        record.decision == Decision.NO_MATCH
        and record.confidence >= min_confidence
        and bool(record.conflicting_fields)
        and record.conflicting_fields != ["category"]
        and len(record.evidence) >= 2
    )


def needs_detail(envelope: CandidateEnvelope, cheap_record: DecisionRecord, min_no_match_confidence: float = 0.86) -> bool:
    return (
        cheap_record.status != "valid"
        or not verifiable_explicit_no(cheap_record, min_no_match_confidence)
        or envelope.relation_hint == RelationType.EVENT_OUTCOME_BRIDGE
    )


def needs_review(cheap_record: DecisionRecord, detail_record: DecisionRecord) -> bool:
    disagreement = (
        cheap_record.status == "valid"
        and detail_record.status == "valid"
        and cheap_record.decision != Decision.AMBIGUOUS
        and detail_record.decision != Decision.AMBIGUOUS
        and cheap_record.decision != detail_record.decision
    )
    return detail_record.status != "valid" or detail_record.decision == Decision.AMBIGUOUS or disagreement


def select_final_records(
    envelopes: Sequence[CandidateEnvelope],
    cheap: Mapping[str, DecisionRecord],
    detail: Mapping[str, DecisionRecord],
    review: Mapping[str, DecisionRecord],
    final_confidence: float,
) -> list[DecisionRecord]:
    final = []
    for envelope in envelopes:
        record = review.get(envelope.pair_id) or detail.get(envelope.pair_id) or cheap[envelope.pair_id]
        if record.status != "valid":
            record.decision = Decision.AMBIGUOUS
            record.relation_type = RelationType.NONE
        if record.decision == Decision.MATCH and record.confidence < final_confidence:
            record.decision = Decision.AMBIGUOUS
            record.relation_type = RelationType.NONE
            record.reason_code = ReasonCode.INSUFFICIENT_EVIDENCE
        final.append(record)
    return final


class HierarchicalVerifier:
    def __init__(
        self,
        provider: LLMProvider,
        config: VerifierConfig | None = None,
        cache: DecisionCache | None = None,
        pricing: PricingCatalog | None = None,
        telemetry: RunTelemetry | None = None,
    ):
        self.provider = provider
        self.config = config or VerifierConfig()
        self.cache = cache or DecisionCache()
        self.pricing = pricing or PricingCatalog()
        self.telemetry = telemetry or RunTelemetry()
        self.records: list[DecisionRecord] = []
        self.spent = 0.0
        self.estimated_spend = 0.0
        self.monthly_spent_at_start = self.telemetry.monthly_cost()
        self.run_id = f"llm-{uuid.uuid4().hex[:12]}"
        self.consensus_routes: dict[str, int] = {}

    def _model_for_stage(self, stage: VerificationStage) -> tuple[str, str | None, int]:
        if stage == VerificationStage.CHEAP:
            return self.config.cheap_model, self.config.cheap_reasoning, self.config.cheap_batch_size
        if stage == VerificationStage.DETAIL:
            return self.config.detail_model, self.config.detail_reasoning, self.config.detail_batch_size
        return self.config.review_model, self.config.review_reasoning, self.config.review_batch_size

    def _process_batch(
        self,
        batch: Sequence[CandidateEnvelope],
        stage: VerificationStage,
        contexts: Mapping[str, Mapping[str, Any]],
        model: str,
        reasoning: str | None,
    ) -> tuple[list[DecisionRecord], list[CandidateEnvelope]]:
        system, user = build_prompt(batch, stage, contexts)
        request = LLMRequest(
            custom_id=f"{self.run_id}-{stage.value}-{batch[0].pair_id}",
            model=model,
            system_message=system,
            user_message=user,
            json_schema=decision_json_schema(),
            max_output_tokens=max(800, 180 * len(batch)),
            reasoning_effort=reasoning,
        )
        estimate = estimate_text_tokens(system + user)
        estimated_usage = Usage(
            input_tokens=estimate.high,
            output_tokens=request.max_output_tokens,
        )
        estimated_cost = self.pricing.cost(model, estimated_usage, self.config.pricing_mode)
        if self.estimated_spend + estimated_cost > self.config.max_cost_usd:
            raise RuntimeError(
                "LLM run budget would be exceeded before provider call: "
                f"~${self.estimated_spend:.4f} + ~${estimated_cost:.4f} > ${self.config.max_cost_usd:.4f}"
            )
        if self.monthly_spent_at_start + self.estimated_spend + estimated_cost > self.config.max_monthly_cost_usd:
            raise RuntimeError(
                "LLM project monthly budget would be exceeded before provider call: "
                f"${self.monthly_spent_at_start:.4f} logged + ~${self.estimated_spend + estimated_cost:.4f} run "
                f"> ${self.config.max_monthly_cost_usd:.4f}"
            )
        self.estimated_spend += estimated_cost
        response = self.provider.complete_json(request)
        by_id = {envelope.pair_id: envelope for envelope in batch}
        hashes = {
            envelope.pair_id: envelope.content_hash(stage, contexts.get(envelope.pair_id, {}))
            for envelope in batch
        }
        if response.status != "valid" or response.payload is None:
            if len(batch) > 1:
                return [], list(batch)
            return [_error_record(batch[0], stage, model, hashes[batch[0].pair_id], response)], []

        parsed: list[DecisionRecord] = []
        seen: set[str] = set()
        raw_decisions = response.payload.get("decisions", []) if isinstance(response.payload, Mapping) else []
        decision_count = max(1, len(raw_decisions) if isinstance(raw_decisions, list) else len(batch))

        def usage_share(index: int) -> Usage:
            def share(total: int) -> int:
                base, remainder = divmod(total, decision_count)
                return base + (1 if index < remainder else 0)

            return Usage(
                input_tokens=share(response.usage.input_tokens),
                cached_input_tokens=share(response.usage.cached_input_tokens),
                output_tokens=share(response.usage.output_tokens),
                reasoning_tokens=share(response.usage.reasoning_tokens),
            )

        for raw_index, raw in enumerate(raw_decisions if isinstance(raw_decisions, list) else []):
            try:
                item = validate_decision_payload({"decisions": [raw]}, set(by_id) - seen)[0]
            except (ValueError, TypeError, KeyError):
                continue
            pair_id = item["pair_id"]
            seen.add(pair_id)
            parsed.append(
                _record_from_payload(
                    item,
                    by_id[pair_id],
                    stage,
                    response,
                    usage_share(raw_index),
                    round(response.latency_ms / decision_count),
                    hashes[pair_id],
                    self.pricing,
                    self.config.pricing_mode,
                )
            )
        missing = [envelope for envelope in batch if envelope.pair_id not in seen]
        return parsed, missing

    def process_stage(
        self,
        envelopes: Sequence[CandidateEnvelope],
        stage: VerificationStage,
        contexts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, DecisionRecord]:
        contexts = contexts or {}
        model, reasoning, batch_size = self._model_for_stage(stage)
        return self.process_model(envelopes, stage, contexts, model, reasoning, batch_size)

    def process_model(
        self,
        envelopes: Sequence[CandidateEnvelope],
        stage: VerificationStage,
        contexts: Mapping[str, Mapping[str, Any]] | None,
        model: str,
        reasoning: str | None,
        batch_size: int,
    ) -> dict[str, DecisionRecord]:
        results: dict[str, DecisionRecord] = {}
        contexts = contexts or {}
        misses: list[CandidateEnvelope] = []
        for envelope in envelopes:
            content_hash = envelope.content_hash(stage, contexts.get(envelope.pair_id, {}))
            cached = self.cache.get(envelope.pair_id, content_hash, stage, model, prompt_version(stage))
            if cached:
                results[envelope.pair_id] = cached
            else:
                misses.append(envelope)

        batches = chunk_envelopes(misses, stage, contexts, batch_size, self.config.max_batch_chars)
        for batch in batches:
            if self.spent >= self.config.max_cost_usd:
                raise RuntimeError(f"LLM run budget exceeded: ${self.spent:.4f} >= ${self.config.max_cost_usd:.4f}")
            parsed, missing = self._process_batch(batch, stage, contexts, model, reasoning)
            if missing and self.config.retry_missing_once and len(batch) > 1:
                for envelope in missing:
                    retried, still_missing = self._process_batch([envelope], stage, contexts, model, reasoning)
                    parsed.extend(retried)
                    if still_missing and not retried:
                        response = ProviderResponse(envelope.pair_id, None, model=model, status="error", error="missing pair_id after retry")
                        parsed.append(_error_record(envelope, stage, model, envelope.content_hash(stage, contexts.get(envelope.pair_id, {})), response))
            for record in parsed:
                self.records.append(record)
                self.telemetry.append(record, self.run_id)
                self.spent += record.cost_usd
                results[record.pair_id] = record
            self.cache.put_many(record for record in parsed if record.status == "valid")
        return results

    def verify_consensus(
        self,
        envelopes: Sequence[CandidateEnvelope],
        contexts: Mapping[str, Mapping[str, Any]],
    ) -> list[DecisionRecord]:
        config = self.config
        stage = VerificationStage.DETAIL
        primary = self.process_model(
            envelopes,
            stage,
            contexts,
            config.primary_model,
            config.primary_reasoning,
            config.consensus_batch_size,
        )
        secondary = self.process_model(
            envelopes,
            stage,
            contexts,
            config.secondary_model,
            config.secondary_reasoning,
            config.consensus_batch_size,
        )

        frontier_inputs = [
            envelope
            for envelope in envelopes
            if needs_frontier(primary[envelope.pair_id], secondary[envelope.pair_id])
        ]
        frontier = self.process_model(
            frontier_inputs,
            stage,
            contexts,
            config.frontier_model,
            config.frontier_reasoning,
            config.consensus_batch_size,
        )
        tiebreak_inputs = [
            envelope
            for envelope in frontier_inputs
            if needs_tiebreak(secondary[envelope.pair_id], frontier[envelope.pair_id])
        ]
        tiebreak = self.process_model(
            tiebreak_inputs,
            stage,
            contexts,
            config.tiebreak_model,
            config.tiebreak_reasoning,
            config.consensus_batch_size,
        )

        selected = {
            envelope.pair_id: select_consensus(
                primary[envelope.pair_id],
                secondary[envelope.pair_id],
                frontier.get(envelope.pair_id),
                tiebreak.get(envelope.pair_id),
            )
            for envelope in envelopes
        }
        self.consensus_routes = route_counts(selected)
        return [selected[envelope.pair_id].record for envelope in envelopes]

    def verify(self, envelopes: Sequence[CandidateEnvelope], contexts: Mapping[str, Mapping[str, Any]]) -> list[DecisionRecord]:
        cheap = self.process_stage(envelopes, VerificationStage.CHEAP)
        detail_inputs = []
        for envelope in envelopes:
            record = cheap[envelope.pair_id]
            if needs_detail(envelope, record, self.config.cheap_no_match_confidence):
                detail_inputs.append(envelope)

        detail = self.process_stage(detail_inputs, VerificationStage.DETAIL, contexts)
        review_inputs = []
        for envelope in detail_inputs:
            cheap_record = cheap[envelope.pair_id]
            detail_record = detail[envelope.pair_id]
            if needs_review(cheap_record, detail_record):
                review_inputs.append(envelope)

        review = self.process_stage(review_inputs, VerificationStage.REVIEW, contexts)
        return select_final_records(envelopes, cheap, detail, review, self.config.final_confidence)


def build_dry_run_report(
    envelopes: Sequence[CandidateEnvelope],
    config: VerifierConfig,
    pricing: PricingCatalog,
    contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    contexts = contexts or {}
    batches = chunk_envelopes(
        envelopes,
        VerificationStage.DETAIL,
        contexts,
        config.consensus_batch_size,
        config.max_batch_chars,
    )
    input_low = input_high = 0
    for batch in batches:
        system, user = build_prompt(batch, VerificationStage.DETAIL, contexts)
        estimate = estimate_text_tokens(system + user)
        input_low += estimate.low
        input_high += estimate.high
    output_low = len(envelopes) * 22
    output_high = len(envelopes) * 80
    route_rates = {
        "primary": 1.0,
        "secondary": 1.0,
        "frontier": 196 / 300,
        "tiebreak": 173 / 300,
    }
    models = {
        "primary": config.primary_model,
        "secondary": config.secondary_model,
        "frontier": config.frontier_model,
        "tiebreak": config.tiebreak_model,
    }
    stage_estimates = {}
    analytical_low = analytical_high = 0.0
    for name, model in models.items():
        multiplier = route_rates[name]
        rates = pricing.rates(model, config.pricing_mode)
        cost_low = multiplier * (input_low * rates["input"] + output_low * rates["output"]) / 1_000_000
        cost_high = multiplier * (input_high * rates["input"] + output_high * rates["output"]) / 1_000_000
        analytical_low += cost_low
        analytical_high += cost_high
        stage_estimates[name] = {
            "model": model,
            "candidate_rate": round(multiplier, 6),
            "candidates": round(len(envelopes) * multiplier),
            "cost_usd": {"low": round(cost_low, 6), "high": round(cost_high, 6)},
        }
    type_counts: dict[str, int] = {}
    for envelope in envelopes:
        type_counts[envelope.candidate_type] = type_counts.get(envelope.candidate_type, 0) + 1
    measured_projection = None
    projection_path = os.path.join(ROOT_DIR, "data", "reports", "llm_calibration", "consensus_cost_projection.json")
    if os.path.exists(projection_path):
        projection = json.loads(Path(projection_path).read_text(encoding="utf-8"))
        basis = projection.get("basis", {})
        per_pair_key = f"{config.pricing_mode}_cost_per_pair_usd"
        per_pair = float(basis.get(per_pair_key, 0.0) or 0.0)
        if per_pair:
            measured_projection = round(per_pair * len(envelopes), 6)
    projected_floor = measured_projection if measured_projection is not None else analytical_low
    return {
        "schema_version": PARSER_SCHEMA_VERSION,
        "policy_version": DECISION_POLICY_VERSION,
        "pricing_version": pricing.version,
        "candidate_count": len(envelopes),
        "candidate_types": dict(sorted(type_counts.items())),
        "prompt_batches_per_full_pass": len(batches),
        "prompt_input_tokens_per_full_pass": {"low": input_low, "high": input_high},
        "prompt_output_tokens_per_full_pass": {"low": output_low, "high": output_high},
        "route_rates_from_300_example_evaluation": route_rates,
        "stage_estimates": stage_estimates,
        "analytical_cost_usd": {"low": round(analytical_low, 6), "high": round(analytical_high, 6)},
        "measured_evaluation_routing_cost_usd": measured_projection,
        "max_run_cost_usd": config.max_cost_usd,
        "budget_override_required": projected_floor > config.max_cost_usd,
    }


def migrate_legacy_confirmed(cache: DecisionCache, envelopes: Sequence[CandidateEnvelope]) -> int:
    if not os.path.exists(CONFIRMED_CSV):
        return 0
    event_pairs = {
        (str(envelope.kalshi.get("event_id", "")), str(envelope.polymarket.get("event_id", ""))): envelope
        for envelope in envelopes
        if envelope.candidate_type == "event_event"
    }
    confirmed = pd.read_csv(CONFIRMED_CSV, dtype=str).fillna("")
    records = []
    for row in confirmed.to_dict("records"):
        envelope = event_pairs.get((str(row.get("Kalshi_ID", "")), str(row.get("Polymarket_ID", ""))))
        if not envelope:
            continue
        existing = cache.get(envelope.pair_id, "legacy", VerificationStage.LEGACY, "legacy", "legacy-v1")
        if existing:
            continue
        relation_value = str(row.get("Relation_Type", RelationType.EVENT_EVENT.value)) or RelationType.EVENT_EVENT.value
        try:
            relation = RelationType(relation_value)
        except ValueError:
            relation = RelationType.EVENT_EVENT
        records.append(
            DecisionRecord(
                pair_id=envelope.pair_id,
                content_hash="legacy",
                decision=Decision.MATCH,
                relation_type=relation,
                reason_code=ReasonCode.SAME_PREDICATE,
                stage=VerificationStage.LEGACY,
                evidence=[str(row.get("Reasoning", ""))[:180]] if row.get("Reasoning") else [],
                confidence=1.0,
                provider="legacy",
                model="legacy",
                prompt_version="legacy-v1",
                parser_schema_version="legacy",
                decision_policy_version="legacy",
                source="legacy_confirmed_csv",
            )
        )
    return cache.import_legacy_matches(records)


def _write_decisions(path: str, records: Iterable[DecisionRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = f"{path}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.pair_id):
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(temp, path)


def _write_ambiguous(
    records: Iterable[DecisionRecord],
    envelopes: Mapping[str, CandidateEnvelope],
    path: str = AMBIGUOUS_CSV,
) -> None:
    rows = []
    for record in records:
        if record.decision != Decision.AMBIGUOUS:
            continue
        envelope = envelopes[record.pair_id]
        rows.append(
            {
                "Pair_ID": record.pair_id,
                "Kalshi_ID": envelope.kalshi.get("event_id", ""),
                "Polymarket_ID": envelope.polymarket.get("event_id", ""),
                "Candidate_Type": envelope.candidate_type,
                "Reason_Code": record.reason_code.value,
                "Missing_Fields": ",".join(record.missing_fields),
                "Model": record.model,
                "Confidence": record.confidence,
            }
        )
    columns = ["Pair_ID", "Kalshi_ID", "Polymarket_ID", "Candidate_Type", "Reason_Code", "Missing_Fields", "Model", "Confidence"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _apply_confirmed(records: Sequence[DecisionRecord], envelopes: Mapping[str, CandidateEnvelope]) -> int:
    selected, _ = consolidate_matches(records, envelopes)
    rows = [confirmed_row(record, envelopes[record.pair_id]) for record in selected]
    new = pd.DataFrame(rows)
    if os.path.exists(CONFIRMED_CSV):
        existing = pd.read_csv(CONFIRMED_CSV, dtype=str).fillna("")
        if "SyncStatus" in existing:
            existing["SyncStatus"] = "Existing"
        if "Relation_Type" not in existing:
            existing["Relation_Type"] = RelationType.EVENT_EVENT.value
        combined = pd.concat([existing, new], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["Kalshi_ID", "Polymarket_ID", "Relation_Type"], keep="last")
    else:
        combined = new
    combined.to_csv(CONFIRMED_CSV, index=False)
    return len(rows)


def run_v2_verifier(
    dry_run: bool = False,
    apply: bool = False,
    include_outcomes: bool = True,
    max_candidates: int | None = None,
    provider: LLMProvider | None = None,
    config: VerifierConfig | None = None,
) -> dict[str, Any]:
    config = config or VerifierConfig(include_outcomes=include_outcomes)
    pricing = PricingCatalog()
    envelopes = load_envelopes(include_outcomes=config.include_outcomes)
    if max_candidates and max_candidates > 0:
        envelopes = envelopes[:max_candidates]
    kalshi_context = build_context_maps(KALSHI_CSV)
    polymarket_context = build_context_maps(POLYMARKET_CSV)
    contexts = targeted_contexts(envelopes, kalshi_context, polymarket_context)
    dry_report = build_dry_run_report(envelopes, config, pricing, contexts)
    cache = DecisionCache()
    migrate_legacy_confirmed(cache, envelopes)
    dry_report["decision_cache"] = cache.stats()
    write_json_atomic(DRY_RUN_REPORT, dry_report)
    if dry_run:
        return {**dry_report, "dry_run": True}

    if provider is None:
        provider = OpenAIProvider()
    verifier = HierarchicalVerifier(provider, config=config, pricing=pricing)
    final = verifier.verify_consensus(envelopes, contexts)
    envelope_map = {envelope.pair_id: envelope for envelope in envelopes}
    output_path = DECISIONS_JSONL if apply else os.path.join(ROOT_DIR, "data", "reports", "llm_decisions_shadow.jsonl")
    _write_decisions(output_path, final)
    ambiguous_path = AMBIGUOUS_CSV if apply else os.path.join(ROOT_DIR, "data", "reports", "ambiguous_matches_shadow.csv")
    _write_ambiguous(final, envelope_map, ambiguous_path)
    applied = _apply_confirmed(final, envelope_map) if apply else 0
    summary = build_run_summary(
        verifier.run_id,
        verifier.records,
        {
            "final_decisions": len(final),
            "final_match": sum(record.decision == Decision.MATCH for record in final),
            "final_no_match": sum(record.decision == Decision.NO_MATCH for record in final),
            "final_ambiguous": sum(record.decision == Decision.AMBIGUOUS for record in final),
            "applied_matches": applied,
            "apply": apply,
            "models": {
                "primary": config.primary_model,
                "secondary": config.secondary_model,
                "frontier": config.frontier_model,
                "tiebreak": config.tiebreak_model,
            },
            "consensus_routes": verifier.consensus_routes,
        },
    )
    write_json_atomic(SHADOW_REPORT if not apply else os.path.join(ROOT_DIR, "data", "reports", "llm_apply_report.json"), summary)
    return summary
