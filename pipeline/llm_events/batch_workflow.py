from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.llm_events.decision_cache import DecisionCache  # noqa: E402
from pipeline.llm_events.consensus import needs_frontier, needs_tiebreak, route_counts, select_consensus  # noqa: E402
from pipeline.llm_events.decision_schema import Usage, VerificationStage, decision_json_schema, validate_decision_payload  # noqa: E402
from pipeline.llm_events.pricing import PricingCatalog, estimate_text_tokens  # noqa: E402
from pipeline.llm_events.prompt_builder import CandidateEnvelope, build_prompt, prompt_version  # noqa: E402
from pipeline.llm_events.provider import LLMRequest, OpenAIProvider, ProviderResponse  # noqa: E402
from pipeline.llm_events.verifier import (  # noqa: E402
    AMBIGUOUS_CSV,
    CONFIRMED_CSV,
    DECISIONS_JSONL,
    KALSHI_CSV,
    POLYMARKET_CSV,
    VerifierConfig,
    _apply_confirmed,
    _record_from_payload,
    _write_ambiguous,
    _write_decisions,
    build_context_maps,
    chunk_envelopes,
    load_envelopes,
    needs_detail,
    needs_review,
    select_final_records,
    targeted_contexts,
)


BATCH_DIR = os.path.join(ROOT_DIR, "data", "reports", "llm_batches")
CONSENSUS_PASSES = ("primary", "secondary", "frontier", "tiebreak")


def _consensus_pass_config(config: VerifierConfig, pass_name: str) -> tuple[str, str | None, int]:
    if pass_name not in CONSENSUS_PASSES:
        raise ValueError(f"unknown consensus pass: {pass_name}")
    return (
        getattr(config, f"{pass_name}_model"),
        getattr(config, f"{pass_name}_reasoning"),
        config.consensus_batch_size,
    )


def _estimated_request_cost(requests: Sequence[LLMRequest], pricing: PricingCatalog) -> dict[str, float | int]:
    input_low = input_high = output_high = 0
    for request in requests:
        estimate = estimate_text_tokens(request.system_message + request.user_message)
        input_low += estimate.low
        input_high += estimate.high
        output_high += request.max_output_tokens
    model = requests[0].model if requests else ""
    rates = pricing.rates(model, "batch") if model else {"input": 0.0, "output": 0.0}
    low = input_low * rates["input"] / 1_000_000
    high = (input_high * rates["input"] + output_high * rates["output"]) / 1_000_000
    return {
        "input_tokens_low": input_low,
        "input_tokens_high": input_high,
        "output_tokens_high": output_high,
        "cost_usd_low": round(low, 6),
        "cost_usd_high": round(high, 6),
    }


def _stage_config(config: VerifierConfig, stage: VerificationStage) -> tuple[str, str | None, int]:
    if stage == VerificationStage.CHEAP:
        return config.cheap_model, config.cheap_reasoning, config.cheap_batch_size
    if stage == VerificationStage.DETAIL:
        return config.detail_model, config.detail_reasoning, config.detail_batch_size
    return config.review_model, config.review_reasoning, config.review_batch_size


def _contexts(envelopes: Sequence[CandidateEnvelope], stage: VerificationStage) -> dict[str, dict[str, Any]]:
    if stage == VerificationStage.CHEAP:
        return {}
    return targeted_contexts(envelopes, build_context_maps(KALSHI_CSV), build_context_maps(POLYMARKET_CSV))


def _cached_map(
    cache: DecisionCache,
    envelopes: Sequence[CandidateEnvelope],
    stage: VerificationStage,
    model: str,
    contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for envelope in envelopes:
        record = cache.get(
            envelope.pair_id,
            envelope.content_hash(stage, contexts.get(envelope.pair_id, {})),
            stage,
            model,
            prompt_version(stage),
        )
        if record:
            result[envelope.pair_id] = record
    return result


def stage_inputs(
    stage: VerificationStage,
    envelopes: Sequence[CandidateEnvelope],
    cache: DecisionCache,
    config: VerifierConfig,
) -> tuple[list[CandidateEnvelope], dict[str, dict[str, Any]]]:
    contexts = _contexts(envelopes, stage)
    if stage == VerificationStage.CHEAP:
        return list(envelopes), contexts

    cheap = _cached_map(cache, envelopes, VerificationStage.CHEAP, config.cheap_model, {})
    if len(cheap) != len(envelopes):
        raise RuntimeError("cheap stage is incomplete; collect it before preparing detail/review")
    detail_candidates = [
        envelope for envelope in envelopes
        if needs_detail(envelope, cheap[envelope.pair_id], config.cheap_no_match_confidence)
    ]
    if stage == VerificationStage.DETAIL:
        return detail_candidates, contexts

    detail_contexts = _contexts(detail_candidates, VerificationStage.DETAIL)
    detail = _cached_map(cache, detail_candidates, VerificationStage.DETAIL, config.detail_model, detail_contexts)
    if len(detail) != len(detail_candidates):
        raise RuntimeError("detail stage is incomplete; collect it before preparing review")
    review_candidates = [
        envelope for envelope in detail_candidates
        if needs_review(cheap[envelope.pair_id], detail[envelope.pair_id])
    ]
    return review_candidates, {pair_id: contexts[pair_id] for pair_id in {item.pair_id for item in review_candidates}}


def prepare_stage(
    stage: VerificationStage,
    config: VerifierConfig | None = None,
    include_outcomes: bool = True,
) -> dict[str, Any]:
    config = config or VerifierConfig(include_outcomes=include_outcomes, pricing_mode="batch")
    cache = DecisionCache()
    envelopes = load_envelopes(include_outcomes)
    candidates, contexts = stage_inputs(stage, envelopes, cache, config)
    model, reasoning, batch_size = _stage_config(config, stage)
    misses = []
    for envelope in candidates:
        if not cache.get(
            envelope.pair_id,
            envelope.content_hash(stage, contexts.get(envelope.pair_id, {})),
            stage,
            model,
            prompt_version(stage),
        ):
            misses.append(envelope)

    batches = chunk_envelopes(misses, stage, contexts, batch_size, config.max_batch_chars)
    requests = []
    manifest_requests = []
    for index, batch in enumerate(batches):
        system, user = build_prompt(batch, stage, contexts)
        custom_id = f"{stage.value}-{index:06d}"
        requests.append(
            LLMRequest(
                custom_id=custom_id,
                model=model,
                system_message=system,
                user_message=user,
                json_schema=decision_json_schema(),
                max_output_tokens=max(800, 180 * len(batch)),
                reasoning_effort=reasoning,
            )
        )
        manifest_requests.append({"custom_id": custom_id, "pair_ids": [item.pair_id for item in batch]})

    os.makedirs(BATCH_DIR, exist_ok=True)
    input_path = os.path.join(BATCH_DIR, f"{stage.value}_input.jsonl")
    manifest_path = os.path.join(BATCH_DIR, f"{stage.value}_manifest.json")
    if requests:
        OpenAIProvider.write_batch_file(requests, input_path)
    else:
        Path(input_path).write_text("", encoding="utf-8")
    estimate = _estimated_request_cost(requests, PricingCatalog())
    manifest = {
        "stage": stage.value,
        "model": model,
        "reasoning_effort": reasoning,
        "prompt_version": prompt_version(stage),
        "include_outcomes": include_outcomes,
        "input_path": input_path,
        "candidate_count": len(candidates),
        "cached_count": len(candidates) - len(misses),
        "request_count": len(requests),
        "estimate": estimate,
        "requests": manifest_requests,
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": manifest_path}


def consensus_pass_inputs(
    pass_name: str,
    envelopes: Sequence[CandidateEnvelope],
    cache: DecisionCache,
    config: VerifierConfig,
    contexts: Mapping[str, Mapping[str, Any]],
) -> list[CandidateEnvelope]:
    if pass_name in {"primary", "secondary"}:
        return list(envelopes)

    primary = _cached_map(cache, envelopes, VerificationStage.DETAIL, config.primary_model, contexts)
    secondary = _cached_map(cache, envelopes, VerificationStage.DETAIL, config.secondary_model, contexts)
    if len(primary) != len(envelopes) or len(secondary) != len(envelopes):
        raise RuntimeError("primary and secondary consensus passes must be collected first")
    frontier_inputs = [
        envelope
        for envelope in envelopes
        if needs_frontier(primary[envelope.pair_id], secondary[envelope.pair_id])
    ]
    if pass_name == "frontier":
        return frontier_inputs

    frontier = _cached_map(cache, frontier_inputs, VerificationStage.DETAIL, config.frontier_model, contexts)
    if len(frontier) != len(frontier_inputs):
        raise RuntimeError("frontier consensus pass must be collected before tiebreak")
    return [
        envelope
        for envelope in frontier_inputs
        if needs_tiebreak(secondary[envelope.pair_id], frontier[envelope.pair_id])
    ]


def prepare_consensus_pass(
    pass_name: str,
    config: VerifierConfig | None = None,
    include_outcomes: bool = True,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    config = config or VerifierConfig(include_outcomes=include_outcomes, pricing_mode="batch")
    cache = DecisionCache()
    envelopes = load_envelopes(include_outcomes)
    if max_candidates and max_candidates > 0:
        envelopes = envelopes[:max_candidates]
    contexts = _contexts(envelopes, VerificationStage.DETAIL)
    candidates = consensus_pass_inputs(pass_name, envelopes, cache, config, contexts)
    model, reasoning, batch_size = _consensus_pass_config(config, pass_name)
    misses = [
        envelope
        for envelope in candidates
        if not cache.get(
            envelope.pair_id,
            envelope.content_hash(VerificationStage.DETAIL, contexts.get(envelope.pair_id, {})),
            VerificationStage.DETAIL,
            model,
            prompt_version(VerificationStage.DETAIL),
        )
    ]
    batches = chunk_envelopes(
        misses,
        VerificationStage.DETAIL,
        contexts,
        batch_size,
        config.max_batch_chars,
    )
    requests = []
    manifest_requests = []
    for index, batch in enumerate(batches):
        system, user = build_prompt(batch, VerificationStage.DETAIL, contexts)
        custom_id = f"consensus-{pass_name}-{index:06d}"
        requests.append(
            LLMRequest(
                custom_id=custom_id,
                model=model,
                system_message=system,
                user_message=user,
                json_schema=decision_json_schema(),
                max_output_tokens=max(800, 180 * len(batch)),
                reasoning_effort=reasoning,
            )
        )
        manifest_requests.append({"custom_id": custom_id, "pair_ids": [item.pair_id for item in batch]})

    os.makedirs(BATCH_DIR, exist_ok=True)
    input_path = os.path.join(BATCH_DIR, f"consensus_{pass_name}_input.jsonl")
    manifest_path = os.path.join(BATCH_DIR, f"consensus_{pass_name}_manifest.json")
    if requests:
        OpenAIProvider.write_batch_file(requests, input_path)
    else:
        Path(input_path).write_text("", encoding="utf-8")
    estimate = _estimated_request_cost(requests, PricingCatalog())
    manifest = {
        "workflow": "consensus-v1",
        "pass_name": pass_name,
        "stage": VerificationStage.DETAIL.value,
        "model": model,
        "reasoning_effort": reasoning,
        "prompt_version": prompt_version(VerificationStage.DETAIL),
        "include_outcomes": include_outcomes,
        "max_candidates": max_candidates,
        "input_path": input_path,
        "candidate_count": len(candidates),
        "cached_count": len(candidates) - len(misses),
        "request_count": len(requests),
        "estimate": estimate,
        "requests": manifest_requests,
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": manifest_path}


def submit_manifest(manifest_path: str) -> dict[str, Any]:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest["request_count"]:
        return {"status": "nothing_to_submit", "manifest_path": manifest_path}
    estimated_high = float((manifest.get("estimate") or {}).get("cost_usd_high", 0.0))
    max_cost = float(os.environ.get("LLM_MAX_BATCH_COST_USD", "0.25"))
    if estimated_high <= 0:
        raise RuntimeError("batch manifest has no positive cost estimate; prepare it again before submission")
    if estimated_high > max_cost:
        raise RuntimeError(
            f"batch estimate ${estimated_high:.4f} exceeds LLM_MAX_BATCH_COST_USD=${max_cost:.4f}; "
            "raise the environment limit explicitly after reviewing the manifest"
        )
    result = OpenAIProvider().submit_batch_file(
        manifest["input_path"],
        metadata={
            "stage": manifest["stage"],
            "pass": manifest.get("pass_name", "legacy"),
            "prompt_version": manifest["prompt_version"],
        },
    )
    manifest["submission"] = result
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _usage(body: Mapping[str, Any]) -> Usage:
    raw = body.get("usage", {}) or {}
    prompt_details = raw.get("prompt_tokens_details", {}) or {}
    completion_details = raw.get("completion_tokens_details", {}) or {}
    return Usage(
        input_tokens=int(raw.get("prompt_tokens", 0) or 0),
        cached_input_tokens=int(prompt_details.get("cached_tokens", 0) or 0),
        output_tokens=int(raw.get("completion_tokens", 0) or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens", 0) or 0),
    )


def collect_manifest(manifest_path: str, result_path: str | None = None) -> dict[str, Any]:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if result_path is None:
        batch_id = manifest.get("submission", {}).get("batch_id")
        if not batch_id:
            raise RuntimeError("manifest has not been submitted")
        result_name = manifest.get("pass_name", manifest["stage"])
        result_path = os.path.join(BATCH_DIR, f"{result_name}_result.jsonl")
        OpenAIProvider().download_batch_results(batch_id, result_path)

    stage = VerificationStage(manifest["stage"])
    config = VerifierConfig(pricing_mode="batch")
    envelopes = load_envelopes(bool(manifest.get("include_outcomes", config.include_outcomes)))
    envelope_map = {envelope.pair_id: envelope for envelope in envelopes}
    contexts = _contexts(envelopes, stage)
    request_map = {item["custom_id"]: item["pair_ids"] for item in manifest["requests"]}
    pricing = PricingCatalog()
    cache = DecisionCache()
    records = []
    missing = []

    for line in Path(result_path).read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        custom_id = raw.get("custom_id", "")
        expected_ids = request_map.get(custom_id, [])
        response = raw.get("response") or {}
        body = response.get("body") or {}
        if int(response.get("status_code", 0) or 0) != 200:
            missing.extend(expected_ids)
            continue
        content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            missing.extend(expected_ids)
            continue
        usage = _usage(body)
        raw_decisions = payload.get("decisions", []) if isinstance(payload, Mapping) else []
        count = max(1, len(raw_decisions))
        seen = set()
        for index, item in enumerate(raw_decisions):
            try:
                validated = validate_decision_payload({"decisions": [item]}, set(expected_ids) - seen)[0]
            except (ValueError, KeyError, TypeError):
                continue
            pair_id = validated["pair_id"]
            seen.add(pair_id)

            def share(total: int) -> int:
                base, remainder = divmod(total, count)
                return base + (1 if index < remainder else 0)

            allocated = Usage(
                input_tokens=share(usage.input_tokens),
                cached_input_tokens=share(usage.cached_input_tokens),
                output_tokens=share(usage.output_tokens),
                reasoning_tokens=share(usage.reasoning_tokens),
            )
            provider_response = ProviderResponse(custom_id, payload, allocated, str(body.get("model", manifest["model"])))
            envelope = envelope_map[pair_id]
            records.append(
                _record_from_payload(
                    validated,
                    envelope,
                    stage,
                    provider_response,
                    allocated,
                    0,
                    envelope.content_hash(stage, contexts.get(pair_id, {})),
                    pricing,
                    "batch",
                )
            )
        missing.extend(pair_id for pair_id in expected_ids if pair_id not in seen)

    cache.put_many(records)
    report = {
        "stage": stage.value,
        "records_cached": len(records),
        "missing_pair_ids": sorted(set(missing)),
        "cost_usd": round(sum(record.cost_usd for record in records), 8),
        "input_tokens": sum(record.usage.input_tokens for record in records),
        "output_tokens": sum(record.usage.output_tokens for record in records),
    }
    collect_name = manifest.get("pass_name", stage.value)
    collect_path = os.path.join(BATCH_DIR, f"{collect_name}_collect_report.json")
    Path(collect_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def finalize_from_cache(apply: bool = False, include_outcomes: bool = True) -> dict[str, Any]:
    config = VerifierConfig(include_outcomes=include_outcomes, pricing_mode="batch")
    envelopes = load_envelopes(include_outcomes)
    envelope_map = {envelope.pair_id: envelope for envelope in envelopes}
    cache = DecisionCache()
    cheap = _cached_map(cache, envelopes, VerificationStage.CHEAP, config.cheap_model, {})
    if len(cheap) != len(envelopes):
        raise RuntimeError(f"cheap cache incomplete: {len(cheap)}/{len(envelopes)}")
    detail_candidates = [
        envelope for envelope in envelopes
        if needs_detail(envelope, cheap[envelope.pair_id], config.cheap_no_match_confidence)
    ]
    detail_contexts = _contexts(detail_candidates, VerificationStage.DETAIL)
    detail = _cached_map(cache, detail_candidates, VerificationStage.DETAIL, config.detail_model, detail_contexts)
    if len(detail) != len(detail_candidates):
        raise RuntimeError(f"detail cache incomplete: {len(detail)}/{len(detail_candidates)}")
    review_candidates = [envelope for envelope in detail_candidates if needs_review(cheap[envelope.pair_id], detail[envelope.pair_id])]
    review_contexts = _contexts(review_candidates, VerificationStage.REVIEW)
    review = _cached_map(cache, review_candidates, VerificationStage.REVIEW, config.review_model, review_contexts)
    if len(review) != len(review_candidates):
        raise RuntimeError(f"review cache incomplete: {len(review)}/{len(review_candidates)}")

    final = select_final_records(envelopes, cheap, detail, review, config.final_confidence)
    _write_decisions(DECISIONS_JSONL, final)
    _write_ambiguous(final, envelope_map)
    applied = _apply_confirmed(final, envelope_map) if apply else 0
    return {
        "final": len(final),
        "matches": sum(record.decision.value == "MATCH" for record in final),
        "no_matches": sum(record.decision.value == "NO_MATCH" for record in final),
        "ambiguous": sum(record.decision.value == "AMBIGUOUS" for record in final),
        "applied": applied,
        "decisions_path": DECISIONS_JSONL,
        "ambiguous_path": AMBIGUOUS_CSV,
        "confirmed_path": CONFIRMED_CSV if apply else "",
    }


def finalize_consensus_from_cache(
    apply: bool = False,
    include_outcomes: bool = True,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    config = VerifierConfig(include_outcomes=include_outcomes, pricing_mode="batch")
    envelopes = load_envelopes(include_outcomes)
    if max_candidates and max_candidates > 0:
        envelopes = envelopes[:max_candidates]
    envelope_map = {envelope.pair_id: envelope for envelope in envelopes}
    contexts = _contexts(envelopes, VerificationStage.DETAIL)
    cache = DecisionCache()
    primary = _cached_map(cache, envelopes, VerificationStage.DETAIL, config.primary_model, contexts)
    secondary = _cached_map(cache, envelopes, VerificationStage.DETAIL, config.secondary_model, contexts)
    if len(primary) != len(envelopes) or len(secondary) != len(envelopes):
        raise RuntimeError(
            f"consensus initial caches incomplete: primary={len(primary)}, secondary={len(secondary)}, total={len(envelopes)}"
        )

    frontier_inputs = [
        envelope
        for envelope in envelopes
        if needs_frontier(primary[envelope.pair_id], secondary[envelope.pair_id])
    ]
    frontier = _cached_map(cache, frontier_inputs, VerificationStage.DETAIL, config.frontier_model, contexts)
    if len(frontier) != len(frontier_inputs):
        raise RuntimeError(f"frontier cache incomplete: {len(frontier)}/{len(frontier_inputs)}")
    tiebreak_inputs = [
        envelope
        for envelope in frontier_inputs
        if needs_tiebreak(secondary[envelope.pair_id], frontier[envelope.pair_id])
    ]
    tiebreak = _cached_map(cache, tiebreak_inputs, VerificationStage.DETAIL, config.tiebreak_model, contexts)

    results = {
        envelope.pair_id: select_consensus(
            primary[envelope.pair_id],
            secondary[envelope.pair_id],
            frontier.get(envelope.pair_id),
            tiebreak.get(envelope.pair_id),
        )
        for envelope in envelopes
    }
    final = [results[envelope.pair_id].record for envelope in envelopes]
    decisions_path = DECISIONS_JSONL if apply else os.path.join(ROOT_DIR, "data", "reports", "llm_decisions_shadow.jsonl")
    ambiguous_path = AMBIGUOUS_CSV if apply else os.path.join(ROOT_DIR, "data", "reports", "ambiguous_matches_shadow.csv")
    _write_decisions(decisions_path, final)
    _write_ambiguous(final, envelope_map, ambiguous_path)
    applied = _apply_confirmed(final, envelope_map) if apply else 0
    return {
        "workflow": "consensus-v1",
        "final": len(final),
        "matches": sum(record.decision.value == "MATCH" for record in final),
        "no_matches": sum(record.decision.value == "NO_MATCH" for record in final),
        "ambiguous": sum(record.decision.value == "AMBIGUOUS" for record in final),
        "routes": route_counts(results),
        "frontier_candidates": len(frontier_inputs),
        "tiebreak_candidates": len(tiebreak_inputs),
        "tiebreak_cached": len(tiebreak),
        "tiebreak_missing_abstained": len(tiebreak_inputs) - len(tiebreak),
        "applied": applied,
        "decisions_path": decisions_path,
        "ambiguous_path": ambiguous_path,
        "confirmed_path": CONFIRMED_CSV if apply else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI Batch workflows for event equivalence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--stage", choices=("cheap", "detail", "review"), required=True)
    prepare.add_argument("--events-only", action="store_true")
    prepare_consensus = subparsers.add_parser("prepare-consensus")
    prepare_consensus.add_argument("--pass-name", choices=CONSENSUS_PASSES, required=True)
    prepare_consensus.add_argument("--events-only", action="store_true")
    prepare_consensus.add_argument("--max-candidates", type=int, default=None)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--manifest", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--batch-id", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--result-file", default=None)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--apply", action="store_true")
    finalize.add_argument("--events-only", action="store_true")
    finalize_consensus = subparsers.add_parser("finalize-consensus")
    finalize_consensus.add_argument("--apply", action="store_true")
    finalize_consensus.add_argument("--events-only", action="store_true")
    finalize_consensus.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_stage(VerificationStage(args.stage), include_outcomes=not args.events_only)
    elif args.command == "prepare-consensus":
        result = prepare_consensus_pass(
            args.pass_name,
            include_outcomes=not args.events_only,
            max_candidates=args.max_candidates,
        )
    elif args.command == "submit":
        result = submit_manifest(args.manifest)
    elif args.command == "status":
        load_dotenv(os.path.join(ROOT_DIR, ".env"))
        result = OpenAIProvider().retrieve_batch(args.batch_id)
    elif args.command == "collect":
        result = collect_manifest(args.manifest, args.result_file)
    elif args.command == "finalize":
        result = finalize_from_cache(args.apply, include_outcomes=not args.events_only)
    else:
        result = finalize_consensus_from_cache(
            args.apply,
            include_outcomes=not args.events_only,
            max_candidates=args.max_candidates,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
