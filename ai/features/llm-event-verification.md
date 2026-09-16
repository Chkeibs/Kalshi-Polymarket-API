# LLM Event Verification

## Purpose

Verify candidate event pairs after deterministic parsing/clustering, producing reliable event matches while avoiding false arbitrage-causing positives and expensive repeated calls.

This is optional research tooling. The default read-only API works from validated versioned mappings and must not require an OpenAI key.

## Current Behavior

- `pipeline/llm_events/incremental_matcher.py` loads candidates, skips cached matches, can dry-run without model calls, and writes confirmed event matches in legacy/current flows.
- `pipeline/llm_events/verifier.py` implements the newer cascade with cache, context targeting, retries, budget checks, shadow/apply mode, and dry-run estimates.
- `pipeline/llm_events/decision_schema.py` defines compact decisions, relation types, reason codes, and semantic validation.
- `pipeline/llm_events/provider.py` abstracts OpenAI/FakeProvider, structured outputs, usage, and Batch primitives.
- `pipeline/llm_events/decision_cache.py`, `pricing.py`, and `telemetry.py` make decisions auditable and cacheable.
- `pipeline/llm_events/batch_workflow.py` prepares/submits/statuses/collects/finalizes consensus passes with cost limits.
- `pipeline/llm_events/consensus.py` and `consolidation.py` apply conservative consensus and deterministic relation consolidation.

## Important Rules

- Do not let LLM search for candidates; it only verifies candidates already generated.
- Missing critical info should become `AMBIGUOUS`, not `NO_MATCH`.
- `MATCH` requires same underlying reality and compatible resolution predicate.
- Event-event and event-outcome bridge relations are different.
- A confidence score cannot override an explicit contradiction.
- Model outputs must be structured and validated by `pair_id`.
- Cache keys must include content hashes, parser/policy/prompt versions, provider, and model.
- Errors, timeouts, invalid JSON, and missing IDs are retryable/error states, never negative labels.
- Full paid runs are blocked unless cost/quality gates and user approval are explicit.
- Never put a provider key in Git or the shared `.env.example`.
- A future Claude/provider adapter is acceptable only if it preserves the same structured, cached, budget-gated contract.

## Related Files

- `pipeline/llm_events/incremental_matcher.py`
- `pipeline/llm_events/verifier.py`
- `pipeline/llm_events/decision_schema.py`
- `pipeline/llm_events/prompt_builder.py`
- `pipeline/llm_events/provider.py`
- `pipeline/llm_events/decision_cache.py`
- `pipeline/llm_events/pricing.py`
- `pipeline/llm_events/telemetry.py`
- `pipeline/llm_events/batch_workflow.py`
- `pipeline/llm_events/consensus.py`
- `pipeline/llm_events/consolidation.py`
- `pipeline/llm_events/*evaluation*.py`
- `config/llm_pricing.json`
- `data/evaluation/*`
- `tests/test_llm_*.py`

## Current Known State

- The newer cascade was documented as implemented in shadow/dry-run mode.
- Historical gates failed on cost and recall targets: precision was good, but projected full-run cost was too high and recall did not meet the target.
- Batch canaries and dry-runs are the safe default.

## Edge Cases

- Multi-outcome bridges may remain ambiguous even when event/outcome semantics are related.
- Greedy first-yes behavior is unsafe; consolidate all decisions globally.
- Descriptions and rules are untrusted data and must be delimited in prompts.

## Open Questions

- Whether to invest next in cheaper verifier routing, human gold labels, or candidate reduction before full activation.
