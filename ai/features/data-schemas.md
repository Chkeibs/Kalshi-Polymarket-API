# Data Artifacts / Schemas

## Purpose

Document the main files that form contracts between pipeline stages.

## Current Behavior

- Schemas are centralized in `pipeline/common/schemas.py`.
- `tests/test_pipeline_schemas.py` enforces required columns/keys, but not exact column order.

## Core Artifacts

- `data/markets/markets_clean_kalshi.csv`: normalized active Kalshi markets/contracts.
- `data/markets/markets_clean_polymarket.csv`: normalized active Polymarket events/outcomes exploded into rows.
- `data/keywords/event_keywords_index.json`: keywords per event.
- `data/clusters/candidate_clusters.json`: event-event candidate pairs.
- `data/clusters/candidate_outcome_pairs.json`: outcome-aware candidate pairs.
- `data/clusters/candidate_outcome_bundles.json`: parent event bundles with outcome mappings.
- `data/clusters/event_profiles.json`: structured event profiles.
- `data/clusters/clustering_run_report.json`: clustering run metrics and policy versions.
- `data/clusters/exclusive_clusters.json`: greedy one-to-one debug artifact.
- `data/clusters/graph_data.json`: graph debug artifact from final candidates.
- `data/matches/confirmed_matches.csv`: verified equivalent event pairs.
- `data/matches/confirmed_matches_bets.csv`: exact contract/outcome mappings consumed by the scanner.
- `data/matches/ambiguous_matches.csv`: review queue for unresolved LLM cases.
- `data/cache/match_cache.jsonl`: legacy/current pair cache.
- `data/cache/llm_decisions_v2.jsonl`: versioned decision cache, generally runtime/ignored.
- `data/evaluation/*`: gold/silver/review/evaluation datasets and manifests.

## Important Rules

- Both event and exact contract/outcome mappings must be validated before an opportunity is API-ready.
- `graph_data.json` is debug; final matching depends on candidates and verifier outputs.
- Runtime reports/caches may be regenerated and should not be treated like source code unless intentionally versioned.

## Related Files

- `pipeline/common/schemas.py`
- `tests/test_pipeline_schemas.py`
- `pipeline/postprocess/clean_confirmed_matches.py`

## Edge Cases

- Old data files may have different column order; tests require presence, not exact order.
- Cleanup must remove invalid confirmed event matches, linked bet matches, and expired cache pairs consistently.

## Open Questions

- Decide which data artifacts should be versioned long-term vs regenerated per environment.
