# Fetch / Sync

## Purpose

Fetch active Kalshi and Polymarket markets, normalize them into comparable CSVs, and maintain sync status for downstream matching.

## Current Behavior

- `pipeline/fetch/fetch_clean_data.py` calls platform APIs, normalizes dates/categories/titles, explodes Polymarket outcomes into comparable rows, and aligns columns to schemas.
- Kalshi uses `external-api.kalshi.com`, cursor pagination, and `with_nested_markets=true`; a bounded event-detail fallback runs only when a response omits nested markets.
- Polymarket uses `GET /events/keyset` with `after_cursor` / `next_cursor` and keeps the existing outcome-explosion schema.
- HTTP calls reuse per-thread sessions and retry bounded transient failures.
- `pipeline/fetch/data_manager.py` manages incremental CSV sync, marks `New` / `Existing`, preserves old statuses, aborts partial-looking fetches, and validates platform/schema contracts.
- Main outputs are `data/markets/markets_clean_kalshi.csv` and `data/markets/markets_clean_polymarket.csv`.

## Important Rules

- Fetch quality matters because all later matching depends on these CSVs.
- Current sync compares a newly fetched full-ish snapshot to local CSVs; it is not yet a true API-side incremental fetch.
- Do not push secrets, runtime logs, caches, or `config/poly_id_map.json`.

## Related Files

- `pipeline/fetch/fetch_clean_data.py`
- `pipeline/fetch/data_manager.py`
- `pipeline/common/schemas.py`
- `tests/test_fetch_sync.py`
- `data/markets/markets_clean_kalshi.csv`
- `data/markets/markets_clean_polymarket.csv`

## Edge Cases

- Kalshi full-fetch performance still needs a measured benchmark, but the previous per-event detail N+1 path is no longer the default.
- Polymarket fetch is much faster in observed benchmarks.
- A future fast event-matching fetch mode should avoid pulling full details unless needed for bets/arbitrage.
- `min_updated_ts` is not used for the main sync because `MarketDataManager` currently expects a full-ish active snapshot before computing removals.

## Open Questions

- Benchmark a bounded full Kalshi refresh before deciding whether series-tag caching is still needed.
- Design a true incremental API-side mode separately from the full-snapshot sync contract.
- Decide whether market CSVs should remain versioned or be treated as regenerable local artifacts.
