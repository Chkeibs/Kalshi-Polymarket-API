# Read-only Arbitrage Scanner / API

## Purpose

Expose validated equivalent market mappings and live arbitrage signals without placing trades.

## Current behavior

- `api/services/arbitrage.py` calculates volume-weighted opportunity size, profit, ROI, direction, and an approximate Kalshi fee from normalized order books.
- `api/services/scanner.py` loads `confirmed_matches_bets.csv`, consumes Kalshi and Polymarket market-data WebSockets, maintains in-memory books, and evaluates mapped pairs.
- Kalshi WebSocket signing authenticates read-only data access only.
- `api/main.py` exposes `/health`, `/pipeline/refresh`, scanner start/stop/status, and `/opportunities`.
- There is no order placement/cancellation, wallet, position, or execution code.
- A typed equivalent-markets endpoint is planned but not implemented.

## Current API compatibility

- Kalshi: recommended external WebSocket host, RSA-PSS digest salt, and fixed-point `yes_dollars_fp`, `no_dollars_fp`, `price_dollars`, and `delta_fp` fields. Legacy cent fixtures remain supported.
- Polymarket: market subscription uses `type`, `assets_ids`, `custom_feature_enabled`, and a text `PING` heartbeat.
- Small read-only live smoke checks succeeded on 2026-08-17 without running the full pipeline.

## Non-negotiable rules

- Do not add trading/execution code to this repo.
- Do not return an opportunity unless exact outcome mapping is validated.
- Include source IDs/titles, mapping direction, calculation inputs, timestamps, and freshness in stable API responses.
- Real Kalshi credentials stay in local `.env`; Polymarket public market data needs no wallet key.
- `config/poly_id_map.json` is a runtime cache and should not be pushed.

## Known gaps

- No `GET /equivalent-markets` endpoint or Pydantic response contract.
- Kalshi fee function is a prototype; fees/ticks can vary by market.
- Polymarket fee schedules are not consumed per market.
- Books lack explicit update timestamps and stale-data rejection.
- Reconnect/resubscribe, partial books, opportunity deduplication, and mapping validation need stronger tests.

## Related files

- `api/main.py`
- `api/services/arbitrage.py`
- `api/services/scanner.py`
- `api/services/pipeline.py`
- `tests/test_arbitrage_engine.py`
- `tests/test_scanner_api_compat.py`
