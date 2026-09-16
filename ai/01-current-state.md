# Current State

Last updated: 2026-09-04

## Product

Build a read-only API that returns:

1. equivalent Kalshi and Polymarket events/contracts;
2. live arbitrage opportunities calculated from mapped order books.

Trading and order execution are permanently out of scope for this repository. No wallet custody, order placement/cancellation, trade signing, positions, or automated execution should be added.

## Pipeline

```text
fetch/normalize
-> deterministic event candidates
-> conservative event verification
-> exact contract/outcome mappings
-> live read-only order books
-> fees/liquidity/freshness
-> equivalents + opportunities API
```

## Implementation state

- Fetch/sync supports the current Kalshi external REST API with nested markets and Polymarket `/events/keyset` pagination.
- Deterministic parsing/clustering is implemented and does not require an LLM.
- `data/matches/confirmed_matches.csv` contains a versioned 381-row event-match snapshot.
- `data/matches/confirmed_matches_bets.csv` contains a 499-row contract/outcome snapshot used by the scanner; systematic validation remains a priority.
- Kalshi and Polymarket WebSocket compatibility was migrated and checked with fixtures plus small read-only live smoke tests.
- FastAPI exposes health, refresh, scanner control/status, and opportunities.
- The required equivalent-markets read endpoint and stable response models do not exist yet.
- Arbitrage calculation is a prototype: current Kalshi fees are approximate and per-market fees/ticks plus stale-book guarantees are missing.
- No execution/trading code exists.

## Default development mode

- Work from versioned snapshots and run targeted tests.
- Do not require an OpenAI key. Model-assisted verification is optional research tooling, disabled from the shared environment template.
- Do not run full fetch/pipeline/paid-model calls unless explicitly required.
- Keep all platform credentials local and untracked.

## Current priority

Turn the existing mappings and scanner into a reliable read-only product API:

1. implement typed/paginated equivalent-market responses;
2. validate exact outcome mappings and Same/Opposite semantics;
3. add fee/tick metadata and order-book timestamps;
4. harden scanner reconnect/freshness behavior;
5. expose only traceable, fresh, fee-aware opportunities.

Canonical remote: `https://github.com/Chkeibs/Kalshi-Polymarket-Arbitrage-Simulation.git`.
