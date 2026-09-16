# Handoff

Last updated: 2026-09-15

## Current product state

This is a read-only API project for equivalent Kalshi/Polymarket events, exact outcome mappings and live arbitrage signals. Trading, wallets, order placement/cancellation and positions are out of scope.

- REST ingestion and deterministic event/outcome candidate generation are implemented.
- Versioned snapshots contain 381 event matches and 499 exact contract/outcome mappings; these require systematic review before production use.
- Live scanner consumers support the API formats validated in August 2026.
- Arbitrage calculation and opportunity endpoints remain prototypes.
- Equivalent-market responses, reviewed mapping schemas, per-market fee metadata, stale-book rejection and scanner resilience remain unfinished.
- Optional model verification is research tooling, not a default API requirement.

## Public snapshot

This repository starts with a single clean initial commit on main. It contains current source, configuration, tests and versioned data, without previous Git history, private environment files, logs or runtime caches.

The public repository was named Kalshi-Polymarket-Arbitrage-Simulation. README title and repository URLs were aligned while retaining a single root commit; no source or dataset changed during renaming.

The README explains the architecture, methods explored, historical metrics and limitations. No pipeline, live scanner session or paid model call was run during export. All keyword and reference datasets were retained.

## Security

Only empty credential placeholders and public API hosts are shared. Keep real credentials local in ignored environment files. Never add wallet or execution credentials.

## Validation

The exported snapshot independently passed all 132 unit tests on 2026-09-15; compileall passed. A scan of all 131 exported files, including materialized CSV data, found no provider tokens, encoded private keys, exact local credentials/key fragments, targeted personal identifiers or references to the private repository. Only public-facing documentation changed; source code and datasets were copied unchanged. No pipeline or external model call ran.

## Next task

Audit exact outcome mappings and introduce a reviewed, versioned schema with Same/Opposite semantics and provenance. Implement typed, paginated GET /equivalent-markets responses, then harden fee metadata and order-book freshness with focused tests.

## Repository

- Branch: main
- Remote: https://github.com/Chkeibs/Kalshi-Polymarket-Arbitrage-Simulation.git
