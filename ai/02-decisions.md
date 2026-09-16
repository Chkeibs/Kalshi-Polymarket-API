# Decisions

- The product is a read-only equivalent-markets and live-arbitrage API, not a trading system.
- Never add order placement/cancellation, wallets, trade signing, positions, or automated execution to this repository.
- Event equivalence must be reliable before an opportunity can be trusted.
- Exact contract/outcome predicates and Same/Opposite payoff mapping are required after event matching.
- Categories and tags route comparisons; they never prove equivalence alone.
- Missing information is not a conflict. Explicit incompatible dimensions should reject a pair early.
- Preserve multi-outcome mappings and source traceability.
- Prefer deterministic parsing, hard rejects, cache, tests, and audit reports before optional model verification.
- Model-assisted verification must stay optional, structured, cached, budget-gated, and outside the default API request path.
- No OpenAI key belongs in shared setup or Git. Kalshi credentials are local-only even though they are used solely for read-only WebSocket authentication.
- Public API responses need typed schemas, timestamps, provenance, filters, pagination, and freshness guarantees.
- Keep current truth in README/current-state/feature docs/handoff; delete obsolete duplicated plans rather than archiving them indefinitely.
