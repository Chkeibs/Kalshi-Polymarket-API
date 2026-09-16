# AI Collaboration Instructions

Read first:

1. `README.md`
2. `ai/00-router.md`
3. `ai/01-current-state.md`

If continuing previous work, also read `ai/handoff.md`. Open only the relevant feature docs in `ai/features/`; do not scan the whole repository documentation by default.

## Product boundary

This repository builds a read-only API for equivalent Kalshi/Polymarket events, exact outcome mappings, and live arbitrage signals. It is not a trading bot.

- Do not add order placement/cancellation, wallets, trade signing, positions, or automated execution.
- Kalshi signing is allowed only for authenticated read-only market-data access.
- Never commit `.env`, OpenAI keys, Kalshi credentials, wallet keys, or any other secret.
- The default API path must not require an OpenAI/GPT key.
- Do not run a full fetch, the full pipeline, paid model calls, or long live sessions unless the task explicitly needs them.

Before large code changes, state which context files you read. Prefer targeted tests that do not run the whole pipeline. After meaningful work, update `ai/handoff.md` with the new state, touched files, validation, next task, and open questions.
