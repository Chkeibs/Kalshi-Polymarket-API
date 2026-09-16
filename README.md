# Kalshi × Polymarket — Arbitrage Simulation

A Python project for answering two questions: **which Kalshi and Polymarket contracts describe the same outcome, and where do their live prices imply an arbitrage opportunity?**

The goal is a read-only market-intelligence API for arbitrage simulation, not a trading bot. It links markets, explains mappings, and calculates hypothetical signals from order-book depth. It does not place orders, manage positions, or hold trading wallets.

> **Status:** matching pipeline and live-data prototype implemented; production API still in development. Existing matches are snapshots, and opportunity estimates are not yet production-grade.

## The idea — and the difficult part

Both platforms can ask about the same real-world event using different titles, categories, market structures, and settlement rules. There is no shared event ID, so comparison starts with discovering equivalent questions.

For example, “Will Team A win the tournament?” may correspond to the Team A outcome inside a multi-outcome “Who will win the tournament?” market. Conversely, “wins the election” and “becomes president” can sound equivalent while having different settlement conditions.

A price difference between unrelated contracts is not arbitrage. The system therefore separates three levels:

- **Candidate:** worth comparing, but not a confirmed match.
- **Equivalent event:** the parent questions concern compatible underlying events.
- **Equivalent contract/outcome:** the exact payoff predicates align, including dates, thresholds, participants, and outcome direction.

Only the last level should feed trusted arbitrage signals. Missing information should lead to review or ambiguity, not an invented match.

## How it works

```text
Kalshi REST + Polymarket REST
  → normalized market rows
  → keywords, entities, dates and structured event profiles
  → ranked event/outcome candidates
  → conservative verification and reviewed mappings
  → exact contract/token outcome pairs
  → live WebSocket order books
  → liquidity + fees + freshness checks
  → equivalent-markets and opportunities API
```

### 1. Fetch and normalize

`pipeline/fetch/` converts both platforms into comparable CSV schemas while retaining source IDs, titles, descriptions, rules, and outcome information. Polymarket outcomes are expanded into rows for comparison with individual Kalshi contracts.

Sync compares an active-market snapshot against local data; it is **not yet true API-side incremental ingestion**. The August 2026 compatibility migration added Kalshi nested-market discovery and Polymarket keyset pagination, validated with fixtures and small live read-only checks. Those checks are historical validation, not a guarantee that external APIs never change.

### 2. Find candidates without comparing everything

`pipeline/clustering/` uses categories as routing hints, canonical keywords/aliases, entities, time periods, participants, locations, and other structured dimensions. Explicit contradictions reject pairs early; one-sided missing fields do not.

Clustering v2 adds anchor/keyword-frequency scoring, reciprocal ranking, candidate budgets, and graph diagnostics. A separate outcome-aware stage handles broad parent events versus individual outcomes.

The saved v2 report reduced **10,859 candidates to 5,190 (52.2%)**. Regression coverage retained **381/381 existing event matches** and **82/82 silver event matches**. This demonstrates preservation of the known reference set—not measured recall across every market or independently proven match quality.

### 3. Verify semantics, then map exact outcomes

`pipeline/llm_events/` contains optional model-assisted verification: structured decisions, targeted context, caching, retries, conservative consensus, shadow/apply controls, and cost limits. The model verifies candidates; it does not search the whole market universe.

`pipeline/llm_bets/` contains the current model-assisted contract/outcome association step. Event matches and exact mappings are stored separately. Same versus Opposite payoff direction must be explicit before using a pair for arbitrage.

The API can operate from existing match snapshots without an OpenAI key. **Autonomous generation of new high-quality matches is still unfinished.**

### 4. Read live books and calculate signals

`api/services/scanner.py` loads exact mappings, resolves Polymarket token IDs, and consumes both market-data WebSockets. It supports Kalshi fixed-point book messages and Polymarket book/price-change messages.

The arbitrage engine walks available depth rather than comparing only best prices. Conceptually, complementary payouts costing less than their combined settlement value can produce a signal; the calculation estimates volume, capital, fees, profit, and ROI in both directions. Opposite mappings require payoff normalization.

This remains an estimate: **fees are approximate, book freshness is not enforced, and settlement assumptions need systematic mapping validation.** No order is executed.

## Techniques explored and lessons learned

| Approach | Role / lesson |
|---|---|
| Categories and keyword overlap | Useful for routing and discovery, but too weak to prove equivalence. Common words create noisy comparison graphs. |
| Structured parsing + clustering v2 | Current deterministic foundation. Reject explicit conflicts and prioritize strong anchors while preserving known regression pairs. |
| LLM semantic verification | Useful for wording/rules differences, but expensive and fallible. Research tooling exists; production activation is incomplete. |
| Cascade, consensus, caching and Batch | Implemented experiments to control cost and uncertainty. More verification does not automatically solve recall or label quality. |
| Human review / reference datasets | Necessary to audit ambiguity and measure quality. Legacy and synthetic labels are diagnostic, not independent proof. |
| Embeddings / semantic retrieval | Possible future candidate-retrieval extension, **not implemented**. Similarity would still require settlement verification. |

Development progressed from normalization and broad candidate matching to structured pruning, verifier experiments, exact outcome snapshots, then a read-only live scanner. Historical verifier evaluations did not meet the desired cost/recall gates. Small prompt canaries also exposed false-positive and malformed/truncated-response risks, so paid verification remains gated rather than silently promoted into production.

The main lesson: improve candidate discovery and audit semantic equivalence before scaling model calls or trusting price signals.

## Where we are

Repository inspection and clean-snapshot validation: **2026-09-15**, **132 unit tests passed**.

| Component | Current state |
|---|---|
| REST ingestion / normalization | Implemented; snapshot sync, bounded retries and schema tests. |
| Deterministic event/outcome candidates | Implemented with regression and audit tooling. |
| Event-match snapshot | 381 rows in `confirmed_matches.csv`; not a freshly regenerated live catalogue. |
| Exact outcome snapshot | 499 rows in `confirmed_matches_bets.csv`; systematic audit still needed. |
| Model verifier | Optional research subsystem; cost/quality gates remain unresolved. |
| Live scanner / arbitrage | Prototype; fees, staleness and resilience need hardening. |
| Equivalent-markets endpoint | **Not implemented yet.** |

The bottleneck is not simply adding endpoints: it is making mappings and live calculations trustworthy. False matches, stale books, incomplete fee metadata, and outdated snapshots can all create misleading opportunities.

## Current API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Process health. |
| `GET` | `/scanner/status` | Running state, tracked pairs, opportunity count and last error. |
| `POST` | `/scanner/start`, `/scanner/stop` | Control read-only WebSocket consumers. |
| `GET` | `/opportunities?min_profit=0&min_roi=0` | In-memory signals filtered/sorted by estimated profit and ROI. |
| `POST` | `/pipeline/refresh?batch_size=40` | Refresh source data and matching artifacts; **not a cheap health check**. |

Planned: `GET /equivalent-markets` with typed responses, filters, pagination, source IDs, mapping status, and provenance. Scanner/refresh endpoints are operational controls, not trading endpoints. Authentication and public deployment hardening are not implemented.

## Run locally

```bash
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive documentation. CSV snapshots use Git LFS; fetch their contents before offline pipeline/schema tests.

Kalshi live market-data access needs local `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY` environment values. Signing authenticates the read-only WebSocket connection, not trades. Polymarket public market data needs no wallet key. No OpenAI key is required to serve existing mappings/signals; paid verifier modes need separate explicit configuration. **Never commit real credentials or `.env`.**

Safe tests without live fetches or paid model calls:

```bash
python3 -m pytest -q
python3 -m compileall -q api pipeline tests main.py
```

Deterministic candidate regeneration from local snapshots, without LLM calls:

```bash
python3 pipeline/run_clustering_pipeline.py
```

That command rewrites generated artifacts. Adding `--fetch`, calling `/pipeline/refresh`, or running the full pipeline calls external services and can replace data. Default refresh verification is dry-run, but environment settings can enable paid modes; inspect configuration first.

## Code and data guide

- `api/` — FastAPI, scanner, volume-weighted arbitrage engine.
- `pipeline/fetch/` — ingestion, normalization and snapshot sync.
- `pipeline/common/`, `config/keywords/` — schemas, parsing, aliases and keyword configuration.
- `pipeline/clustering/` — profiles, candidate scoring/pruning and outcome bundles.
- `pipeline/llm_events/`, `pipeline/llm_bets/` — optional verification research and exact outcome association.
- `tests/` — unit, schema, regression and API-format compatibility checks.

Key stage contracts are `data/markets/markets_clean_*.csv`, `data/clusters/candidate_clusters.json`, outcome candidate/bundle files, and the two `data/matches/confirmed_matches*.csv` snapshots. Candidates are not confirmed matches. Runtime reports, decision caches and `config/poly_id_map.json` are ignored by Git; keyword/reference data is retained for reproducible development.

## Next milestones

1. Audit exact outcome mappings and define a reviewed, versioned schema with Same/Opposite semantics and provenance.
2. Implement typed, paginated `/equivalent-markets` responses from those mappings.
3. Add per-market fee metadata, book timestamps and stale/partial-book rejection.
4. Harden reconnect/resubscribe behavior, task health and opportunity deduplication with focused tests.
5. Make opportunity responses explain their inputs and limitations; improve match generation using independently reviewed labels before scaling paid verification.

Order execution, trading wallets and position management remain outside this repository's scope.

## Working with Claude or Codex

Start with [AGENTS.md](AGENTS.md), [the context router](ai/00-router.md), [current state](ai/01-current-state.md), and [handoff](ai/handoff.md). Read only the relevant feature docs after that. Preserve the read-only boundary, distinguish implemented behavior from plans, avoid unrequested full/paid runs, and update the handoff after meaningful work.
