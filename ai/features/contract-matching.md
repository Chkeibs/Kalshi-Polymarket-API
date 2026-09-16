# Contract / Outcome Matching

## Purpose

Map exact Kalshi contracts/outcomes to Polymarket token outcomes after parent events are judged equivalent. This mapping is a required input to the read-only arbitrage API.

## Current behavior

- `pipeline/llm_bets/verify_bets.py` contains the current model-assisted association step.
- `pipeline/clustering/outcome_candidate_generator.py` produces outcome-aware candidates and parent bundles.
- `data/matches/confirmed_matches_bets.csv` is the scanner input and currently contains a 499-row versioned snapshot.

## Rules

- Never treat an event match as sufficient for arbitrage.
- Resolution predicates, dates, thresholds, participants, outcome meaning, and cancellation rules must be compatible.
- Store whether payoff mapping is `Same` or `Opposite`.
- Preserve legitimate multi-outcome bridges and source traceability.
- A broad event and a specific outcome may discuss the same reality but resolve differently.
- Do not require an OpenAI key in the production API path; validated versioned/manual mappings are valid inputs.

## Next work

- Define and enforce a deterministic schema for reviewed exact mappings.
- Audit the existing snapshot before treating all rows as API-ready.
- Add mapping status/provenance/version fields and tests.
- Feed only validated mappings to equivalent-market and opportunity responses.

## Related files

- `pipeline/llm_bets/verify_bets.py`
- `pipeline/clustering/outcome_candidate_generator.py`
- `data/matches/confirmed_matches.csv`
- `data/matches/confirmed_matches_bets.csv`
- `data/clusters/candidate_outcome_pairs.json`
- `data/clusters/candidate_outcome_bundles.json`
