# Event Candidate Matching / Clustering

## Purpose

Generate a small, high-recall set of Kalshi/Polymarket event candidates for later deterministic and LLM verification.

## Current Behavior

- `pipeline/run_clustering_pipeline.py` runs optional fetch, keyword indexing, clustering v2, outcome bundles, and audit without LLM verification.
- `pipeline/clustering/event_profiles.py` builds `event-profile-v2` with participants, groups, districts, locations, periods, scopes, occurrence/proposition keys, and explicit conflicts.
- `pipeline/clustering/clustering_features.py` computes DF/IDF, anchors, occurrence/proposition scores, reciprocal ranking, fanout, components, and versioned pruning policy.
- `pipeline/clustering/group_events_cluster.py` writes `candidate_clusters.json`, `exclusive_clusters.json`, `event_signatures.csv`, and `graph_data.json`.
- `pipeline/clustering/outcome_candidate_generator.py` writes `candidate_outcome_pairs.json` and `candidate_outcome_bundles.json`.
- `pipeline/clustering/clustering_audit.py` produces offline deterministic audit reports.

## Important Rules

- Clustering should find candidates, not final matches.
- Preserve recall for known confirmed and silver MATCH pairs while reducing noisy graph fanout.
- Missing information is not a conflict.
- Hard rejects require explicit incompatible structured values.
- Do not deduplicate by title alone; include domain, subject/participants, time, competition/jurisdiction, and stage/group.
- Outcome rows must remain traceable even when grouped into parent bundles.

## Related Files

- `pipeline/run_clustering_pipeline.py`
- `pipeline/clustering/event_profiles.py`
- `pipeline/clustering/clustering_features.py`
- `pipeline/clustering/group_events_cluster.py`
- `pipeline/clustering/outcome_candidate_generator.py`
- `pipeline/clustering/clustering_audit.py`
- `data/clusters/candidate_clusters.json`
- `data/clusters/candidate_outcome_pairs.json`
- `data/clusters/candidate_outcome_bundles.json`
- `data/clusters/event_profiles.json`
- `data/clusters/clustering_run_report.json`
- `tests/test_event_profiles.py`
- `tests/test_clustering_audit.py`
- `tests/test_clustering_v2_regression.py`
- `tests/test_group_events_cluster_categories.py`
- `tests/test_outcome_candidate_generator.py`

## Current Metrics From Historical Notes

- Policy v2 reduced event candidates from about `10859` to `5190` while preserving `381/381` confirmed event matches and `82/82` silver MATCH events.
- Outcome rows remained `3714`, grouped into `800` parent bundles.
- No LLM calls are required for clustering or its audit.

## Edge Cases

- Group identifiers such as Group A/B must be extracted before short-token removal.
- Sports fixtures require order-independent participant comparison.
- Broad-vs-specific questions should route to outcome-aware candidates rather than strict event-level equivalence.
- Large graph components and fanout are warning signs, not automatic proof of false matches.

## Open Questions

- Future gold labels may reveal recall gaps not covered by current confirmed/silver regression sets.
