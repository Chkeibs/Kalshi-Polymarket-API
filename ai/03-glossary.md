# Glossary

- Arbitrage: Buying logically equivalent outcomes across platforms when prices imply risk-free or near risk-free profit.
- Event: The high-level market subject, such as an election, match, tournament, award, or economic release.
- Contract / Bet: A specific tradable market or outcome under an event.
- Outcome: A selectable result, especially in Polymarket multi-outcome markets.
- Event match: A Kalshi event and Polymarket event that represent the same underlying reality and compatible predicate.
- Contract match: A Kalshi contract/outcome and Polymarket token/outcome that resolve the same way.
- Candidate: A possible Kalshi/Polymarket pair generated before final verification.
- Event profile: Structured extracted representation of an event, including participants, jurisdiction, time, scope, proposition, and conflicts.
- Occurrence key: Canonical key for the underlying real-world occurrence.
- Proposition key: Occurrence key plus action/metric/scope/direction/threshold/outcome type.
- Anchor: Typed evidence that can strongly route or support a candidate, such as participants, jurisdiction+office, competition+stage, or asset+metric.
- Hard reject: Deterministic rejection when both sides explicitly contain incompatible facts.
- Missing information: A dimension present on one side but absent on the other; this may make a pair ambiguous but should not reject it by itself.
- Event-outcome bridge: Relation where a global event and a specific outcome share a reality but are not strict event-event equivalents.
- Silver set: Evaluation labels created from consensus/human resolution, useful for engineering validation but not a fully human gold truth.
- Gold set: Manually audited labels intended as stronger ground truth.
- Shadow mode: Run the new verifier or policy without writing final production outputs.
- Batch canary: Small OpenAI Batch preparation/test used to estimate behavior without submitting a full costly run.
