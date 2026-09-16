"""
Shared schema contracts for the core matching pipeline.

These are intentionally small and dependency-free. They define the columns/keys
that each stage must preserve so later refactors can be checked quickly.
"""

KALSHI_MARKET_REQUIRED_COLUMNS = (
    "Platform",
    "Event ID",
    "Bet ID",
    "Market Title",
    "Bet Title",
    "Category",
    "Sub-Category",
    "Series Ticker",
    "Description",
    "Bet_Rules",
    "Event_Description",
    "Status",
    "Expiration Date",
    "SyncStatus",
)

POLYMARKET_MARKET_REQUIRED_COLUMNS = (
    "Platform",
    "Event ID",
    "Bet ID",
    "Market Title",
    "Bet Title",
    "Category",
    "Sub-Category",
    "Description",
    "Bet_Rules",
    "Event_Description",
    "Status",
    "Expiration Date",
    "Outcomes",
    "Token_IDs",
    "Original_Market_ID",
    "Outcome_Name",
    "SyncStatus",
)

CONFIRMED_MATCH_REQUIRED_COLUMNS = (
    "Kalshi_ID",
    "Kalshi_Title",
    "Polymarket_ID",
    "Polymarket_Title",
    "Reasoning",
    "Common_Keywords",
    "Match_Score",
    "SyncStatus",
)

AMBIGUOUS_MATCH_REQUIRED_COLUMNS = (
    "Pair_ID",
    "Kalshi_ID",
    "Polymarket_ID",
    "Candidate_Type",
    "Reason_Code",
    "Missing_Fields",
    "Model",
    "Confidence",
)

LLM_DECISION_REQUIRED_KEYS = (
    "pair_id",
    "content_hash",
    "decision",
    "relation_type",
    "reason_code",
    "stage",
    "confidence",
    "model",
    "prompt_version",
    "parser_schema_version",
    "decision_policy_version",
    "usage",
    "cost_usd",
    "status",
)

CANDIDATE_CLUSTER_REQUIRED_KEYS = (
    "cluster_id",
    "kalshi",
    "polymarket",
    "common_keywords",
    "match_score",
)

CANDIDATE_CLUSTER_OPTIONAL_KEYS = (
    "llm_verified",
)

CANDIDATE_KALSHI_REQUIRED_KEYS = (
    "event_id",
    "title",
    "category",
    "keywords",
)

CANDIDATE_POLYMARKET_REQUIRED_KEYS = (
    "event_id",
    "title",
    "category",
    "keywords",
)

CANDIDATE_OUTCOME_PAIR_REQUIRED_KEYS = (
    "candidate_id",
    "candidate_type",
    "kalshi",
    "polymarket",
    "shared_market_keywords",
    "shared_outcome_keywords",
    "match_score",
)

CANDIDATE_OUTCOME_KALSHI_REQUIRED_KEYS = (
    "event_id",
    "bet_id",
    "market_title",
    "bet_title",
    "outcome_subject",
    "outcome_key",
    "market_intent",
    "category",
    "sub_category",
)

CANDIDATE_OUTCOME_POLYMARKET_REQUIRED_KEYS = (
    "event_id",
    "bet_id",
    "original_market_id",
    "market_title",
    "bet_title",
    "outcome_subject",
    "outcome_key",
    "outcome_name",
    "market_intent",
    "category",
    "sub_category",
)


def missing_columns(actual_columns, required_columns):
    return [column for column in required_columns if column not in set(actual_columns)]


def missing_keys(mapping, required_keys):
    return [key for key in required_keys if key not in mapping]


def align_columns(df, required_columns, default_value=""):
    """Return a copy with required columns present and ordered first."""
    aligned = df.copy()
    for column in required_columns:
        if column not in aligned.columns:
            aligned[column] = default_value
    extra_columns = [column for column in aligned.columns if column not in required_columns]
    return aligned[list(required_columns) + extra_columns]
