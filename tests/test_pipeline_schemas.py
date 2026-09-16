import json
import os
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.common.schemas import (
    AMBIGUOUS_MATCH_REQUIRED_COLUMNS,
    CANDIDATE_CLUSTER_REQUIRED_KEYS,
    CANDIDATE_OUTCOME_KALSHI_REQUIRED_KEYS,
    CANDIDATE_OUTCOME_PAIR_REQUIRED_KEYS,
    CANDIDATE_OUTCOME_POLYMARKET_REQUIRED_KEYS,
    CANDIDATE_KALSHI_REQUIRED_KEYS,
    CANDIDATE_POLYMARKET_REQUIRED_KEYS,
    CONFIRMED_MATCH_REQUIRED_COLUMNS,
    KALSHI_MARKET_REQUIRED_COLUMNS,
    LLM_DECISION_REQUIRED_KEYS,
    POLYMARKET_MARKET_REQUIRED_COLUMNS,
    missing_columns,
    missing_keys,
)
from pipeline.llm_events.decision_schema import (  # noqa: E402
    Decision,
    DecisionRecord,
    ReasonCode,
    RelationType,
    VerificationStage,
)


def assert_csv_schema(path, required_columns):
    df = pd.read_csv(os.path.join(ROOT_DIR, path), nrows=0)
    missing = missing_columns(df.columns, required_columns)
    assert not missing, f"{path} missing columns: {missing}"


def assert_candidate_cluster_schema(path):
    with open(os.path.join(ROOT_DIR, path), "r") as handle:
        clusters = json.load(handle)

    assert isinstance(clusters, list), f"{path} must contain a list"
    if not clusters:
        return

    for idx, cluster in enumerate(clusters[:100]):
        missing = missing_keys(cluster, CANDIDATE_CLUSTER_REQUIRED_KEYS)
        assert not missing, f"{path}[{idx}] missing keys: {missing}"

        for side, required in (
            ("kalshi", CANDIDATE_KALSHI_REQUIRED_KEYS),
            ("polymarket", CANDIDATE_POLYMARKET_REQUIRED_KEYS),
        ):
            assert isinstance(cluster[side], dict), f"{path}[{idx}].{side} must be an object"
            side_missing = missing_keys(cluster[side], required)
            assert not side_missing, f"{path}[{idx}].{side} missing keys: {side_missing}"
            assert isinstance(cluster[side]["keywords"], list), f"{path}[{idx}].{side}.keywords must be a list"

        assert isinstance(cluster["common_keywords"], list), f"{path}[{idx}].common_keywords must be a list"


def assert_candidate_outcome_schema(path):
    full_path = os.path.join(ROOT_DIR, path)
    if not os.path.exists(full_path):
        return

    with open(full_path, "r") as handle:
        candidates = json.load(handle)

    assert isinstance(candidates, list), f"{path} must contain a list"
    if not candidates:
        return

    for idx, candidate in enumerate(candidates[:100]):
        missing = missing_keys(candidate, CANDIDATE_OUTCOME_PAIR_REQUIRED_KEYS)
        assert not missing, f"{path}[{idx}] missing keys: {missing}"

        for side, required in (
            ("kalshi", CANDIDATE_OUTCOME_KALSHI_REQUIRED_KEYS),
            ("polymarket", CANDIDATE_OUTCOME_POLYMARKET_REQUIRED_KEYS),
        ):
            assert isinstance(candidate[side], dict), f"{path}[{idx}].{side} must be an object"
            side_missing = missing_keys(candidate[side], required)
            assert not side_missing, f"{path}[{idx}].{side} missing keys: {side_missing}"

        assert isinstance(candidate["shared_market_keywords"], list), f"{path}[{idx}].shared_market_keywords must be a list"
        assert isinstance(candidate["shared_outcome_keywords"], list), f"{path}[{idx}].shared_outcome_keywords must be a list"


def run() -> None:
    assert_csv_schema("data/markets/markets_clean_kalshi.csv", KALSHI_MARKET_REQUIRED_COLUMNS)
    assert_csv_schema("data/markets/markets_clean_polymarket.csv", POLYMARKET_MARKET_REQUIRED_COLUMNS)
    assert_csv_schema("data/matches/confirmed_matches.csv", CONFIRMED_MATCH_REQUIRED_COLUMNS)
    assert_candidate_cluster_schema("data/clusters/candidate_clusters.json")
    assert_candidate_outcome_schema("data/clusters/candidate_outcome_pairs.json")
    sample_decision = DecisionRecord(
        pair_id="pair",
        content_hash="hash",
        decision=Decision.AMBIGUOUS,
        relation_type=RelationType.NONE,
        reason_code=ReasonCode.MISSING_CRITICAL_INFO,
        stage=VerificationStage.CHEAP,
    ).to_dict()
    assert not missing_keys(sample_decision, LLM_DECISION_REQUIRED_KEYS)
    assert len(AMBIGUOUS_MATCH_REQUIRED_COLUMNS) == 8


if __name__ == "__main__":
    run()
    print("ok")
