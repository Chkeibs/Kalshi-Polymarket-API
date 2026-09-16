import csv
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.clustering.clustering_features import connected_component_stats
from pipeline.clustering.group_events_cluster import (
    build_pairs_to_compare,
    load_category_mapping,
    load_existing_signatures,
    optimize_candidate_pairs,
)


def load_inputs():
    with open(os.path.join(ROOT_DIR, "data", "keywords", "event_keywords_index.json")) as handle:
        index = json.load(handle)
    kalshi_signatures, polymarket_signatures, _, _ = load_existing_signatures(force_full=False)
    pairs = build_pairs_to_compare(
        index,
        load_category_mapping(),
        kalshi_signatures,
        polymarket_signatures,
    )
    return index, pairs


def load_confirmed():
    path = os.path.join(ROOT_DIR, "data", "matches", "confirmed_matches.csv")
    with open(path, newline="") as handle:
        return {
            (row["Kalshi_ID"], row["Polymarket_ID"])
            for row in csv.DictReader(handle)
        }


def load_silver_event_matches():
    path = os.path.join(ROOT_DIR, "data", "evaluation", "silver_consensus_v1.jsonl")
    matches = set()
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("candidate_type") != "event_event" or row.get("expected_decision") != "MATCH":
                continue
            candidate = row["candidate"]
            matches.add((
                str(candidate["kalshi"]["event_id"]),
                str(candidate["polymarket"]["event_id"]),
            ))
    return matches


def test_v2_reduces_graph_without_losing_regression_matches():
    index, legacy_pairs = load_inputs()
    optimized, _, report = optimize_candidate_pairs(legacy_pairs, index)
    optimized_keys = {(pair["kalshi_id"], pair["poly_id"]) for pair in optimized}

    confirmed = load_confirmed()
    silver_matches = load_silver_event_matches()
    assert confirmed.issubset(optimized_keys)
    assert silver_matches.issubset(optimized_keys)
    assert report["reduction_ratio"] >= 0.35
    assert report["output_pairs"] < report["input_pairs"]

    before_largest = connected_component_stats(legacy_pairs)[0]["edges"]
    after_largest = connected_component_stats(optimized)[0]["edges"]
    assert after_largest < before_largest


def run():
    test_v2_reduces_graph_without_losing_regression_matches()
    print("ok")


if __name__ == "__main__":
    run()
