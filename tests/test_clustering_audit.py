import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.clustering.clustering_audit import build_report


def test_audit_is_deterministic_and_tracks_confirmed_coverage():
    index = {
        "K": {
            "platform": "kalshi",
            "title": "World Cup Group A Winner",
            "category": "Sports",
            "sub_category": "Soccer",
            "keywords": ["world", "cup", "group:A"],
        },
        "P": {
            "platform": "polymarket",
            "title": "FIFA World Cup Group A Winner",
            "category": "Sports",
            "sub_category": "Soccer",
            "keywords": ["world", "cup", "group:A"],
        },
    }
    candidates = [{
        "kalshi_id": "K",
        "poly_id": "P",
        "common_keywords": ["world", "cup", "group:A"],
        "candidate_reason": {
            "quality_tier": "strong",
            "quality_score": 500,
            "title_min_overlap": 0.8,
            "title_jaccard": 0.6,
        },
    }]

    first, first_samples = build_report(candidates, index, {("K", "P")})
    second, second_samples = build_report(candidates, index, {("K", "P")})

    assert first == second
    assert first_samples == second_samples
    assert first["candidate_count"] == 1
    assert first["confirmed_covered"] == 1
    assert first["confirmed_missing"] == []


def run():
    test_audit_is_deterministic_and_tracks_confirmed_coverage()
    print("ok")


if __name__ == "__main__":
    run()
