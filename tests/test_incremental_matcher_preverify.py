import os
import sys
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.common.match_cache import MatchCache  # noqa: E402
import pipeline.llm_events.incremental_matcher as matcher  # noqa: E402
from pipeline.llm_events.incremental_matcher import (  # noqa: E402
    deterministic_reject_reason,
    filter_clusters_by_min_quality,
    rank_clusters_for_llm,
    run_incremental_pipeline,
    split_preverified_clusters,
)


def cluster(
    kalshi_title,
    polymarket_title,
    common_keywords,
    kalshi_event_id="K1",
    polymarket_event_id="P1",
    kalshi_category="Sports",
    kalshi_sub_category="Esports",
    polymarket_category="Sports",
    polymarket_sub_category="Esports",
    candidate_reason=None,
):
    c = {
        "kalshi": {
            "event_id": kalshi_event_id,
            "title": kalshi_title,
            "category": kalshi_category,
            "sub_category": kalshi_sub_category,
        },
        "polymarket": {
            "event_id": polymarket_event_id,
            "title": polymarket_title,
            "category": polymarket_category,
            "sub_category": polymarket_sub_category,
        },
        "common_keywords": common_keywords,
        "match_score": len(common_keywords),
    }
    if candidate_reason is not None:
        c["candidate_reason"] = candidate_reason
    return c


def test_pre_rejects_map_or_match_vs_season_winner_scope():
    c = cluster(
        "LPL 2026",
        "LoL: LPL 2026 Season Winner",
        ["lpl"],
        kalshi_event_id="KXLOLMAP-26MAR03IGAL-1",
        polymarket_event_id="86364",
    )

    assert deterministic_reject_reason(c) == "pre_reject_scope_granularity"


def test_pre_rejects_broad_specific_single_generic_keyword():
    c = cluster(
        "Who will be arrested before 2027?",
        "Lee Jae-myung arrested before 2027?",
        ["arrested"],
        kalshi_category="Politics",
        kalshi_sub_category="SCOTUS & courts",
        polymarket_category="Politics",
        polymarket_sub_category="World, Politics",
    )

    assert deterministic_reject_reason(c) == "pre_reject_broad_specific_single_keyword"


def test_pre_rejects_broad_specific_even_when_year_matches():
    c = cluster(
        "Who will be arrested before 2027?",
        "Lee Jae-myung arrested before 2027?",
        ["arrested", "year:2027"],
        kalshi_category="Politics",
        kalshi_sub_category="SCOTUS & courts",
        polymarket_category="Politics",
        polymarket_sub_category="World, Politics",
    )

    assert deterministic_reject_reason(c) == "pre_reject_broad_specific_single_keyword"


def test_pre_rejects_sports_domain_conflict():
    c = cluster(
        "Great Britain vs Italy",
        "First Round Pool B: Italy vs. Great Britain",
        ["great britain", "italy"],
        kalshi_sub_category="Basketball",
        polymarket_sub_category="Games, baseball, WBC, World Baseball Classic",
    )

    assert deterministic_reject_reason(c) == "pre_reject_sports_domain_conflict"


def test_pre_rejects_direction_date_number_and_location_conflicts():
    assert deterministic_reject_reason(
        cluster(
            "How high will Trump's approval rating get before 2027?",
            "How low will Trump's approval rating go in 2026?",
            ["approval", "rating", "trump"],
            kalshi_category="Politics",
            kalshi_sub_category="N/A",
            polymarket_category="Politics",
            polymarket_sub_category="Trump",
        )
    ) == "pre_reject_direction_conflict"

    assert deterministic_reject_reason(
        cluster(
            "Ethereum price on Mar 6, 2026 at 5pm EST?",
            "Ethereum price on March 3?",
            ["ethereum", "price"],
            kalshi_category="Crypto",
            kalshi_sub_category="ETH, Hourly",
            polymarket_category="Crypto",
            polymarket_sub_category="Crypto Prices",
        )
    ) == "pre_reject_date_scope_conflict"

    assert deterministic_reject_reason(
        cluster(
            "Will there be an at least 8.0 magnitude earthquake in California before 2028?",
            "9.0 or above earthquake before 2027?",
            ["earthquake"],
            kalshi_category="Science",
            kalshi_sub_category="N/A",
            polymarket_category="Science",
            polymarket_sub_category="Earthquakes",
        )
    ) == "pre_reject_numeric_threshold_conflict"

    assert deterministic_reject_reason(
        cluster(
            "Will John Cornyn win the Texas Republican Senate primary?",
            "Iowa Republican Senate Primary Winner",
            ["primary", "republican", "senate"],
            kalshi_category="Politics",
            kalshi_sub_category="Primaries",
            polymarket_category="Politics",
            polymarket_sub_category="Primaries",
        )
    ) == "pre_reject_location_conflict"


def test_split_preverified_clusters_keeps_valid_exact_match():
    valid = cluster(
        "Which bills will become law in 2026?",
        "Which bills will become law in 2026?",
        ["bills", "law"],
        kalshi_category="Politics",
        kalshi_sub_category="Congress",
        polymarket_category="Politics",
        polymarket_sub_category="Trump, Politics",
    )
    invalid = cluster(
        "LPL 2026",
        "LoL: LPL 2026 Season Winner",
        ["lpl"],
        kalshi_event_id="KXLOLMAP-26MAR03IGAL-1",
    )

    llm_clusters, rejected = split_preverified_clusters([valid, invalid])

    assert llm_clusters == [valid]
    assert rejected[0][0] == invalid
    assert rejected[0][1] == "pre_reject_scope_granularity"


def test_match_cache_accepts_uppercase_and_lowercase_match_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "match_cache.jsonl")
        cache = MatchCache(cache_file=cache_path, confirmed_csv=None)

        cache.add_matches_from_list(
            [
                {"Kalshi_ID": "K1", "Polymarket_ID": "P1", "Reasoning": "upper"},
                {"kalshi_id": "K2", "polymarket_id": "P2", "Reasoning": "lower"},
                {"Kalshi_ID": "", "Polymarket_ID": "P3", "Reasoning": "bad"},
            ]
        )

        assert cache.get_active_pair_ids() == {("K1", "P1"), ("K2", "P2")}


def test_incremental_pipeline_dry_run_stops_before_llm():
    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_path = os.path.join(tmpdir, "candidate_clusters.json")
        original_candidate_path = matcher.CANDIDATE_CLUSTERS_JSON
        matcher.CANDIDATE_CLUSTERS_JSON = candidate_path
        try:
            with open(candidate_path, "w") as handle:
                import json

                json.dump(
                    [
                        cluster(
                            "Which bills will become law in 2026?",
                            "Which bills will become law in 2026?",
                            ["bills", "law"],
                            kalshi_category="Politics",
                            kalshi_sub_category="Congress",
                            polymarket_category="Politics",
                            polymarket_sub_category="Trump, Politics",
                        ),
                        cluster(
                            "LPL 2026",
                            "LoL: LPL 2026 Season Winner",
                            ["lpl"],
                            kalshi_event_id="KXLOLMAP-26MAR03IGAL-1",
                        ),
                    ],
                    handle,
                )

            result = run_incremental_pipeline(full_retry=True, dry_run=True, engine="legacy")

            assert result["dry_run"] is True
            assert result["clusters_processed"] == 1
            assert result["pre_rejected"] == 1
            assert result["new_event_matches"] == 0
        finally:
            matcher.CANDIDATE_CLUSTERS_JSON = original_candidate_path


def test_rank_clusters_prioritizes_stronger_exact_candidates():
    weak = cluster(
        "Will a company release a product?",
        "Will a company release a product?",
        ["company"],
        kalshi_event_id="KWEAK",
        polymarket_event_id="PWEAK",
    )
    strong = cluster(
        "Will SpaceX launch Starship Flight Test 12?",
        "SpaceX Starship Flight Test 12",
        ["spacex", "starship", "flight", "test", "12"],
        kalshi_event_id="KSTRONG",
        polymarket_event_id="PSTRONG",
    )

    ranked = rank_clusters_for_llm([weak, strong])

    assert ranked[0] == strong


def test_min_quality_filter_defers_weak_candidates():
    weak = cluster(
        "Will a generic thing happen?",
        "Another generic thing?",
        ["thing"],
        candidate_reason={"quality_tier": "weak", "quality_score": 120},
    )
    medium = cluster(
        "AL-02 House Election Winner",
        "Which party will win the House race for AL-02?",
        ["al-02", "house"],
        candidate_reason={"quality_tier": "medium", "quality_score": 310},
    )

    kept, deferred = filter_clusters_by_min_quality([weak, medium], min_quality="medium")

    assert kept == [medium]
    assert deferred == [weak]


def test_incremental_pipeline_dry_run_applies_llm_budget_after_ranking():
    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_path = os.path.join(tmpdir, "candidate_clusters.json")
        original_candidate_path = matcher.CANDIDATE_CLUSTERS_JSON
        matcher.CANDIDATE_CLUSTERS_JSON = candidate_path
        try:
            with open(candidate_path, "w") as handle:
                import json

                json.dump(
                    [
                        cluster("A weak candidate?", "A weak candidate?", ["weak"], "K1", "P1"),
                        cluster(
                            "Will SpaceX launch Starship Flight Test 12?",
                            "SpaceX Starship Flight Test 12",
                            ["spacex", "starship", "flight", "test", "12"],
                            "K2",
                            "P2",
                            candidate_reason={"quality_tier": "strong", "quality_score": 520},
                        ),
                        cluster("Another weak candidate?", "Another weak candidate?", ["weak"], "K3", "P3"),
                    ],
                    handle,
                )

            result = run_incremental_pipeline(full_retry=True, dry_run=True, max_llm_clusters=1, engine="legacy")

            assert result["dry_run"] is True
            assert result["clusters_processed"] == 1
            assert result["deferred_by_budget"] == 2
            assert result["priority_counts"]["high"] >= 1
        finally:
            matcher.CANDIDATE_CLUSTERS_JSON = original_candidate_path


def run() -> None:
    test_pre_rejects_map_or_match_vs_season_winner_scope()
    test_pre_rejects_broad_specific_single_generic_keyword()
    test_pre_rejects_broad_specific_even_when_year_matches()
    test_pre_rejects_sports_domain_conflict()
    test_pre_rejects_direction_date_number_and_location_conflicts()
    test_split_preverified_clusters_keeps_valid_exact_match()
    test_match_cache_accepts_uppercase_and_lowercase_match_keys()
    test_incremental_pipeline_dry_run_stops_before_llm()
    test_rank_clusters_prioritizes_stronger_exact_candidates()
    test_min_quality_filter_defers_weak_candidates()
    test_incremental_pipeline_dry_run_applies_llm_budget_after_ranking()


if __name__ == "__main__":
    run()
    print("ok")
