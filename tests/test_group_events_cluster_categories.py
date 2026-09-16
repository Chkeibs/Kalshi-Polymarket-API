import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.clustering.group_events_cluster import (  # noqa: E402
    build_exclusive_clusters,
    build_graph_data,
    build_pairs_to_compare,
    candidate_reason,
    check_incompatibility,
    hard_reject_reason,
    is_broad_specific_single_keyword,
)
from pipeline.common.category_utils import normalize_category_mapping  # noqa: E402


def pair_ids(pairs):
    return {(pair["kalshi_id"], pair["poly_id"]) for pair in pairs}


def test_build_pairs_uses_na_and_polymarket_subcategory_tags():
    index = {
        "KOSCARS": {
            "platform": "kalshi",
            "title": "Will Dune win Best Picture at the Oscars?",
            "category": "Culture",
            "sub_category": "N/A",
            "keywords": ["dune", "best", "picture", "oscars"],
        },
        "POSCARS": {
            "platform": "polymarket",
            "title": "Dune Best Picture Oscar?",
            "category": "Culture",
            "sub_category": "Oscars, Movies",
            "keywords": ["dune", "best", "picture", "oscars"],
        },
        "KPRIMARY": {
            "platform": "kalshi",
            "title": "Will John Cornyn win the Texas Senate primary?",
            "category": "Politics",
            "sub_category": "Primaries, US Elections",
            "keywords": ["john cornyn", "texas", "senate", "primary"],
        },
        "PPRIMARY": {
            "platform": "polymarket",
            "title": "Texas Senate Republican primary winner",
            "category": "Politics",
            "sub_category": "Elections, Primaries, Texas Senate",
            "keywords": ["john cornyn", "texas", "senate", "primary"],
        },
    }
    mapping = normalize_category_mapping(
        {
            "Culture / N/A": ["Culture"],
            "Politics / Primaries, US Elections": ["Politics", "Primaries"],
        }
    )

    pairs = build_pairs_to_compare(index, mapping)
    ids = pair_ids(pairs)

    assert ("KOSCARS", "POSCARS") in ids
    assert ("KPRIMARY", "PPRIMARY") in ids


def test_build_pairs_rejects_incompatible_dates_and_scopes():
    index = {
        "KQ4": {
            "platform": "kalshi",
            "title": "Will S&P 500 close above 7000 in Q4 2026?",
            "category": "Economics",
            "sub_category": "Stocks",
            "keywords": ["s&p 500", "close", "above", "7000"],
        },
        "PQ4": {
            "platform": "polymarket",
            "title": "S&P 500 above 7000 in Q4 2025?",
            "category": "Equities",
            "sub_category": "Stocks",
            "keywords": ["s&p 500", "close", "above", "7000"],
        },
        "KPRES": {
            "platform": "kalshi",
            "title": "Which party will win the 2028 presidential election?",
            "category": "Politics",
            "sub_category": "Elections",
            "keywords": ["democratic", "party", "election"],
        },
        "PHOUSE": {
            "platform": "polymarket",
            "title": "California 45th District House Election Winner",
            "category": "Politics",
            "sub_category": "Elections, US Elections",
            "keywords": ["democratic", "party", "election"],
        },
    }
    mapping = normalize_category_mapping(
        {
            "Economics / Stocks": ["Equities"],
            "Politics / Elections": ["Politics", "Elections"],
        }
    )

    pairs = build_pairs_to_compare(index, mapping)
    ids = pair_ids(pairs)

    assert ("KQ4", "PQ4") not in ids
    assert ("KPRES", "PHOUSE") not in ids


def test_person_conflict_normalizes_possessives_and_team_names():
    assert not check_incompatibility({"PERSON:harry styles's '"}, {"PERSON:harry styles '"})
    assert not check_incompatibility({"PERSON:hc barys"}, {"PERSON:barys astana"})
    assert check_incompatibility({"PERSON:trump"}, {"PERSON:biden"})


def test_strong_textual_fallback_recovers_category_tag_misses():
    index = {
        "KBILLS": {
            "platform": "kalshi",
            "title": "Which bills will become law in 2026?",
            "category": "Politics",
            "sub_category": "Congress",
            "keywords": ["bills", "law"],
        },
        "PBILLS": {
            "platform": "polymarket",
            "title": "Which bills will become law in 2026?",
            "category": "Politics",
            "sub_category": "Trump, Politics",
            "keywords": ["bills", "law"],
        },
        "KLBJ": {
            "platform": "kalshi",
            "title": "LeBron James announces retirement before the 2026-27 season?",
            "category": "Sports",
            "sub_category": "N/A",
            "keywords": ["announces", "lebron james", "retirement", "season"],
        },
        "PLBJ": {
            "platform": "polymarket",
            "title": "Will LeBron James retire before next NBA season?",
            "category": "Sports",
            "sub_category": "Sports",
            "keywords": ["lebron james", "nba", "retire", "season"],
        },
    }
    mapping = normalize_category_mapping(
        {
            "Politics / Congress": ["Politics"],
            "Sports / N/A": ["Sports"],
        }
    )

    ids = pair_ids(build_pairs_to_compare(index, mapping))

    assert ("KBILLS", "PBILLS") in ids
    assert ("KLBJ", "PLBJ") in ids


def test_strong_textual_fallback_respects_sports_domain_conflicts():
    index = {
        "KFIBA": {
            "platform": "kalshi",
            "title": "Great Britain vs Italy",
            "category": "Sports",
            "sub_category": "Basketball",
            "keywords": ["great britain", "italy"],
        },
        "PWBC": {
            "platform": "polymarket",
            "title": "First Round Pool B: Italy vs. Great Britain",
            "category": "Sports",
            "sub_category": "Games, baseball, WBC, World Baseball Classic",
            "keywords": ["first round pool", "great britain", "italy"],
        },
    }
    mapping = normalize_category_mapping({"Sports / Basketball": ["Basketball"]})

    ids = pair_ids(build_pairs_to_compare(index, mapping))

    assert ("KFIBA", "PWBC") not in ids


def test_specific_category_resolution_is_recorded_without_broad_expansion():
    index = {
        "KRUGBY": {
            "platform": "kalshi",
            "title": "Will France win the Rugby World Cup?",
            "category": "Sports",
            "sub_category": "Rugby",
            "keywords": ["france", "rugby", "world cup"],
        },
        "PRUGBY": {
            "platform": "polymarket",
            "title": "France to win the Rugby World Cup?",
            "category": "Sports",
            "sub_category": "Games, Rugby, World Cup",
            "keywords": ["france", "rugby", "world cup"],
        },
        "PSOCCER": {
            "platform": "polymarket",
            "title": "France to win the FIFA World Cup?",
            "category": "Sports",
            "sub_category": "Games, Soccer, FIFA World Cup",
            "keywords": ["france", "world cup"],
        },
    }
    mapping = normalize_category_mapping({"Sports / N/A": ["Sports"]})

    pairs = build_pairs_to_compare(index, mapping)
    by_pair = {(pair["kalshi_id"], pair["poly_id"]): pair for pair in pairs}

    assert ("KRUGBY", "PRUGBY") in by_pair
    assert by_pair[("KRUGBY", "PRUGBY")]["candidate_reason"]["category_resolution_mode"] == "inferred_specific"
    assert by_pair[("KRUGBY", "PRUGBY")]["candidate_reason"]["category_match_keys"] == ["rugby"]
    assert ("KRUGBY", "PSOCCER") not in by_pair


def test_broad_category_fallback_requires_multiple_supported_keywords():
    index = {
        "KMOTOR": {
            "platform": "kalshi",
            "title": "Will Lando Norris win the 2026 drivers championship?",
            "category": "Sports",
            "sub_category": "Motorsport",
            "keywords": ["lando norris", "drivers championship", "year:2026"],
        },
        "PMOTOR": {
            "platform": "polymarket",
            "title": "Lando Norris 2026 drivers championship winner?",
            "category": "Sports",
            "sub_category": "Games",
            "keywords": ["lando norris", "drivers championship", "year:2026"],
        },
        "PNOISE": {
            "platform": "polymarket",
            "title": "Will Lando Norris attend an awards ceremony?",
            "category": "Sports",
            "sub_category": "Games",
            "keywords": ["lando norris"],
        },
    }
    mapping = normalize_category_mapping(
        {
            "Sports / Motorsport": ["Sports"],
            "Sports / N/A": ["Sports"],
        }
    )

    pairs = build_pairs_to_compare(index, mapping)
    by_pair = {(pair["kalshi_id"], pair["poly_id"]): pair for pair in pairs}

    assert ("KMOTOR", "PMOTOR") in by_pair
    assert by_pair[("KMOTOR", "PMOTOR")]["candidate_reason"]["path"] == "broad_category_keywords"
    assert by_pair[("KMOTOR", "PMOTOR")]["candidate_reason"]["category_resolution_mode"] == "broad_fallback"
    assert ("KMOTOR", "PNOISE") not in by_pair


def test_build_pairs_rejects_weak_single_keyword_noise():
    index = {
        "KTRUMP": {
            "platform": "kalshi",
            "title": "Will Trump resign during his term?",
            "category": "Politics",
            "sub_category": "N/A",
            "keywords": ["trump"],
        },
        "PTRUMP": {
            "platform": "polymarket",
            "title": "Trump out as President before 2027?",
            "category": "Politics",
            "sub_category": "Trump",
            "keywords": ["trump"],
        },
    }
    mapping = normalize_category_mapping({"Politics / N/A": ["Politics"]})

    ids = pair_ids(build_pairs_to_compare(index, mapping))

    assert ("KTRUMP", "PTRUMP") not in ids


def test_broad_specific_detection_keeps_broad_to_broad_events():
    assert is_broad_specific_single_keyword(
        "Who will be arrested before 2027?",
        "Lee Jae-myung arrested before 2027?",
        {"arrested", "year:2027"},
    )
    assert not is_broad_specific_single_keyword(
        "Who will release a new album in 2026?",
        "Which artists will release new albums in 2026?",
        {"album", "year:2026"},
    )


def test_plural_aliases_and_year_do_not_count_as_three_independent_proofs():
    index = {
        "KTEXAS": {
            "platform": "kalshi",
            "title": "Will Democrats win any statewide election in Texas in 2026?",
            "category": "Politics",
            "sub_category": "N/A",
            "keywords": ["democrat", "democrats", "texas", "year:2026"],
        },
        "PHOUSE": {
            "platform": "polymarket",
            "title": "2026 U.S. House election: Republicans flip the Democrats?",
            "category": "Politics",
            "sub_category": "Midterms, Elections, US Election",
            "keywords": ["democrat", "democrats", "house", "republican", "year:2026"],
        },
    }
    mapping = normalize_category_mapping({"Politics / N/A": ["Politics"]})

    ids = pair_ids(build_pairs_to_compare(index, mapping))

    assert ("KTEXAS", "PHOUSE") not in ids


def test_close_titles_recover_valid_pairs_after_keyword_canonicalization():
    index = {
        "KRETIRE": {
            "platform": "kalshi",
            "title": "LeBron James announces retirement before the 2026-27 season?",
            "category": "Sports",
            "sub_category": "N/A",
            "keywords": ["jame", "james", "lebron", "retirement", "season", "year:2026"],
        },
        "PRETIRE": {
            "platform": "polymarket",
            "title": "Will LeBron James retire before next NBA season?",
            "category": "Sports",
            "sub_category": "Sports",
            "keywords": ["jame", "james", "lebron", "nba", "retire", "season"],
        },
        "KPARTY": {
            "platform": "kalshi",
            "title": "2028 Presidential Election winner? (Party)",
            "category": "Politics",
            "sub_category": "N/A",
            "keywords": ["party", "presidential", "year:2028"],
        },
        "PPARTY": {
            "platform": "polymarket",
            "title": "Presidential Election Winner 2028",
            "category": "Politics",
            "sub_category": "Elections, Politics",
            "keywords": ["presidential", "year:2028"],
        },
    }
    mapping = normalize_category_mapping(
        {
            "Sports / N/A": ["Sports"],
            "Politics / N/A": ["Politics"],
        }
    )

    ids = pair_ids(build_pairs_to_compare(index, mapping))

    assert ("KRETIRE", "PRETIRE") in ids
    assert ("KPARTY", "PPARTY") in ids


def test_two_specific_shared_terms_recover_a_cross_category_tag_error():
    index = {
        "KSTARSHIP": {
            "platform": "kalshi",
            "title": "SpaceX Starship 12th launch?",
            "category": "Economics",
            "sub_category": "N/A",
            "keywords": ["12th", "spacex", "starship"],
        },
        "PSTARSHIP": {
            "platform": "polymarket",
            "title": "SpaceX Starship Flight Test 12",
            "category": "Science",
            "sub_category": "Science, Big Tech, Elon Musk",
            "keywords": ["flight", "spacex", "starship", "test"],
        },
    }
    mapping = normalize_category_mapping({"Economics / N/A": ["Economics"]})

    ids = pair_ids(build_pairs_to_compare(index, mapping))

    assert ("KSTARSHIP", "PSTARSHIP") in ids


def test_hard_rejects_direction_dates_numbers_and_locations():
    assert hard_reject_reason(
        "KAPPROVAL",
        "PAPPROVAL",
        {"title": "How high will Trump's approval rating get before 2027?", "category": "Politics", "sub_category": "N/A"},
        {"title": "How low will Trump's approval rating go in 2026?", "category": "Politics", "sub_category": "Trump"},
        {},
        {},
    ) == "direction_conflict"

    assert hard_reject_reason(
        "KETH",
        "PETH",
        {"title": "Ethereum price on Mar 6, 2026 at 5pm EST?", "category": "Crypto", "sub_category": "ETH, Hourly"},
        {"title": "Ethereum price on March 3?", "category": "Crypto", "sub_category": "Crypto Prices"},
        {},
        {},
    ) == "date_scope_conflict"

    assert not hard_reject_reason(
        "KNYC",
        "PNYC",
        {"title": "Highest temperature in NYC on Mar 2, 2026?", "category": "Science", "sub_category": "Daily temperature"},
        {"title": "Highest temperature in NYC on March 2?", "category": "Science", "sub_category": "Weather"},
        {},
        {},
    )

    assert hard_reject_reason(
        "KQUAKE",
        "PQUAKE",
        {"title": "Will there be an at least 8.0 magnitude earthquake in California before 2028?", "category": "Science", "sub_category": "N/A"},
        {"title": "9.0 or above earthquake before 2027?", "category": "Science", "sub_category": "Earthquakes"},
        {},
        {},
    ) == "numeric_threshold_conflict"

    assert hard_reject_reason(
        "KTEXAS",
        "PIOWA",
        {"title": "Texas Senate Republican primary margin of victory?", "category": "Politics", "sub_category": "Primaries"},
        {"title": "Iowa Republican Senate Primary Winner", "category": "Politics", "sub_category": "Primaries"},
        {},
        {},
    ) == "title_scope_conflict"

    assert hard_reject_reason(
        "KREL",
        "PREL",
        {"title": "Will Israel and Syria normalize relations during Trump's term?", "category": "Politics", "sub_category": "International"},
        {"title": "Israel and Saudi Arabia normalize relations by March 31?", "category": "Politics", "sub_category": "Foreign Policy"},
        {},
        {},
    ) == "location_conflict"

    assert hard_reject_reason(
        "KWV02",
        "PWV01",
        {"title": "Which party will win the House race for WV-02?", "category": "Politics", "sub_category": "Elections"},
        {"title": "WV-01 House Election Winner", "category": "Politics", "sub_category": "Elections"},
        {},
        {},
    ) == "location_conflict"

    assert hard_reject_reason(
        "KVERMONT",
        "PAKAL",
        {"title": "Which party will win the House race for Vermont?", "category": "Politics", "sub_category": "Elections"},
        {"title": "AK-AL House Election Winner", "category": "Politics", "sub_category": "Elections"},
        {},
        {},
    ) == "location_conflict"


def test_hard_rejects_explicitly_incompatible_market_dimensions():
    assert hard_reject_reason(
        "KALBUM",
        "PARTIST",
        {"title": "#2 Album on Spotify in 2026?", "category": "Culture", "sub_category": "Music"},
        {"title": "#2 Spotify artist in March?", "category": "Culture", "sub_category": "Music"},
        {},
        {},
    ) == "title_scope_conflict"

    assert hard_reject_reason(
        "KDRIVER",
        "PCONSTRUCTOR",
        {"title": "F1 Drivers Champion", "category": "Sports", "sub_category": "Motorsport"},
        {"title": "F1 Constructors Champion", "category": "Sports", "sub_category": "F1"},
        {},
        {},
    ) == "title_scope_conflict"

    assert hard_reject_reason(
        "KWINNER",
        "PRELEGATION",
        {"title": "Ligue 1 Winner", "category": "Sports", "sub_category": "Soccer"},
        {"title": "Ligue 1 - Which Clubs Get Relegated?", "category": "Sports", "sub_category": "Soccer"},
        {},
        {},
    ) == "title_scope_conflict"

    assert not hard_reject_reason(
        "KTOP",
        "PTOP",
        {"title": "Premier League Top 4 Finishers", "category": "Sports", "sub_category": "Soccer"},
        {"title": "Premier League - Top Four Finish", "category": "Sports", "sub_category": "Soccer"},
        {},
        {},
    )


def test_hard_rejects_match_or_game_scope_against_season_scope():
    assert hard_reject_reason(
        "KXLOLMAP-26MAR03IGAL-1",
        "86364",
        {"title": "LPL 2026", "category": "Sports", "sub_category": "Esports"},
        {"title": "LoL: LPL 2026 Season Winner", "category": "League Of Legends", "sub_category": "Esports"},
        {},
        {},
    ) == "scope_granularity_conflict"

    assert hard_reject_reason(
        "KXCOVEREA-27",
        "72208",
        {"title": "Who will be the cover athlete for the next EA Sports college basketball video game?", "category": "Sports", "sub_category": "N/A"},
        {"title": "SEC Men's College Basketball Regular Season Champion", "category": "Sports", "sub_category": "Basketball"},
        {},
        {},
    ) == "scope_granularity_conflict"

    assert not hard_reject_reason(
        "KSTARSHIP",
        "PSTARSHIP",
        {"title": "SpaceX Starship 12th launch?", "category": "Economics", "sub_category": "N/A"},
        {"title": "SpaceX Starship Flight Test 12", "category": "Science", "sub_category": "Science"},
        {},
        {},
    )


def test_build_pairs_keeps_shared_outright_scope_and_plural_aliases():
    index = {
        "KPRIMARY": {
            "platform": "kalshi",
            "title": "Will a candidate win outright in the Texas Republican Senate primary?",
            "category": "Politics",
            "sub_category": "Primaries, US Elections",
            "keywords": ["candidate", "outright", "primary", "republican", "senate", "texas"],
        },
        "PPRIMARY": {
            "platform": "polymarket",
            "title": "Will any candidate win the Texas Republican Senate Primary outright?",
            "category": "Politics",
            "sub_category": "US Election, Elections, Primaries",
            "keywords": ["candidate", "outright", "primary", "republican", "senate", "texas"],
        },
        "KOSCAR": {
            "platform": "kalshi",
            "title": "Oscar for Best Director?",
            "category": "Culture",
            "sub_category": "Awards, Movies",
            "keywords": ["oscar", "director"],
        },
        "POSCAR": {
            "platform": "polymarket",
            "title": "Oscars 2026: Best Director Winner",
            "category": "Culture",
            "sub_category": "Oscars, Movies",
            "keywords": ["oscars", "oscar", "director", "year:2026"],
        },
    }
    mapping = normalize_category_mapping(
        {
            "Politics / Primaries, US Elections": ["Politics", "Primaries"],
            "Culture / Awards, Movies": ["Culture", "Oscars", "Movies"],
        }
    )

    ids = pair_ids(build_pairs_to_compare(index, mapping))

    assert ("KPRIMARY", "PPRIMARY") in ids
    assert ("KOSCAR", "POSCAR") in ids


def test_candidate_pairs_include_decision_trace():
    index = {
        "KOSCARS": {
            "platform": "kalshi",
            "title": "Will Dune win Best Picture at the Oscars?",
            "category": "Culture",
            "sub_category": "N/A",
            "keywords": ["dune", "best", "picture", "oscars"],
        },
        "POSCARS": {
            "platform": "polymarket",
            "title": "Dune Best Picture Oscar?",
            "category": "Culture",
            "sub_category": "Oscars, Movies",
            "keywords": ["dune", "best", "picture", "oscars"],
        },
    }
    mapping = normalize_category_mapping({"Culture / N/A": ["Culture"]})

    pairs = build_pairs_to_compare(index, mapping)

    assert pairs[0]["candidate_reason"]["path"] == "category_keywords"
    assert pairs[0]["candidate_reason"]["category_match"] is True
    assert pairs[0]["candidate_reason"]["quality_tier"] in {"strong", "exact"}
    assert pairs[0]["candidate_reason"]["quality_score"] > 0
    assert "dune" in pairs[0]["candidate_reason"]["evidence_keywords"]


def test_candidate_quality_marks_generic_single_keyword_as_weak():
    reason = candidate_reason(
        True,
        True,
        False,
        {"house"},
        {"title": "Which party will win the next Australian House election?"},
        {"title": "CA-45 House Election Winner"},
    )

    assert reason["quality_tier"] == "weak"


def test_candidate_quality_promotes_supported_single_keyword_matches():
    reason = candidate_reason(
        True,
        True,
        False,
        {"bitcoin"},
        {"title": "Bitcoin price on Mar 6, 2026 at 5pm EST?", "category": "Crypto", "sub_category": "BTC, Hourly"},
        {"title": "Bitcoin price on March 6?", "category": "Crypto", "sub_category": "Bitcoin, Crypto Prices"},
    )

    assert reason["quality_tier"] == "medium"


def test_graph_data_uses_final_candidate_pairs():
    pairs = [
        {"kalshi_id": "K1", "poly_id": "P1", "score": 3},
        {"kalshi_id": "K2", "poly_id": "P2", "score": 1},
    ]
    details = {
        "K1": {"event_id": "K1", "platform": "kalshi"},
        "P1": {"event_id": "P1", "platform": "polymarket"},
        "K2": {"event_id": "K2", "platform": "kalshi"},
        "P2": {"event_id": "P2", "platform": "polymarket"},
    }

    graph = build_graph_data(pairs, details)

    assert set(graph["nodes"]) == {"K1", "P1", "K2", "P2"}
    assert graph["edges"] == [
        {"source": "K1", "target": "P1", "weight": 3},
        {"source": "K2", "target": "P2", "weight": 1},
    ]


def test_exclusive_clusters_are_deterministic_and_quality_first():
    low_quality = {
        "kalshi_id": "K1",
        "poly_id": "P1",
        "score": 4,
        "common_keywords": ["a", "b", "c", "d"],
        "candidate_reason": {"quality_score": 250, "title_min_overlap": 0.5, "title_jaccard": 0.3},
    }
    high_quality = {
        "kalshi_id": "K1",
        "poly_id": "P2",
        "score": 3,
        "common_keywords": ["a", "b", "c"],
        "candidate_reason": {"quality_score": 500, "title_min_overlap": 0.9, "title_jaccard": 0.8},
    }
    details = {
        "K1": {"title": "Kalshi", "category": "Sports", "sub_category": "Rugby", "keywords": []},
        "P1": {"title": "Poly one", "category": "Sports", "sub_category": "Rugby", "keywords": []},
        "P2": {"title": "Poly two", "category": "Sports", "sub_category": "Rugby", "keywords": []},
    }

    forward, _, _ = build_exclusive_clusters([low_quality, high_quality], details)
    reverse, _, _ = build_exclusive_clusters([high_quality, low_quality], details)

    assert forward == reverse
    assert forward[0]["polymarket"]["event_id"] == "P2"


def run() -> None:
    test_build_pairs_uses_na_and_polymarket_subcategory_tags()
    test_build_pairs_rejects_incompatible_dates_and_scopes()
    test_person_conflict_normalizes_possessives_and_team_names()
    test_strong_textual_fallback_recovers_category_tag_misses()
    test_strong_textual_fallback_respects_sports_domain_conflicts()
    test_specific_category_resolution_is_recorded_without_broad_expansion()
    test_broad_category_fallback_requires_multiple_supported_keywords()
    test_build_pairs_rejects_weak_single_keyword_noise()
    test_broad_specific_detection_keeps_broad_to_broad_events()
    test_plural_aliases_and_year_do_not_count_as_three_independent_proofs()
    test_close_titles_recover_valid_pairs_after_keyword_canonicalization()
    test_two_specific_shared_terms_recover_a_cross_category_tag_error()
    test_hard_rejects_direction_dates_numbers_and_locations()
    test_hard_rejects_explicitly_incompatible_market_dimensions()
    test_hard_rejects_match_or_game_scope_against_season_scope()
    test_build_pairs_keeps_shared_outright_scope_and_plural_aliases()
    test_candidate_pairs_include_decision_trace()
    test_candidate_quality_marks_generic_single_keyword_as_weak()
    test_candidate_quality_promotes_supported_single_keyword_matches()
    test_graph_data_uses_final_candidate_pairs()
    test_exclusive_clusters_are_deterministic_and_quality_first()


if __name__ == "__main__":
    run()
    print("ok")
