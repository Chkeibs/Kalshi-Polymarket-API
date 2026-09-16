import math
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.common.category_utils import (  # noqa: E402
    kalshi_category_key,
    mapped_polymarket_keys_for_kalshi,
    normalize_category_mapping,
    normalize_label,
    polymarket_category_keys,
    resolve_polymarket_keys_for_kalshi,
)


def test_normalize_missing_category_values():
    assert normalize_label(None) == "N/A"
    assert normalize_label(math.nan) == "N/A"
    assert normalize_label("nan") == "N/A"
    assert normalize_label("<NA>") == "N/A"
    assert normalize_label("") == "N/A"


def test_kalshi_category_key_normalizes_nan_to_na():
    assert kalshi_category_key("Politics", math.nan) == "politics / n/a"


def test_polymarket_category_keys_include_tags():
    keys = polymarket_category_keys("Sports", "Basketball, NCAA")

    assert {"sports", "basketball", "ncaa"}.issubset(keys)


def test_specific_mapping_prefers_subcategory_tags_over_broad_categories():
    mapping = normalize_category_mapping(
        {
            "Sports / Basketball": ["Sports", "Basketball", "NBA"],
            "Sports / N/A": ["Sports"],
        }
    )

    keys = mapped_polymarket_keys_for_kalshi("Sports / Basketball", mapping)

    assert "basketball" in keys
    assert "nba" in keys
    assert "sports" not in keys


def test_missing_specific_mapping_falls_back_to_subcategory_and_broad_category():
    mapping = normalize_category_mapping({"Sports / N/A": ["Sports"]})

    keys = mapped_polymarket_keys_for_kalshi("Sports / Rugby", mapping)

    assert {"rugby", "sports"}.issubset(keys)


def test_available_specific_mapping_does_not_expand_to_broad_category():
    mapping = normalize_category_mapping(
        {
            "Sports / Basketball": ["Sports", "Basketball", "NBA"],
            "Sports / N/A": ["Sports"],
        }
    )

    resolution = resolve_polymarket_keys_for_kalshi(
        "Sports / Basketball",
        mapping,
        {"sports", "basketball", "nba"},
    )

    assert resolution.mode == "exact_specific"
    assert resolution.keys == frozenset({"basketball", "nba"})


def test_na_mapping_stays_broad_even_when_a_broad_alias_expands_to_specific_terms():
    mapping = normalize_category_mapping(
        {"Crypto / N/A": ["Crypto", "Crypto Prices"]}
    )

    resolution = resolve_polymarket_keys_for_kalshi(
        "Crypto / N/A",
        mapping,
        {"crypto", "crypto prices", "bitcoin", "btc"},
    )

    assert resolution.mode == "exact_broad"
    assert resolution.keys == frozenset({"crypto", "crypto prices"})


def test_missing_specific_tag_uses_traceable_broad_fallback():
    mapping = normalize_category_mapping(
        {
            "Sports / Motorsport": ["Motorsport"],
            "Sports / N/A": ["Sports"],
        }
    )

    resolution = resolve_polymarket_keys_for_kalshi(
        "Sports / Motorsport",
        mapping,
        {"sports", "basketball"},
    )

    assert resolution.mode == "broad_fallback"
    assert resolution.keys == frozenset({"sports"})
    assert "motorsport" in resolution.requested_specific_keys


def test_domain_alias_can_resolve_to_an_available_specific_tag():
    mapping = normalize_category_mapping(
        {
            "Sports / Motorsport": ["Sports"],
            "Sports / N/A": ["Sports"],
        }
    )

    resolution = resolve_polymarket_keys_for_kalshi(
        "Sports / Motorsport",
        mapping,
        {"sports", "f1", "formula 1"},
    )

    assert resolution.mode == "exact_specific"
    assert resolution.keys == frozenset({"f1", "formula 1"})


def test_unmapped_subcategory_uses_available_specific_tag_only():
    mapping = normalize_category_mapping({"Sports / N/A": ["Sports"]})

    resolution = resolve_polymarket_keys_for_kalshi(
        "Sports / Rugby",
        mapping,
        {"sports", "rugby"},
    )

    assert resolution.mode == "inferred_specific"
    assert resolution.keys == frozenset({"rugby"})


def test_unmapped_subcategory_without_specific_tag_uses_broad_fallback():
    mapping = normalize_category_mapping({"Sports / N/A": ["Sports"]})

    resolution = resolve_polymarket_keys_for_kalshi(
        "Sports / Lacrosse",
        mapping,
        {"sports", "basketball"},
    )

    assert resolution.mode == "broad_fallback"
    assert resolution.keys == frozenset({"sports"})


def test_resolution_stays_unmapped_when_no_compatible_key_exists():
    mapping = normalize_category_mapping({"Sports / N/A": ["Sports"]})

    resolution = resolve_polymarket_keys_for_kalshi(
        "Sports / Lacrosse",
        mapping,
        {"politics", "world"},
    )

    assert resolution.mode == "unmapped"
    assert not resolution.keys


def test_empty_exact_mapping_uses_available_same_name_broad_category():
    mapping = normalize_category_mapping({"Health / N/A": []})

    resolution = resolve_polymarket_keys_for_kalshi(
        "Health / N/A",
        mapping,
        {"health", "politics"},
    )

    assert resolution.mode == "broad_fallback"
    assert resolution.keys == frozenset({"health"})


def test_data_aliases_resolve_available_specific_tags():
    mapping = normalize_category_mapping(
        {
            "Crypto / DOGE": ["Crypto"],
            "Science / Snow and rain": ["Science"],
        }
    )

    doge = resolve_polymarket_keys_for_kalshi(
        "Crypto / DOGE",
        mapping,
        {"crypto", "dogecoin"},
    )
    weather = resolve_polymarket_keys_for_kalshi(
        "Science / Snow and rain",
        mapping,
        {"science", "weather", "climate & weather"},
    )

    assert doge.mode == "exact_specific"
    assert doge.keys == frozenset({"dogecoin"})
    assert weather.mode == "exact_specific"
    assert weather.keys == frozenset({"weather", "climate & weather"})


def test_politics_primary_aliases_match_polymarket_tags():
    mapping = normalize_category_mapping({"Politics / Primaries": ["Politics", "Primaries"]})

    mapped = mapped_polymarket_keys_for_kalshi("Politics / Primaries", mapping)
    poly_keys = polymarket_category_keys(
        "Politics",
        "Primaries, primary elections, Governor Primary, Republican Primary",
    )

    assert mapped.intersection(poly_keys)


def test_house_aliases_match_congress_tags():
    mapping = normalize_category_mapping({"Politics / House": ["Politics"]})

    mapped = mapped_polymarket_keys_for_kalshi("Politics / House", mapping)
    poly_keys = polymarket_category_keys("Politics", "Congress, house of representatives")

    assert mapped.intersection(poly_keys)


def test_domain_aliases_cover_common_polymarket_tag_variants():
    mapping = normalize_category_mapping(
        {
            "Politics / International": ["Politics", "World"],
            "Politics / Foreign Elections": ["Politics", "World"],
            "Culture / Music": ["Culture", "Spotify"],
            "Crypto / BTC": ["Crypto", "Crypto Prices"],
            "Economics / Growth": ["Economics"],
            "Economics / Fed": ["Economics"],
        }
    )

    assert mapped_polymarket_keys_for_kalshi("Politics / International", mapping).intersection(
        polymarket_category_keys("Politics", "Foreign Policy, Middle East, Geopolitics")
    )
    assert mapped_polymarket_keys_for_kalshi("Politics / Foreign Elections", mapping).intersection(
        polymarket_category_keys("World", "Global Elections, World Elections")
    )
    assert mapped_polymarket_keys_for_kalshi("Culture / Music", mapping).intersection(
        polymarket_category_keys("Culture", "billboard, hot 100")
    )
    assert mapped_polymarket_keys_for_kalshi("Crypto / BTC", mapping).intersection(
        polymarket_category_keys("Crypto", "Bitcoin, Crypto Prices")
    )
    assert mapped_polymarket_keys_for_kalshi("Economics / Growth", mapping).intersection(
        polymarket_category_keys("Economics", "GDP, Macro Indicators")
    )
    assert mapped_polymarket_keys_for_kalshi("Economics / Fed", mapping).intersection(
        polymarket_category_keys("Economics", "Fed Rates, Economic Policy")
    )


def run() -> None:
    test_normalize_missing_category_values()
    test_kalshi_category_key_normalizes_nan_to_na()
    test_polymarket_category_keys_include_tags()
    test_specific_mapping_prefers_subcategory_tags_over_broad_categories()
    test_missing_specific_mapping_falls_back_to_subcategory_and_broad_category()
    test_available_specific_mapping_does_not_expand_to_broad_category()
    test_na_mapping_stays_broad_even_when_a_broad_alias_expands_to_specific_terms()
    test_missing_specific_tag_uses_traceable_broad_fallback()
    test_domain_alias_can_resolve_to_an_available_specific_tag()
    test_unmapped_subcategory_uses_available_specific_tag_only()
    test_unmapped_subcategory_without_specific_tag_uses_broad_fallback()
    test_resolution_stays_unmapped_when_no_compatible_key_exists()
    test_empty_exact_mapping_uses_available_same_name_broad_category()
    test_data_aliases_resolve_available_specific_tags()
    test_politics_primary_aliases_match_polymarket_tags()
    test_house_aliases_match_congress_tags()
    test_domain_aliases_cover_common_polymarket_tag_variants()


if __name__ == "__main__":
    run()
    print("ok")
