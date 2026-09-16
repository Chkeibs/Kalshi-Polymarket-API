"""
Shared category normalization for Kalshi/Polymarket clustering.

The clustering code compares Kalshi category/sub-category keys against
Polymarket categories and tags. Keeping this normalization in one place avoids
silent misses from values like NaN, "", "<NA>", or different casing.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass


MISSING_CATEGORY = "N/A"

SUPER_BROAD_POLYMARKET_CATEGORIES = {
    "approval",
    "climate & science",
    "crypto",
    "crypto prices",
    "culture",
    "economics",
    "equities",
    "esports",
    "health",
    "mentions",
    "other",
    "politics",
    "science",
    "social",
    "sports",
    "up or down",
    "world",
    "world affairs",
}

CATEGORY_TERM_ALIASES = {
    "basketball": {"nba", "ncaa basketball", "ncaa cbb", "cbb"},
    "baseball": {"mlb", "wbc", "world baseball classic"},
    "big tech": {"tech", "business"},
    "bitcoin": {"btc", "crypto prices", "hit price"},
    "btc": {"bitcoin", "crypto prices", "hit price"},
    "business": {"economics", "economy"},
    "charts": {"music charts", "billboard", "hot 100"},
    "congress": {"house", "house elections", "house of representatives"},
    "crypto prices": {"crypto", "bitcoin", "btc", "hit price"},
    "doge": {"dogecoin"},
    "dogecoin": {"doge"},
    "economic policy": {"economics", "economy"},
    "economics": {"economy", "economic policy", "business", "macro indicators"},
    "economy": {"economics", "economic policy", "business"},
    "election": {"elections"},
    "elections": {"election"},
    "fed": {"fed rates", "economic policy", "economics", "economy"},
    "fed rates": {"fed", "economic policy", "economics", "economy"},
    "foreign elections": {"global elections", "world elections", "elections"},
    "foreign policy": {"geopolitics", "international", "world", "world affairs"},
    "football": {"nfl", "ncaa football", "cfb"},
    "f1": {"formula 1", "motorsport"},
    "formula 1": {"f1", "motorsport"},
    "gdp": {"growth", "macro indicators", "economics"},
    "geopolitics": {"foreign policy", "international", "world", "world affairs"},
    "global elections": {"foreign elections", "world elections", "elections"},
    "governor": {"governorship", "gubernatorial"},
    "governor primary": {"governor", "primaries", "primary elections"},
    "growth": {"gdp", "macro indicators", "economics"},
    "hot 100": {"billboard", "charts", "music charts"},
    "house": {"congress", "house elections", "house of representatives"},
    "house elections": {"house", "congress", "elections"},
    "house of representatives": {"house", "congress"},
    "international": {"foreign policy", "geopolitics", "world", "world affairs", "global elections", "world elections"},
    "macro indicators": {"economics", "gdp", "growth"},
    "middle east": {"foreign policy", "geopolitics", "international", "world"},
    "mlb": {"baseball", "wbc", "world baseball classic"},
    "midterms": {"elections", "us election", "us elections", "nov 4 elections"},
    "music": {"music charts", "charts", "billboard", "hot 100", "celebrities"},
    "music charts": {"music", "charts", "billboard", "hot 100"},
    "motorsport": {"f1", "formula 1", "nascar", "indycar"},
    "nasdaq": {"stocks", "equities", "nasdaq 100"},
    "nba": {"basketball"},
    "ncaa basketball": {"basketball", "ncaa cbb", "cbb"},
    "nfl": {"football", "ncaa football", "cfb"},
    "nhl": {"hockey"},
    "nov 4 elections": {"elections", "us election", "us elections", "midterms"},
    "primary": {"primaries", "primary elections"},
    "primary elections": {"primary", "primaries"},
    "primaries": {"primary", "primary elections"},
    "republican primary": {"republican party", "primaries", "primary elections"},
    "democratic primary": {"democratic party", "primaries", "primary elections"},
    "senate": {"senator"},
    "senate primary": {"senate", "primaries", "primary elections"},
    "soccer": {"world cup"},
    "snow and rain": {"weather", "climate & weather"},
    "sol": {"solana"},
    "solana": {"sol"},
    "tech": {"big tech", "business"},
    "us election": {"us elections", "elections", "midterms", "nov 4 elections"},
    "us elections": {"us election", "elections", "midterms", "nov 4 elections"},
    "world elections": {"foreign elections", "global elections", "elections"},
    "ufc": {"mma"},
    "wbc": {"baseball", "world baseball classic"},
    "world baseball classic": {"baseball", "wbc"},
}


@dataclass(frozen=True)
class CategoryResolution:
    keys: frozenset[str]
    mode: str
    has_exact_mapping: bool
    requested_specific_keys: frozenset[str]

    @property
    def is_broad_fallback(self) -> bool:
        return self.mode == "broad_fallback"


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text.lower() in {"", "nan", "none", "null", "na", "n/a", "<na>"}


def normalize_label(value, *, missing: str = MISSING_CATEGORY) -> str:
    if _is_missing(value):
        return missing
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(value) -> str:
    return normalize_label(value).lower()


def expand_category_term(value) -> set[str]:
    term = normalize_key(value)
    if term == "n/a":
        return set()

    expanded = {term}
    expanded.update(CATEGORY_TERM_ALIASES.get(term, set()))

    if term.endswith(" primary"):
        expanded.update({"primaries", "primary elections"})
    if "governor" in term and "primary" in term:
        expanded.update({"governor", "primaries"})
    if "senate" in term and "primary" in term:
        expanded.update({"senate", "primaries"})
    if term in {"house primary", "congressional primary"}:
        expanded.update({"house", "congress", "primaries"})

    return {normalize_key(item) for item in expanded if normalize_key(item) != "n/a"}


def split_tags(value) -> set[str]:
    if _is_missing(value):
        return set()
    tags = set()
    for raw in str(value).split(","):
        tags.update(expand_category_term(raw))
    return tags


def split_kalshi_category_key(category_key: str) -> tuple[str, str]:
    key = normalize_key(category_key)
    if " / " not in key:
        return key, "n/a"
    category, sub_category = key.split(" / ", 1)
    return normalize_key(category), normalize_key(sub_category)


def kalshi_category_key(category, sub_category=None) -> str:
    category_key = normalize_key(category)
    sub_key = normalize_key(sub_category)
    return f"{category_key} / {sub_key}"


def kalshi_category_terms(category, sub_category=None) -> set[str]:
    terms = expand_category_term(category)
    terms.update(split_tags(sub_category))
    return {term for term in terms if term != "n/a"}


def polymarket_category_keys(category, sub_category=None) -> set[str]:
    keys = expand_category_term(category)
    keys.update(split_tags(sub_category))
    return {key for key in keys if key != "n/a"}


def normalize_category_mapping(raw_mapping: dict[str, Iterable[str]]) -> dict[str, set[str]]:
    normalized = {}
    for raw_kalshi_key, raw_poly_keys in raw_mapping.items():
        category, sub_category = split_kalshi_category_key(raw_kalshi_key)
        key = kalshi_category_key(category, sub_category)
        values = set()
        for value in raw_poly_keys:
            values.update(expand_category_term(value))
        normalized[key] = values
    return normalized


def _available_category_keys(values: Iterable[str], available_keys: set[str] | None) -> set[str]:
    normalized = {normalize_key(value) for value in values if normalize_key(value) != "n/a"}
    if available_keys is None:
        return normalized
    return normalized.intersection(available_keys)


def resolve_polymarket_keys_for_kalshi(
    kalshi_key: str,
    category_mapping: dict[str, set[str]],
    available_poly_keys: Iterable[str] | None = None,
) -> CategoryResolution:
    """Resolve category keys using tags that exist in the current Polymarket data.

    Specific tags are preferred. A broad parent category is used only when no
    requested specific tag is currently available, and the caller can then
    require stronger textual evidence for that weaker fallback.
    """
    category, sub_category = split_kalshi_category_key(kalshi_key)
    key = kalshi_category_key(category, sub_category)
    has_exact_mapping = key in category_mapping
    available = None
    if available_poly_keys is not None:
        available = {
            normalize_key(value)
            for value in available_poly_keys
            if normalize_key(value) != "n/a"
        }

    exact_values = set(category_mapping.get(key, set()))
    sub_category_values = split_tags(sub_category) if sub_category != "n/a" else set()
    requested_values = exact_values.union(sub_category_values)
    requested_specific = {
        value for value in requested_values
        if value not in SUPER_BROAD_POLYMARKET_CATEGORIES
    }
    exact_broad = {
        value for value in exact_values
        if value in SUPER_BROAD_POLYMARKET_CATEGORIES
    }
    if has_exact_mapping and sub_category == "n/a" and exact_broad:
        selected = _available_category_keys(exact_broad, available)
        return CategoryResolution(
            keys=frozenset(selected),
            mode="exact_broad" if selected else "unmapped",
            has_exact_mapping=True,
            requested_specific_keys=frozenset(),
        )

    available_specific = _available_category_keys(requested_specific, available)

    if available_specific:
        mode = "exact_specific" if has_exact_mapping else "inferred_specific"
        return CategoryResolution(
            keys=frozenset(available_specific),
            mode=mode,
            has_exact_mapping=has_exact_mapping,
            requested_specific_keys=frozenset(requested_specific),
        )

    if has_exact_mapping and not requested_specific:
        selected = _available_category_keys(exact_broad or exact_values, available)
        if selected:
            return CategoryResolution(
                keys=frozenset(selected),
                mode="exact_broad",
                has_exact_mapping=True,
                requested_specific_keys=frozenset(),
            )

    broad_key = kalshi_category_key(category, MISSING_CATEGORY)
    broad_values = set(category_mapping.get(broad_key, set()))
    broad_values.update(exact_broad)
    broad_values.update(expand_category_term(category))
    broad_candidates = {
        value for value in broad_values
        if value in SUPER_BROAD_POLYMARKET_CATEGORIES
    }
    selected_broad = _available_category_keys(broad_candidates, available)

    if selected_broad:
        return CategoryResolution(
            keys=frozenset(selected_broad),
            mode="broad_fallback",
            has_exact_mapping=has_exact_mapping,
            requested_specific_keys=frozenset(requested_specific),
        )

    return CategoryResolution(
        keys=frozenset(),
        mode="unmapped",
        has_exact_mapping=has_exact_mapping,
        requested_specific_keys=frozenset(requested_specific),
    )


def mapped_polymarket_keys_for_kalshi(kalshi_key: str, category_mapping: dict[str, set[str]]) -> set[str]:
    category, sub_category = split_kalshi_category_key(kalshi_key)
    key = kalshi_category_key(category, sub_category)
    has_exact_mapping = key in category_mapping
    exact = set(category_mapping.get(key, set()))
    if sub_category != "n/a":
        exact.update(split_tags(sub_category))

    if has_exact_mapping and exact and sub_category != "n/a":
        specific = {value for value in exact if value not in SUPER_BROAD_POLYMARKET_CATEGORIES}
        if specific:
            return specific

    if has_exact_mapping and exact:
        return exact

    fallback = set()
    if sub_category != "n/a":
        fallback.update(split_tags(sub_category))
        fallback.add(sub_category)

    broad_key = kalshi_category_key(category, MISSING_CATEGORY)
    broad_values = set(category_mapping.get(broad_key, set()))
    if broad_values:
        fallback.update(broad_values)

    if not fallback:
        fallback.add(category)

    return {value for value in fallback if value and value != "n/a"}


def categories_overlap(kalshi_category, kalshi_sub_category, polymarket_category, polymarket_sub_category) -> bool:
    kalshi_terms = kalshi_category_terms(kalshi_category, kalshi_sub_category)
    poly_terms = polymarket_category_keys(polymarket_category, polymarket_sub_category)
    if kalshi_terms.intersection(poly_terms):
        return True

    # Politics/World and Science/Climate are close enough for candidate generation;
    # exact resolution is handled later by hard rules and LLM verification.
    compatible_groups = (
        {"politics", "world", "world affairs", "nov 4 elections", "primaries"},
        {"science", "climate & science"},
        {"culture", "celebrities"},
        {"economics", "equities", "crypto", "crypto prices"},
    )
    for group in compatible_groups:
        if kalshi_terms.intersection(group) and poly_terms.intersection(group):
            return True
    return False
