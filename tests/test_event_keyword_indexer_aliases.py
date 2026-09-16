import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.clustering.event_keyword_indexer import (  # noqa: E402
    apply_alias,
    extract_keywords_simple,
    is_sports_context,
)


class DummyToken:
    def __init__(self, text, index):
        self.text = text
        self.i = index
        self.is_punct = False
        self.is_space = False
        self.is_digit = text.isdigit()
        self.like_num = text.isdigit()


class DummyDoc(list):
    ents = []


class DummyEntity(list):
    def __init__(self, text, label, tokens):
        super().__init__(tokens)
        self.text = text
        self.label_ = label


def dummy_nlp(text):
    return DummyDoc(DummyToken(token, idx) for idx, token in enumerate(text.split()))


def entity_nlp(text):
    tokens = [DummyToken(token, idx) for idx, token in enumerate(text.split())]
    doc = DummyDoc(tokens)
    doc.ents = [DummyEntity(text, "ORG", tokens)]
    return doc


def test_sports_aliases_are_contextual():
    keywords = extract_keywords_simple(
        "Will Mamdani open a city grocery store?",
        dummy_nlp,
        category="Politics",
        sub_category="N/A",
    )

    assert "city" in keywords
    assert "manchester city" not in keywords


def test_sports_aliases_still_apply_in_match_context():
    keywords = extract_keywords_simple(
        "Manchester City vs Arsenal",
        dummy_nlp,
        category="Sports",
        sub_category="Soccer",
    )

    assert "manchester city" in keywords
    assert "arsenal" in keywords


def test_global_aliases_apply_without_sports_context():
    assert apply_alias("USA", sports_context=False) == "united states"
    assert is_sports_context("Real Madrid vs Barcelona", category="Culture", sub_category="Games")


def test_parser_adds_simple_singular_variants():
    keywords = extract_keywords_simple(
        "Oscars 2026: Best Director Winner",
        dummy_nlp,
        category="Culture",
        sub_category="Awards, Movies",
    )

    assert "oscars" in keywords
    assert "oscar" in keywords
    assert "year:2026" in keywords


def test_parser_does_not_create_bad_singular_variants():
    keywords = extract_keywords_simple(
        "Will Democrats win any statewide election in Texas?",
        dummy_nlp,
        category="Politics",
        sub_category="US Elections",
    )
    country_keywords = extract_keywords_simple(
        "Which countries will qualify before 2027?",
        dummy_nlp,
        category="Politics",
        sub_category="Foreign Elections",
    )

    assert "texas" in keywords
    assert "texa" not in keywords
    assert "country" in country_keywords
    assert "countrie" not in country_keywords


def test_parser_preserves_typed_district_and_group_ids():
    district = extract_keywords_simple(
        "WY-AL House Election Winner",
        dummy_nlp,
        category="Politics",
        sub_category="House Elections",
    )
    group = extract_keywords_simple(
        "FIFA World Cup Group A Winner",
        dummy_nlp,
        category="Sports",
        sub_category="Soccer",
    )

    assert "district:WY-AL" in district
    assert "group:A" in group


def test_entity_phrases_keep_component_keywords_for_cross_platform_overlap():
    keywords = extract_keywords_simple(
        "Alabama Senate Election Winner",
        entity_nlp,
        category="Politics",
        sub_category="Elections",
    )

    assert "alabama senate election winner" in keywords
    assert "alabama" in keywords
    assert "senate" in keywords


def run() -> None:
    test_sports_aliases_are_contextual()
    test_sports_aliases_still_apply_in_match_context()
    test_global_aliases_apply_without_sports_context()
    test_parser_adds_simple_singular_variants()
    test_parser_does_not_create_bad_singular_variants()
    test_parser_preserves_typed_district_and_group_ids()
    test_entity_phrases_keep_component_keywords_for_cross_platform_overlap()


if __name__ == "__main__":
    run()
    print("ok")
