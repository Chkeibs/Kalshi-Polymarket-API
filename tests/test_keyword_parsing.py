import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.clustering import event_keyword_indexer
from pipeline.clustering import group_events_cluster
from pipeline.common.keyword_config import canonicalize_keyword, is_noise_keyword


def load_models():
    nlp = event_keyword_indexer.load_spacy()
    group_events_cluster.nlp = nlp
    return nlp


def test_keyword_config_canonicalizes_aliases_and_noise():
    assert canonicalize_keyword("BTC") == "bitcoin"
    assert canonicalize_keyword("PSG") == "paris saint-germain"
    assert canonicalize_keyword("U.S.") == "united states"
    assert is_noise_keyword("named")
    assert is_noise_keyword("win")


def test_event_keyword_indexer_keeps_entities_years_and_drops_generic_words():
    nlp = load_models()

    btc_keywords = event_keyword_indexer.extract_keywords_simple(
        "Will BTC hit $100,000 before July 2026?",
        nlp,
    )
    assert "bitcoin" in btc_keywords
    assert "year:2026" in btc_keywords
    assert "hit" not in btc_keywords
    assert "will" not in btc_keywords

    xi_keywords = event_keyword_indexer.extract_keywords_simple(
        "Who will be named as Xi Jinping's successor?",
        nlp,
    )
    assert "xi jinping" in xi_keywords
    assert "successor" in xi_keywords
    assert "named" not in xi_keywords
    assert "name" not in xi_keywords

    costa_terms = event_keyword_indexer.extract_keywords_and_entities(
        "Margin of victory in the 1st round of the Costa Rica Presidential election?",
        nlp,
    )
    assert "costa rica" in costa_terms["entities"]["gpe"]
    assert "margin" in costa_terms["keywords"]

    costa_rican_terms = event_keyword_indexer.extract_keywords_and_entities(
        "Who will win the 2026 Costa Rican Presidential election?",
        nlp,
    )
    assert "costa rica" in costa_rican_terms["entities"]["gpe"]
    assert "year:2026" in costa_rican_terms["keywords"]


def test_event_keyword_indexer_uses_sports_and_macro_aliases():
    nlp = load_models()

    sports_keywords = event_keyword_indexer.extract_keywords_simple(
        "PSG vs Barcelona: who will win?",
        nlp,
    )
    assert "paris saint-germain" in sports_keywords
    assert "fc barcelona" in sports_keywords
    assert "vs" not in sports_keywords
    assert "win" not in sports_keywords

    macro_keywords = event_keyword_indexer.extract_keywords_simple(
        "Will the Fed cut rates by 25 bps in June 2026?",
        nlp,
    )
    assert "federal reserve" in macro_keywords
    assert "basis points" in macro_keywords
    assert "year:2026" in macro_keywords


def test_signature_details_filter_noisy_hpw_tokens():
    load_models()

    signature, details = group_events_cluster.extract_signature_details(
        "PSG vs Barcelona: who will win?",
        "Sports",
    )

    assert "HPW:paris saint-germain" in signature
    assert "HPW:fc barcelona" in signature
    assert "CAT:sports" in signature
    assert "HPW:vs" not in signature
    assert "HPW:win" not in signature
    assert details["keyword_config_version"] == group_events_cluster.CONFIG_VERSION

    btc_signature, _ = group_events_cluster.extract_signature_details(
        "Will BTC hit $100,000 before July 2026?",
        "Crypto",
    )
    assert "ORG:bitcoin" in btc_signature
    assert "YEAR:2026" in btc_signature
    assert "HPW:hit" not in btc_signature
    assert "HPW:$" not in btc_signature


def test_candidate_hard_conflict_uses_structured_entities_and_years():
    costa = {
        "entities": {"gpe": ["costa rica"]},
        "years": ["2026"],
    }
    brazil = {
        "entities": {"gpe": ["brazil"]},
        "years": ["2026"],
    }
    same_country = {
        "entities": {"gpe": ["costa rica"]},
        "years": ["2026"],
    }
    far_year = {
        "entities": {"gpe": ["costa rica"]},
        "years": ["2030"],
    }

    assert group_events_cluster.has_candidate_hard_conflict(costa, brazil)
    assert not group_events_cluster.has_candidate_hard_conflict(costa, same_country)
    assert group_events_cluster.has_candidate_hard_conflict(costa, far_year)


def run() -> None:
    test_keyword_config_canonicalizes_aliases_and_noise()
    test_event_keyword_indexer_keeps_entities_years_and_drops_generic_words()
    test_event_keyword_indexer_uses_sports_and_macro_aliases()
    test_signature_details_filter_noisy_hpw_tokens()
    test_candidate_hard_conflict_uses_structured_entities_and_years()


if __name__ == "__main__":
    run()
    print("ok")
