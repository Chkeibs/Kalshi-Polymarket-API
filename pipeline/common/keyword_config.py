"""
Central keyword parsing configuration.

This module keeps the parsing rules data-driven: stop words, market-generic
words, aliases and important entities live in config/keywords/*.json instead
of being duplicated across clustering scripts.
"""

import hashlib
import json
import os
import re
from functools import lru_cache


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_KEYWORDS_DIR = os.path.join(ROOT_DIR, "config", "keywords")

STOP_WORDS_FILE = os.path.join(CONFIG_KEYWORDS_DIR, "stop_words.json")
GENERIC_MARKET_WORDS_FILE = os.path.join(CONFIG_KEYWORDS_DIR, "generic_market_words.json")
ALIASES_FILE = os.path.join(CONFIG_KEYWORDS_DIR, "aliases.json")
IMPORTANT_ENTITIES_FILE = os.path.join(CONFIG_KEYWORDS_DIR, "important_entities.json")
DOMAIN_KEYWORDS_FILE = os.path.join(CONFIG_KEYWORDS_DIR, "domain_keywords.json")

CONFIG_FILES = (
    STOP_WORDS_FILE,
    GENERIC_MARKET_WORDS_FILE,
    ALIASES_FILE,
    IMPORTANT_ENTITIES_FILE,
    DOMAIN_KEYWORDS_FILE,
)

ENTITY_TYPE_TO_LABEL = {
    "person": "PERSON",
    "org": "ORG",
    "gpe": "GPE",
    "loc": "LOC",
    "topic": "TOPIC",
}


def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def normalize_phrase(value):
    """Normalize text while preserving meaningful symbols like %, $, hyphen."""
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"['’]s\b", "", text)
    text = text.replace("_", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,!?;:\"()[]{}")


def compact_lookup_key(value):
    """Alias lookup key that tolerates punctuation differences like U.S. vs US."""
    return re.sub(r"[^a-z0-9:%$+-]+", "", normalize_phrase(value))


def _add_alias(alias_map, variant, canonical):
    variant_norm = normalize_phrase(variant)
    canonical_norm = normalize_phrase(canonical)
    if not variant_norm or not canonical_norm:
        return
    alias_map[variant_norm] = canonical_norm
    compact = compact_lookup_key(variant_norm)
    if compact and compact != variant_norm:
        alias_map[compact] = canonical_norm


@lru_cache(maxsize=1)
def load_keyword_config():
    stop_words = {
        normalize_phrase(word)
        for word in load_json_file(STOP_WORDS_FILE, [])
        if normalize_phrase(word)
    }
    generic_market_words = {
        normalize_phrase(word)
        for word in load_json_file(GENERIC_MARKET_WORDS_FILE, [])
        if normalize_phrase(word)
    }

    aliases = {}
    aliases_raw = load_json_file(ALIASES_FILE, {})
    if isinstance(aliases_raw, dict):
        for variant, canonical in aliases_raw.items():
            _add_alias(aliases, variant, canonical)

    important_entities = {}
    entities_raw = load_json_file(IMPORTANT_ENTITIES_FILE, {})
    if isinstance(entities_raw, dict):
        for canonical, info in entities_raw.items():
            canonical_norm = normalize_phrase(canonical)
            if not canonical_norm:
                continue
            entity_type = "topic"
            variants = []
            if isinstance(info, dict):
                entity_type = normalize_phrase(info.get("type", "topic")) or "topic"
                variants = info.get("variants", [])
            important_entities[canonical_norm] = {
                "type": entity_type,
                "variants": [normalize_phrase(v) for v in variants if normalize_phrase(v)],
            }
            _add_alias(aliases, canonical_norm, canonical_norm)
            for variant in variants:
                _add_alias(aliases, variant, canonical_norm)

    domain_entity_tokens = {}
    domain_raw = load_json_file(DOMAIN_KEYWORDS_FILE, {})
    if isinstance(domain_raw, dict):
        for group, words in domain_raw.items():
            label = ENTITY_TYPE_TO_LABEL.get(normalize_phrase(group), "TOPIC")
            if not isinstance(words, list):
                continue
            for word in words:
                word_norm = normalize_phrase(word)
                if not word_norm:
                    continue
                canonical = aliases.get(word_norm) or aliases.get(compact_lookup_key(word_norm)) or word_norm
                domain_entity_tokens[word_norm] = f"{label}:{canonical}"
                domain_entity_tokens[canonical] = f"{label}:{canonical}"

    return {
        "stop_words": stop_words,
        "generic_market_words": generic_market_words,
        "noise_words": stop_words | generic_market_words,
        "aliases": aliases,
        "important_entities": important_entities,
        "domain_entity_tokens": domain_entity_tokens,
    }


def load_aliases():
    return dict(load_keyword_config()["aliases"])


def canonicalize_keyword(value):
    norm = normalize_phrase(value)
    if not norm:
        return ""
    config = load_keyword_config()
    return config["aliases"].get(norm) or config["aliases"].get(compact_lookup_key(norm)) or norm


def is_noise_keyword(value):
    canonical = canonicalize_keyword(value)
    if not canonical:
        return True
    compact = compact_lookup_key(canonical)
    config = load_keyword_config()
    if canonical in config["noise_words"] or compact in config["noise_words"]:
        return True
    if re.fullmatch(r"[$€£¥]+", canonical):
        return True
    if len(re.sub(r"[^a-z0-9]", "", canonical)) < 3 and not canonical.isdigit():
        return True
    return False


def add_keyword(target_set, value):
    canonical = canonicalize_keyword(value)
    if not is_noise_keyword(canonical):
        target_set.add(canonical)


def find_alias_matches(text):
    """Find configured aliases/entities explicitly mentioned in a title."""
    normalized_text = f" {normalize_phrase(text)} "
    if not normalized_text.strip():
        return set()

    matches = set()
    for variant, canonical in load_keyword_config()["aliases"].items():
        if not variant or len(variant) < 2:
            continue
        escaped = re.escape(variant)
        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", normalized_text):
            if not is_noise_keyword(canonical):
                matches.add(canonical)
    return matches


def find_domain_entity_tokens(text):
    """Return structured fallback tokens like PERSON:trump or ORG:bitcoin."""
    normalized_text = f" {normalize_phrase(text)} "
    tokens = set()
    for phrase, token in load_keyword_config()["domain_entity_tokens"].items():
        escaped = re.escape(phrase)
        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", normalized_text):
            tokens.add(token)
    return tokens


def entity_label_for_keyword(value):
    canonical = canonicalize_keyword(value)
    info = load_keyword_config()["important_entities"].get(canonical)
    if not info:
        return ""
    return ENTITY_TYPE_TO_LABEL.get(info.get("type", "topic"), "TOPIC")


def signature_noise_tokens():
    config = load_keyword_config()
    return {f"HPW:{word}" for word in config["noise_words"]}


def config_fingerprint():
    digest = hashlib.sha256()
    digest.update(b"keyword-config-v1")
    for path in CONFIG_FILES:
        relative_path = os.path.relpath(path, ROOT_DIR).replace(os.sep, "/")
        digest.update(relative_path.encode("utf-8"))
        if os.path.exists(path):
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()[:16]
