# Parsing / Keywords

## Purpose

Normalize categories, aliases, entities, keywords, and schemas so event candidates are generated from structured evidence rather than raw title overlap.

## Current Behavior

- `pipeline/common/category_utils.py` normalizes `NaN`/`N/A`, builds category keys, reads Polymarket tags, and applies controlled category variants.
- `pipeline/common/keyword_config.py` loads `config/keywords/*.json`, centralizing stop words, generic words, aliases, important entities, canonicalization, and config fingerprints.
- `pipeline/common/abbreviations.py` expands aliases and abbreviations, enriched by keyword config and cleaned normalized keywords when present.
- `pipeline/clustering/event_keyword_indexer.py` extracts event keywords with SpaCy when available and regex fallback otherwise.
- `pipeline/clustering/entity_extractor.py` and `keyword_consolidator.py` support keyword dictionary enrichment.

## Important Rules

- Categories route comparisons but are not positive proof.
- Generic words, offices, years, and broad sports/politics terms should not validate a pair alone.
- `year:YYYY` is useful context but insufficient by itself.
- Contextual sports aliases such as city/team words should only apply in sports contexts.
- Preserve singular/plural canonicalization without creating bogus variants.

## Related Files

- `config/category_mapping.json`
- `config/keywords/*.json`
- `config/polymarket_topic_tags.json`
- `data/keywords/event_keywords_index.json`
- `pipeline/common/category_utils.py`
- `pipeline/common/keyword_config.py`
- `pipeline/common/abbreviations.py`
- `pipeline/clustering/event_keyword_indexer.py`
- `tests/test_category_utils.py`
- `tests/test_event_keyword_indexer_aliases.py`
- `tests/test_keyword_parsing.py`

## Edge Cases

- `New Mexico` must remain a US state and not become country `Mexico`.
- National adjectives such as `Slovenian` should map to country context when appropriate.
- SpaCy may classify teams/titles as people; conflict logic normalizes tokens to reduce false person conflicts.

## Open Questions

- Which aliases/entities should be promoted from observed data into stable config after human review?
