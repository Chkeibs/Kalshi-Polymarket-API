"""Corpus-aware evidence, scoring, and graph controls for event clustering."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import math
from typing import Iterable

try:
    from pipeline.clustering.event_profiles import (
        EventProfile,
        explicit_conflicts,
        one_sided_fields,
        participant_sets_compatible,
    )
except ImportError:
    from event_profiles import EventProfile, explicit_conflicts, one_sided_fields, participant_sets_compatible


CLUSTERING_POLICY_VERSION = "clustering-policy-v2"


def canonical_keyword(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("year:"):
        return text
    if text.endswith(("ss", "us", "is", "as")):
        return text
    if text.endswith("ies") and len(text) > 5:
        return text[:-3] + "y"
    if text.endswith("s") and len(text) > 4:
        return text[:-1]
    return text


@dataclass(frozen=True)
class KeywordStatistics:
    event_count: int
    document_frequency: dict[str, int]
    kalshi_frequency: dict[str, int]
    polymarket_frequency: dict[str, int]

    def df(self, keyword: object) -> int:
        return int(self.document_frequency.get(canonical_keyword(keyword), 0))

    def idf(self, keyword: object) -> float:
        return math.log((self.event_count + 1) / (self.df(keyword) + 1)) + 1.0


def build_keyword_statistics(index: dict[str, dict[str, object]]) -> KeywordStatistics:
    total = Counter()
    per_platform = {"kalshi": Counter(), "polymarket": Counter()}
    for data in index.values():
        keywords = {
            canonical_keyword(value)
            for value in data.get("keywords", []) or []
            if value and not str(value).lower().startswith("year:")
        }
        for keyword in keywords:
            total[keyword] += 1
            platform = str(data.get("platform", "") or "").lower()
            if platform in per_platform:
                per_platform[platform][keyword] += 1
    return KeywordStatistics(
        event_count=len(index),
        document_frequency=dict(total),
        kalshi_frequency=dict(per_platform["kalshi"]),
        polymarket_frequency=dict(per_platform["polymarket"]),
    )


def _shared(left: Iterable[str], right: Iterable[str]) -> set[str]:
    return set(left).intersection(right)


def pair_semantic_features(
    left: EventProfile,
    right: EventProfile,
    common_keywords: Iterable[str],
    statistics: KeywordStatistics,
) -> dict[str, object]:
    canonical_common = {
        canonical_keyword(value)
        for value in common_keywords
        if value and not str(value).lower().startswith("year:")
    }
    keyword_evidence = sorted(
        (
            {
                "keyword": keyword,
                "document_frequency": statistics.df(keyword),
                "idf": round(statistics.idf(keyword), 4),
            }
            for keyword in canonical_common
        ),
        key=lambda item: (-float(item["idf"]), str(item["keyword"])),
    )
    rarest_df = min((int(item["document_frequency"]) for item in keyword_evidence), default=None)

    shared_participants = _shared(left.participants, right.participants)
    exact_fixture = bool(participant_sets_compatible(left.participants, right.participants))
    shared_entities = _shared(left.entities, right.entities)
    shared_districts = _shared(left.districts, right.districts)
    shared_locations = _shared(left.locations, right.locations)
    shared_groups = _shared(left.groups, right.groups)
    shared_dates = _shared(left.dates, right.dates)
    shared_periods = _shared(left.periods, right.periods)
    shared_seasons = _shared(left.seasons, right.seasons)
    shared_years = _shared(left.years, right.years)
    shared_actions = _shared(left.actions, right.actions)
    shared_scopes = _shared(left.scopes, right.scopes)
    shared_layers = _shared(left.market_layers, right.market_layers)

    primary_anchors = []
    if exact_fixture:
        primary_anchors.append("fixture")
    if shared_districts:
        primary_anchors.append("district")
    if shared_locations:
        primary_anchors.append("location")
    if shared_entities:
        primary_anchors.append("entity")
    if shared_groups and (shared_entities or shared_scopes):
        primary_anchors.append("group_context")
    if rarest_df is not None and rarest_df <= 25:
        primary_anchors.append("rare_keyword")

    secondary_anchors = []
    if shared_dates or shared_periods or shared_seasons or shared_years:
        secondary_anchors.append("time")
    if shared_actions:
        secondary_anchors.append("action")
    if shared_scopes:
        secondary_anchors.append("scope")
    if shared_layers:
        secondary_anchors.append("market_layer")

    occurrence_score = 0.0
    occurrence_score += 12.0 if exact_fixture else min(4.0, len(shared_participants) * 2.0)
    occurrence_score += min(12.0, len(shared_entities) * 4.0)
    occurrence_score += len(shared_districts) * 10.0
    occurrence_score += len(shared_locations) * 8.0
    occurrence_score += len(shared_groups) * 4.0
    occurrence_score += len(shared_dates) * 5.0
    occurrence_score += len(shared_periods) * 4.0
    occurrence_score += len(shared_seasons) * 4.0
    occurrence_score += len(shared_years) * 2.0
    occurrence_score += min(12.0, sum(min(4.0, statistics.idf(value)) for value in canonical_common))

    proposition_score = 0.0
    proposition_score += len(shared_actions) * 4.0
    proposition_score += len(shared_scopes) * 3.0
    proposition_score += len(shared_layers) * 3.0
    proposition_score += len(_shared(left.directions, right.directions)) * 4.0
    proposition_score += len(_shared(left.thresholds, right.thresholds)) * 5.0

    conflicts = explicit_conflicts(left, right)
    missing = one_sided_fields(left, right)
    conflict_penalty = len(conflicts) * 20.0
    missing_penalty = len(missing) * 0.75

    return {
        "policy_version": CLUSTERING_POLICY_VERSION,
        "profile_schema_version": left.schema_version,
        "primary_anchors": sorted(set(primary_anchors)),
        "secondary_anchors": sorted(set(secondary_anchors)),
        "keyword_evidence": keyword_evidence,
        "rarest_keyword_df": rarest_df,
        "explicit_conflicts": list(conflicts),
        "one_sided_fields": list(missing),
        "occurrence_score": round(occurrence_score, 3),
        "proposition_score": round(proposition_score, 3),
        "conflict_penalty": round(conflict_penalty, 3),
        "missing_penalty": round(missing_penalty, 3),
        "semantic_score": round(occurrence_score + proposition_score - conflict_penalty - missing_penalty, 3),
        "same_occurrence_key": bool(left.occurrence_key and left.occurrence_key == right.occurrence_key),
        "same_proposition_key": bool(left.proposition_key and left.proposition_key == right.proposition_key),
    }


def annotate_pairs(
    pairs: list[dict[str, object]],
    profiles: dict[str, EventProfile],
    statistics: KeywordStatistics,
) -> list[dict[str, object]]:
    annotated = []
    for pair in pairs:
        kalshi_id = str(pair["kalshi_id"])
        poly_id = str(pair["poly_id"])
        left = profiles[kalshi_id]
        right = profiles[poly_id]
        semantic = pair_semantic_features(left, right, pair.get("common_keywords", []), statistics)
        reason = dict(pair.get("candidate_reason", {}))
        reason["semantic"] = semantic
        annotated.append({**pair, "candidate_reason": reason})
    return annotated


def add_reciprocal_ranks(pairs: list[dict[str, object]]) -> None:
    by_kalshi = defaultdict(list)
    by_poly = defaultdict(list)
    for pair in pairs:
        by_kalshi[str(pair["kalshi_id"])].append(pair)
        by_poly[str(pair["poly_id"])].append(pair)

    def order_key(pair: dict[str, object]) -> tuple[float, float, str, str]:
        reason = pair.get("candidate_reason", {})
        semantic = reason.get("semantic", {})
        return (
            -float(semantic.get("semantic_score", 0.0)),
            -float(reason.get("quality_score", 0.0)),
            str(pair["kalshi_id"]),
            str(pair["poly_id"]),
        )

    for values in by_kalshi.values():
        values.sort(key=order_key)
        for rank, pair in enumerate(values, 1):
            pair["candidate_reason"]["semantic"]["kalshi_rank"] = rank
            pair["candidate_reason"]["semantic"]["kalshi_fanout"] = len(values)
    for values in by_poly.values():
        values.sort(key=order_key)
        for rank, pair in enumerate(values, 1):
            pair["candidate_reason"]["semantic"]["polymarket_rank"] = rank
            pair["candidate_reason"]["semantic"]["polymarket_fanout"] = len(values)


def connected_component_stats(pairs: list[dict[str, object]]) -> list[dict[str, int]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        left = f"k:{pair['kalshi_id']}"
        right = f"p:{pair['poly_id']}"
        adjacency[left].add(right)
        adjacency[right].add(left)

    components = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen)
        queue = deque([root])
        nodes = set()
        edge_twice = 0
        while queue:
            node = queue.popleft()
            if node in nodes:
                continue
            nodes.add(node)
            unseen.discard(node)
            edge_twice += len(adjacency[node])
            queue.extend(adjacency[node].difference(nodes))
        components.append({"nodes": len(nodes), "edges": edge_twice // 2})
    return sorted(components, key=lambda item: (-item["edges"], -item["nodes"]))


def should_prune_pair(pair: dict[str, object], max_reciprocal_rank: int = 2) -> tuple[bool, str]:
    reason = pair.get("candidate_reason", {})
    semantic = reason.get("semantic", {})
    conflicts = set(semantic.get("explicit_conflicts", []))
    if conflicts.intersection({"groups", "districts", "locations", "participants", "periods", "scopes"}):
        return True, "explicit_identity_conflict"

    tier = str(reason.get("quality_tier", "weak"))
    primary = set(semantic.get("primary_anchors", []))
    rarest_df = semantic.get("rarest_keyword_df")
    min_overlap = float(reason.get("title_min_overlap", 0.0))
    jaccard = float(reason.get("title_jaccard", 0.0))
    if tier == "weak" and not primary and min_overlap < 0.75:
        return True, "weak_without_primary_anchor"
    if (
        rarest_df is not None
        and int(rarest_df) > 100
        and not primary
        and min_overlap < 0.75
        and jaccard < 0.55
    ):
        return True, "high_frequency_only"

    kalshi_rank = int(semantic.get("kalshi_rank", 1))
    poly_rank = int(semantic.get("polymarket_rank", 1))
    best_rank = min(kalshi_rank, poly_rank)
    strong_title = min_overlap >= 0.80
    structured_identity = bool(primary.intersection({"fixture", "district"}))
    entity_supported_title = "entity" in primary and min_overlap >= 0.60
    if not (
        best_rank <= max_reciprocal_rank
        or strong_title
        or structured_identity
        or entity_supported_title
    ):
        return True, "outside_candidate_budget"
    return False, ""


def prune_pairs(
    pairs: list[dict[str, object]],
    max_reciprocal_rank: int = 2,
) -> tuple[list[dict[str, object]], Counter]:
    add_reciprocal_ranks(pairs)
    kept = []
    rejected = Counter()
    for pair in pairs:
        prune, reason = should_prune_pair(pair, max_reciprocal_rank=max_reciprocal_rank)
        if prune:
            rejected[reason] += 1
            continue
        kept.append(pair)
    return kept, rejected
