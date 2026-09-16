"""Reproducible, offline audit for clustering candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

try:
    from pipeline.clustering.clustering_features import (
        annotate_pairs,
        build_keyword_statistics,
        connected_component_stats,
    )
    from pipeline.clustering.event_profiles import build_event_profiles, profiles_fingerprint
except ImportError:
    from clustering_features import annotate_pairs, build_keyword_statistics, connected_component_stats
    from event_profiles import build_event_profiles, profiles_fingerprint


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT_DIR / "data" / "clusters" / "candidate_clusters.json"
DEFAULT_INDEX = ROOT_DIR / "data" / "keywords" / "event_keywords_index.json"
DEFAULT_CONFIRMED = ROOT_DIR / "data" / "matches" / "confirmed_matches.csv"
DEFAULT_REPORT = ROOT_DIR / "data" / "reports" / "clustering" / "clustering_audit.json"
DEFAULT_SAMPLES = ROOT_DIR / "data" / "reports" / "clustering" / "clustering_samples.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_confirmed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (str(row.get("Kalshi_ID", "")), str(row.get("Polymarket_ID", "")))
            for row in csv.DictReader(handle)
            if row.get("Kalshi_ID") and row.get("Polymarket_ID")
        }


def pair_key(pair: dict[str, object]) -> tuple[str, str]:
    if "kalshi_id" in pair:
        return str(pair["kalshi_id"]), str(pair["poly_id"])
    return str(pair["kalshi"]["event_id"]), str(pair["polymarket"]["event_id"])


def internal_pair(pair: dict[str, object]) -> dict[str, object]:
    if "kalshi_id" in pair:
        return pair
    return {
        "kalshi_id": str(pair["kalshi"]["event_id"]),
        "poly_id": str(pair["polymarket"]["event_id"]),
        "common_keywords": pair.get("common_keywords", []),
        "candidate_reason": pair.get("candidate_reason", {}),
    }


def risk_segment(pair: dict[str, object]) -> str:
    reason = pair.get("candidate_reason", {})
    semantic = reason.get("semantic", {})
    conflicts = set(semantic.get("explicit_conflicts", []))
    if conflicts.intersection({"groups", "districts"}):
        return "explicit_identity_conflict"
    if semantic.get("one_sided_fields"):
        return "one_sided_structured_field"
    if reason.get("quality_tier") == "weak":
        return "weak"
    rarest = semantic.get("rarest_keyword_df")
    if rarest is not None and int(rarest) > 100:
        return "high_frequency_only"
    if int(semantic.get("kalshi_fanout", 1)) > 10 or int(semantic.get("polymarket_fanout", 1)) > 10:
        return "high_fanout"
    return "bounded"


def build_report(
    candidates: list[dict[str, object]],
    index: dict[str, dict[str, object]],
    confirmed: set[tuple[str, str]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    profiles = build_event_profiles(index)
    statistics = build_keyword_statistics(index)
    pairs = annotate_pairs([internal_pair(pair) for pair in candidates], profiles, statistics)

    # Ranking annotates fanout fields in-place without filtering.
    try:
        from pipeline.clustering.clustering_features import add_reciprocal_ranks
    except ImportError:
        from clustering_features import add_reciprocal_ranks
    add_reciprocal_ranks(pairs)

    tier_counts = Counter()
    category_counts = Counter()
    segment_counts = Counter()
    segment_confirmed = Counter()
    kalshi_degree = Counter()
    polymarket_degree = Counter()
    samples = []
    seen_sample_segments = Counter()
    candidate_keys = set()

    for pair in pairs:
        key = pair_key(pair)
        candidate_keys.add(key)
        reason = pair.get("candidate_reason", {})
        tier_counts[str(reason.get("quality_tier", "unknown"))] += 1
        category = str(index.get(key[0], {}).get("category", "unknown"))
        category_counts[category] += 1
        segment = risk_segment(pair)
        segment_counts[segment] += 1
        if key in confirmed:
            segment_confirmed[segment] += 1
        kalshi_degree[key[0]] += 1
        polymarket_degree[key[1]] += 1

        if seen_sample_segments[segment] < 25:
            semantic = reason.get("semantic", {})
            samples.append({
                "segment": segment,
                "is_confirmed": key in confirmed,
                "kalshi_id": key[0],
                "polymarket_id": key[1],
                "kalshi_title": index.get(key[0], {}).get("title", ""),
                "polymarket_title": index.get(key[1], {}).get("title", ""),
                "quality_tier": reason.get("quality_tier", ""),
                "semantic_score": semantic.get("semantic_score", ""),
                "primary_anchors": ",".join(semantic.get("primary_anchors", [])),
                "explicit_conflicts": ",".join(semantic.get("explicit_conflicts", [])),
                "one_sided_fields": ",".join(semantic.get("one_sided_fields", [])),
            })
            seen_sample_segments[segment] += 1

    components = connected_component_stats(pairs)
    covered_confirmed = confirmed.intersection(candidate_keys)
    report = {
        "schema_version": "clustering-audit-v1",
        "candidate_count": len(pairs),
        "event_index_count": len(index),
        "confirmed_total": len(confirmed),
        "confirmed_covered": len(covered_confirmed),
        "confirmed_missing": [list(key) for key in sorted(confirmed.difference(candidate_keys))],
        "profile_fingerprint": profiles_fingerprint(profiles),
        "quality_tiers": dict(sorted(tier_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "risk_segments": {
            name: {"candidates": count, "confirmed": segment_confirmed[name]}
            for name, count in sorted(segment_counts.items())
        },
        "fanout": {
            "kalshi_max": max(kalshi_degree.values(), default=0),
            "polymarket_max": max(polymarket_degree.values(), default=0),
            "pairs_either_degree_gt_10": sum(
                1 for pair in pairs
                if kalshi_degree[pair_key(pair)[0]] > 10 or polymarket_degree[pair_key(pair)[1]] > 10
            ),
        },
        "components": {
            "count": len(components),
            "largest": components[:20],
        },
    }
    return report, samples


def write_report(report: dict[str, object], samples: Iterable[dict[str, object]], report_path: Path, samples_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows = list(samples)
    fieldnames = list(rows[0]) if rows else ["segment"]
    with samples_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(candidates_path: Path, index_path: Path, confirmed_path: Path, report_path: Path, samples_path: Path) -> dict[str, object]:
    candidates = json.loads(candidates_path.read_text())
    index = json.loads(index_path.read_text())
    confirmed = load_confirmed(confirmed_path)
    report, samples = build_report(candidates, index, confirmed)
    report["input_hashes"] = {
        "candidates": file_sha256(candidates_path),
        "event_index": file_sha256(index_path),
        "confirmed": file_sha256(confirmed_path) if confirmed_path.exists() else "",
    }
    write_report(report, samples, report_path, samples_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit clustering candidates without network or LLM calls.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--confirmed", type=Path, default=DEFAULT_CONFIRMED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    args = parser.parse_args()
    report = run(args.candidates, args.index, args.confirmed, args.report, args.samples)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
