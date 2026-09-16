import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.clustering.event_profiles import (
    build_event_profile,
    explicit_conflicts,
    one_sided_fields,
)


def event(title, keywords=None, category="Sports"):
    return {
        "platform": "kalshi",
        "title": title,
        "category": category,
        "sub_category": "",
        "keywords": keywords or [],
    }


def test_group_ids_are_preserved_before_short_token_removal():
    group_a = build_event_profile("A", event("World Cup Group A Winner", ["world", "cup"]))
    group_b = build_event_profile("B", event("FIFA World Cup Group B Winner", ["world", "cup"]))
    missing = build_event_profile("M", event("World Cup Group Winner", ["world", "cup"]))

    assert group_a.groups == frozenset({"a"})
    assert group_b.groups == frozenset({"b"})
    assert "groups" in explicit_conflicts(group_a, group_b)
    assert "groups" not in explicit_conflicts(group_a, missing)
    assert "groups" in one_sided_fields(group_a, missing)


def test_fixture_participants_are_order_independent_and_layered():
    kalshi = build_event_profile(
        "K",
        event("Manchester City at Real Madrid", ["manchester city", "real madrid"]),
    )
    polymarket = build_event_profile(
        "P",
        event(
            "Real Madrid CF vs. Manchester City FC - More Markets",
            ["manchester city", "real madrid"],
        ),
    )

    assert kalshi.participants == polymarket.participants
    assert not explicit_conflicts(kalshi, polymarket)
    assert kalshi.occurrence_key == polymarket.occurrence_key
    assert kalshi.proposition_key != polymarket.proposition_key

    different_opponent = build_event_profile(
        "D",
        event("Arsenal vs. Manchester City", ["arsenal", "manchester city"]),
    )
    assert "participants" in explicit_conflicts(kalshi, different_opponent)


def test_explicit_district_and_threshold_conflicts_are_typed():
    district_one = build_event_profile("K", event("CA-03 Democratic nominee?", category="Politics"))
    district_two = build_event_profile("P", event("CA-12 Democratic nominee?", category="Politics"))
    above = build_event_profile("A", event("Will CPI be above 3%?", category="Economics"))
    below = build_event_profile("B", event("Will CPI be below 2%?", category="Economics"))

    assert "districts" in explicit_conflicts(district_one, district_two)
    assert {"directions", "thresholds"}.issubset(explicit_conflicts(above, below))


def test_aggregate_period_location_and_group_scopes_are_structured():
    district = build_event_profile("K", event("Which party will win the House race for CA-03?", category="Politics"))
    chamber = build_event_profile("P", event("Which party will control the House?", category="Politics"))
    march = build_event_profile("M", event("Fed decision in March 2026?", category="Economics"))
    april = build_event_profile("A", event("Fed decision in April?", category="Economics"))
    austin = build_event_profile("T1", event("Highest temperature in Austin on Mar 2, 2026?"))
    london = build_event_profile("T2", event("Highest temperature in London on March 2?"))
    group = build_event_profile("G", event("World Cup Group A Winner"))
    overall = build_event_profile("O", event("2026 FIFA World Cup Winner"))

    assert "scopes" in explicit_conflicts(district, chamber)
    assert "periods" in explicit_conflicts(march, april)
    assert "dates" not in explicit_conflicts(austin, london)
    assert "locations" in explicit_conflicts(austin, london)
    assert "scopes" in explicit_conflicts(group, overall)

    meeting = build_event_profile("FM", event("Fed decision in April 2026?", category="Economics"))
    annual = build_event_profile("FY", event("What will the Fed rate be at the end of 2026?", category="Economics"))
    december = build_event_profile("FD", event("Fed funds rate after Dec 2026 meeting?", category="Economics"))
    assert "scopes" in explicit_conflicts(meeting, annual)
    assert "scopes" not in explicit_conflicts(december, annual)


def run():
    test_group_ids_are_preserved_before_short_token_removal()
    test_fixture_participants_are_order_independent_and_layered()
    test_explicit_district_and_threshold_conflicts_are_typed()
    test_aggregate_period_location_and_group_scopes_are_structured()
    print("ok")


if __name__ == "__main__":
    run()
