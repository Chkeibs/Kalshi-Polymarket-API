from pipeline.common.schemas import (
    KALSHI_MARKET_REQUIRED_COLUMNS,
    POLYMARKET_MARKET_REQUIRED_COLUMNS,
    missing_columns,
)
from pipeline.fetch import fetch_clean_data


def kalshi_event(*, nested=True):
    event = {
        "event_ticker": "KXTEST-26",
        "series_ticker": "KXTEST",
        "title": "Test event: 2026",
        "category": "Politics",
        "sub_title": "Event context",
    }
    if nested:
        event["markets"] = [
            {
                "ticker": "KXTEST-26-YES",
                "status": "active",
                "title": "Test event",
                "yes_sub_title": "Candidate A",
                "rules_primary": "Resolves Yes if Candidate A wins.",
                "expiration_time": "2026-12-31T00:00:00Z",
            }
        ]
    return event


def test_kalshi_uses_recommended_host_and_nested_markets(monkeypatch):
    calls = []

    def fake_request(url, params=None, **kwargs):
        calls.append((url, params))
        return {"events": [kalshi_event()], "cursor": ""}

    monkeypatch.setattr(fetch_clean_data, "request_json", fake_request)
    monkeypatch.setattr(fetch_clean_data, "fetch_kalshi_series_tags", lambda tickers: {"KXTEST": "Elections"})

    result = fetch_clean_data.fetch_kalshi_markets(max_pages=1)

    assert fetch_clean_data.KALSHI_API_BASE == "https://external-api.kalshi.com/trade-api/v2"
    assert len(calls) == 1
    assert calls[0][0].endswith("/events")
    assert calls[0][1]["with_nested_markets"] == "true"
    assert calls[0][1]["status"] == "open"
    assert len(result) == 1
    assert not missing_columns(result.columns, KALSHI_MARKET_REQUIRED_COLUMNS)
    assert result.iloc[0]["Bet ID"] == "KXTEST-26-YES"
    assert result.iloc[0]["Sub-Category"] == "Elections"


def test_kalshi_falls_back_only_when_nested_markets_are_missing(monkeypatch):
    calls = []

    def fake_request(url, params=None, **kwargs):
        calls.append(url)
        if url.endswith("/events"):
            return {"events": [kalshi_event(nested=False)], "cursor": ""}
        return {"event": kalshi_event(nested=False), "markets": kalshi_event()["markets"]}

    monkeypatch.setattr(fetch_clean_data, "request_json", fake_request)
    monkeypatch.setattr(fetch_clean_data, "fetch_kalshi_series_tags", lambda tickers: {})

    result = fetch_clean_data.fetch_kalshi_markets(max_pages=1)

    assert calls == [
        "https://external-api.kalshi.com/trade-api/v2/events",
        "https://external-api.kalshi.com/trade-api/v2/events/KXTEST-26",
    ]
    assert list(result["Bet ID"]) == ["KXTEST-26-YES"]


def polymarket_event(event_id, token_prefix):
    return {
        "id": event_id,
        "title": f"Event {event_id}",
        "description": "Resolution context",
        "active": True,
        "closed": False,
        "tags": [{"label": "Politics"}, {"label": "Elections"}],
        "markets": [
            {
                "id": f"market-{event_id}",
                "question": f"Question {event_id}",
                "active": True,
                "closed": False,
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": f'["{token_prefix}1", "{token_prefix}2"]',
                "endDate": "2026-12-31T00:00:00Z",
            }
        ],
    }


def test_polymarket_uses_keyset_pagination_and_preserves_schema(monkeypatch):
    calls = []
    responses = iter(
        [
            {"events": [polymarket_event("1", "111111111111111111111")], "next_cursor": "cursor-2"},
            {"events": [polymarket_event("2", "222222222222222222222")]},
        ]
    )

    def fake_request(url, params=None, **kwargs):
        calls.append((url, dict(params or {})))
        return next(responses)

    monkeypatch.setattr(fetch_clean_data, "request_json", fake_request)
    result = fetch_clean_data.fetch_polymarket_markets(max_pages=2)

    assert [call[0] for call in calls] == [
        "https://gamma-api.polymarket.com/events/keyset",
        "https://gamma-api.polymarket.com/events/keyset",
    ]
    assert "offset" not in calls[0][1]
    assert "after_cursor" not in calls[0][1]
    assert calls[1][1]["after_cursor"] == "cursor-2"
    assert calls[0][1]["limit"] == 500
    assert len(result) == 4
    assert not missing_columns(result.columns, POLYMARKET_MARKET_REQUIRED_COLUMNS)
    assert set(result["Outcome_Name"]) == {"Yes", "No"}
