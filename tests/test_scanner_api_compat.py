from api.services.scanner import KALSHI_WS_URL, LiveScanner


def test_kalshi_fixed_point_snapshot_preserves_subcent_prices():
    scanner = LiveScanner()
    scanner.process_kalshi_snapshot(
        {
            "market_ticker": "KXTEST",
            "yes_dollars_fp": [["0.1234", "2.50"], ["0.1200", "3.00"]],
            "no_dollars_fp": [["0.8765", "4.25"]],
        }
    )

    book = scanner.orderbooks["KXTEST"]
    assert KALSHI_WS_URL == "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
    assert book["bids"] == [(0.1234, 2.5), (0.12, 3.0)]
    assert book["asks"] == [(0.1235, 4.25)]
    assert book["best_bid"] == 0.1234
    assert book["best_ask"] == 0.1235


def test_kalshi_fixed_point_delta_updates_and_removes_levels():
    scanner = LiveScanner()
    scanner.process_kalshi_snapshot(
        {
            "market_ticker": "KXTEST",
            "yes_dollars_fp": [["0.1234", "2.50"]],
            "no_dollars_fp": [["0.8765", "4.25"]],
        }
    )

    scanner.process_kalshi_delta(
        {"market_ticker": "KXTEST", "side": "yes", "price_dollars": "0.1240", "delta_fp": "1.25"}
    )
    scanner.process_kalshi_delta(
        {"market_ticker": "KXTEST", "side": "no", "price_dollars": "0.8750", "delta_fp": "2.00"}
    )
    assert scanner.orderbooks["KXTEST"]["best_bid"] == 0.124
    assert scanner.orderbooks["KXTEST"]["best_ask"] == 0.1235
    assert (0.125, 2.0) in scanner.orderbooks["KXTEST"]["asks"]

    scanner.process_kalshi_delta(
        {"market_ticker": "KXTEST", "side": "no", "price_dollars": "0.8765", "delta_fp": "-4.25"}
    )
    assert scanner.orderbooks["KXTEST"]["best_ask"] == 0.125

    scanner.process_kalshi_delta(
        {"market_ticker": "KXTEST", "side": "yes", "price_dollars": "0.1240", "delta_fp": "-1.25"}
    )
    assert scanner.orderbooks["KXTEST"]["best_bid"] == 0.1234


def test_polymarket_subscription_and_book_match_current_schema():
    scanner = LiveScanner()
    scanner.p_id_to_clob_id = {"poly-a": "token-2", "poly-b": "token-1"}
    scanner.clob_id_to_p_id = {"token-1": "poly-b", "token-2": "poly-a"}

    assert scanner.build_polymarket_subscription() == {
        "assets_ids": ["token-1", "token-2"],
        "type": "market",
        "custom_feature_enabled": True,
    }

    scanner.process_poly_update(
        {
            "event_type": "book",
            "asset_id": "token-1",
            "bids": [{"price": ".48", "size": "30"}],
            "asks": [{"price": ".52", "size": "25"}],
        }
    )
    assert scanner.orderbooks["poly-b"]["best_bid"] == 0.48
    assert scanner.orderbooks["poly-b"]["best_ask"] == 0.52

    scanner.process_poly_update(
        {
            "event_type": "price_change",
            "price_changes": [
                {
                    "asset_id": "token-1",
                    "price": "0.49",
                    "size": "12",
                    "side": "BUY",
                    "best_bid": "0.49",
                    "best_ask": "0.52",
                }
            ],
        }
    )
    assert scanner.orderbooks["poly-b"]["bids"] == [(0.49, 12.0), (0.48, 30.0)]
    assert scanner.orderbooks["poly-b"]["best_bid"] == 0.49

    scanner.process_poly_update(
        {
            "event_type": "price_change",
            "price_changes": [
                {
                    "asset_id": "token-1",
                    "price": "0.49",
                    "size": "0",
                    "side": "BUY",
                    "best_bid": "0.48",
                    "best_ask": "0.52",
                }
            ],
        }
    )
    assert scanner.orderbooks["poly-b"]["bids"] == [(0.48, 30.0)]
    assert scanner.orderbooks["poly-b"]["best_bid"] == 0.48
