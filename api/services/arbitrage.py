from __future__ import annotations

from typing import Any


def kalshi_fee(probability: float) -> float:
    return 0.07 * probability * (1.0 - probability)


def normalize_opposite_orderbook(orderbook: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_bid": round(1.0 - orderbook["best_ask"], 4) if orderbook.get("best_ask") else 0.0,
        "best_ask": round(1.0 - orderbook["best_bid"], 4) if orderbook.get("best_bid") else 1.0,
        "bids": sorted([(round(1.0 - price, 4), volume) for price, volume in orderbook.get("asks", [])], key=lambda item: -item[0]),
        "asks": sorted([(round(1.0 - price, 4), volume) for price, volume in orderbook.get("bids", [])], key=lambda item: item[0]),
        "platform": orderbook.get("platform", ""),
    }


def calculate_volume_weighted_arb(ob_buy: dict[str, Any], ob_sell: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    asks = ob_buy.get("asks", [])
    bids = ob_sell.get("bids", [])
    buy_platform = ob_buy.get("platform", "")
    sell_platform = ob_sell.get("platform", "")

    if not asks or not bids:
        return 0, 0, 0, 0, 0, 0, 0

    total_volume = 0.0
    total_profit = 0.0
    total_buy_inv = 0.0
    total_sell_inv = 0.0
    total_fees = 0.0

    ask_idx = 0
    bid_idx = 0
    ask_remaining = asks[0][1]
    bid_remaining = bids[0][1]

    while ask_idx < len(asks) and bid_idx < len(bids):
        ask_price, _ = asks[ask_idx]
        bid_price, _ = bids[bid_idx]

        fee_buy = kalshi_fee(ask_price) if buy_platform == "Kalshi" else 0.0
        fee_sell = kalshi_fee(1.0 - bid_price) if sell_platform == "Kalshi" else 0.0
        if bid_price - ask_price - fee_buy - fee_sell <= 0:
            break

        trade_qty = min(ask_remaining, bid_remaining)
        unit_buy_cost = ask_price + fee_buy
        unit_sell_cost = (1.0 - bid_price) + fee_sell

        total_volume += trade_qty
        total_profit += (1.0 - (unit_buy_cost + unit_sell_cost)) * trade_qty
        total_buy_inv += unit_buy_cost * trade_qty
        total_sell_inv += unit_sell_cost * trade_qty
        total_fees += (fee_buy + fee_sell) * trade_qty

        ask_remaining -= trade_qty
        bid_remaining -= trade_qty

        if ask_remaining <= 0:
            ask_idx += 1
            if ask_idx < len(asks):
                ask_remaining = asks[ask_idx][1]
        if bid_remaining <= 0:
            bid_idx += 1
            if bid_idx < len(bids):
                bid_remaining = bids[bid_idx][1]

    best_ask = asks[0][0] if asks else 0.0
    best_bid = bids[0][0] if bids else 0.0
    net_spread_pct = (best_bid - best_ask) / best_ask if best_ask > 0 else 0.0
    total_capital = total_buy_inv + total_sell_inv
    roi_pct = total_profit / total_capital if total_capital > 0 else 0.0

    return total_volume, roi_pct, net_spread_pct, total_profit, total_buy_inv, total_sell_inv, total_fees


def evaluate_arbitrage(source_id: str, source_orderbook: dict[str, Any], target_orderbook: dict[str, Any], pair_info: dict[str, Any]) -> dict[str, Any]:
    processed_target = normalize_opposite_orderbook(target_orderbook) if pair_info.get("mapping") == "Opposite" else target_orderbook

    vol1, roi1, spread1, profit1, buy_inv1, sell_inv1, fees1 = calculate_volume_weighted_arb(source_orderbook, processed_target)
    vol2, roi2, spread2, profit2, buy_inv2, sell_inv2, fees2 = calculate_volume_weighted_arb(processed_target, source_orderbook)

    best = {
        "arb_volume": 0.0,
        "roi": 0.0,
        "net_spread": 0.0,
        "net_profit": 0.0,
        "investment": 0.0,
        "k_invest": 0.0,
        "p_invest": 0.0,
        "total_fees": 0.0,
        "direction": "",
        "buy_platform": "",
        "sell_platform": "",
    }

    is_source_poly = pair_info.get("platform") == "Polymarket"

    if profit1 > profit2 and profit1 > 0:
        best.update(
            {
                "arb_volume": vol1,
                "roi": roi1,
                "net_spread": spread1,
                "net_profit": profit1,
                "investment": buy_inv1 + sell_inv1,
                "total_fees": fees1,
                "direction": f"{'Poly' if is_source_poly else 'Kalshi'}_Ask(${buy_inv1:.2f}) + {'Kalshi' if is_source_poly else 'Poly'}_Ask(${sell_inv1:.2f}) -> ${vol1:.0f}",
                "buy_platform": "Polymarket" if is_source_poly else "Kalshi",
                "sell_platform": "Kalshi" if is_source_poly else "Polymarket",
                "k_invest": sell_inv1 if is_source_poly else buy_inv1,
                "p_invest": buy_inv1 if is_source_poly else sell_inv1,
            }
        )
    elif profit2 > 0:
        best.update(
            {
                "arb_volume": vol2,
                "roi": roi2,
                "net_spread": spread2,
                "net_profit": profit2,
                "investment": buy_inv2 + sell_inv2,
                "total_fees": fees2,
                "direction": f"{'Kalshi' if is_source_poly else 'Poly'}_Ask(${buy_inv2:.2f}) + {'Poly' if is_source_poly else 'Kalshi'}_Ask(${sell_inv2:.2f}) -> ${vol2:.0f}",
                "buy_platform": "Kalshi" if is_source_poly else "Polymarket",
                "sell_platform": "Polymarket" if is_source_poly else "Kalshi",
                "k_invest": buy_inv2 if is_source_poly else sell_inv2,
                "p_invest": sell_inv2 if is_source_poly else buy_inv2,
            }
        )

    return {
        "id": source_id,
        "bid": source_orderbook.get("best_bid", 0.0),
        "ask": source_orderbook.get("best_ask", 0.0),
        "platform": pair_info.get("platform", ""),
        **best,
    }

