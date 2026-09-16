import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.services.arbitrage import calculate_volume_weighted_arb


def run() -> None:
    buy = {
        "asks": [(0.40, 100)],
        "bids": [(0.35, 100)],
        "best_ask": 0.40,
        "best_bid": 0.35,
        "platform": "Polymarket",
    }
    sell = {
        "asks": [(0.45, 100)],
        "bids": [(0.60, 100)],
        "best_ask": 0.45,
        "best_bid": 0.60,
        "platform": "Polymarket",
    }
    volume, roi, _, profit, buy_inv, sell_inv, _ = calculate_volume_weighted_arb(buy, sell)
    assert abs(volume - 100) < 0.01
    assert abs(buy_inv - 40) < 0.01
    assert abs(sell_inv - 40) < 0.01
    assert abs(profit - 20) < 0.01
    assert abs(roi - 0.25) < 0.01


if __name__ == "__main__":
    run()
    print("ok")
