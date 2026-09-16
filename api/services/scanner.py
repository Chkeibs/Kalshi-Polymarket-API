from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import traceback
from typing import Any

import aiohttp
import pandas as pd
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from api.services.arbitrage import evaluate_arbitrage


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
MATCH_FILE = os.path.join(ROOT_DIR, "data", "matches", "confirmed_matches_bets.csv")
ID_MAP_FILE = os.path.join(CONFIG_DIR, "poly_id_map.json")

KALSHI_WS_URL = os.getenv("KALSHI_WS_URL", "wss://external-api-ws.kalshi.com/trade-api/ws/v2")
POLY_GAMMA_URL = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com/markets?id={}")
POLY_WS_URL = os.getenv("POLYMARKET_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market")


class LiveScanner:
    def __init__(self) -> None:
        self.running = False
        self.tasks: list[asyncio.Task[Any]] = []
        self.orderbooks: dict[str, dict[str, Any]] = {}
        self.mappings: dict[str, dict[str, Any]] = {}
        self.items: list[dict[str, Any]] = []
        self.p_id_to_clob_id: dict[str, str] = {}
        self.clob_id_to_p_id: dict[str, str] = {}
        self.opportunities: dict[str, dict[str, Any]] = {}
        self.last_error = ""
        self.started_at: float | None = None

    def load_matches(self) -> int:
        if not os.path.exists(MATCH_FILE):
            raise FileNotFoundError(f"Missing match file: {MATCH_FILE}")

        df = pd.read_csv(MATCH_FILE, dtype=str).fillna("")
        items: list[dict[str, Any]] = []
        self.orderbooks.clear()
        self.mappings.clear()
        self.opportunities.clear()

        for _, row in df.iterrows():
            k_id = str(row.get("Kalshi_Bet_ID", "")).strip()
            p_id = str(row.get("Polymarket_Bet_ID", "")).strip()
            if not k_id or not p_id:
                continue

            item = {
                "k_id": k_id,
                "p_id": p_id,
                "match_type": str(row.get("MatchType", "Identical")),
                "mapping": str(row.get("Mapping", "Same")),
                "kalshi_event_title": str(row.get("Kalshi_Event_Title", "")),
                "kalshi_bet_title": str(row.get("Kalshi_Bet_Title", "")),
                "polymarket_event_title": str(row.get("Polymarket_Event_Title", "")),
                "polymarket_bet_title": str(row.get("Polymarket_Bet_Title", "")),
            }
            items.append(item)

            self.mappings[k_id] = {"pair": p_id, "platform": "Kalshi", **item}
            self.mappings[p_id] = {"pair": k_id, "platform": "Polymarket", **item}
            self.orderbooks[k_id] = {"bids": [], "asks": [], "best_bid": 0.0, "best_ask": 1.0, "platform": "Kalshi"}
            self.orderbooks[p_id] = {"bids": [], "asks": [], "best_bid": 0.0, "best_ask": 1.0, "platform": "Polymarket"}

        self.items = items
        self._load_id_cache()
        return len(items)

    def _load_id_cache(self) -> None:
        if not os.path.exists(ID_MAP_FILE):
            return
        try:
            with open(ID_MAP_FILE, "r") as handle:
                cached = json.load(handle)
            self.p_id_to_clob_id.update(cached.get("p_to_clob", {}))
            self.clob_id_to_p_id.update(cached.get("clob_to_p", {}))
        except Exception:
            pass

    def _save_id_cache(self) -> None:
        os.makedirs(os.path.dirname(ID_MAP_FILE), exist_ok=True)
        with open(ID_MAP_FILE, "w") as handle:
            json.dump({"p_to_clob": self.p_id_to_clob_id, "clob_to_p": self.clob_id_to_p_id}, handle)

    async def start(self) -> dict[str, Any]:
        if self.running:
            return self.status()

        tracked = self.load_matches()
        if tracked == 0:
            raise ValueError("No matched bet pairs available in confirmed_matches_bets.csv")

        self.running = True
        self.started_at = time.time()
        self.last_error = ""
        self.tasks = [
            asyncio.create_task(self.run_kalshi(), name="kalshi-ws"),
            asyncio.create_task(self.run_polymarket(), name="polymarket-ws"),
        ]
        return self.status()

    async def stop(self) -> dict[str, Any]:
        self.running = False
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "tracked_pairs": len(self.items),
            "opportunities": len(self.opportunities),
            "started_at": self.started_at,
            "last_error": self.last_error,
        }

    def list_opportunities(self, min_profit: float = 0.0, min_roi: float = 0.0) -> list[dict[str, Any]]:
        values = [payload for payload in self.opportunities.values() if payload["net_profit"] >= min_profit and payload["roi"] >= min_roi]
        values.sort(key=lambda payload: (payload["net_profit"], payload["roi"]), reverse=True)
        return values

    def ensure_orderbook(self, ticker: str, platform: str) -> None:
        self.orderbooks.setdefault(ticker, {"bids": [], "asks": [], "best_bid": 0.0, "best_ask": 1.0, "platform": platform})

    def _get_kalshi_signer(self) -> tuple[str, Any]:
        key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
        private_key = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
        if not key_id or not private_key:
            raise ValueError("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY are required")
        if not private_key.startswith("-----BEGIN"):
            private_key = f"-----BEGIN RSA PRIVATE KEY-----\n{private_key}\n-----END RSA PRIVATE KEY-----"
        signer = serialization.load_pem_private_key(private_key.encode(), password=None)
        return key_id, signer

    async def run_kalshi(self) -> None:
        tickers = list({item["k_id"] for item in self.items})
        if not tickers:
            return
        key_id, signer = self._get_kalshi_signer()
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    timestamp = str(int(time.time() * 1000))
                    path = "/trade-api/ws/v2"
                    signature = signer.sign(
                        (timestamp + "GET" + path).encode(),
                        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                        hashes.SHA256(),
                    )
                    headers = {
                        "KALSHI-ACCESS-KEY": key_id,
                        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
                        "KALSHI-ACCESS-TIMESTAMP": timestamp,
                    }
                    async with session.ws_connect(KALSHI_WS_URL, headers=headers, heartbeat=20) as ws:
                        await ws.send_str(json.dumps({"id": int(time.time()), "cmd": "subscribe", "params": {"channels": ["orderbook_delta"], "market_tickers": tickers}}))
                        async for msg in ws:
                            if not self.running:
                                break
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            if data.get("type") == "orderbook_snapshot":
                                self.process_kalshi_snapshot(data.get("msg", {}))
                            elif data.get("type") == "orderbook_delta":
                                self.process_kalshi_delta(data.get("msg", {}))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.last_error = f"Kalshi WS: {exc}"
                    await asyncio.sleep(5)

    def process_kalshi_snapshot(self, data: dict[str, Any]) -> None:
        ticker = data.get("market_ticker")
        if not ticker:
            return
        self.ensure_orderbook(ticker, "Kalshi")
        if "yes_dollars_fp" in data or "no_dollars_fp" in data:
            yes_levels = data.get("yes_dollars_fp") or []
            no_levels = data.get("no_dollars_fp") or []
            bids = sorted([(round(float(price), 4), float(size)) for price, size in yes_levels], key=lambda item: -item[0])
            asks = sorted([(round(1.0 - float(price), 4), float(size)) for price, size in no_levels], key=lambda item: item[0])
        else:
            # Compatibility with saved legacy fixtures/messages.
            bids = sorted([(float(price) / 100.0, float(size)) for price, size in data.get("yes", [])], key=lambda item: -item[0])
            asks = sorted([((100.0 - float(price)) / 100.0, float(size)) for price, size in data.get("no", [])], key=lambda item: item[0])
        self.orderbooks[ticker].update({"bids": bids, "asks": asks, "best_bid": bids[0][0] if bids else 0.0, "best_ask": asks[0][0] if asks else 1.0})
        self.check_arbitrage(ticker)

    def process_kalshi_delta(self, data: dict[str, Any]) -> None:
        ticker = data.get("market_ticker")
        if not ticker:
            return
        self.ensure_orderbook(ticker, "Kalshi")
        side = data.get("side")
        uses_fixed_point = data.get("price_dollars") is not None or data.get("delta_fp") is not None
        price_raw = data.get("price_dollars") if uses_fixed_point else data.get("price")
        delta_qty = data.get("delta_fp") if uses_fixed_point else data.get("delta")
        if side is None or price_raw is None or delta_qty is None:
            return

        def apply_delta(levels: list[tuple[float, float]], raw_price: float, delta: float, is_yes_bid: bool) -> list[tuple[float, float]]:
            current = {price: qty for price, qty in levels}
            quoted_price = float(raw_price) if uses_fixed_point else float(raw_price) / 100.0
            price = quoted_price if is_yes_bid else 1.0 - quoted_price
            price = round(price, 4)
            new_qty = current.get(price, 0.0) + float(delta)
            if new_qty <= 0:
                current.pop(price, None)
            else:
                current[price] = new_qty
            out = [(price, qty) for price, qty in current.items()]
            out.sort(key=lambda item: -item[0] if is_yes_bid else item[0])
            return out

        if side == "yes":
            self.orderbooks[ticker]["bids"] = apply_delta(self.orderbooks[ticker]["bids"], price_raw, delta_qty, True)
        elif side == "no":
            self.orderbooks[ticker]["asks"] = apply_delta(self.orderbooks[ticker]["asks"], price_raw, delta_qty, False)
        self.orderbooks[ticker]["best_bid"] = self.orderbooks[ticker]["bids"][0][0] if self.orderbooks[ticker]["bids"] else 0.0
        self.orderbooks[ticker]["best_ask"] = self.orderbooks[ticker]["asks"][0][0] if self.orderbooks[ticker]["asks"] else 1.0
        self.check_arbitrage(ticker)

    async def fetch_clob_id(self, session: aiohttp.ClientSession, pid: str, sem: asyncio.Semaphore) -> bool:
        pid_str = str(pid).strip().split(".")[0]
        if not pid_str or pid_str == "nan":
            return False
        if len(pid_str) > 20:
            self.p_id_to_clob_id[pid] = pid_str
            self.clob_id_to_p_id[pid_str] = pid
            return True
        async with sem:
            try:
                async with session.get(POLY_GAMMA_URL.format(pid_str), timeout=5) as response:
                    if response.status != 200:
                        return False
                    data = await response.json()
                    if not data:
                        return False
                    clob_ids_str = data[0].get("clobTokenIds")
                    if not clob_ids_str:
                        return False
                    clob_ids = json.loads(clob_ids_str)
                    if not clob_ids:
                        return False
                    cid = clob_ids[0]
                    self.p_id_to_clob_id[pid] = cid
                    self.clob_id_to_p_id[cid] = pid
                    return True
            except Exception:
                return False

    async def init_polymarket_mappings(self, session: aiohttp.ClientSession) -> None:
        unique_p_ids = list({item["p_id"] for item in self.items})
        missing = [pid for pid in unique_p_ids if pid not in self.p_id_to_clob_id]
        if missing:
            sem = asyncio.Semaphore(10)
            await asyncio.gather(*(self.fetch_clob_id(session, pid, sem) for pid in missing))
            self._save_id_cache()

    async def run_polymarket(self) -> None:
        async with aiohttp.ClientSession() as session:
            await self.init_polymarket_mappings(session)
            if not self.p_id_to_clob_id:
                raise ValueError("Unable to map any Polymarket token IDs")
            while self.running:
                try:
                    async with session.ws_connect(POLY_WS_URL, heartbeat=20) as ws:
                        await ws.send_str(json.dumps(self.build_polymarket_subscription()))
                        heartbeat_task = asyncio.create_task(self._send_polymarket_heartbeats(ws))
                        try:
                            async for msg in ws:
                                if not self.running:
                                    break
                                if msg.type != aiohttp.WSMsgType.TEXT or msg.data == "PONG":
                                    continue
                                data = json.loads(msg.data)
                                updates = data if isinstance(data, list) else [data]
                                for update in updates:
                                    if update.get("event_type") == "book" or "price_changes" in update:
                                        self.process_poly_update(update)
                        finally:
                            heartbeat_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await heartbeat_task
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.last_error = f"Polymarket WS: {exc}"
                    traceback.print_exc()
                    await asyncio.sleep(5)

    def build_polymarket_subscription(self) -> dict[str, Any]:
        return {
            "assets_ids": sorted(set(self.p_id_to_clob_id.values())),
            "type": "market",
            "custom_feature_enabled": True,
        }

    async def _send_polymarket_heartbeats(self, ws: Any) -> None:
        while self.running:
            await asyncio.sleep(10)
            await ws.send_str("PING")

    def process_poly_update(self, update: dict[str, Any]) -> None:
        if "price_changes" in update:
            for price_change in update.get("price_changes", []):
                self.process_poly_price_change(price_change)
            return

        clob_id = update.get("asset_id")
        p_id = self.clob_id_to_p_id.get(clob_id)
        if not p_id:
            return

        bids = sorted([(float(level["price"]), float(level["size"])) for level in update.get("bids", [])], key=lambda item: -item[0])
        asks = sorted([(float(level["price"]), float(level["size"])) for level in update.get("asks", [])], key=lambda item: item[0])
        self.ensure_orderbook(p_id, "Polymarket")
        self.orderbooks[p_id]["bids"] = bids
        self.orderbooks[p_id]["asks"] = asks
        self.orderbooks[p_id]["best_bid"] = bids[0][0] if bids else 0.0
        self.orderbooks[p_id]["best_ask"] = asks[0][0] if asks else 1.0
        self.check_arbitrage(p_id)

    def process_poly_price_change(self, update: dict[str, Any]) -> None:
        clob_id = update.get("asset_id")
        p_id = self.clob_id_to_p_id.get(clob_id)
        if not p_id:
            return
        self.ensure_orderbook(p_id, "Polymarket")
        side = str(update.get("side", "")).upper()
        price_raw = update.get("price")
        size_raw = update.get("size")
        if side in {"BUY", "SELL"} and price_raw is not None and size_raw is not None:
            levels_key = "bids" if side == "BUY" else "asks"
            levels = {price: size for price, size in self.orderbooks[p_id][levels_key]}
            price = float(price_raw)
            size = float(size_raw)
            if size <= 0:
                levels.pop(price, None)
            else:
                levels[price] = size
            self.orderbooks[p_id][levels_key] = sorted(
                levels.items(),
                key=(lambda item: -item[0]) if side == "BUY" else (lambda item: item[0]),
            )

        best_bid = update.get("best_bid")
        best_ask = update.get("best_ask")
        if best_bid is not None:
            self.orderbooks[p_id]["best_bid"] = float(best_bid)
        elif self.orderbooks[p_id]["bids"]:
            self.orderbooks[p_id]["best_bid"] = self.orderbooks[p_id]["bids"][0][0]
        else:
            self.orderbooks[p_id]["best_bid"] = 0.0
        if best_ask is not None:
            self.orderbooks[p_id]["best_ask"] = float(best_ask)
        elif self.orderbooks[p_id]["asks"]:
            self.orderbooks[p_id]["best_ask"] = self.orderbooks[p_id]["asks"][0][0]
        else:
            self.orderbooks[p_id]["best_ask"] = 1.0
        self.check_arbitrage(p_id)

    def check_arbitrage(self, source_id: str) -> None:
        pair_info = self.mappings.get(source_id)
        if not pair_info:
            return
        target_id = pair_info["pair"]
        source_orderbook = self.orderbooks.get(source_id, {})
        target_orderbook = self.orderbooks.get(target_id, {})
        if not source_orderbook.get("bids") or not source_orderbook.get("asks") or not target_orderbook.get("bids") or not target_orderbook.get("asks"):
            return
        if min(source_orderbook["best_bid"], source_orderbook["best_ask"], target_orderbook["best_bid"], target_orderbook["best_ask"]) <= 0:
            return
        if max(source_orderbook["best_bid"], source_orderbook["best_ask"], target_orderbook["best_bid"], target_orderbook["best_ask"]) >= 1:
            return

        result = evaluate_arbitrage(source_id, source_orderbook, target_orderbook, pair_info)
        if result["net_profit"] <= 0:
            self.opportunities.pop(source_id, None)
            return

        result.update(
            {
                "pair_id": target_id,
                "match_type": pair_info.get("match_type", "Identical"),
                "mapping": pair_info.get("mapping", "Same"),
                "kalshi_event_title": pair_info.get("kalshi_event_title", ""),
                "kalshi_bet_title": pair_info.get("kalshi_bet_title", ""),
                "polymarket_event_title": pair_info.get("polymarket_event_title", ""),
                "polymarket_bet_title": pair_info.get("polymarket_bet_title", ""),
                "updated_at": time.time(),
            }
        )
        self.opportunities[source_id] = result
