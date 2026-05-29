"""
DataService — Binance futures WebSocket feed.

Connects to three combined streams:
  btcusdt@aggTrade    → TRADE events
  btcusdt@kline_1m    → CANDLE_M1 events (from exchange)
  btcusdt@markPrice   → FUNDING events (every 3s)

Reconnects automatically with exponential backoff.
Heartbeat monitor forces reconnect if no message arrives within 30s.
"""
import asyncio
import json
import time
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from live.event_bus import Event, EventBus, EventType

_FUTURES_WS_BASE = "wss://fstream.binance.com/stream"
_STREAMS = [
    "btcusdt@aggTrade",
    "btcusdt@kline_1m",
    "btcusdt@markPrice",
]
_SYMBOL = "BTCUSDT"
_HEARTBEAT_TIMEOUT_S = 30.0
_RECONNECT_DELAY_INIT = 1.0
_RECONNECT_DELAY_MAX = 60.0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _parse_agg_trade(data: dict) -> Event:
    # m=True means buyer is market maker → aggressor is seller
    return Event(
        type=EventType.TRADE,
        ts_event=data["T"],
        ts_local=_now_ms(),
        data={
            "symbol": _SYMBOL,
            "trade_id": data["l"],
            "price": float(data["p"]),
            "qty": float(data["q"]),
            "side": "sell" if data["m"] else "buy",
        },
    )


def _parse_kline(data: dict) -> Event:
    k = data["k"]
    return Event(
        type=EventType.CANDLE_M1,
        ts_event=k["t"],
        ts_local=_now_ms(),
        data={
            "symbol": _SYMBOL,
            "tf": "M1",
            "ts_open": k["t"],
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "n_trades": int(k["n"]),
            "is_closed": bool(k["x"]),
        },
    )


def _parse_mark_price(data: dict) -> Event:
    return Event(
        type=EventType.FUNDING,
        ts_event=data.get("T", _now_ms()),
        ts_local=_now_ms(),
        data={
            "symbol": _SYMBOL,
            "mark_price": float(data["p"]),
            "funding_rate": float(data["r"]),
            "next_funding_time": data.get("T"),
        },
    )


def _parse_message(msg: dict) -> Optional[Event]:
    stream = msg.get("stream", "")
    data = msg.get("data", msg)
    if "aggTrade" in stream:
        return _parse_agg_trade(data)
    if "kline" in stream:
        return _parse_kline(data)
    if "markPrice" in stream:
        return _parse_mark_price(data)
    return None


class DataService:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._running = False
        self._last_msg_time: float = 0.0

    async def start(self) -> None:
        self._running = True
        url = _FUTURES_WS_BASE + "?streams=" + "/".join(_STREAMS)
        delay = _RECONNECT_DELAY_INIT

        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    print(f"[DataService] connected — {len(_STREAMS)} streams active")
                    self._last_msg_time = time.monotonic()
                    delay = _RECONNECT_DELAY_INIT
                    await asyncio.gather(
                        self._recv_loop(ws),
                        self._heartbeat_loop(ws),
                    )
            except asyncio.CancelledError:
                break
            except (ConnectionClosed, WebSocketException, OSError, Exception) as exc:
                print(f"[DataService] disconnected ({type(exc).__name__}: {exc}) — retry in {delay:.1f}s")

            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RECONNECT_DELAY_MAX)

    async def stop(self) -> None:
        self._running = False

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            self._last_msg_time = time.monotonic()
            try:
                event = _parse_message(json.loads(raw))
                if event:
                    await self._bus.publish(event)
            except Exception as exc:
                print(f"[DataService] parse error: {exc}")

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(5.0)
            lag = time.monotonic() - self._last_msg_time
            if lag > _HEARTBEAT_TIMEOUT_S:
                print(f"[DataService] heartbeat timeout ({lag:.1f}s) — forcing reconnect")
                await ws.close()
                return
            await self._bus.publish(Event(
                type=EventType.HEARTBEAT,
                ts_event=_now_ms(),
                ts_local=_now_ms(),
                data={"lag_s": round(lag, 2)},
            ))
