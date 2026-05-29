"""
ReplayEngine — reproduces any historical period through the EventBus.

Two sources:
  1. replay_candles()   — from live storage (data/live/market.db)
  2. replay_from_csv()  — from research dataset (data/raw/BTCUSDT_M1.csv)

Both emit identical event types to the live DataService feed.
Replay events carry _replay=True so strategies can detect test mode.

speed=0.0 → instant (all events pushed with no delay)
speed=1.0 → real-time (sleeps proportional to inter-candle interval)
speed=N   → N× faster than real-time
"""
import asyncio
import time
from pathlib import Path
from typing import Optional

from live.event_bus import Event, EventBus, EventType
from live.storage import Storage


class ReplayEngine:
    def __init__(self, bus: EventBus, storage: Optional[Storage] = None) -> None:
        self._bus     = bus
        self._storage = storage

    async def replay_candles(
        self,
        symbol: str,
        tf: str,
        ts_start: int,
        ts_end: int,
        speed: float = 0.0,
    ) -> None:
        assert self._storage is not None, "Storage required for replay_candles"
        rows = self._storage.query_candles(symbol, tf, ts_start, ts_end)
        await self._emit_rows(rows, tf, speed)

    async def replay_from_csv(
        self,
        csv_path: str,
        symbol: str = "BTCUSDT",
        tf: str = "M1",
        ts_start: Optional[int] = None,
        ts_end: Optional[int] = None,
        speed: float = 0.0,
    ) -> None:
        import pandas as pd

        df = pd.read_csv(csv_path)
        df["ts_open"] = df["open_time"].astype("int64")

        if ts_start is not None:
            df = df[df["ts_open"] >= ts_start]
        if ts_end is not None:
            df = df[df["ts_open"] < ts_end]

        rows = [
            {
                "symbol":   symbol,
                "tf":       tf,
                "ts_open":  int(row["ts_open"]),
                "open":     float(row["open"]),
                "high":     float(row["high"]),
                "low":      float(row["low"]),
                "close":    float(row["close"]),
                "volume":   float(row["volume"]),
                "n_trades": int(row.get("num_trades", 0)),
            }
            for _, row in df.iterrows()
        ]
        await self._emit_rows(rows, tf, speed)

    async def _emit_rows(self, rows: list[dict], tf: str, speed: float) -> None:
        evt_type = EventType[f"CANDLE_{tf}"]
        prev_ts: Optional[int] = None

        for row in rows:
            if speed > 0.0 and prev_ts is not None:
                delta_ms = row["ts_open"] - prev_ts
                await asyncio.sleep(delta_ms / 1000.0 / speed)
            prev_ts = row["ts_open"]

            await self._bus.publish(Event(
                type=evt_type,
                ts_event=row["ts_open"],
                ts_local=int(time.time() * 1000),
                data={**row, "is_closed": True, "_replay": True},
            ))
