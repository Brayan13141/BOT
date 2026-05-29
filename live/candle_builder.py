"""
CandleBuilder — aggregates closed M1 candles into M15, H1, H4.

Subscribes to CANDLE_M1 events. Publishes CANDLE_M15/H1/H4 on close.
A HTF candle closes when the last M1 that falls within it is received.

Alignment: uses UTC floor division so candle boundaries are always
at exact multiples of the interval (00:00, 00:15, 01:00, 04:00, ...).
"""
from dataclasses import dataclass
from typing import Optional

from live.event_bus import Event, EventBus, EventType

_TIMEFRAMES: dict[str, int] = {
    "M15": 15 * 60 * 1000,
    "H1":  60 * 60 * 1000,
    "H4":  4 * 60 * 60 * 1000,
}
_M1_MS = 60 * 1000


@dataclass
class _Bar:
    tf: str
    ts_open: int
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0
    n_trades: int = 0

    def update(self, m1: dict) -> None:
        if self.n_trades == 0:
            self.open = m1["open"]
            self.high = m1["high"]
            self.low  = m1["low"]
        else:
            self.high = max(self.high, m1["high"])
            self.low  = min(self.low,  m1["low"])
        self.close    = m1["close"]
        self.volume  += m1["volume"]
        self.n_trades += m1["n_trades"]

    def to_dict(self, symbol: str) -> dict:
        return {
            "symbol":   symbol,
            "tf":       self.tf,
            "ts_open":  self.ts_open,
            "open":     self.open,
            "high":     self.high,
            "low":      self.low,
            "close":    self.close,
            "volume":   self.volume,
            "n_trades": self.n_trades,
            "is_closed": True,
        }


class CandleBuilder:
    def __init__(self, bus: EventBus, symbol: str = "BTCUSDT") -> None:
        self._bus    = bus
        self._symbol = symbol
        self._bars: dict[str, Optional[_Bar]] = {tf: None for tf in _TIMEFRAMES}
        bus.subscribe(EventType.CANDLE_M1, self._on_m1)

    @staticmethod
    def _bar_open(tf_ms: int, m1_ts_open: int) -> int:
        return (m1_ts_open // tf_ms) * tf_ms

    async def _on_m1(self, event: Event) -> None:
        m1 = event.data
        if not m1.get("is_closed"):
            return

        m1_ts = m1["ts_open"]

        for tf, tf_ms in _TIMEFRAMES.items():
            expected_open = self._bar_open(tf_ms, m1_ts)
            bar = self._bars[tf]

            if bar is None or bar.ts_open != expected_open:
                if bar is not None:
                    await self._emit(bar, event.ts_local)
                self._bars[tf] = _Bar(tf=tf, ts_open=expected_open)

            self._bars[tf].update(m1)

            # emit HTF close when this M1 is the last in the interval
            if (m1_ts + _M1_MS) % tf_ms == 0:
                await self._emit(self._bars[tf], event.ts_local)
                self._bars[tf] = None

    async def _emit(self, bar: _Bar, ts_local: int) -> None:
        evt_type = EventType[f"CANDLE_{bar.tf}"]
        await self._bus.publish(Event(
            type=evt_type,
            ts_event=bar.ts_open,
            ts_local=ts_local,
            data=bar.to_dict(self._symbol),
        ))
