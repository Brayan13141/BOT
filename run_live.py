"""
Phase 1A entry point — starts live market data collection.

What runs:
  DataService  — connects to Binance futures WebSocket, parses raw messages
  CandleBuilder — aggregates M1 → M15, H1, H4 on candle close
  Storage      — writes everything to data/live/market.db (SQLite WAL)

Press Ctrl+C to stop gracefully.
"""
import asyncio

from live.event_bus import EventBus, EventType, Event
from live.data_service import DataService
from live.candle_builder import CandleBuilder
from live.storage import Storage


async def _log_candle(event: Event) -> None:
    d = event.data
    if d.get("is_closed"):
        import datetime
        ts = datetime.datetime.utcfromtimestamp(d["ts_open"] / 1000).strftime("%Y-%m-%d %H:%M")
        print(
            f"[{d['tf']:>3}] {ts}  "
            f"O={d['open']:>10.2f}  H={d['high']:>10.2f}  "
            f"L={d['low']:>10.2f}  C={d['close']:>10.2f}  "
            f"V={d['volume']:>10.4f}  n={d['n_trades']}"
        )


async def _log_funding(event: Event) -> None:
    d = event.data
    print(f"[FND] mark={d['mark_price']:.2f}  rate={d['funding_rate']:.6f}")


async def main() -> None:
    bus     = EventBus()
    storage = Storage(bus)
    _builder = CandleBuilder(bus)
    service = DataService(bus)

    for tf in ("M1", "M15", "H1", "H4"):
        bus.subscribe(EventType[f"CANDLE_{tf}"], _log_candle)
    bus.subscribe(EventType.FUNDING, _log_funding)

    print("[run_live] starting — Ctrl+C to stop")
    print(f"[run_live] storing to: data/live/market.db")

    try:
        await service.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await service.stop()
        storage.stop()
        print("[run_live] stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
