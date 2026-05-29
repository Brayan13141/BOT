"""
Storage — SQLite WAL persistence for live market data.

Writer runs in a dedicated daemon thread; the async _enqueue handler
puts events into a thread-safe queue without blocking the event loop.

Tables: trades, candles (all timeframes), funding.
All writes are append-only except candles (UPSERT on ts_open).
"""
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from live.event_bus import Event, EventBus, EventType

_DB_PATH = Path("data/live/market.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT    NOT NULL,
    trade_id  INTEGER,
    price     REAL    NOT NULL,
    qty       REAL    NOT NULL,
    side      TEXT    NOT NULL,
    ts_event  INTEGER NOT NULL,
    ts_local  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts_event);

CREATE TABLE IF NOT EXISTS candles (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT    NOT NULL,
    tf        TEXT    NOT NULL,
    ts_open   INTEGER NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    n_trades  INTEGER NOT NULL,
    ts_local  INTEGER NOT NULL,
    UNIQUE(symbol, tf, ts_open)
);
CREATE INDEX IF NOT EXISTS idx_candles ON candles(symbol, tf, ts_open);

CREATE TABLE IF NOT EXISTS funding (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT    NOT NULL,
    mark_price        REAL    NOT NULL,
    funding_rate      REAL    NOT NULL,
    next_funding_time INTEGER,
    ts_event          INTEGER NOT NULL,
    ts_local          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_funding_ts ON funding(ts_event);
"""

_CANDLE_TYPES = {
    EventType.CANDLE_M1,
    EventType.CANDLE_M15,
    EventType.CANDLE_H1,
    EventType.CANDLE_H4,
}
_SUBSCRIBED = _CANDLE_TYPES | {EventType.TRADE, EventType.FUNDING}

_SENTINEL = object()


class Storage:
    def __init__(self, bus: EventBus, db_path: Path = _DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=10_000)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="storage-writer")
        self._init_schema()
        self._thread.start()
        for evt_type in _SUBSCRIBED:
            bus.subscribe(evt_type, self._enqueue)

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    async def _enqueue(self, event: Event) -> None:
        try:
            self._q.put_nowait(event)
        except queue.Full:
            print("[Storage] queue full — dropping event")

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        while True:
            try:
                item = self._q.get(timeout=1.0)
                if item is _SENTINEL:
                    break
                self._write(conn, item)
                conn.commit()
            except queue.Empty:
                pass
            except Exception as exc:
                print(f"[Storage] write error: {exc}")

    def _write(self, conn: sqlite3.Connection, event: Event) -> None:
        d = event.data
        if event.type == EventType.TRADE:
            conn.execute(
                "INSERT INTO trades (symbol, trade_id, price, qty, side, ts_event, ts_local) VALUES (?,?,?,?,?,?,?)",
                (d["symbol"], d.get("trade_id"), d["price"], d["qty"], d["side"], event.ts_event, event.ts_local),
            )
        elif event.type in _CANDLE_TYPES:
            conn.execute(
                """INSERT INTO candles (symbol, tf, ts_open, open, high, low, close, volume, n_trades, ts_local)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol, tf, ts_open) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume,
                     n_trades=excluded.n_trades, ts_local=excluded.ts_local""",
                (d["symbol"], d["tf"], d["ts_open"], d["open"], d["high"],
                 d["low"], d["close"], d["volume"], d["n_trades"], event.ts_local),
            )
        elif event.type == EventType.FUNDING:
            conn.execute(
                "INSERT INTO funding (symbol, mark_price, funding_rate, next_funding_time, ts_event, ts_local) VALUES (?,?,?,?,?,?)",
                (d["symbol"], d["mark_price"], d["funding_rate"], d.get("next_funding_time"), event.ts_event, event.ts_local),
            )

    def stop(self) -> None:
        self._q.put(_SENTINEL)
        self._thread.join(timeout=5.0)

    # ── Query API (used by Replay and strategy harness) ────────────────────

    def query_candles(
        self,
        symbol: str,
        tf: str,
        ts_start: int,
        ts_end: int,
    ) -> list[dict]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM candles WHERE symbol=? AND tf=? AND ts_open>=? AND ts_open<? ORDER BY ts_open",
            (symbol, tf, ts_start, ts_end),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def query_funding(
        self,
        symbol: str,
        ts_start: int,
        ts_end: int,
    ) -> list[dict]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM funding WHERE symbol=? AND ts_event>=? AND ts_event<? ORDER BY ts_event",
            (symbol, ts_start, ts_end),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def first_ts(self, symbol: str, tf: str) -> Optional[int]:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT MIN(ts_open) FROM candles WHERE symbol=? AND tf=?", (symbol, tf)
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] is not None else None

    def last_ts(self, symbol: str, tf: str) -> Optional[int]:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT MAX(ts_open) FROM candles WHERE symbol=? AND tf=?", (symbol, tf)
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] is not None else None

    def count_candles(self, symbol: str, tf: str) -> int:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol=? AND tf=?", (symbol, tf)
        ).fetchone()
        conn.close()
        return row[0] if row else 0
