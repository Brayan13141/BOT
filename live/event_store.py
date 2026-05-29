"""
Event Store — append-only dual-write JSONL + SQLite for execution events.

Architecture:
    - Four independent streams: raw_events, domain_events, transitions, fills.
    - Each stream writes to both a .jsonl file and an SQLite table simultaneously.
    - JSONL: canonical, portable, corruption-recoverable, exact replay.
    - SQLite: indexed for queries (by order_id, by event_ts_ms).
    - Never UPDATE. Never DELETE. Append-only.

Usage:
    store = EventStore(logs_dir=Path("logs"))
    store.append_domain_event(OrderSubmitted(...))
    store.close()

    # Or as a context manager:
    with EventStore(logs_dir=Path("logs")) as store:
        store.append_domain_event(...)

Query:
    events = store.get_domain_events_by_order("order-001")

Separation of concerns:
    - This module does NOT know about Order objects.
    - It accepts only serializable event dataclasses from event_models.py.
    - Serialization is delegated to serializers.py.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from live.event_models import (
    BaseEvent,
    CostLedgerEntry,
    PersistedFill,
    PersistedTransition,
    RawExchangeEvent,
)
from live.serializers import event_to_json, json_to_event

# SQLite schema — all Decimal fields stored as TEXT to preserve precision.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id             TEXT    NOT NULL UNIQUE,
    schema_version       INTEGER NOT NULL DEFAULT 1,
    exchange             TEXT    NOT NULL,
    stream               TEXT    NOT NULL,
    event_ts_ms          INTEGER NOT NULL,
    local_receive_ts_ms  INTEGER NOT NULL,
    payload_json         TEXT    NOT NULL,
    checksum             TEXT,
    causation_id         TEXT,
    correlation_id       TEXT,
    created_at           INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS domain_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT    NOT NULL UNIQUE,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    event_type      TEXT    NOT NULL,
    order_id        TEXT,
    event_ts_ms     INTEGER NOT NULL,
    causation_id    TEXT,
    correlation_id  TEXT,
    payload_json    TEXT    NOT NULL,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_domain_events_order
    ON domain_events(order_id, event_ts_ms);

CREATE TABLE IF NOT EXISTS order_transitions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT    NOT NULL UNIQUE,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    order_id        TEXT    NOT NULL,
    from_state      TEXT    NOT NULL,
    to_state        TEXT    NOT NULL,
    trigger         TEXT    NOT NULL,
    event_ts_ms     INTEGER NOT NULL,
    causation_id    TEXT,
    correlation_id  TEXT,
    payload_json    TEXT    NOT NULL,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transitions_order
    ON order_transitions(order_id, event_ts_ms);

CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT    NOT NULL UNIQUE,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    order_id        TEXT    NOT NULL,
    fill_id         TEXT    NOT NULL UNIQUE,
    price           TEXT    NOT NULL,
    qty             TEXT    NOT NULL,
    fee             TEXT    NOT NULL,
    fee_asset       TEXT    NOT NULL,
    fee_model       TEXT    NOT NULL,
    liquidity_role  TEXT    NOT NULL,
    trade_id        TEXT,
    event_ts_ms     INTEGER NOT NULL,
    causation_id    TEXT,
    correlation_id  TEXT,
    payload_json    TEXT    NOT NULL,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_order
    ON fills(order_id, event_ts_ms);

CREATE TABLE IF NOT EXISTS cost_ledger (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id         TEXT    NOT NULL UNIQUE,
    schema_version   INTEGER NOT NULL DEFAULT 1,
    order_id         TEXT    NOT NULL,
    fill_id          TEXT    NOT NULL UNIQUE,
    side             TEXT    NOT NULL,
    fill_price       TEXT    NOT NULL,
    fill_qty         TEXT    NOT NULL,
    arrival_price    TEXT    NOT NULL,
    slippage         TEXT    NOT NULL,
    fee_model        TEXT    NOT NULL,
    maker_rebate     TEXT    NOT NULL,
    taker_fee        TEXT    NOT NULL,
    latency_cost     TEXT    NOT NULL,
    inventory_cost   TEXT    NOT NULL,
    net_cost         TEXT    NOT NULL,
    event_ts_ms      INTEGER NOT NULL,
    causation_id     TEXT,
    correlation_id   TEXT,
    payload_json     TEXT    NOT NULL,
    created_at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cost_ledger_order
    ON cost_ledger(order_id, event_ts_ms);
"""


class EventStore:
    """
    Append-only dual-write store for execution domain events.

    Thread safety: not thread-safe. Designed for single-threaded paper trading.
    For multi-threaded use, add a threading.Lock around append methods.
    """

    def __init__(self, logs_dir: Path) -> None:
        self._logs_dir = logs_dir
        self._closed = False
        logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            # JSONL file handles (append mode, line-buffered)
            self._f_raw    = open(logs_dir / "raw_events.jsonl",    "a", encoding="utf-8")
            self._f_domain = open(logs_dir / "domain_events.jsonl", "a", encoding="utf-8")
            self._f_trans  = open(logs_dir / "transitions.jsonl",   "a", encoding="utf-8")
            self._f_fills  = open(logs_dir / "fills.jsonl",         "a", encoding="utf-8")
            self._f_costs  = open(logs_dir / "cost_ledger.jsonl",   "a", encoding="utf-8")

            # SQLite connection (WAL mode for better read concurrency)
            self._conn = sqlite3.connect(logs_dir / "events.db")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except Exception:
            self.close()
            raise

    # ── Append interface ──────────────────────────────────────────────────────

    def append_raw_event(self, event: RawExchangeEvent) -> None:
        """Persist an exact exchange payload (pre-normalisation)."""
        json_str = event_to_json(event)
        # SQLite first: if INSERT fails (e.g. duplicate event_id), JSONL is not written
        self._conn.execute(
            """INSERT INTO raw_events
               (event_id, schema_version, exchange, stream,
                event_ts_ms, local_receive_ts_ms, payload_json, checksum,
                causation_id, correlation_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.event_id, event.schema_version,
                event.exchange, event.stream,
                event.event_ts_ms, event.local_receive_ts_ms,
                event.payload_json, event.checksum,
                event.causation_id, event.correlation_id,
                _now_ms(),
            ),
        )
        self._conn.commit()
        self._f_raw.write(json_str + "\n")
        self._f_raw.flush()

    def append_domain_event(self, event: BaseEvent) -> None:
        """
        Persist a normalised domain event (OrderSubmitted, OrderFillReceived, etc.).

        Extracts order_id from the event (if present) for indexing.
        Full event stored in payload_json for exact round-trip reconstruction.
        """
        json_str = event_to_json(event)
        order_id: str | None = getattr(event, "order_id", None)
        # SQLite first: if INSERT fails (e.g. duplicate event_id), JSONL is not written
        self._conn.execute(
            """INSERT INTO domain_events
               (event_id, schema_version, event_type, order_id, event_ts_ms,
                causation_id, correlation_id, payload_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                event.event_id, event.schema_version,
                type(event).__name__, order_id, event.event_ts_ms,
                event.causation_id, event.correlation_id,
                json_str, _now_ms(),
            ),
        )
        self._conn.commit()
        self._f_domain.write(json_str + "\n")
        self._f_domain.flush()

    def append_transition(self, event: PersistedTransition) -> None:
        """Persist a state machine transition (FSM audit log)."""
        json_str = event_to_json(event)
        # SQLite first: if INSERT fails (e.g. duplicate event_id), JSONL is not written
        self._conn.execute(
            """INSERT INTO order_transitions
               (event_id, schema_version, order_id, from_state, to_state, trigger,
                event_ts_ms, causation_id, correlation_id, payload_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.event_id, event.schema_version,
                event.order_id, event.from_state, event.to_state, event.trigger,
                event.event_ts_ms, event.causation_id, event.correlation_id,
                json_str, _now_ms(),
            ),
        )
        self._conn.commit()
        self._f_trans.write(json_str + "\n")
        self._f_trans.flush()

    def append_fill(self, event: PersistedFill) -> None:
        """Persist a fill ledger entry (financial audit)."""
        json_str = event_to_json(event)
        # SQLite first: if INSERT fails (e.g. duplicate event_id), JSONL is not written
        self._conn.execute(
            """INSERT INTO fills
               (event_id, schema_version, order_id, fill_id,
                price, qty, fee, fee_asset, fee_model, liquidity_role, trade_id,
                event_ts_ms, causation_id, correlation_id, payload_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.event_id, event.schema_version,
                event.order_id, event.fill_id,
                str(event.price), str(event.qty), str(event.fee),
                event.fee_asset, event.fee_model, event.liquidity_role, event.trade_id,
                event.event_ts_ms, event.causation_id, event.correlation_id,
                json_str, _now_ms(),
            ),
        )
        self._conn.commit()
        self._f_fills.write(json_str + "\n")
        self._f_fills.flush()

    def append_cost_entry(self, event: CostLedgerEntry) -> None:
        """Persist an economic cost record for a single fill."""
        json_str = event_to_json(event)
        self._conn.execute(
            """INSERT INTO cost_ledger
               (event_id, schema_version, order_id, fill_id, side,
                fill_price, fill_qty, arrival_price, slippage,
                fee_model, maker_rebate, taker_fee, latency_cost, inventory_cost,
                net_cost, event_ts_ms, causation_id, correlation_id,
                payload_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.event_id, event.schema_version,
                event.order_id, event.fill_id, event.side,
                str(event.fill_price), str(event.fill_qty),
                str(event.arrival_price), str(event.slippage),
                event.fee_model,
                str(event.maker_rebate), str(event.taker_fee),
                str(event.latency_cost), str(event.inventory_cost),
                str(event.net_cost),
                event.event_ts_ms, event.causation_id, event.correlation_id,
                json_str, _now_ms(),
            ),
        )
        self._conn.commit()
        self._f_costs.write(json_str + "\n")
        self._f_costs.flush()

    def get_cost_entries_by_order(self, order_id: str) -> list[CostLedgerEntry]:
        """Return all cost entries for an order, ordered by event_ts_ms ASC."""
        cur = self._conn.execute(
            """SELECT payload_json FROM cost_ledger
               WHERE order_id = ?
               ORDER BY event_ts_ms ASC, id ASC""",
            (order_id,),
        )
        return [json_to_event(row[0]) for row in cur]  # type: ignore[return-value]

    # ── Query interface ───────────────────────────────────────────────────────

    def get_domain_events_by_order(self, order_id: str) -> list[BaseEvent]:
        """
        Return all domain events for an order, ordered by event_ts_ms ASC.
        Ties broken by insertion order (id ASC). Used by ExecutionReplay.
        """
        cur = self._conn.execute(
            """SELECT payload_json FROM domain_events
               WHERE order_id = ?
               ORDER BY event_ts_ms ASC, id ASC""",
            (order_id,),
        )
        return [json_to_event(row[0]) for row in cur]

    def get_transitions_by_order(self, order_id: str) -> list[PersistedTransition]:
        """Return all transitions for an order, ordered by event_ts_ms ASC."""
        cur = self._conn.execute(
            """SELECT payload_json FROM order_transitions
               WHERE order_id = ?
               ORDER BY event_ts_ms ASC, id ASC""",
            (order_id,),
        )
        return [json_to_event(row[0]) for row in cur]  # type: ignore[return-value]

    def get_fills_by_order(self, order_id: str) -> list[PersistedFill]:
        """Return all fills for an order, ordered by event_ts_ms ASC."""
        cur = self._conn.execute(
            """SELECT payload_json FROM fills
               WHERE order_id = ?
               ORDER BY event_ts_ms ASC, id ASC""",
            (order_id,),
        )
        return [json_to_event(row[0]) for row in cur]  # type: ignore[return-value]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for f in (self._f_raw, self._f_domain, self._f_trans, self._f_fills, self._f_costs):
            if not f.closed:
                f.flush()
                f.close()
        if hasattr(self, "_conn"):
            self._conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _now_ms() -> int:
    return int(time.time() * 1000)
