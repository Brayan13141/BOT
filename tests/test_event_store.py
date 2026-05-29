"""Tests for EventStore — append-only dual-write JSONL+SQLite, query interface."""
import sqlite3
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from live.event_models import (
    OrderAcknowledged,
    OrderCancelled,
    OrderExpired,
    OrderFillReceived,
    OrderRejected,
    OrderSubmitted,
    PersistedFill,
    PersistedTransition,
    RawExchangeEvent,
)
from live.event_store import EventStore
from live.serializers import json_to_event


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(logs_dir=tmp_path)


def _make_submitted(order_id: str = "o1", ts: int = 1_001_000) -> OrderSubmitted:
    return OrderSubmitted(
        event_id=str(uuid.uuid4()),
        schema_version=1,
        event_ts_ms=ts,
        correlation_id=order_id,
        order_id=order_id,
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        qty=Decimal("0.01"),
        created_ts_ms=ts - 1000,
        limit_price=Decimal("65000"),
    )


def _make_acknowledged(order_id: str = "o1", ts: int = 1_050_000) -> OrderAcknowledged:
    return OrderAcknowledged(
        event_id=str(uuid.uuid4()),
        schema_version=1,
        event_ts_ms=ts,
        correlation_id=order_id,
        order_id=order_id,
        submitted_ts_ms=ts - 49_000,
    )


def _make_fill(order_id: str = "o1", fill_id: str = "f1",
               ts: int = 1_100_000) -> OrderFillReceived:
    return OrderFillReceived(
        event_id=str(uuid.uuid4()),
        schema_version=1,
        event_ts_ms=ts,
        correlation_id=order_id,
        order_id=order_id,
        fill_id=fill_id,
        price=Decimal("65100"),
        qty=Decimal("0.01"),
        fee=Decimal("0.1302"),
        fee_asset="USDT",
        fee_model="MAKER",
    )


def _make_transition(order_id: str = "o1", ts: int = 1_001_000,
                     from_s: str = "CREATED", to_s: str = "SUBMITTED",
                     trigger: str = "SUBMIT") -> PersistedTransition:
    return PersistedTransition(
        event_id=str(uuid.uuid4()),
        schema_version=1,
        event_ts_ms=ts,
        correlation_id=order_id,
        order_id=order_id,
        from_state=from_s,
        to_state=to_s,
        trigger=trigger,
    )


def _make_fill_ledger(order_id: str = "o1", fill_id: str = "f1",
                      ts: int = 1_100_000) -> PersistedFill:
    return PersistedFill(
        event_id=str(uuid.uuid4()),
        schema_version=1,
        event_ts_ms=ts,
        correlation_id=order_id,
        order_id=order_id,
        fill_id=fill_id,
        price=Decimal("65100"),
        qty=Decimal("0.01"),
        fee=Decimal("0.1302"),
        fee_asset="USDT",
        fee_model="MAKER",
        liquidity_role="MAKER",
    )


# ── EventStore initialisation ─────────────────────────────────────────────────

def test_event_store_creates_log_files(tmp_path: Path):
    store = EventStore(logs_dir=tmp_path)
    store.close()
    assert (tmp_path / "domain_events.jsonl").exists()
    assert (tmp_path / "transitions.jsonl").exists()
    assert (tmp_path / "fills.jsonl").exists()
    assert (tmp_path / "raw_events.jsonl").exists()


def test_event_store_creates_sqlite_db(tmp_path: Path):
    store = EventStore(logs_dir=tmp_path)
    store.close()
    assert (tmp_path / "events.db").exists()


def test_event_store_context_manager(tmp_path: Path):
    with EventStore(logs_dir=tmp_path) as store:
        store.append_domain_event(_make_submitted())
    # After exiting context manager, files should be flushed and closed
    assert (tmp_path / "domain_events.jsonl").exists()


# ── append_domain_event ───────────────────────────────────────────────────────

def test_append_domain_event_writes_to_jsonl(store: EventStore, tmp_path: Path):
    evt = _make_submitted(order_id="o1")
    store.append_domain_event(evt)
    lines = (tmp_path / "domain_events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    restored = json_to_event(lines[0])
    assert isinstance(restored, OrderSubmitted)
    assert restored.order_id == "o1"


def test_append_domain_event_writes_to_sqlite(store: EventStore, tmp_path: Path):
    evt = _make_submitted(order_id="o1")
    store.append_domain_event(evt)
    conn = sqlite3.connect(tmp_path / "events.db")
    row = conn.execute("SELECT event_id, event_type, order_id FROM domain_events").fetchone()
    conn.close()
    assert row[0] == evt.event_id
    assert row[1] == "OrderSubmitted"
    assert row[2] == "o1"


def test_append_multiple_domain_events_ordered(store: EventStore, tmp_path: Path):
    e1 = _make_submitted(order_id="o1", ts=1_001_000)
    e2 = _make_acknowledged(order_id="o1", ts=1_050_000)
    e3 = _make_fill(order_id="o1", ts=1_100_000)
    store.append_domain_event(e1)
    store.append_domain_event(e2)
    store.append_domain_event(e3)
    lines = (tmp_path / "domain_events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3


def test_append_domain_event_dual_write_consistent(store: EventStore, tmp_path: Path):
    """JSONL and SQLite must agree on event_id and payload."""
    evt = _make_fill(order_id="o1", fill_id="f1")
    store.append_domain_event(evt)
    # JSONL
    jsonl_line = (tmp_path / "domain_events.jsonl").read_text().strip()
    restored_jsonl = json_to_event(jsonl_line)
    # SQLite
    conn = sqlite3.connect(tmp_path / "events.db")
    row = conn.execute("SELECT event_id, payload_json FROM domain_events").fetchone()
    conn.close()
    restored_sqlite = json_to_event(row[1])
    assert restored_jsonl.event_id == restored_sqlite.event_id
    assert restored_jsonl == restored_sqlite


# ── append_transition ─────────────────────────────────────────────────────────

def test_append_transition_writes_to_jsonl(store: EventStore, tmp_path: Path):
    t = _make_transition(order_id="o1")
    store.append_transition(t)
    lines = (tmp_path / "transitions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    restored = json_to_event(lines[0])
    assert isinstance(restored, PersistedTransition)
    assert restored.trigger == "SUBMIT"


def test_append_transition_writes_to_sqlite(store: EventStore, tmp_path: Path):
    t = _make_transition(order_id="o1", from_s="ACKNOWLEDGED", to_s="FILLED",
                         trigger="FULL_FILL")
    store.append_transition(t)
    conn = sqlite3.connect(tmp_path / "events.db")
    row = conn.execute(
        "SELECT order_id, from_state, to_state, trigger FROM order_transitions"
    ).fetchone()
    conn.close()
    assert row == ("o1", "ACKNOWLEDGED", "FILLED", "FULL_FILL")


# ── append_fill ───────────────────────────────────────────────────────────────

def test_append_fill_writes_to_jsonl(store: EventStore, tmp_path: Path):
    f = _make_fill_ledger(order_id="o1")
    store.append_fill(f)
    lines = (tmp_path / "fills.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    restored = json_to_event(lines[0])
    assert isinstance(restored, PersistedFill)


def test_append_fill_writes_to_sqlite(store: EventStore, tmp_path: Path):
    f = _make_fill_ledger(order_id="o1", fill_id="f99")
    store.append_fill(f)
    conn = sqlite3.connect(tmp_path / "events.db")
    row = conn.execute("SELECT order_id, fill_id, price FROM fills").fetchone()
    conn.close()
    assert row[0] == "o1"
    assert row[1] == "f99"
    assert row[2] == "65100"  # stored as TEXT (Decimal str)


# ── append_raw_event ──────────────────────────────────────────────────────────

def test_append_raw_event_writes_to_jsonl(store: EventStore, tmp_path: Path):
    r = RawExchangeEvent(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=1_000_000,
        exchange="binance", stream="executionReport",
        local_receive_ts_ms=1_000_050,
        payload_json='{"e":"executionReport"}',
    )
    store.append_raw_event(r)
    lines = (tmp_path / "raw_events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_append_raw_event_writes_to_sqlite(store: EventStore, tmp_path: Path):
    r = RawExchangeEvent(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=1_000_000,
        exchange="binance", stream="executionReport",
        local_receive_ts_ms=1_000_050,
        payload_json='{"test":true}',
    )
    store.append_raw_event(r)
    conn = sqlite3.connect(tmp_path / "events.db")
    row = conn.execute("SELECT exchange, stream FROM raw_events").fetchone()
    conn.close()
    assert row == ("binance", "executionReport")


# ── query interface ───────────────────────────────────────────────────────────

def test_get_domain_events_by_order_returns_in_ts_order(store: EventStore):
    e1 = _make_submitted(order_id="o1", ts=1_001_000)
    e2 = _make_acknowledged(order_id="o1", ts=1_050_000)
    e3 = _make_fill(order_id="o1", ts=1_100_000)
    store.append_domain_event(e3)  # append out of ts order
    store.append_domain_event(e1)
    store.append_domain_event(e2)
    events = store.get_domain_events_by_order("o1")
    assert len(events) == 3
    assert events[0].event_ts_ms == 1_001_000
    assert events[1].event_ts_ms == 1_050_000
    assert events[2].event_ts_ms == 1_100_000


def test_get_domain_events_by_order_filters_by_order_id(store: EventStore):
    store.append_domain_event(_make_submitted(order_id="o1", ts=1_001_000))
    store.append_domain_event(_make_submitted(order_id="o2", ts=1_002_000))
    events = store.get_domain_events_by_order("o1")
    assert len(events) == 1
    assert isinstance(events[0], OrderSubmitted)
    assert events[0].order_id == "o1"


def test_get_domain_events_by_order_empty_returns_empty_list(store: EventStore):
    events = store.get_domain_events_by_order("nonexistent-order")
    assert events == []


def test_get_transitions_by_order(store: EventStore):
    t1 = _make_transition("o1", ts=1_001_000, from_s="CREATED", to_s="SUBMITTED",
                           trigger="SUBMIT")
    t2 = _make_transition("o1", ts=1_050_000, from_s="SUBMITTED",
                           to_s="ACKNOWLEDGED", trigger="ACK_RECEIVED")
    store.append_transition(t1)
    store.append_transition(t2)
    transitions = store.get_transitions_by_order("o1")
    assert len(transitions) == 2
    assert transitions[0].trigger == "SUBMIT"
    assert transitions[1].trigger == "ACK_RECEIVED"


def test_get_fills_by_order(store: EventStore):
    f1 = _make_fill_ledger("o1", fill_id="f1", ts=1_100_000)
    f2 = _make_fill_ledger("o1", fill_id="f2", ts=1_200_000)
    store.append_fill(f1)
    store.append_fill(f2)
    fills = store.get_fills_by_order("o1")
    assert len(fills) == 2
    assert fills[0].fill_id == "f1"
    assert fills[1].fill_id == "f2"
    assert isinstance(fills[0].price, Decimal)


def test_get_fills_by_order_filters_by_order_id(store: EventStore):
    store.append_fill(_make_fill_ledger("o1", fill_id="f1"))
    store.append_fill(_make_fill_ledger("o2", fill_id="f2"))
    fills = store.get_fills_by_order("o1")
    assert len(fills) == 1
    assert fills[0].fill_id == "f1"


# ── duplicate event_id raises ─────────────────────────────────────────────────

def test_duplicate_domain_event_id_raises(store: EventStore):
    """Duplicate event_id = bug in the system. Must not be silently ignored."""
    evt = _make_submitted(order_id="o1")
    store.append_domain_event(evt)
    with pytest.raises(Exception):  # sqlite3.IntegrityError
        store.append_domain_event(evt)
