# Event-Sourced Execution Logging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build append-only dual-write (JSONL + SQLite) event logging for execution events, enabling deterministic replay, audit trail, post-mortem debugging, and reconciliation with the exchange.

**Architecture:** Four persisted event streams — raw exchange events (exact bytes), domain events (OrderSubmitted, OrderAcknowledged, OrderFillReceived, OrderCancelled, OrderExpired, OrderRejected), state transitions (FSM audit log), and a fill ledger (financial entries). The OSM stays pure/stateless — no disk writes inside it. The EventStore is a separate append-only service. ExecutionReplay reconstructs Order state by replaying domain events through a fresh OSM instance. All financial quantities use Decimal serialized as str.

**Tech Stack:** Python 3.12, stdlib only: `dataclasses`, `decimal`, `json`, `sqlite3`, `pathlib`, `uuid`, `time`. No third-party dependencies. `pytest 8.0+` for tests.

**Architectural constraints (from design session):**
- `storage.py` owns market data (trades/candles/funding). `event_store.py` owns execution events. Never mix.
- `event_store.py` never accepts `Order` objects — only serializable event dataclasses.
- `replay.py` owns market data replay. `execution_replay.py` owns execution event replay.
- Append-only: no `UPDATE`, no `DELETE`, no overwrite.
- Decimal MUST be serialized as `str(d)` and deserialized as `Decimal(s)`. Float prohibited.
- `schema_version: int = 1` on every event. Increment only on breaking schema changes.
- `causation_id` + `correlation_id` on every event. `correlation_id` = `order_id` for all order events.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `live/event_models.py` | **Create** | Frozen dataclasses: `BaseEvent`, `RawExchangeEvent`, 6 domain events, `PersistedTransition`, `PersistedFill` |
| `live/serializers.py` | **Create** | `event_to_json()`, `json_to_event()`, Decimal registry, event type registry |
| `live/event_store.py` | **Create** | `EventStore`: dual-write JSONL+SQLite, append + query interface |
| `live/execution_replay.py` | **Create** | `ExecutionReplay`: reconstruct `Order` from domain events via OSM |
| `tests/test_event_models.py` | **Create** | Immutability, field types, all event types |
| `tests/test_serializers.py` | **Create** | JSON round-trip for all event types, Decimal precision |
| `tests/test_event_store.py` | **Create** | Append + query for all 4 streams, dual-write consistency |
| `tests/test_execution_replay.py` | **Create** | Order reconstruction, determinism, all lifecycle paths |

---

## Valid Transition Graph (reference for replay)

```
OrderSubmitted   → osm.create_order() + osm.submit()
OrderAcknowledged → osm.acknowledge()
OrderFillReceived → osm.fill()
OrderCancelled   → osm.cancel()
OrderExpired     → osm.expire()
OrderRejected    → osm.reject()
```

---

## Task 1: event_models.py — frozen dataclasses for all persisted events

**Files:** `live/event_models.py` (create), `tests/test_event_models.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_event_models.py`:
```python
"""Tests for event_models.py — frozen dataclasses, field types, BaseEvent contract."""
from decimal import Decimal

import pytest

from live.event_models import (
    BaseEvent,
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


# ── BaseEvent contract ────────────────────────────────────────────────────────

def test_base_event_fields():
    """BaseEvent has the five required fields."""
    e = BaseEvent(
        event_id="evt-001",
        schema_version=1,
        event_ts_ms=1_000_000,
    )
    assert e.event_id == "evt-001"
    assert e.schema_version == 1
    assert e.event_ts_ms == 1_000_000
    assert e.causation_id is None
    assert e.correlation_id is None


def test_base_event_is_immutable():
    e = BaseEvent(event_id="evt-001", schema_version=1, event_ts_ms=1_000_000)
    with pytest.raises((AttributeError, TypeError)):
        e.event_id = "tampered"  # type: ignore[misc]


def test_base_event_with_optional_ids():
    e = BaseEvent(
        event_id="evt-002",
        schema_version=1,
        event_ts_ms=1_000_000,
        causation_id="cause-001",
        correlation_id="order-001",
    )
    assert e.causation_id == "cause-001"
    assert e.correlation_id == "order-001"


# ── RawExchangeEvent ─────────────────────────────────────────────────────────

def test_raw_exchange_event_fields():
    e = RawExchangeEvent(
        event_id="raw-001",
        schema_version=1,
        event_ts_ms=1_000_000,
        exchange="binance",
        stream="executionReport",
        local_receive_ts_ms=1_000_050,
        payload_json='{"e":"executionReport","s":"BTCUSDT"}',
    )
    assert e.exchange == "binance"
    assert e.stream == "executionReport"
    assert e.local_receive_ts_ms == 1_000_050
    assert e.checksum is None


def test_raw_exchange_event_is_immutable():
    e = RawExchangeEvent(
        event_id="raw-001", schema_version=1, event_ts_ms=1_000_000,
        exchange="binance", stream="s", local_receive_ts_ms=1_000_050,
        payload_json='{}',
    )
    with pytest.raises((AttributeError, TypeError)):
        e.payload_json = "tampered"  # type: ignore[misc]


# ── OrderSubmitted ────────────────────────────────────────────────────────────

def test_order_submitted_fields():
    e = OrderSubmitted(
        event_id="evt-003",
        schema_version=1,
        event_ts_ms=1_001_000,
        correlation_id="order-001",
        order_id="order-001",
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        qty=Decimal("0.01"),
        created_ts_ms=1_000_000,
        limit_price=Decimal("65000"),
    )
    assert e.order_id == "order-001"
    assert e.symbol == "BTCUSDT"
    assert e.side == "BUY"
    assert isinstance(e.qty, Decimal)
    assert isinstance(e.limit_price, Decimal)
    assert e.created_ts_ms == 1_000_000


def test_order_submitted_limit_price_optional():
    e = OrderSubmitted(
        event_id="e", schema_version=1, event_ts_ms=1_001_000,
        order_id="o1", symbol="BTCUSDT", side="BUY",
        order_type="MARKET", qty=Decimal("0.01"), created_ts_ms=1_000_000,
    )
    assert e.limit_price is None


def test_order_submitted_is_immutable():
    e = OrderSubmitted(
        event_id="e", schema_version=1, event_ts_ms=1_001_000,
        order_id="o1", symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"), created_ts_ms=1_000_000,
        limit_price=Decimal("65000"),
    )
    with pytest.raises((AttributeError, TypeError)):
        e.order_id = "tampered"  # type: ignore[misc]


# ── OrderAcknowledged ─────────────────────────────────────────────────────────

def test_order_acknowledged_fields():
    e = OrderAcknowledged(
        event_id="e", schema_version=1, event_ts_ms=1_050_000,
        order_id="o1", submitted_ts_ms=1_001_000,
    )
    assert e.order_id == "o1"
    assert e.submitted_ts_ms == 1_001_000
    assert e.event_ts_ms == 1_050_000


# ── OrderFillReceived ─────────────────────────────────────────────────────────

def test_order_fill_received_fields():
    e = OrderFillReceived(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1",
        fill_id="fill-001",
        price=Decimal("65100.50"),
        qty=Decimal("0.01"),
        fee=Decimal("0.001302010"),
        fee_asset="USDT",
        fee_model="MAKER",
    )
    assert e.fill_id == "fill-001"
    assert isinstance(e.price, Decimal)
    assert isinstance(e.qty, Decimal)
    assert isinstance(e.fee, Decimal)
    assert e.fee_model == "MAKER"


def test_order_fill_received_is_immutable():
    e = OrderFillReceived(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1", fill_id="f1", price=Decimal("65000"),
        qty=Decimal("0.01"), fee=Decimal("0.13"), fee_asset="USDT",
        fee_model="MAKER",
    )
    with pytest.raises((AttributeError, TypeError)):
        e.price = Decimal("0")  # type: ignore[misc]


# ── OrderCancelled ────────────────────────────────────────────────────────────

def test_order_cancelled_fields():
    e = OrderCancelled(
        event_id="e", schema_version=1, event_ts_ms=1_200_000,
        order_id="o1", trigger="USER_CANCEL",
    )
    assert e.trigger == "USER_CANCEL"


# ── OrderExpired ──────────────────────────────────────────────────────────────

def test_order_expired_fields():
    e = OrderExpired(
        event_id="e", schema_version=1, event_ts_ms=1_500_000,
        order_id="o1",
    )
    assert e.order_id == "o1"


# ── OrderRejected ─────────────────────────────────────────────────────────────

def test_order_rejected_fields():
    e = OrderRejected(
        event_id="e", schema_version=1, event_ts_ms=1_002_000,
        order_id="o1",
    )
    assert e.reason is None


def test_order_rejected_with_reason():
    e = OrderRejected(
        event_id="e", schema_version=1, event_ts_ms=1_002_000,
        order_id="o1", reason="INSUFFICIENT_BALANCE",
    )
    assert e.reason == "INSUFFICIENT_BALANCE"


# ── PersistedTransition ───────────────────────────────────────────────────────

def test_persisted_transition_fields():
    e = PersistedTransition(
        event_id="e", schema_version=1, event_ts_ms=1_001_000,
        order_id="o1",
        from_state="CREATED",
        to_state="SUBMITTED",
        trigger="SUBMIT",
    )
    assert e.from_state == "CREATED"
    assert e.to_state == "SUBMITTED"
    assert e.trigger == "SUBMIT"
    assert e.bid_price is None
    assert e.ask_price is None


def test_persisted_transition_with_spread_snapshot():
    e = PersistedTransition(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1",
        from_state="ACKNOWLEDGED",
        to_state="FILLED",
        trigger="FULL_FILL",
        bid_price=Decimal("64999"),
        ask_price=Decimal("65001"),
    )
    assert isinstance(e.bid_price, Decimal)
    assert isinstance(e.ask_price, Decimal)


def test_persisted_transition_is_immutable():
    e = PersistedTransition(
        event_id="e", schema_version=1, event_ts_ms=1_001_000,
        order_id="o1", from_state="CREATED", to_state="SUBMITTED",
        trigger="SUBMIT",
    )
    with pytest.raises((AttributeError, TypeError)):
        e.trigger = "TAMPERED"  # type: ignore[misc]


# ── PersistedFill ─────────────────────────────────────────────────────────────

def test_persisted_fill_fields():
    e = PersistedFill(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1",
        fill_id="fill-001",
        price=Decimal("65100"),
        qty=Decimal("0.01"),
        fee=Decimal("0.1302"),
        fee_asset="USDT",
        fee_model="MAKER",
        liquidity_role="MAKER",
    )
    assert e.fill_id == "fill-001"
    assert isinstance(e.price, Decimal)
    assert isinstance(e.qty, Decimal)
    assert isinstance(e.fee, Decimal)
    assert e.fee_model == "MAKER"
    assert e.liquidity_role == "MAKER"
    assert e.trade_id is None


def test_persisted_fill_with_trade_id():
    e = PersistedFill(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1", fill_id="f1", price=Decimal("65000"),
        qty=Decimal("0.01"), fee=Decimal("0.13"), fee_asset="USDT",
        fee_model="TAKER", liquidity_role="TAKER", trade_id="trade-12345",
    )
    assert e.trade_id == "trade-12345"


def test_persisted_fill_is_immutable():
    e = PersistedFill(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1", fill_id="f1", price=Decimal("65000"),
        qty=Decimal("0.01"), fee=Decimal("0.13"), fee_asset="USDT",
        fee_model="MAKER", liquidity_role="MAKER",
    )
    with pytest.raises((AttributeError, TypeError)):
        e.price = Decimal("0")  # type: ignore[misc]


# ── schema_version is always 1 for v1 events ─────────────────────────────────

def test_all_events_have_schema_version_1():
    """All v1 events must carry schema_version=1 for forward compatibility."""
    events = [
        BaseEvent(event_id="e", schema_version=1, event_ts_ms=1_000_000),
        OrderSubmitted(event_id="e", schema_version=1, event_ts_ms=1_001_000,
                       order_id="o1", symbol="BTCUSDT", side="BUY",
                       order_type="LIMIT", qty=Decimal("0.01"),
                       created_ts_ms=1_000_000, limit_price=Decimal("65000")),
        OrderAcknowledged(event_id="e", schema_version=1, event_ts_ms=1_050_000,
                          order_id="o1", submitted_ts_ms=1_001_000),
        PersistedFill(event_id="e", schema_version=1, event_ts_ms=1_100_000,
                      order_id="o1", fill_id="f1", price=Decimal("65000"),
                      qty=Decimal("0.01"), fee=Decimal("0.13"), fee_asset="USDT",
                      fee_model="MAKER", liquidity_role="MAKER"),
    ]
    for e in events:
        assert e.schema_version == 1
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_event_models.py -v`

Expected: `ModuleNotFoundError: No module named 'live.event_models'`

- [ ] **Step 3: Create `live/event_models.py`**

```python
"""
Domain event models for execution event sourcing.

All events are immutable frozen dataclasses.
All financial quantities (Decimal) are stored as Python Decimal in memory.
Serialization to/from str is handled by serializers.py — never use float here.

BaseEvent fields on every event:
    event_id:        UUID4 — globally unique per event instance
    schema_version:  int — 1 for all v1 events; increment on breaking changes
    event_ts_ms:     int — canonical exchange timestamp (Unix ms, int64)
    causation_id:    str | None — event_id that triggered this event
    correlation_id:  str | None — groups related events (use order_id for order events)

Child classes use kw_only=True to allow required fields after BaseEvent's optional fields.

Doctrinal rules:
    - No mutable state. Ever.
    - No float. Ever.
    - No Order objects. Events are serializable primitives only.
    - Enum values stored as .value strings ("BUY", "MAKER", "FILLED"), not enum instances.
      This keeps event_models.py independent of order_types.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class BaseEvent:
    """Root of the event hierarchy. Every persisted event extends this."""
    event_id: str
    schema_version: int
    event_ts_ms: int
    causation_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class RawExchangeEvent(BaseEvent):
    """
    Exact bytes/JSON received from the exchange — never normalised.
    Stored as-is for reproducible parsing and bug diagnosis.

    CRITICAL: Never derive execution decisions from this event.
    Use domain events (OrderSubmitted etc.) for all business logic.
    """
    exchange: str           # "binance"
    stream: str             # "executionReport", "!trade@arr"
    local_receive_ts_ms: int
    payload_json: str       # raw JSON string from exchange WS
    checksum: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class OrderSubmitted(BaseEvent):
    """
    An order was created locally and the submit request was sent to exchange.

    event_ts_ms  = when osm.submit() was called (submitted to exchange).
    created_ts_ms = when osm.create_order() was called (Order object created).

    Both timestamps are needed for exact replay through the OSM.
    """
    order_id: str
    symbol: str
    side: str           # OrderSide.value: "BUY" | "SELL"
    order_type: str     # OrderType.value: "LIMIT" | "MARKET"
    qty: Decimal
    created_ts_ms: int  # for osm.create_order() replay
    limit_price: Optional[Decimal] = None


@dataclass(frozen=True, kw_only=True)
class OrderAcknowledged(BaseEvent):
    """Exchange confirmed the order is resting in the book.

    event_ts_ms  = when ACK was received (= osm.acknowledge ts).
    submitted_ts_ms = when submit was sent (for ack_latency_ms = event_ts_ms - submitted_ts_ms).
    """
    order_id: str
    submitted_ts_ms: int


@dataclass(frozen=True, kw_only=True)
class OrderFillReceived(BaseEvent):
    """A fill (partial or full) received from exchange.

    fill_id = execution_id from exchange (used by OSM for idempotency).
    fee is the actual fee charged, in fee_asset units.
    fee_model = "MAKER" | "TAKER" (determines fee rate and liquidity_role).
    event_ts_ms = fill timestamp from exchange.
    """
    order_id: str
    fill_id: str        # execution_id → passed to osm.fill(execution_id=...)
    price: Decimal
    qty: Decimal
    fee: Decimal        # computed fee amount (price * qty * rate)
    fee_asset: str      # "USDT" for USDT-margined perps
    fee_model: str      # FeeModel.value: "MAKER" | "TAKER"


@dataclass(frozen=True, kw_only=True)
class OrderCancelled(BaseEvent):
    """Order was cancelled by user or system.

    trigger = "USER_CANCEL" | "SYSTEM_CANCEL" (passed to osm.cancel).
    """
    order_id: str
    trigger: str


@dataclass(frozen=True, kw_only=True)
class OrderExpired(BaseEvent):
    """Order expired due to TTL without being fully filled."""
    order_id: str


@dataclass(frozen=True, kw_only=True)
class OrderRejected(BaseEvent):
    """Exchange rejected the order (insufficient margin, invalid params, etc.)."""
    order_id: str
    reason: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class PersistedTransition(BaseEvent):
    """
    OSM state machine transition — audit log entry for every state change.

    event_ts_ms = same exchange timestamp as the triggering event.
    from_state / to_state = OrderState.value strings.
    trigger = OSM trigger string (e.g. "FULL_FILL", "USER_CANCEL").
    bid_price / ask_price = optional spread snapshot at transition time.
    """
    order_id: str
    from_state: str         # OrderState.value
    to_state: str           # OrderState.value
    trigger: str
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None


@dataclass(frozen=True, kw_only=True)
class PersistedFill(BaseEvent):
    """
    Financial ledger entry for a single fill.

    Created AFTER the OSM processes an OrderFillReceived, so liquidity_role
    is available (derived by OSM from the full fill history of the order).
    trade_id is the exchange's trade identifier (if provided).
    """
    order_id: str
    fill_id: str                # matches OrderFillReceived.fill_id
    price: Decimal
    qty: Decimal
    fee: Decimal
    fee_asset: str
    fee_model: str              # FeeModel.value
    liquidity_role: str         # LiquidityRole.value (MAKER | TAKER | MIXED)
    trade_id: Optional[str] = None
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `pytest tests/test_event_models.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit** *(skip — Bryan commits manually)*

---

## Task 2: serializers.py — Decimal/JSON round-trip + event type registry

**Files:** `live/serializers.py` (create), `tests/test_serializers.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_serializers.py`:
```python
"""Tests for serializers.py — JSON round-trip, Decimal precision, event registry."""
from decimal import Decimal

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
from live.serializers import event_to_json, json_to_event


# ── OrderSubmitted round-trip ─────────────────────────────────────────────────

def test_order_submitted_round_trip():
    original = OrderSubmitted(
        event_id="evt-001",
        schema_version=1,
        event_ts_ms=1_001_000,
        causation_id="raw-001",
        correlation_id="order-001",
        order_id="order-001",
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        qty=Decimal("0.01"),
        created_ts_ms=1_000_000,
        limit_price=Decimal("65000.50"),
    )
    json_str = event_to_json(original)
    restored = json_to_event(json_str)
    assert isinstance(restored, OrderSubmitted)
    assert restored.event_id == original.event_id
    assert restored.order_id == original.order_id
    assert restored.qty == original.qty
    assert restored.limit_price == original.limit_price
    assert isinstance(restored.qty, Decimal)
    assert isinstance(restored.limit_price, Decimal)


def test_order_submitted_without_limit_price_round_trip():
    original = OrderSubmitted(
        event_id="e", schema_version=1, event_ts_ms=1_001_000,
        order_id="o1", symbol="BTCUSDT", side="BUY",
        order_type="MARKET", qty=Decimal("0.01"), created_ts_ms=1_000_000,
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, OrderSubmitted)
    assert restored.limit_price is None


# ── OrderFillReceived round-trip ──────────────────────────────────────────────

def test_order_fill_received_decimal_precision():
    """Decimal precision is preserved — float would lose it."""
    original = OrderFillReceived(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1", fill_id="fill-001",
        price=Decimal("65000.12345678"),
        qty=Decimal("0.00100000"),
        fee=Decimal("0.00130001302"),
        fee_asset="USDT",
        fee_model="MAKER",
    )
    restored = json_to_event(event_to_json(original))
    assert restored.price == Decimal("65000.12345678")
    assert restored.qty == Decimal("0.00100000")
    assert restored.fee == Decimal("0.00130001302")
    assert isinstance(restored.price, Decimal)
    assert isinstance(restored.qty, Decimal)
    assert isinstance(restored.fee, Decimal)


def test_order_fill_received_round_trip():
    original = OrderFillReceived(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1", fill_id="f1", price=Decimal("65100"),
        qty=Decimal("0.01"), fee=Decimal("0.1302"), fee_asset="USDT",
        fee_model="TAKER",
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, OrderFillReceived)
    assert restored.fee_model == "TAKER"
    assert restored.fill_id == "f1"


# ── OrderAcknowledged round-trip ──────────────────────────────────────────────

def test_order_acknowledged_round_trip():
    original = OrderAcknowledged(
        event_id="e", schema_version=1, event_ts_ms=1_050_000,
        order_id="o1", submitted_ts_ms=1_001_000,
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, OrderAcknowledged)
    assert restored.submitted_ts_ms == 1_001_000


# ── OrderCancelled round-trip ─────────────────────────────────────────────────

def test_order_cancelled_round_trip():
    original = OrderCancelled(
        event_id="e", schema_version=1, event_ts_ms=1_200_000,
        order_id="o1", trigger="SYSTEM_CANCEL",
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, OrderCancelled)
    assert restored.trigger == "SYSTEM_CANCEL"


# ── OrderExpired round-trip ───────────────────────────────────────────────────

def test_order_expired_round_trip():
    original = OrderExpired(
        event_id="e", schema_version=1, event_ts_ms=1_500_000,
        order_id="o1",
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, OrderExpired)
    assert restored.order_id == "o1"


# ── OrderRejected round-trip ──────────────────────────────────────────────────

def test_order_rejected_round_trip_with_reason():
    original = OrderRejected(
        event_id="e", schema_version=1, event_ts_ms=1_002_000,
        order_id="o1", reason="INSUFFICIENT_MARGIN",
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, OrderRejected)
    assert restored.reason == "INSUFFICIENT_MARGIN"


def test_order_rejected_round_trip_no_reason():
    original = OrderRejected(
        event_id="e", schema_version=1, event_ts_ms=1_002_000,
        order_id="o1",
    )
    restored = json_to_event(event_to_json(original))
    assert restored.reason is None


# ── PersistedTransition round-trip ────────────────────────────────────────────

def test_persisted_transition_round_trip_with_spread():
    original = PersistedTransition(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1",
        from_state="ACKNOWLEDGED",
        to_state="FILLED",
        trigger="FULL_FILL",
        bid_price=Decimal("64999.50"),
        ask_price=Decimal("65000.50"),
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, PersistedTransition)
    assert restored.bid_price == Decimal("64999.50")
    assert restored.ask_price == Decimal("65000.50")
    assert isinstance(restored.bid_price, Decimal)


def test_persisted_transition_round_trip_no_spread():
    original = PersistedTransition(
        event_id="e", schema_version=1, event_ts_ms=1_001_000,
        order_id="o1", from_state="CREATED", to_state="SUBMITTED",
        trigger="SUBMIT",
    )
    restored = json_to_event(event_to_json(original))
    assert restored.bid_price is None
    assert restored.ask_price is None


# ── PersistedFill round-trip ──────────────────────────────────────────────────

def test_persisted_fill_round_trip():
    original = PersistedFill(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1", fill_id="f1", price=Decimal("65000"),
        qty=Decimal("0.01"), fee=Decimal("0.13"), fee_asset="USDT",
        fee_model="MAKER", liquidity_role="MAKER", trade_id="trade-99",
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, PersistedFill)
    assert restored.trade_id == "trade-99"
    assert isinstance(restored.price, Decimal)


# ── RawExchangeEvent round-trip ───────────────────────────────────────────────

def test_raw_exchange_event_round_trip():
    original = RawExchangeEvent(
        event_id="r1", schema_version=1, event_ts_ms=1_000_000,
        exchange="binance", stream="executionReport",
        local_receive_ts_ms=1_000_050,
        payload_json='{"e":"executionReport","s":"BTCUSDT","q":"0.01"}',
        checksum="abc123",
    )
    restored = json_to_event(event_to_json(original))
    assert isinstance(restored, RawExchangeEvent)
    assert restored.payload_json == original.payload_json
    assert restored.checksum == "abc123"


# ── JSON format invariants ────────────────────────────────────────────────────

def test_event_to_json_contains_event_type():
    """Serialized JSON must contain __event_type__ for deserialization."""
    import json
    e = OrderSubmitted(
        event_id="e", schema_version=1, event_ts_ms=1_001_000,
        order_id="o1", symbol="BTCUSDT", side="BUY", order_type="LIMIT",
        qty=Decimal("0.01"), created_ts_ms=1_000_000, limit_price=Decimal("65000"),
    )
    data = json.loads(event_to_json(e))
    assert data["__event_type__"] == "OrderSubmitted"


def test_decimal_serialized_as_string_not_float():
    """Decimal must become a JSON string, not a float (would lose precision)."""
    import json
    e = OrderFillReceived(
        event_id="e", schema_version=1, event_ts_ms=1_100_000,
        order_id="o1", fill_id="f1", price=Decimal("65000.1"),
        qty=Decimal("0.01"), fee=Decimal("0.1300002"), fee_asset="USDT",
        fee_model="MAKER",
    )
    data = json.loads(event_to_json(e))
    assert isinstance(data["price"], str)
    assert isinstance(data["qty"], str)
    assert isinstance(data["fee"], str)


def test_none_decimal_field_serializes_as_null():
    """Optional[Decimal] = None must serialize as JSON null, not omitted."""
    import json
    e = OrderSubmitted(
        event_id="e", schema_version=1, event_ts_ms=1_001_000,
        order_id="o1", symbol="BTCUSDT", side="BUY", order_type="MARKET",
        qty=Decimal("0.01"), created_ts_ms=1_000_000,
    )
    data = json.loads(event_to_json(e))
    assert "limit_price" in data
    assert data["limit_price"] is None


def test_unknown_event_type_raises_key_error():
    """Deserializing an unknown event type must fail fast."""
    import json
    bad_json = json.dumps({"__event_type__": "UnknownEvent", "event_id": "x",
                           "schema_version": 1, "event_ts_ms": 1_000_000})
    with pytest.raises(KeyError):
        json_to_event(bad_json)
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_serializers.py -v`

Expected: `ModuleNotFoundError: No module named 'live.serializers'`

- [ ] **Step 3: Create `live/serializers.py`**

```python
"""
Serializers for execution domain events.

Responsibilities:
    - Convert event dataclasses to/from JSON strings (one line per event in JSONL).
    - Serialize Decimal as str, deserialize str back to Decimal.
    - Embed and recover __event_type__ for polymorphic deserialization.

Decimal protocol (mandatory):
    serialize:   str(decimal_value)
    deserialize: Decimal(string_value)
    NEVER use float at any point in the pipeline.

Adding a new event type:
    1. Add the class to live/event_models.py.
    2. Add an entry to _EVENT_REGISTRY below.
    3. Add the Decimal field names to _DECIMAL_FIELDS below.
    That's all — serialization is generic.
"""
from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from typing import Any

from live.event_models import (
    BaseEvent,
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

# ── Event type registry ───────────────────────────────────────────────────────
# Maps __event_type__ string → dataclass. Add new event types here.

_EVENT_REGISTRY: dict[str, type[BaseEvent]] = {
    "RawExchangeEvent": RawExchangeEvent,
    "OrderSubmitted":   OrderSubmitted,
    "OrderAcknowledged": OrderAcknowledged,
    "OrderFillReceived": OrderFillReceived,
    "OrderCancelled":   OrderCancelled,
    "OrderExpired":     OrderExpired,
    "OrderRejected":    OrderRejected,
    "PersistedTransition": PersistedTransition,
    "PersistedFill":    PersistedFill,
}

# ── Decimal field registry ────────────────────────────────────────────────────
# Maps event type name → set of field names that hold Decimal values.
# Optional[Decimal] fields are included here; None values are handled gracefully.

_DECIMAL_FIELDS: dict[str, frozenset[str]] = {
    "RawExchangeEvent":  frozenset(),
    "OrderSubmitted":    frozenset({"qty", "limit_price"}),
    "OrderAcknowledged": frozenset(),
    "OrderFillReceived": frozenset({"price", "qty", "fee"}),
    "OrderCancelled":    frozenset(),
    "OrderExpired":      frozenset(),
    "OrderRejected":     frozenset(),
    "PersistedTransition": frozenset({"bid_price", "ask_price"}),
    "PersistedFill":     frozenset({"price", "qty", "fee"}),
}


# ── Public API ────────────────────────────────────────────────────────────────

def event_to_json(event: BaseEvent) -> str:
    """
    Serialize a domain event to a single-line JSON string.

    - Adds __event_type__ for round-trip deserialization.
    - Converts all Decimal fields to str (preserving full precision).
    - None values are serialized as JSON null (not omitted).
    """
    event_type = type(event).__name__
    data = dataclasses.asdict(event)
    decimal_fields = _DECIMAL_FIELDS.get(event_type, frozenset())

    for field_name in decimal_fields:
        value = data.get(field_name)
        if value is not None:
            # dataclasses.asdict() returns Decimal as-is (not converted)
            data[field_name] = str(value)

    data["__event_type__"] = event_type
    return json.dumps(data, ensure_ascii=False)


def json_to_event(json_str: str) -> BaseEvent:
    """
    Deserialize a JSON string back to the correct event dataclass.

    - Reads __event_type__ to find the right class in _EVENT_REGISTRY.
    - Converts Decimal field strings back to Decimal instances.
    - Raises KeyError if __event_type__ is not registered.
    """
    data = json.loads(json_str)
    event_type = data.pop("__event_type__")
    cls = _EVENT_REGISTRY[event_type]  # KeyError on unknown type — fail fast

    decimal_fields = _DECIMAL_FIELDS.get(event_type, frozenset())
    for field_name in decimal_fields:
        value = data.get(field_name)
        if value is not None:
            data[field_name] = Decimal(value)

    return cls(**data)
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `pytest tests/test_serializers.py -v`

Expected: all tests pass.

- [ ] **Step 5: Run full suite so far**

Run: `pytest tests/ -v`

Expected: all tests pass (test_event_models + test_serializers + existing OSM tests).

---

## Task 3: event_store.py — dual-write JSONL + SQLite, append interface

**Files:** `live/event_store.py` (create), `tests/test_event_store.py` (create, append + basic queries)

- [ ] **Step 1: Write failing tests**

Create `tests/test_event_store.py`:
```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_event_store.py -v`

Expected: `ModuleNotFoundError: No module named 'live.event_store'`

- [ ] **Step 3: Create `live/event_store.py`**

```python
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
from typing import Optional

from live.event_models import (
    BaseEvent,
    OrderFillReceived,
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
    exchange_ts_ms       INTEGER NOT NULL,
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
"""


class EventStore:
    """
    Append-only dual-write store for execution domain events.

    Thread safety: not thread-safe. Designed for single-threaded paper trading.
    For multi-threaded use, add a threading.Lock around append methods.
    """

    def __init__(self, logs_dir: Path) -> None:
        self._logs_dir = logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)

        # JSONL file handles (append mode, line-buffered)
        self._f_raw    = open(logs_dir / "raw_events.jsonl",    "a", encoding="utf-8")
        self._f_domain = open(logs_dir / "domain_events.jsonl", "a", encoding="utf-8")
        self._f_trans  = open(logs_dir / "transitions.jsonl",   "a", encoding="utf-8")
        self._f_fills  = open(logs_dir / "fills.jsonl",         "a", encoding="utf-8")

        # SQLite connection (WAL mode for better read concurrency)
        self._conn = sqlite3.connect(logs_dir / "events.db")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Append interface ──────────────────────────────────────────────────────

    def append_raw_event(self, event: RawExchangeEvent) -> None:
        """Persist an exact exchange payload (pre-normalisation)."""
        json_str = event_to_json(event)
        self._f_raw.write(json_str + "\n")
        self._f_raw.flush()
        self._conn.execute(
            """INSERT INTO raw_events
               (event_id, schema_version, exchange, stream,
                exchange_ts_ms, local_receive_ts_ms, payload_json, checksum,
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

    def append_domain_event(self, event: BaseEvent) -> None:
        """
        Persist a normalised domain event (OrderSubmitted, OrderFillReceived, etc.).

        Extracts order_id from the event (if present) for indexing.
        Full event stored in payload_json for exact round-trip reconstruction.
        """
        json_str = event_to_json(event)
        self._f_domain.write(json_str + "\n")
        self._f_domain.flush()
        order_id: Optional[str] = getattr(event, "order_id", None)
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

    def append_transition(self, event: PersistedTransition) -> None:
        """Persist a state machine transition (FSM audit log)."""
        json_str = event_to_json(event)
        self._f_trans.write(json_str + "\n")
        self._f_trans.flush()
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

    def append_fill(self, event: PersistedFill) -> None:
        """Persist a fill ledger entry (financial audit)."""
        json_str = event_to_json(event)
        self._f_fills.write(json_str + "\n")
        self._f_fills.flush()
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
        for f in (self._f_raw, self._f_domain, self._f_trans, self._f_fills):
            f.flush()
            f.close()
        self._conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _now_ms() -> int:
    return int(time.time() * 1000)
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `pytest tests/test_event_store.py -v`

Expected: all tests pass.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -v`

Expected: all tests pass.

---

## Task 4: execution_replay.py — reconstruct Order from domain events

**Files:** `live/execution_replay.py` (create), `tests/test_execution_replay.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_replay.py`:
```python
"""
Tests for ExecutionReplay — deterministic Order reconstruction from domain events.

Key invariant under test:
    Replaying the same sequence of domain events through a fresh OSM always
    produces the same Order state (same state, filled_qty, avg_fill_price,
    transition triggers, fill execution_ids).

This is the central guarantee of the execution event sourcing system.
"""
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
)
from live.event_store import EventStore
from live.execution_replay import ExecutionReplay
from live.order_types import OrderState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(logs_dir=tmp_path)


@pytest.fixture
def replay(store: EventStore) -> ExecutionReplay:
    return ExecutionReplay(event_store=store)


def _evt(ts: int = 1_000_000) -> dict:
    return {"event_id": str(uuid.uuid4()), "schema_version": 1, "event_ts_ms": ts}


# ── Helper to store a complete lifecycle ──────────────────────────────────────

def _store_full_lifecycle(store: EventStore, order_id: str = "replay-01"):
    """Store: submitted → acknowledged → partial fill → full fill."""
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=order_id,
        order_id=order_id, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.02"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(1_050_000), correlation_id=order_id,
        order_id=order_id, submitted_ts_ms=1_000_000,
    ))
    store.append_domain_event(OrderFillReceived(
        **_evt(1_100_000), correlation_id=order_id,
        order_id=order_id, fill_id="fill-01",
        price=Decimal("65100"), qty=Decimal("0.01"),
        fee=Decimal("0.1302"), fee_asset="USDT", fee_model="MAKER",
    ))
    store.append_domain_event(OrderFillReceived(
        **_evt(1_200_000), correlation_id=order_id,
        order_id=order_id, fill_id="fill-02",
        price=Decimal("65200"), qty=Decimal("0.01"),
        fee=Decimal("0.1304"), fee_asset="USDT", fee_model="MAKER",
    ))


# ── Basic replay paths ────────────────────────────────────────────────────────

def test_replay_submitted_only(store: EventStore, replay: ExecutionReplay):
    """Replay an order that only reached SUBMITTED state."""
    oid = "r-submit"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.SUBMITTED
    assert order.order_id == oid
    assert order.qty == Decimal("0.01")


def test_replay_acknowledged(store: EventStore, replay: ExecutionReplay):
    oid = "r-ack"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(1_050_000), correlation_id=oid,
        order_id=oid, submitted_ts_ms=1_000_000,
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.ACKNOWLEDGED
    assert order.acknowledged_ts_ms == 1_050_000


def test_replay_full_fill(store: EventStore, replay: ExecutionReplay):
    oid = "r-filled"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    assert order.state == OrderState.FILLED
    assert order.filled_qty == Decimal("0.02")
    assert order.remaining_qty == Decimal("0")


def test_replay_partial_fill_then_cancel(store: EventStore, replay: ExecutionReplay):
    oid = "r-partial-cancel"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.03"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(1_050_000), correlation_id=oid,
        order_id=oid, submitted_ts_ms=1_000_000,
    ))
    store.append_domain_event(OrderFillReceived(
        **_evt(1_100_000), correlation_id=oid,
        order_id=oid, fill_id="f1",
        price=Decimal("65000"), qty=Decimal("0.01"),
        fee=Decimal("0.13"), fee_asset="USDT", fee_model="MAKER",
    ))
    store.append_domain_event(OrderCancelled(
        **_evt(1_200_000), correlation_id=oid,
        order_id=oid, trigger="USER_CANCEL",
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.CANCELLED
    assert order.filled_qty == Decimal("0.01")


def test_replay_expired(store: EventStore, replay: ExecutionReplay):
    oid = "r-expired"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(1_050_000), correlation_id=oid,
        order_id=oid, submitted_ts_ms=1_000_000,
    ))
    store.append_domain_event(OrderExpired(
        **_evt(1_500_000), correlation_id=oid, order_id=oid,
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.EXPIRED


def test_replay_rejected(store: EventStore, replay: ExecutionReplay):
    oid = "r-rejected"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderRejected(
        **_evt(1_001_000), correlation_id=oid,
        order_id=oid, reason="INSUFFICIENT_MARGIN",
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.REJECTED


def test_replay_nonexistent_order_raises(replay: ExecutionReplay):
    with pytest.raises(ValueError, match="No domain events"):
        replay.replay_order("ghost-order-id")


# ── Deterministic replay — core guarantee ─────────────────────────────────────

def test_replay_avg_fill_price_matches_live(store: EventStore, replay: ExecutionReplay):
    """Weighted avg fill price reconstructed exactly with Decimal arithmetic."""
    oid = "r-avg"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    # avg = (65100 * 0.01 + 65200 * 0.01) / 0.02 = 65150
    expected_avg = (Decimal("65100") * Decimal("0.01") +
                    Decimal("65200") * Decimal("0.01")) / Decimal("0.02")
    assert order.avg_fill_price == expected_avg
    assert isinstance(order.avg_fill_price, Decimal)


def test_replay_fill_count_matches(store: EventStore, replay: ExecutionReplay):
    oid = "r-fills"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    assert len(order.fills) == 2


def test_replay_fill_execution_ids_preserved(store: EventStore, replay: ExecutionReplay):
    oid = "r-exec-ids"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    execution_ids = {f.execution_id for f in order.fills}
    assert execution_ids == {"fill-01", "fill-02"}


def test_replay_transition_triggers_match_live(store: EventStore, replay: ExecutionReplay):
    """Transition trigger sequence must be identical in live vs replay."""
    oid = "r-triggers"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    triggers = [t.trigger for t in order.transitions]
    assert triggers == [
        "ORDER_CREATED", "SUBMIT", "ACK_RECEIVED", "PARTIAL_FILL", "FULL_FILL"
    ]


def test_replay_is_deterministic_across_two_runs(store: EventStore, replay: ExecutionReplay):
    """Two replay runs on the same events produce identical Order state."""
    oid = "r-det"
    _store_full_lifecycle(store, oid)
    order_a = replay.replay_order(oid)
    order_b = replay.replay_order(oid)
    assert order_a.state == order_b.state
    assert order_a.filled_qty == order_b.filled_qty
    assert order_a.avg_fill_price == order_b.avg_fill_price
    assert order_a.terminal_ts_ms == order_b.terminal_ts_ms
    assert [t.trigger for t in order_a.transitions] == [t.trigger for t in order_b.transitions]


def test_two_independent_orders_replay_correctly(store: EventStore, replay: ExecutionReplay):
    """Events for different orders don't interfere."""
    _store_full_lifecycle(store, "order-A")
    store.append_domain_event(OrderSubmitted(
        **_evt(2_000_000), correlation_id="order-B",
        order_id="order-B", symbol="ETHUSDT", side="SELL",
        order_type="LIMIT", qty=Decimal("1.0"),
        created_ts_ms=1_999_000, limit_price=Decimal("3000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(2_050_000), correlation_id="order-B",
        order_id="order-B", submitted_ts_ms=2_000_000,
    ))
    order_a = replay.replay_order("order-A")
    order_b = replay.replay_order("order-B")
    assert order_a.state == OrderState.FILLED
    assert order_b.state == OrderState.ACKNOWLEDGED
    assert order_a.symbol == "BTCUSDT"
    assert order_b.symbol == "ETHUSDT"
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_execution_replay.py -v`

Expected: `ModuleNotFoundError: No module named 'live.execution_replay'`

- [ ] **Step 3: Create `live/execution_replay.py`**

```python
"""
Execution Replay — reconstruct Order state from persisted domain events.

Replays domain events (OrderSubmitted, OrderAcknowledged, etc.) through a fresh
OrderStateMachine instance to rebuild the complete Order lifecycle.

Guarantees:
    - Deterministic: same events → same Order state, every time.
    - No side effects: replay never writes to EventStore.
    - Isolated: uses a fresh OSM per call; no shared mutable state.
    - Decimal-exact: financial quantities preserve full precision.

This module depends on:
    - live.event_models: domain event dataclasses
    - live.event_store: EventStore.get_domain_events_by_order()
    - live.order_types: enums (OrderSide, OrderType, FeeModel)
    - live.order_state_machine: OrderStateMachine

It does NOT depend on live.storage (market data) or live.replay (market replay).
"""
from __future__ import annotations

from live.event_models import (
    BaseEvent,
    OrderAcknowledged,
    OrderCancelled,
    OrderExpired,
    OrderFillReceived,
    OrderRejected,
    OrderSubmitted,
)
from live.event_store import EventStore
from live.order_state_machine import OrderStateMachine
from live.order_types import FeeModel, Order, OrderSide, OrderType


class ExecutionReplay:
    """
    Reconstruct an Order's state by replaying its domain events through OSM.

    Usage:
        replay = ExecutionReplay(event_store=store)
        order = replay.replay_order("order-001")
        assert order.state == OrderState.FILLED

    The replay algorithm:
        1. Fetch all domain events for order_id, ordered by event_ts_ms ASC.
        2. For each event, call the corresponding OSM method.
        3. Return the fully reconstructed Order.

    OrderSubmitted drives TWO OSM calls:
        - osm.create_order(..., event_ts_ms=event.created_ts_ms)
        - osm.submit(order, event_ts_ms=event.event_ts_ms)
      Both timestamps are stored in OrderSubmitted specifically for this replay.
    """

    def __init__(self, event_store: EventStore) -> None:
        self._store = event_store

    def replay_order(self, order_id: str) -> Order:
        """
        Reconstruct the Order lifecycle from persisted domain events.

        Raises:
            ValueError: if no domain events are found for order_id.
            InvalidTransitionError / CausalityViolationError: if stored events
                are inconsistent (should never happen with a healthy event log).
        """
        events: list[BaseEvent] = self._store.get_domain_events_by_order(order_id)
        if not events:
            raise ValueError(
                f"No domain events found for order_id={order_id!r}. "
                "Cannot replay an order with no history."
            )

        osm = OrderStateMachine()
        order: Order | None = None

        for event in events:
            if isinstance(event, OrderSubmitted):
                order = osm.create_order(
                    symbol=event.symbol,
                    side=OrderSide(event.side),
                    order_type=OrderType(event.order_type),
                    qty=event.qty,
                    limit_price=event.limit_price,
                    event_ts_ms=event.created_ts_ms,
                    order_id=event.order_id,
                )
                osm.submit(order, event_ts_ms=event.event_ts_ms)

            elif isinstance(event, OrderAcknowledged):
                assert order is not None
                osm.acknowledge(order, event_ts_ms=event.event_ts_ms)

            elif isinstance(event, OrderFillReceived):
                assert order is not None
                osm.fill(
                    order,
                    fill_price=event.price,
                    fill_qty=event.qty,
                    event_ts_ms=event.event_ts_ms,
                    fee_model=FeeModel(event.fee_model),
                    execution_id=event.fill_id,
                )

            elif isinstance(event, OrderCancelled):
                assert order is not None
                osm.cancel(order, event_ts_ms=event.event_ts_ms, trigger=event.trigger)

            elif isinstance(event, OrderExpired):
                assert order is not None
                osm.expire(order, event_ts_ms=event.event_ts_ms)

            elif isinstance(event, OrderRejected):
                assert order is not None
                osm.reject(order, event_ts_ms=event.event_ts_ms)

        assert order is not None  # guaranteed since events is non-empty
        return order
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `pytest tests/test_execution_replay.py -v`

Expected: all tests pass.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -v`

Expected: all tests pass. Note final count (~130+ tests).

---

## Task 5: Deterministic integration test — live processing == replay

**Files:** `tests/test_execution_replay.py` (append)

This is the critical end-to-end test that validates the entire pipeline:
raw OSM calls → EventStore → ExecutionReplay → identical Order state.

- [ ] **Step 1: Append integration test**

Append to `tests/test_execution_replay.py`:
```python
# ── Integration: live OSM calls → EventStore → replay → identical state ──────

def test_live_vs_replay_full_lifecycle(tmp_path: Path):
    """
    CRITICAL DETERMINISM TEST.

    Process a complete order lifecycle using OSM directly.
    Log all domain events to EventStore.
    Replay from EventStore.
    Assert final state is byte-for-byte identical to live processing.
    """
    from live.order_state_machine import OrderStateMachine
    from live.order_types import FeeModel, OrderSide, OrderState, OrderType

    store = EventStore(logs_dir=tmp_path / "live_test")
    osm = OrderStateMachine()

    order_id = "integration-001"
    created_ts = 1_000_000
    submit_ts  = 1_001_000
    ack_ts     = 1_050_000
    fill1_ts   = 1_100_000
    fill2_ts   = 1_200_000

    # ── Live OSM processing ──────────────────────────────────────────────────
    live_order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.02"),
        limit_price=Decimal("65000"),
        event_ts_ms=created_ts,
        order_id=order_id,
    )

    # Log OrderSubmitted domain event
    store.append_domain_event(OrderSubmitted(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=submit_ts,
        correlation_id=order_id, order_id=order_id,
        symbol="BTCUSDT", side="BUY", order_type="LIMIT",
        qty=Decimal("0.02"), created_ts_ms=created_ts,
        limit_price=Decimal("65000"),
    ))
    osm.submit(live_order, event_ts_ms=submit_ts)

    # Log OrderAcknowledged
    store.append_domain_event(OrderAcknowledged(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=ack_ts,
        correlation_id=order_id, order_id=order_id, submitted_ts_ms=submit_ts,
    ))
    osm.acknowledge(live_order, event_ts_ms=ack_ts)

    # Log first partial fill
    store.append_domain_event(OrderFillReceived(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=fill1_ts,
        correlation_id=order_id, order_id=order_id, fill_id="exec-001",
        price=Decimal("65100"), qty=Decimal("0.01"),
        fee=Decimal("0.1302"), fee_asset="USDT", fee_model="MAKER",
    ))
    osm.fill(live_order, fill_price=Decimal("65100"), fill_qty=Decimal("0.01"),
             event_ts_ms=fill1_ts, fee_model=FeeModel.MAKER, execution_id="exec-001")

    # Log second (final) fill
    store.append_domain_event(OrderFillReceived(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=fill2_ts,
        correlation_id=order_id, order_id=order_id, fill_id="exec-002",
        price=Decimal("65200"), qty=Decimal("0.01"),
        fee=Decimal("0.1304"), fee_asset="USDT", fee_model="MAKER",
    ))
    osm.fill(live_order, fill_price=Decimal("65200"), fill_qty=Decimal("0.01"),
             event_ts_ms=fill2_ts, fee_model=FeeModel.MAKER, execution_id="exec-002")

    # ── Replay ───────────────────────────────────────────────────────────────
    replay = ExecutionReplay(event_store=store)
    replayed_order = replay.replay_order(order_id)

    # ── Assert identical final state ─────────────────────────────────────────
    assert live_order.state == replayed_order.state == OrderState.FILLED
    assert live_order.filled_qty == replayed_order.filled_qty == Decimal("0.02")
    assert live_order.remaining_qty == replayed_order.remaining_qty == Decimal("0")

    # Decimal-exact average price
    expected_avg = (Decimal("65100") * Decimal("0.01") +
                    Decimal("65200") * Decimal("0.01")) / Decimal("0.02")
    assert live_order.avg_fill_price == replayed_order.avg_fill_price == expected_avg

    # Timestamps
    assert live_order.terminal_ts_ms == replayed_order.terminal_ts_ms == fill2_ts

    # Transition sequence
    live_triggers = [t.trigger for t in live_order.transitions]
    replay_triggers = [t.trigger for t in replayed_order.transitions]
    assert live_triggers == replay_triggers == [
        "ORDER_CREATED", "SUBMIT", "ACK_RECEIVED", "PARTIAL_FILL", "FULL_FILL"
    ]

    # Fill execution IDs
    live_exec_ids = {f.execution_id for f in live_order.fills}
    replay_exec_ids = {f.execution_id for f in replayed_order.fills}
    assert live_exec_ids == replay_exec_ids == {"exec-001", "exec-002"}

    # Liquidity role
    assert live_order.liquidity_role == replayed_order.liquidity_role

    store.close()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_execution_replay.py::test_live_vs_replay_full_lifecycle -v`

Expected: PASS.

- [ ] **Step 3: Final full suite**

Run: `pytest tests/ -v`

Expected: all tests pass. Note final count.

---

## Self-Review

### Spec coverage

| Requirement | Covered by |
|---|---|
| Raw exchange event store | Task 3 `append_raw_event`, Task 3 `test_append_raw_event_*` |
| Domain event log | Task 3 `append_domain_event`, all domain event classes Task 1 |
| Transition log | Task 3 `append_transition`, Task 3 `test_append_transition_*` |
| Fill ledger | Task 3 `append_fill`, Task 3 `test_append_fill_*` |
| Append-only (no UPDATE/DELETE) | `event_store.py` never calls UPDATE/DELETE; UNIQUE constraint on event_id |
| Dual-write JSONL + SQLite | Task 3 `test_append_domain_event_dual_write_consistent` |
| event_id on every event | `BaseEvent.event_id` required field, Task 1 tests |
| schema_version on every event | `BaseEvent.schema_version`, Task 1 `test_all_events_have_schema_version_1` |
| causation_id + correlation_id | `BaseEvent` optional fields, stored in all SQLite tables |
| Decimal as str (never float) | Task 2 `test_decimal_serialized_as_string_not_float`, `test_order_fill_received_decimal_precision` |
| SQLite WAL mode | `event_store.py` PRAGMA journal_mode=WAL |
| JSONL append-only files | four `.jsonl` files per `EventStore` instance |
| OSM stays pure (no writes inside) | `execution_replay.py` uses OSM as read-only. EventStore is separate. |
| Replay determinism | Task 4 `test_replay_is_deterministic_across_two_runs` |
| live == replay | Task 5 `test_live_vs_replay_full_lifecycle` |
| All lifecycle paths | Task 4: submit, ack, fill, cancel, expire, reject tested separately |
| Separation from storage.py (market data) | `event_store.py` has no import of or dependency on `storage.py` |
| Separation from replay.py (market replay) | `execution_replay.py` has no import of `replay.py` |

### Placeholder scan

No TBD, TODO, or "implement later" phrases. All code is complete and runnable.

### Type consistency

- `EventStore.append_domain_event(event: BaseEvent)` — matches all domain event types (all extend BaseEvent) ✓
- `EventStore.get_domain_events_by_order(order_id: str) -> list[BaseEvent]` — returns BaseEvent, `ExecutionReplay.replay_order()` uses `isinstance()` checks ✓
- `ExecutionReplay.replay_order(order_id: str) -> Order` — returns `live.order_types.Order` ✓
- `FeeModel(event.fee_model)` — `event.fee_model` is always a string like "MAKER", `FeeModel` is a `str` enum ✓
- `OrderSide(event.side)` / `OrderType(event.order_type)` — same pattern ✓
- `PersistedTransition.bid_price: Optional[Decimal]` — matches `StateTransition.bid_price: Optional[Decimal]` from order_types.py ✓
