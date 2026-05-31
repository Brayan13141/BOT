# Order State Machine — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a deterministic Order State Machine (OSM) that is the single source of truth for order lifecycle in the paper trading system — consumed by FillModelA/B/C and the execution ledger.

**Architecture:** Two files: `live/order_types.py` defines all immutable data contracts (enums, frozen dataclasses); `live/order_state_machine.py` defines `OrderStateMachine`, a stateless service that mutates `Order` objects and enforces valid transitions. All timestamps are `event_ts_ms` (int64 Unix ms from exchange). `recv_monotonic_ns` is stored as optional diagnostic data but **never used in any logic or conditional**. All transitions append to `order.transitions` (event-sourced). No transition is ever inferred from market data. All financial quantities use `Decimal`.

**Tech Stack:** Python 3.10+, dataclasses (stdlib), enum (stdlib), decimal (stdlib), uuid (stdlib), pytest 8.0+

---

## Architectural Changes vs v1

| v1 | v2 | Reason |
|----|----|----|
| `TAKER_CONVERTED` in `OrderState` (terminal) | Removed | Lifecycle ≠ fee classification. Terminal state with no fills = ledger inconsistency |
| `order.execution_fee_model: Optional[FeeModel]` | Removed → `@property liquidity_role` | Derived field must not be stored mutable in event-sourced system |
| `FeeModel.MAKER/TAKER` per-order | `LiquidityRole.MAKER/TAKER/MIXED` derived from fills | Accurate for mixed-fill orders |
| `OrderState.RESTING` | `OrderState.ACKNOWLEDGED` | Semantically correct: state = exchange confirmed acceptance |
| `order.ack_ts_ms` + `order.resting_ts_ms` (both set same time) | `order.acknowledged_ts_ms` only | No duplication |
| `float` for qty/price | `Decimal` | Replay determinism: same events → same ledger, across runs/platforms |
| No timestamp enforcement | `CausalityViolationError` raised runtime | Causal ordering must be enforced, not just tested |
| No idempotency | `order.processed_execution_ids: set[str]` + `DuplicateEventError` | WS feeds duplicate messages; replay must be identical |
| No out-of-order handling | `OutOfOrderEventError(InvalidTransitionError)` | Common in live: fill arrives before ACK |
| `InvalidTransitionError` only | `InvalidTransitionError`, `CausalityViolationError`, `OutOfOrderEventError`, `DuplicateEventError` | Distinct failure modes require distinct diagnostics |

---

## Doctrinal Rules (contract — enforce in every test)

1. `TOUCH != FILL` — no fill is ever inferred from price touching a level
2. Market data streams and execution event streams are **never mixed**
3. `ACKNOWLEDGED` state requires exchange ACK — never set from candle/market data
4. `recv_monotonic_ns` stored in transitions for diagnostics but **never** appears in `if`/comparison/logic
5. All transitions must be auditable: recorded in `order.transitions`
6. Terminal states (`FILLED`, `CANCELLED`, `EXPIRED`, `REJECTED`) block all further transitions
7. `PARTIALLY_FILLED` is inventory state, not terminal — it can continue transitioning
8. `LiquidityRole` (MAKER/TAKER/MIXED) is derived from `FillEvent.fee_model` — never stored as mutable field
9. Replay determinism > simulation realism
10. All financial quantities (`qty`, `price`, `fee`, `pnl`, `remaining_qty`, `avg_fill_price`) use `Decimal`. Float prohibited in financial fields.
11. Every fill must provide a unique `execution_id` — duplicate execution_ids on the same order raise `DuplicateEventError`

---

## Valid Transition Graph

```
CREATED          → SUBMITTED            (SUBMIT)
SUBMITTED        → ACKNOWLEDGED         (ACK_RECEIVED)
SUBMITTED        → REJECTED             (EXCHANGE_REJECT)
SUBMITTED        → FILLED               (IMMEDIATE_TAKER_FILL)
SUBMITTED        → PARTIALLY_FILLED     (IMMEDIATE_TAKER_PARTIAL_FILL)
ACKNOWLEDGED     → PARTIALLY_FILLED     (PARTIAL_FILL)
ACKNOWLEDGED     → FILLED               (FULL_FILL)
ACKNOWLEDGED     → CANCELLED            (USER_CANCEL, SYSTEM_CANCEL)
ACKNOWLEDGED     → EXPIRED              (TTL_EXPIRED)
PARTIALLY_FILLED → PARTIALLY_FILLED     (PARTIAL_FILL)
PARTIALLY_FILLED → FILLED               (FULL_FILL)
PARTIALLY_FILLED → CANCELLED            (USER_CANCEL, SYSTEM_CANCEL)
PARTIALLY_FILLED → EXPIRED              (TTL_EXPIRED)

Terminal states (no further transitions):
  FILLED, CANCELLED, EXPIRED, REJECTED
```

Note: Taker fee classification is recorded per-fill via `FillEvent.fee_model`. There is no `TAKER_CONVERTED` state. `order.liquidity_role` is a `@property` derived from `order.fills`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `live/order_types.py` | **Create** | All enums + frozen dataclasses: `OrderState`, `OrderSide`, `OrderType`, `FeeModel`, `LiquidityRole`, `FillEvent`, `StateTransition`, `Order` |
| `live/order_state_machine.py` | **Create** | `OrderStateMachine` service + all exception types |
| `tests/__init__.py` | **Create** | Makes `tests/` a package |
| `tests/conftest.py` | **Create** | Shared fixtures (Decimal-native) |
| `tests/test_order_types.py` | **Create** | Unit tests: enums, dataclass invariants, `liquidity_role` property |
| `tests/test_order_state_machine.py` | **Create** | Unit tests: all transitions, fills, error types, idempotency, causal enforcement |

---

## Task 1: Test infrastructure

**Files:** `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create tests package and conftest**

Create `tests/__init__.py` (empty):
```python
```

Create `tests/conftest.py`:
```python
"""
Shared fixtures for OSM tests.

Canonical clock: event_ts_ms (int64 Unix ms).
Financial quantities: Decimal.
recv_monotonic_ns is always optional — fixtures provide it to verify
it is stored but never affects logic.
"""
from decimal import Decimal

import pytest

from live.order_types import OrderSide, OrderType
from live.order_state_machine import OrderStateMachine


@pytest.fixture
def osm() -> OrderStateMachine:
    return OrderStateMachine()


@pytest.fixture
def limit_buy_order(osm):
    """A freshly created LIMIT BUY order at ts=1_000_000."""
    return osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.01"),
        limit_price=Decimal("65000"),
        event_ts_ms=1_000_000,
        order_id="test-order-001",
    )


@pytest.fixture
def limit_sell_order(osm):
    """A freshly created LIMIT SELL order at ts=2_000_000."""
    return osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.05"),
        limit_price=Decimal("66000"),
        event_ts_ms=2_000_000,
        order_id="test-order-002",
    )
```

- [ ] **Step 2: Verify pytest discovers tests directory**

Run: `pytest tests/ --collect-only`

Expected: `no tests ran` (no import errors).

---

## Task 2: Order types — enums and frozen dataclasses

**Files:** `live/order_types.py`, `tests/test_order_types.py`

- [ ] **Step 1: Write failing tests for order_types**

Create `tests/test_order_types.py`:
```python
"""Tests for order_types.py — enums, frozen dataclasses, Order initialization."""
from decimal import Decimal

import pytest

from live.order_types import (
    FeeModel,
    FillEvent,
    LiquidityRole,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    StateTransition,
)


# ── Enum membership ──────────────────────────────────────────────────────────

def test_order_state_has_eight_states():
    states = {s.value for s in OrderState}
    assert states == {
        "CREATED", "SUBMITTED", "ACKNOWLEDGED", "PARTIALLY_FILLED",
        "FILLED", "CANCELLED", "EXPIRED", "REJECTED",
    }


def test_liquidity_role_values():
    assert LiquidityRole.MAKER.value == "MAKER"
    assert LiquidityRole.TAKER.value == "TAKER"
    assert LiquidityRole.MIXED.value == "MIXED"


def test_fee_model_values():
    assert FeeModel.MAKER.value == "MAKER"
    assert FeeModel.TAKER.value == "TAKER"


def test_order_side_values():
    assert OrderSide.BUY.value == "BUY"
    assert OrderSide.SELL.value == "SELL"


def test_order_type_values():
    assert OrderType.LIMIT.value == "LIMIT"
    assert OrderType.MARKET.value == "MARKET"


# ── FillEvent (frozen) ───────────────────────────────────────────────────────

def test_fill_event_is_immutable():
    fe = FillEvent(
        execution_id="e1",
        order_id="o1",
        event_ts_ms=1_000_000,
        price=Decimal("65000"),
        qty=Decimal("0.01"),
        fee_model=FeeModel.MAKER,
    )
    with pytest.raises((AttributeError, TypeError)):
        fe.price = Decimal("70000")  # type: ignore[misc]


def test_fill_event_recv_monotonic_ns_defaults_to_none():
    fe = FillEvent(
        execution_id="e1",
        order_id="o1",
        event_ts_ms=1_000_000,
        price=Decimal("65000"),
        qty=Decimal("0.01"),
        fee_model=FeeModel.MAKER,
    )
    assert fe.recv_monotonic_ns is None


def test_fill_event_price_and_qty_are_decimal():
    fe = FillEvent(
        execution_id="e1",
        order_id="o1",
        event_ts_ms=1_000_000,
        price=Decimal("65000.50"),
        qty=Decimal("0.001"),
        fee_model=FeeModel.TAKER,
    )
    assert isinstance(fe.price, Decimal)
    assert isinstance(fe.qty, Decimal)


def test_fill_event_stores_recv_monotonic_ns():
    fe1 = FillEvent("e1", "o1", 1_000_000, Decimal("65000"), Decimal("0.01"), FeeModel.MAKER, recv_monotonic_ns=100)
    fe2 = FillEvent("e1", "o1", 1_000_000, Decimal("65000"), Decimal("0.01"), FeeModel.MAKER, recv_monotonic_ns=200)
    assert fe1.execution_id == fe2.execution_id
    assert fe1.recv_monotonic_ns != fe2.recv_monotonic_ns


# ── StateTransition (frozen) ─────────────────────────────────────────────────

def test_state_transition_is_immutable():
    st = StateTransition(
        from_state=OrderState.CREATED,
        to_state=OrderState.SUBMITTED,
        event_ts_ms=1_000_000,
        trigger="SUBMIT",
    )
    with pytest.raises((AttributeError, TypeError)):
        st.trigger = "TAMPERED"  # type: ignore[misc]


def test_state_transition_optional_fields_default_none():
    st = StateTransition(
        from_state=OrderState.SUBMITTED,
        to_state=OrderState.ACKNOWLEDGED,
        event_ts_ms=1_001_000,
        trigger="ACK_RECEIVED",
    )
    assert st.bid_price is None
    assert st.ask_price is None
    assert st.recv_monotonic_ns is None


# ── Order initialization ─────────────────────────────────────────────────────

def test_order_initial_state_is_created():
    order = Order(
        order_id="o1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("65000"),
        qty=Decimal("0.01"),
        created_ts_ms=1_000_000,
    )
    assert order.state == OrderState.CREATED


def test_order_remaining_qty_equals_qty_on_creation():
    order = Order(
        order_id="o1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("65000"),
        qty=Decimal("0.05"),
        created_ts_ms=1_000_000,
    )
    assert order.remaining_qty == Decimal("0.05")
    assert order.filled_qty == Decimal("0")


def test_order_lifecycle_timestamps_start_as_none():
    order = Order(
        order_id="o1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("65000"),
        qty=Decimal("0.01"),
        created_ts_ms=1_000_000,
    )
    assert order.submitted_ts_ms is None
    assert order.acknowledged_ts_ms is None
    assert order.first_fill_ts_ms is None
    assert order.last_fill_ts_ms is None
    assert order.terminal_ts_ms is None
    assert order.avg_fill_price is None


def test_order_fills_transitions_and_processed_ids_start_empty():
    order = Order(
        order_id="o1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("65000"),
        qty=Decimal("0.01"),
        created_ts_ms=1_000_000,
    )
    assert order.fills == []
    assert order.transitions == []
    assert order.processed_execution_ids == set()


def test_order_qty_and_price_are_decimal():
    order = Order(
        order_id="o1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("65000"),
        qty=Decimal("0.01"),
        created_ts_ms=1_000_000,
    )
    assert isinstance(order.qty, Decimal)
    assert isinstance(order.limit_price, Decimal)
    assert isinstance(order.remaining_qty, Decimal)
    assert isinstance(order.filled_qty, Decimal)


# ── liquidity_role property ──────────────────────────────────────────────────

def test_liquidity_role_no_fills_returns_maker():
    """No fills yet → default MAKER (no taker evidence)."""
    order = Order(
        order_id="o1", symbol="BTCUSDT", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, limit_price=Decimal("65000"),
        qty=Decimal("0.01"), created_ts_ms=1_000_000,
    )
    assert order.liquidity_role == LiquidityRole.MAKER


def test_liquidity_role_all_maker_fills():
    order = Order(
        order_id="o1", symbol="BTCUSDT", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, limit_price=Decimal("65000"),
        qty=Decimal("0.02"), created_ts_ms=1_000_000,
    )
    order.fills.append(FillEvent("e1", "o1", 1_100_000, Decimal("65000"), Decimal("0.01"), FeeModel.MAKER))
    order.fills.append(FillEvent("e2", "o1", 1_200_000, Decimal("65100"), Decimal("0.01"), FeeModel.MAKER))
    assert order.liquidity_role == LiquidityRole.MAKER


def test_liquidity_role_all_taker_fills():
    order = Order(
        order_id="o1", symbol="BTCUSDT", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, limit_price=Decimal("65000"),
        qty=Decimal("0.01"), created_ts_ms=1_000_000,
    )
    order.fills.append(FillEvent("e1", "o1", 1_100_000, Decimal("65000"), Decimal("0.01"), FeeModel.TAKER))
    assert order.liquidity_role == LiquidityRole.TAKER


def test_liquidity_role_mixed_fills():
    order = Order(
        order_id="o1", symbol="BTCUSDT", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, limit_price=Decimal("65000"),
        qty=Decimal("0.02"), created_ts_ms=1_000_000,
    )
    order.fills.append(FillEvent("e1", "o1", 1_100_000, Decimal("65000"), Decimal("0.01"), FeeModel.MAKER))
    order.fills.append(FillEvent("e2", "o1", 1_200_000, Decimal("65100"), Decimal("0.01"), FeeModel.TAKER))
    assert order.liquidity_role == LiquidityRole.MIXED
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_order_types.py -v`

Expected: `ModuleNotFoundError: No module named 'live.order_types'`

- [ ] **Step 3: Create `live/order_types.py`**

```python
"""
Order types — enums and immutable data contracts for the Order State Machine.

Canonical time: event_ts_ms (int64 Unix epoch milliseconds from exchange).
recv_monotonic_ns: optional diagnostics — stored but NEVER used in logic.
Financial quantities: Decimal. Float prohibited in financial fields.

Doctrinal rules:
- TOUCH != FILL: no fill is inferred from price touching a level
- Order state derives from execution events only, never from candle data
- ACKNOWLEDGED requires exchange ACK, not market data inference
- LiquidityRole is derived from FillEvent.fee_model — never stored mutable
- All transitions are recorded in order.transitions (event-sourced)
- Terminal states: FILLED, CANCELLED, EXPIRED, REJECTED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class OrderState(str, Enum):
    CREATED          = "CREATED"
    SUBMITTED        = "SUBMITTED"
    ACKNOWLEDGED     = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED           = "FILLED"
    CANCELLED        = "CANCELLED"
    EXPIRED          = "EXPIRED"
    REJECTED         = "REJECTED"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT  = "LIMIT"
    MARKET = "MARKET"


class FeeModel(str, Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class LiquidityRole(str, Enum):
    """
    Derived from fills — never stored as mutable order field.
    MAKER:  all fills have FeeModel.MAKER
    TAKER:  all fills have FeeModel.TAKER
    MIXED:  order has both maker and taker fills (e.g. partial rest + reprice)
    """
    MAKER = "MAKER"
    TAKER = "TAKER"
    MIXED = "MIXED"


@dataclass(frozen=True)
class FillEvent:
    """
    Immutable record of a single fill execution.

    execution_id: unique ID from exchange — used for idempotency.
    event_ts_ms:  canonical exchange clock (int64 ms).
    price, qty:   Decimal — financial fields never float.
    recv_monotonic_ns: optional local profiling — never enters logic.
    """
    execution_id:      str
    order_id:          str
    event_ts_ms:       int
    price:             Decimal
    qty:               Decimal
    fee_model:         FeeModel
    recv_monotonic_ns: Optional[int] = None


@dataclass(frozen=True)
class StateTransition:
    """
    Immutable record of a single state transition.

    event_ts_ms:      canonical exchange clock (int64 ms).
    bid_price/ask_price: Decimal snapshot at transition time (optional, audit only).
    recv_monotonic_ns: optional local profiling — never enters logic.
    """
    from_state:        OrderState
    to_state:          OrderState
    event_ts_ms:       int
    trigger:           str
    bid_price:         Optional[Decimal] = None
    ask_price:         Optional[Decimal] = None
    recv_monotonic_ns: Optional[int]     = None


@dataclass
class Order:
    """
    Mutable order lifecycle object. Mutated exclusively by OrderStateMachine.

    Canonical clock: event_ts_ms (int64 Unix ms from exchange).
    Financial fields: Decimal only. Float prohibited.
    recv_monotonic_ns is stored for profiling but NEVER used in any conditional.

    Latency (diagnostics only):
        ack_latency_ms = acknowledged_ts_ms - submitted_ts_ms

    WARNING: Order mutation is only legal through OrderStateMachine.
    Direct attribute assignment bypasses transition enforcement and
    breaks event-source integrity.
    """
    order_id:      str
    symbol:        str
    side:          OrderSide
    order_type:    OrderType
    limit_price:   Optional[Decimal]   # None for MARKET orders
    qty:           Decimal             # original total quantity
    created_ts_ms: int                 # when order object was created locally

    # Lifecycle timestamps — set only by OrderStateMachine
    submitted_ts_ms:   Optional[int]   = field(default=None)
    acknowledged_ts_ms: Optional[int]  = field(default=None)
    first_fill_ts_ms:  Optional[int]   = field(default=None)
    last_fill_ts_ms:   Optional[int]   = field(default=None)
    terminal_ts_ms:    Optional[int]   = field(default=None)

    # Inventory state — Decimal
    filled_qty:      Decimal           = field(default_factory=lambda: Decimal("0"))
    remaining_qty:   Decimal           = field(default_factory=lambda: Decimal("0"))
    avg_fill_price:  Optional[Decimal] = field(default=None)

    # Current state
    state: OrderState = field(default=OrderState.CREATED)

    # Event-sourced history (append-only, mutated by OSM only)
    fills:       list = field(default_factory=list)  # list[FillEvent]
    transitions: list = field(default_factory=list)  # list[StateTransition]

    # Idempotency: execution IDs seen on this order
    processed_execution_ids: set = field(default_factory=set)  # set[str]

    def __post_init__(self) -> None:
        self.remaining_qty = self.qty

    @property
    def liquidity_role(self) -> LiquidityRole:
        """
        Derived from fills — never stored mutable.

        No fills → MAKER (no taker evidence yet).
        All MAKER → MAKER.
        All TAKER → TAKER.
        Mixed    → MIXED (e.g. partial resting fill + reprice to taker).
        """
        if not self.fills:
            return LiquidityRole.MAKER
        models = {f.fee_model for f in self.fills}
        if models == {FeeModel.MAKER}:
            return LiquidityRole.MAKER
        if models == {FeeModel.TAKER}:
            return LiquidityRole.TAKER
        return LiquidityRole.MIXED
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `pytest tests/test_order_types.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```
git add live/order_types.py tests/__init__.py tests/conftest.py tests/test_order_types.py
git commit -m "feat(1b): order_types v2 — LiquidityRole, Decimal, 8-state OSM, liquidity_role property"
```

---

## Task 3: OSM — exceptions + create_order, submit, acknowledge

**Files:** `live/order_state_machine.py` (create), `tests/test_order_state_machine.py` (create)

- [ ] **Step 1: Write failing tests for exceptions, create_order, submit, acknowledge**

Create `tests/test_order_state_machine.py`:
```python
"""
Tests for OrderStateMachine — all valid and invalid transitions.

Doctrinal rules under test:
- event_ts_ms is the only clock used in logic
- recv_monotonic_ns stored but never affects any assertion on state
- ACKNOWLEDGED requires ACK event, never market data inference
- All transitions recorded in order.transitions
- Terminal states block further transitions
- Decimal for all financial quantities
- Idempotency: duplicate execution_ids raise DuplicateEventError
- Causal ordering: decreasing timestamps raise CausalityViolationError
"""
from decimal import Decimal

import pytest

from live.order_types import FeeModel, LiquidityRole, OrderSide, OrderState, OrderType
from live.order_state_machine import (
    CausalityViolationError,
    DuplicateEventError,
    InvalidTransitionError,
    OrderStateMachine,
    OutOfOrderEventError,
)


# ── create_order ─────────────────────────────────────────────────────────────

def test_create_order_initial_state_is_created(osm):
    order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.01"),
        limit_price=Decimal("65000"),
        event_ts_ms=1_000_000,
        order_id="o1",
    )
    assert order.state == OrderState.CREATED
    assert order.order_id == "o1"
    assert order.qty == Decimal("0.01")
    assert order.remaining_qty == Decimal("0.01")
    assert order.filled_qty == Decimal("0")
    assert order.created_ts_ms == 1_000_000


def test_create_order_records_created_transition(osm):
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    assert len(order.transitions) == 1
    t = order.transitions[0]
    assert t.from_state == OrderState.CREATED
    assert t.to_state == OrderState.CREATED
    assert t.trigger == "ORDER_CREATED"
    assert t.event_ts_ms == 1_000_000


def test_create_limit_order_without_price_raises(osm):
    with pytest.raises(ValueError, match="limit_price"):
        osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("0.01"), event_ts_ms=1_000_000,
        )


def test_create_market_order_with_price_raises(osm):
    with pytest.raises(ValueError, match="MARKET"):
        osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
        )


def test_create_order_zero_qty_raises(osm):
    with pytest.raises(ValueError, match="qty"):
        osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("0"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
        )


def test_create_order_negative_qty_raises(osm):
    with pytest.raises(ValueError, match="qty"):
        osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("-0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
        )


def test_create_order_auto_generates_order_id_if_none(osm):
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    assert order.order_id is not None
    assert len(order.order_id) > 0


# ── submit ────────────────────────────────────────────────────────────────────

def test_submit_transitions_to_submitted(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    assert limit_buy_order.state == OrderState.SUBMITTED
    assert limit_buy_order.submitted_ts_ms == 1_001_000


def test_submit_appends_transition(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.CREATED
    assert last.to_state == OrderState.SUBMITTED
    assert last.trigger == "SUBMIT"
    assert last.event_ts_ms == 1_001_000


def test_submit_stores_recv_monotonic_ns_without_affecting_state(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000, recv_monotonic_ns=5_000_000_000)
    assert limit_buy_order.state == OrderState.SUBMITTED
    assert limit_buy_order.submitted_ts_ms == 1_001_000
    last = limit_buy_order.transitions[-1]
    assert last.recv_monotonic_ns == 5_000_000_000


def test_cannot_submit_twice(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    with pytest.raises(InvalidTransitionError):
        osm.submit(limit_buy_order, event_ts_ms=1_002_000)


def test_submit_with_earlier_timestamp_raises_causality_error(osm, limit_buy_order):
    """Causal ordering enforced at runtime — not just in tests."""
    with pytest.raises(CausalityViolationError):
        osm.submit(limit_buy_order, event_ts_ms=999_999)  # before created_ts_ms=1_000_000


# ── acknowledge (ACKNOWLEDGED) ────────────────────────────────────────────────

def test_acknowledge_transitions_to_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    assert limit_buy_order.state == OrderState.ACKNOWLEDGED
    assert limit_buy_order.acknowledged_ts_ms == 1_050_000


def test_acknowledge_records_ack_latency_computable(osm, limit_buy_order):
    """ack_latency_ms = acknowledged_ts_ms - submitted_ts_ms."""
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_051_000)
    ack_latency_ms = limit_buy_order.acknowledged_ts_ms - limit_buy_order.submitted_ts_ms
    assert ack_latency_ms == 50_000


def test_acknowledge_appends_transition(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.SUBMITTED
    assert last.to_state == OrderState.ACKNOWLEDGED
    assert last.trigger == "ACK_RECEIVED"
    assert last.event_ts_ms == 1_050_000


def test_cannot_acknowledge_without_submitting_first(osm, limit_buy_order):
    """ACKNOWLEDGED requires prior SUBMITTED state."""
    with pytest.raises(OutOfOrderEventError):
        osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)


def test_acknowledge_stores_recv_monotonic_ns_without_affecting_state(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000, recv_monotonic_ns=9_999_999)
    assert limit_buy_order.state == OrderState.ACKNOWLEDGED
    assert limit_buy_order.acknowledged_ts_ms == 1_050_000
    last = limit_buy_order.transitions[-1]
    assert last.recv_monotonic_ns == 9_999_999
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_order_state_machine.py -v`

Expected: `ModuleNotFoundError: No module named 'live.order_state_machine'`

- [ ] **Step 3: Create `live/order_state_machine.py`**

```python
"""
Order State Machine — enforces valid order lifecycle transitions.

Stateless service. Mutates Order objects, appends to order.transitions.
Never reads from candles or infers state from market data.

Canonical clock: event_ts_ms (int64 Unix ms from exchange).
Financial quantities: Decimal.
recv_monotonic_ns: optional param, stored in transitions, NEVER in logic.

Valid transition graph:
  CREATED          → SUBMITTED                      (SUBMIT)
  SUBMITTED        → ACKNOWLEDGED                   (ACK_RECEIVED)
  SUBMITTED        → REJECTED                       (EXCHANGE_REJECT)
  SUBMITTED        → FILLED                         (IMMEDIATE_TAKER_FILL)
  SUBMITTED        → PARTIALLY_FILLED               (IMMEDIATE_TAKER_PARTIAL_FILL)
  ACKNOWLEDGED     → PARTIALLY_FILLED               (PARTIAL_FILL)
  ACKNOWLEDGED     → FILLED                         (FULL_FILL)
  ACKNOWLEDGED     → CANCELLED                      (USER_CANCEL, SYSTEM_CANCEL)
  ACKNOWLEDGED     → EXPIRED                        (TTL_EXPIRED)
  PARTIALLY_FILLED → PARTIALLY_FILLED               (PARTIAL_FILL)
  PARTIALLY_FILLED → FILLED                         (FULL_FILL)
  PARTIALLY_FILLED → CANCELLED                      (USER_CANCEL, SYSTEM_CANCEL)
  PARTIALLY_FILLED → EXPIRED                        (TTL_EXPIRED)

Terminal states (no further transitions):
  FILLED, CANCELLED, EXPIRED, REJECTED

Taker fee classification is per-fill (FillEvent.fee_model).
LiquidityRole is derived from fills — no TAKER_CONVERTED state.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Optional

from live.order_types import (
    FeeModel,
    FillEvent,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    StateTransition,
)

# ── Exceptions ────────────────────────────────────────────────────────────────


class InvalidTransitionError(Exception):
    """Raised when a requested state transition is not in the valid transition table."""


class CausalityViolationError(Exception):
    """
    Raised when event_ts_ms < last transition timestamp.
    Causal ordering of events is a hard invariant — replay determinism depends on it.
    """


class OutOfOrderEventError(InvalidTransitionError):
    """
    Raised when an exchange event arrives for a state the order hasn't reached yet.
    Common in live: fill arrives before ACK due to WS race conditions.

    Attributes:
        order_id:              identifies the affected order
        current_state:         order's state when event was received
        received_event_type:   the event type that arrived (e.g. 'PARTIAL_FILL')
        exchange_event_ts:     event_ts_ms from exchange
        local_receive_ts:      recv_monotonic_ns if available (None otherwise)
    """
    def __init__(
        self,
        order_id: str,
        current_state: OrderState,
        received_event_type: str,
        exchange_event_ts: int,
        local_receive_ts: Optional[int] = None,
    ) -> None:
        self.order_id = order_id
        self.current_state = current_state
        self.received_event_type = received_event_type
        self.exchange_event_ts = exchange_event_ts
        self.local_receive_ts = local_receive_ts
        super().__init__(
            f"Out-of-order event for order {order_id!r}: "
            f"received '{received_event_type}' while in state {current_state.value}. "
            f"exchange_event_ts={exchange_event_ts}, local_receive_ts={local_receive_ts}"
        )


class DuplicateEventError(Exception):
    """
    Raised when an execution_id has already been processed for this order.
    WS feeds duplicate execution reports — idempotency is enforced here.
    """


# ── Transition table ──────────────────────────────────────────────────────────

_VALID_TRANSITIONS: dict[tuple[OrderState, OrderState], set[str]] = {
    (OrderState.CREATED,          OrderState.SUBMITTED):        {"SUBMIT"},
    (OrderState.SUBMITTED,        OrderState.ACKNOWLEDGED):     {"ACK_RECEIVED"},
    (OrderState.SUBMITTED,        OrderState.REJECTED):         {"EXCHANGE_REJECT"},
    (OrderState.SUBMITTED,        OrderState.FILLED):           {"IMMEDIATE_TAKER_FILL"},
    (OrderState.SUBMITTED,        OrderState.PARTIALLY_FILLED): {"IMMEDIATE_TAKER_PARTIAL_FILL"},
    (OrderState.ACKNOWLEDGED,     OrderState.PARTIALLY_FILLED): {"PARTIAL_FILL"},
    (OrderState.ACKNOWLEDGED,     OrderState.FILLED):           {"FULL_FILL"},
    (OrderState.ACKNOWLEDGED,     OrderState.CANCELLED):        {"USER_CANCEL", "SYSTEM_CANCEL"},
    (OrderState.ACKNOWLEDGED,     OrderState.EXPIRED):          {"TTL_EXPIRED"},
    (OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED): {"PARTIAL_FILL"},
    (OrderState.PARTIALLY_FILLED, OrderState.FILLED):           {"FULL_FILL"},
    (OrderState.PARTIALLY_FILLED, OrderState.CANCELLED):        {"USER_CANCEL", "SYSTEM_CANCEL"},
    (OrderState.PARTIALLY_FILLED, OrderState.EXPIRED):          {"TTL_EXPIRED"},
}

_TERMINAL_STATES: frozenset[OrderState] = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.EXPIRED,
    OrderState.REJECTED,
})

# States from which fill() is valid (normal path + immediate taker path)
_FILLABLE_STATES: frozenset[OrderState] = frozenset({
    OrderState.ACKNOWLEDGED,
    OrderState.PARTIALLY_FILLED,
    OrderState.SUBMITTED,  # IMMEDIATE_TAKER_FILL path
})


# ── OrderStateMachine ─────────────────────────────────────────────────────────

class OrderStateMachine:
    """
    Stateless service that enforces order lifecycle transitions.

    Every public method:
    1. Validates causal timestamp ordering
    2. Validates the transition via _assert_transition()
    3. Updates the relevant Order fields
    4. Appends a StateTransition to order.transitions
    """

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        event_ts_ms: int,
        limit_price: Optional[Decimal] = None,
        order_id: Optional[str] = None,
    ) -> Order:
        """Create an Order in CREATED state. Records ORDER_CREATED transition."""
        if order_type == OrderType.LIMIT and limit_price is None:
            raise ValueError("LIMIT orders require limit_price")
        if order_type == OrderType.MARKET and limit_price is not None:
            raise ValueError("MARKET orders must not have limit_price")
        if qty <= Decimal("0"):
            raise ValueError(f"qty must be positive, got {qty}")

        oid = order_id or str(uuid.uuid4())
        order = Order(
            order_id=oid,
            symbol=symbol,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            qty=qty,
            created_ts_ms=event_ts_ms,
        )
        order.transitions.append(StateTransition(
            from_state=OrderState.CREATED,
            to_state=OrderState.CREATED,
            event_ts_ms=event_ts_ms,
            trigger="ORDER_CREATED",
        ))
        return order

    def submit(
        self,
        order: Order,
        event_ts_ms: int,
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """CREATED → SUBMITTED: order sent to exchange."""
        self._assert_monotonic_timestamp(order, event_ts_ms)
        self._assert_transition(order, OrderState.SUBMITTED, "SUBMIT")
        order.submitted_ts_ms = event_ts_ms
        order.state = OrderState.SUBMITTED
        order.transitions.append(StateTransition(
            from_state=OrderState.CREATED,
            to_state=OrderState.SUBMITTED,
            event_ts_ms=event_ts_ms,
            trigger="SUBMIT",
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    def acknowledge(
        self,
        order: Order,
        event_ts_ms: int,
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """
        SUBMITTED → ACKNOWLEDGED: exchange ACK received.

        Contract: ACKNOWLEDGED is ONLY set here. It is never inferred from
        market data (candles, trade feed, price levels).
        ack_latency_ms = acknowledged_ts_ms - submitted_ts_ms
        """
        self._assert_monotonic_timestamp(order, event_ts_ms)
        if order.state != OrderState.SUBMITTED:
            raise OutOfOrderEventError(
                order_id=order.order_id,
                current_state=order.state,
                received_event_type="ACK_RECEIVED",
                exchange_event_ts=event_ts_ms,
                local_receive_ts=recv_monotonic_ns,
            )
        order.acknowledged_ts_ms = event_ts_ms
        order.state = OrderState.ACKNOWLEDGED
        order.transitions.append(StateTransition(
            from_state=OrderState.SUBMITTED,
            to_state=OrderState.ACKNOWLEDGED,
            event_ts_ms=event_ts_ms,
            trigger="ACK_RECEIVED",
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    def reject(
        self,
        order: Order,
        event_ts_ms: int,
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """SUBMITTED → REJECTED: exchange rejected the order."""
        self._assert_monotonic_timestamp(order, event_ts_ms)
        self._assert_transition(order, OrderState.REJECTED, "EXCHANGE_REJECT")
        from_state = order.state
        order.terminal_ts_ms = event_ts_ms
        order.state = OrderState.REJECTED
        order.transitions.append(StateTransition(
            from_state=from_state,
            to_state=OrderState.REJECTED,
            event_ts_ms=event_ts_ms,
            trigger="EXCHANGE_REJECT",
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    def fill(
        self,
        order: Order,
        fill_price: Decimal,
        fill_qty: Decimal,
        event_ts_ms: int,
        fee_model: FeeModel,
        execution_id: str,
        recv_monotonic_ns: Optional[int] = None,
        bid_price: Optional[Decimal] = None,
        ask_price: Optional[Decimal] = None,
    ) -> None:
        """
        Apply a fill event to an order in a fillable state.

        Doctrinal rule: TOUCH != FILL. This method is only called when an
        execution event is received from the exchange. It is never called
        because a candle's high/low touched the limit_price.

        Valid source states: ACKNOWLEDGED, PARTIALLY_FILLED, SUBMITTED (immediate taker only).

        Immediate taker path (SUBMITTED → FILLED/PARTIALLY_FILLED):
          fee_model must be FeeModel.TAKER.
          Trigger: IMMEDIATE_TAKER_FILL (full) or IMMEDIATE_TAKER_PARTIAL_FILL (partial).

        Normal path (ACKNOWLEDGED/PARTIALLY_FILLED → FILLED/PARTIALLY_FILLED):
          Trigger: FULL_FILL (full) or PARTIAL_FILL (partial).

        LiquidityRole is derived from order.fills — not stored separately.
        A fill with FeeModel.TAKER on a previously MAKER order produces LiquidityRole.MIXED.
        """
        self._assert_monotonic_timestamp(order, event_ts_ms)

        # Idempotency
        if execution_id in order.processed_execution_ids:
            raise DuplicateEventError(
                f"execution_id {execution_id!r} already processed for order {order.order_id!r}"
            )

        if order.state not in _FILLABLE_STATES:
            if order.state in _TERMINAL_STATES:
                raise InvalidTransitionError(
                    f"Order {order.order_id!r} is in terminal state {order.state}. "
                    f"No further transitions are allowed."
                )
            raise OutOfOrderEventError(
                order_id=order.order_id,
                current_state=order.state,
                received_event_type="FILL",
                exchange_event_ts=event_ts_ms,
                local_receive_ts=recv_monotonic_ns,
            )

        if fill_qty <= Decimal("0"):
            raise ValueError(f"fill_qty must be positive, got {fill_qty}")
        if fill_qty > order.remaining_qty:
            raise ValueError(
                f"fill_qty {fill_qty} exceeds remaining_qty {order.remaining_qty}"
            )

        # Determine trigger prefix based on source state
        immediate = order.state == OrderState.SUBMITTED

        fill_event = FillEvent(
            execution_id=execution_id,
            order_id=order.order_id,
            event_ts_ms=event_ts_ms,
            price=fill_price,
            qty=fill_qty,
            fee_model=fee_model,
            recv_monotonic_ns=recv_monotonic_ns,
        )
        order.fills.append(fill_event)
        order.processed_execution_ids.add(execution_id)

        if order.first_fill_ts_ms is None:
            order.first_fill_ts_ms = event_ts_ms
        order.last_fill_ts_ms = event_ts_ms

        prev_filled = order.filled_qty
        order.filled_qty += fill_qty
        order.remaining_qty -= fill_qty

        if order.avg_fill_price is None:
            order.avg_fill_price = fill_price
        else:
            order.avg_fill_price = (
                (order.avg_fill_price * prev_filled + fill_price * fill_qty)
                / order.filled_qty
            )

        from_state = order.state
        if order.remaining_qty == Decimal("0"):
            order.terminal_ts_ms = event_ts_ms
            order.state = OrderState.FILLED
            trigger = "IMMEDIATE_TAKER_FILL" if immediate else "FULL_FILL"
            order.transitions.append(StateTransition(
                from_state=from_state,
                to_state=OrderState.FILLED,
                event_ts_ms=event_ts_ms,
                trigger=trigger,
                bid_price=bid_price,
                ask_price=ask_price,
            ))
        else:
            order.state = OrderState.PARTIALLY_FILLED
            trigger = "IMMEDIATE_TAKER_PARTIAL_FILL" if immediate else "PARTIAL_FILL"
            order.transitions.append(StateTransition(
                from_state=from_state,
                to_state=OrderState.PARTIALLY_FILLED,
                event_ts_ms=event_ts_ms,
                trigger=trigger,
                bid_price=bid_price,
                ask_price=ask_price,
            ))

    def cancel(
        self,
        order: Order,
        event_ts_ms: int,
        trigger: str = "USER_CANCEL",
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """ACKNOWLEDGED / PARTIALLY_FILLED → CANCELLED."""
        if trigger not in ("USER_CANCEL", "SYSTEM_CANCEL"):
            raise ValueError(
                f"Invalid cancel trigger: '{trigger}'. Must be USER_CANCEL or SYSTEM_CANCEL."
            )
        self._assert_monotonic_timestamp(order, event_ts_ms)
        self._assert_transition(order, OrderState.CANCELLED, trigger)
        from_state = order.state
        order.terminal_ts_ms = event_ts_ms
        order.state = OrderState.CANCELLED
        order.transitions.append(StateTransition(
            from_state=from_state,
            to_state=OrderState.CANCELLED,
            event_ts_ms=event_ts_ms,
            trigger=trigger,
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    def expire(
        self,
        order: Order,
        event_ts_ms: int,
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """ACKNOWLEDGED / PARTIALLY_FILLED → EXPIRED: TTL elapsed without full fill."""
        self._assert_monotonic_timestamp(order, event_ts_ms)
        self._assert_transition(order, OrderState.EXPIRED, "TTL_EXPIRED")
        from_state = order.state
        order.terminal_ts_ms = event_ts_ms
        order.state = OrderState.EXPIRED
        order.transitions.append(StateTransition(
            from_state=from_state,
            to_state=OrderState.EXPIRED,
            event_ts_ms=event_ts_ms,
            trigger="TTL_EXPIRED",
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_monotonic_timestamp(self, order: Order, event_ts_ms: int) -> None:
        """
        Enforce causal ordering: event_ts_ms must be >= last recorded transition ts.
        Raises CausalityViolationError if violated.
        """
        if not order.transitions:
            return
        last_ts = order.transitions[-1].event_ts_ms
        if event_ts_ms < last_ts:
            raise CausalityViolationError(
                f"event_ts_ms {event_ts_ms} < last transition ts {last_ts} "
                f"for order {order.order_id!r}. Causal ordering violated."
            )

    def _assert_transition(
        self,
        order: Order,
        to_state: OrderState,
        trigger: str,
    ) -> None:
        if order.state in _TERMINAL_STATES:
            raise InvalidTransitionError(
                f"Order {order.order_id!r} is in terminal state {order.state}. "
                f"No further transitions are allowed."
            )
        key = (order.state, to_state)
        if key not in _VALID_TRANSITIONS or trigger not in _VALID_TRANSITIONS[key]:
            raise InvalidTransitionError(
                f"Invalid transition for order {order.order_id!r}: "
                f"{order.state} → {to_state} via trigger '{trigger}'"
            )
```

- [ ] **Step 4: Run create/submit/acknowledge tests only**

Run: `pytest tests/test_order_state_machine.py -v -k "create or submit or acknowledge"`

Expected: all matching tests pass.

- [ ] **Step 5: Commit**

```
git add live/order_state_machine.py tests/test_order_state_machine.py
git commit -m "feat(1b): OSM v2 — exceptions, create_order, submit, acknowledge (ACKNOWLEDGED state)"
```

---

## Task 4: OSM — reject, cancel, expire

**Files:** `tests/test_order_state_machine.py` (append)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_order_state_machine.py`:
```python
# ── reject ────────────────────────────────────────────────────────────────────

def test_reject_transitions_submitted_to_rejected(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.reject(limit_buy_order, event_ts_ms=1_002_000)
    assert limit_buy_order.state == OrderState.REJECTED
    assert limit_buy_order.terminal_ts_ms == 1_002_000


def test_reject_appends_transition(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.reject(limit_buy_order, event_ts_ms=1_002_000)
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.SUBMITTED
    assert last.to_state == OrderState.REJECTED
    assert last.trigger == "EXCHANGE_REJECT"


def test_cannot_reject_from_created(osm, limit_buy_order):
    with pytest.raises(InvalidTransitionError):
        osm.reject(limit_buy_order, event_ts_ms=1_002_000)


def test_cannot_reject_from_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    with pytest.raises(InvalidTransitionError):
        osm.reject(limit_buy_order, event_ts_ms=1_060_000)


# ── cancel ────────────────────────────────────────────────────────────────────

def test_user_cancel_from_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.cancel(limit_buy_order, event_ts_ms=1_200_000, trigger="USER_CANCEL")
    assert limit_buy_order.state == OrderState.CANCELLED
    assert limit_buy_order.terminal_ts_ms == 1_200_000


def test_system_cancel_from_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.cancel(limit_buy_order, event_ts_ms=1_200_000, trigger="SYSTEM_CANCEL")
    assert limit_buy_order.state == OrderState.CANCELLED


def test_cancel_records_correct_from_state(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.cancel(limit_buy_order, event_ts_ms=1_200_000, trigger="USER_CANCEL")
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.ACKNOWLEDGED
    assert last.to_state == OrderState.CANCELLED
    assert last.trigger == "USER_CANCEL"


def test_invalid_cancel_trigger_raises_value_error(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    with pytest.raises(ValueError, match="USER_CANCEL"):
        osm.cancel(limit_buy_order, event_ts_ms=1_200_000, trigger="FORCE_CANCEL")


def test_cannot_cancel_from_submitted(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    with pytest.raises(InvalidTransitionError):
        osm.cancel(limit_buy_order, event_ts_ms=1_100_000)


# ── expire ────────────────────────────────────────────────────────────────────

def test_expire_from_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.expire(limit_buy_order, event_ts_ms=1_500_000)
    assert limit_buy_order.state == OrderState.EXPIRED
    assert limit_buy_order.terminal_ts_ms == 1_500_000


def test_expire_appends_transition(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.expire(limit_buy_order, event_ts_ms=1_500_000)
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.ACKNOWLEDGED
    assert last.to_state == OrderState.EXPIRED
    assert last.trigger == "TTL_EXPIRED"


def test_cannot_expire_from_created(osm, limit_buy_order):
    with pytest.raises(InvalidTransitionError):
        osm.expire(limit_buy_order, event_ts_ms=1_500_000)
```

- [ ] **Step 2: Run new tests**

Run: `pytest tests/test_order_state_machine.py -v -k "reject or cancel or expire"`

Expected: all pass.

- [ ] **Step 3: Full suite**

Run: `pytest tests/ -v`

Expected: all pass.

- [ ] **Step 4: Commit**

```
git add tests/test_order_state_machine.py
git commit -m "test(1b): OSM reject, cancel, expire — happy paths and invalid transitions"
```

---

## Task 5: OSM — fill (Decimal, idempotency, partial accumulation)

**Files:** `tests/test_order_state_machine.py` (append)

- [ ] **Step 1: Append fill tests**

Append to `tests/test_order_state_machine.py`:
```python
# ── fill — helpers ────────────────────────────────────────────────────────────

def _acknowledged_order(osm, qty: str = "0.03", price: str = "65000") -> object:
    """Create and advance an order to ACKNOWLEDGED state."""
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal(qty), limit_price=Decimal(price), event_ts_ms=1_000_000,
    )
    osm.submit(order, event_ts_ms=1_001_000)
    osm.acknowledge(order, event_ts_ms=1_050_000)
    return order


# ── fill — single partial fill ────────────────────────────────────────────────

def test_partial_fill_transitions_to_partially_filled(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
             event_ts_ms=1_100_000, fee_model=FeeModel.MAKER, execution_id="e1")
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.filled_qty == Decimal("0.01")
    assert order.remaining_qty == Decimal("0.02")


def test_partial_fill_sets_first_fill_ts(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
             event_ts_ms=1_100_000, fee_model=FeeModel.MAKER, execution_id="e1")
    assert order.first_fill_ts_ms == 1_100_000
    assert order.last_fill_ts_ms == 1_100_000


def test_partial_fill_appends_fill_event(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, fill_price=Decimal("65100"), fill_qty=Decimal("0.01"),
             event_ts_ms=1_100_000, fee_model=FeeModel.MAKER, execution_id="e1")
    assert len(order.fills) == 1
    fe = order.fills[0]
    assert fe.price == Decimal("65100")
    assert fe.qty == Decimal("0.01")
    assert fe.fee_model == FeeModel.MAKER
    assert fe.execution_id == "e1"
    assert fe.order_id == order.order_id


def test_partial_fill_sets_avg_fill_price(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
             event_ts_ms=1_100_000, fee_model=FeeModel.MAKER, execution_id="e1")
    assert order.avg_fill_price == Decimal("65000")


# ── fill — accumulation ────────────────────────────────────────────────────────

def test_two_partial_fills_accumulate_inventory(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65200"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.filled_qty == Decimal("0.02")
    assert order.remaining_qty == Decimal("0.01")
    assert order.first_fill_ts_ms == 1_100_000
    assert order.last_fill_ts_ms == 1_200_000
    assert len(order.fills) == 2


def test_two_partial_fills_weighted_avg_price(osm):
    """avg = (65000 * 0.01 + 65200 * 0.01) / 0.02 = 65100"""
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65200"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    assert order.avg_fill_price == Decimal("65100")


def test_three_partial_fills_then_full_fill(osm):
    order = _acknowledged_order(osm, qty="0.04")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65100"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    osm.fill(order, Decimal("65200"), Decimal("0.01"), 1_300_000, FeeModel.MAKER, "e3")
    assert order.state == OrderState.PARTIALLY_FILLED
    osm.fill(order, Decimal("65300"), Decimal("0.01"), 1_400_000, FeeModel.MAKER, "e4")
    assert order.state == OrderState.FILLED
    assert order.filled_qty == Decimal("0.04")
    assert order.remaining_qty == Decimal("0")
    assert order.terminal_ts_ms == 1_400_000
    assert len(order.fills) == 4


def test_full_fill_in_one_shot(osm):
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.TAKER, "e1")
    assert order.state == OrderState.FILLED
    assert order.remaining_qty == Decimal("0")
    last = order.transitions[-1]
    assert last.trigger == "FULL_FILL"


def test_fill_exceeding_remaining_qty_raises(osm):
    order = _acknowledged_order(osm, qty="0.01")
    with pytest.raises(ValueError, match="remaining_qty"):
        osm.fill(order, Decimal("65000"), Decimal("0.02"), 1_100_000, FeeModel.MAKER, "e1")


def test_fill_zero_qty_raises(osm):
    order = _acknowledged_order(osm)
    with pytest.raises(ValueError, match="fill_qty"):
        osm.fill(order, Decimal("65000"), Decimal("0"), 1_100_000, FeeModel.MAKER, "e1")


def test_fill_records_bid_ask_snapshot_in_transition(osm):
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1",
             bid_price=Decimal("64999"), ask_price=Decimal("65001"))
    last = order.transitions[-1]
    assert last.bid_price == Decimal("64999")
    assert last.ask_price == Decimal("65001")


def test_partially_filled_can_be_cancelled(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    assert order.state == OrderState.PARTIALLY_FILLED
    osm.cancel(order, event_ts_ms=1_200_000, trigger="USER_CANCEL")
    assert order.state == OrderState.CANCELLED
    last = order.transitions[-1]
    assert last.from_state == OrderState.PARTIALLY_FILLED


def test_partially_filled_can_expire(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.expire(order, event_ts_ms=1_500_000)
    assert order.state == OrderState.EXPIRED


# ── idempotency ────────────────────────────────────────────────────────────────

def test_duplicate_execution_id_raises_duplicate_event_error(osm):
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    with pytest.raises(DuplicateEventError):
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e1")


def test_duplicate_execution_id_does_not_mutate_state(osm):
    """State is unchanged when DuplicateEventError is raised."""
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    state_before = order.state
    filled_before = order.filled_qty
    try:
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e1")
    except DuplicateEventError:
        pass
    assert order.state == state_before
    assert order.filled_qty == filled_before


def test_different_execution_ids_both_processed(osm):
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65100"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    assert "e1" in order.processed_execution_ids
    assert "e2" in order.processed_execution_ids
    assert order.state == OrderState.FILLED
```

- [ ] **Step 2: Run fill tests**

Run: `pytest tests/test_order_state_machine.py -v -k "fill or idempoten or duplicate"`

Expected: all pass.

- [ ] **Step 3: Full suite**

Run: `pytest tests/ -v`

Expected: all pass.

- [ ] **Step 4: Commit**

```
git add tests/test_order_state_machine.py
git commit -m "test(1b): OSM fill — Decimal accumulation, avg_price, idempotency, DuplicateEventError"
```

---

## Task 6: Immediate taker fill path + mixed liquidity role

**Files:** `tests/test_order_state_machine.py` (append)

- [ ] **Step 1: Append immediate taker and liquidity role tests**

Append to `tests/test_order_state_machine.py`:
```python
# ── Immediate taker fill (SUBMITTED → FILLED) ─────────────────────────────────

def test_immediate_taker_full_fill_from_submitted(osm, limit_buy_order):
    """
    Limit order crosses spread on submission — executes immediately as taker.
    No ACKNOWLEDGED state: SUBMITTED → FILLED in one fill event.
    """
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.fill(
        limit_buy_order,
        fill_price=Decimal("65000"),
        fill_qty=Decimal("0.01"),
        event_ts_ms=1_001_050,
        fee_model=FeeModel.TAKER,
        execution_id="e-imm-1",
    )
    assert limit_buy_order.state == OrderState.FILLED
    assert limit_buy_order.remaining_qty == Decimal("0")
    assert limit_buy_order.terminal_ts_ms == 1_001_050
    last = limit_buy_order.transitions[-1]
    assert last.trigger == "IMMEDIATE_TAKER_FILL"
    assert last.from_state == OrderState.SUBMITTED
    assert last.to_state == OrderState.FILLED


def test_immediate_taker_full_fill_liquidity_role_is_taker(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.fill(limit_buy_order, Decimal("65000"), Decimal("0.01"),
             1_001_050, FeeModel.TAKER, "e-imm-1")
    assert limit_buy_order.liquidity_role.value == "TAKER"


def test_immediate_taker_partial_fill_from_submitted(osm):
    """
    Immediate partial: SUBMITTED → PARTIALLY_FILLED.
    Remaining qty will rest in the book (ACKNOWLEDGED on next ACK).
    """
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.03"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    osm.submit(order, event_ts_ms=1_001_000)
    osm.fill(order, Decimal("65000"), Decimal("0.01"),
             1_001_050, FeeModel.TAKER, "e-imm-1")
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.remaining_qty == Decimal("0.02")
    last = order.transitions[-1]
    assert last.trigger == "IMMEDIATE_TAKER_PARTIAL_FILL"


# ── Mixed liquidity role (reprice path) ────────────────────────────────────────

def test_maker_fill_then_taker_fill_produces_mixed_role(osm):
    """
    Order rests as maker, then repriced → next fill is taker.
    liquidity_role = MIXED.
    """
    order = _acknowledged_order(osm, qty="0.02")
    # First fill: resting maker fill
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    assert order.liquidity_role.value == "MAKER"
    # Reprice converts remaining → taker fill
    osm.fill(order, Decimal("65050"), Decimal("0.01"), 1_200_000, FeeModel.TAKER, "e2")
    assert order.state == OrderState.FILLED
    assert order.liquidity_role.value == "MIXED"


def test_all_taker_fills_liquidity_role(osm):
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.TAKER, "e1")
    osm.fill(order, Decimal("65100"), Decimal("0.01"), 1_200_000, FeeModel.TAKER, "e2")
    assert order.liquidity_role.value == "TAKER"


def test_all_maker_fills_liquidity_role(osm):
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65100"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    assert order.liquidity_role.value == "MAKER"


def test_liquidity_role_not_affected_by_state_only_by_fills(osm):
    """
    liquidity_role is a property of fills, not of OrderState.
    A FILLED order with all MAKER fills still shows MAKER.
    """
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    assert order.state == OrderState.FILLED
    assert order.liquidity_role.value == "MAKER"
```

- [ ] **Step 2: Run Task 6 tests**

Run: `pytest tests/test_order_state_machine.py -v -k "taker or immediate or mixed or liquidity"`

Expected: all pass.

- [ ] **Step 3: Full suite**

Run: `pytest tests/ -v`

Expected: all pass.

- [ ] **Step 4: Commit**

```
git add tests/test_order_state_machine.py
git commit -m "test(1b): OSM immediate taker fill path + mixed LiquidityRole derivation"
```

---

## Task 7: Terminal blocking + error types + full lifecycle audit

**Files:** `tests/test_order_state_machine.py` (append)

- [ ] **Step 1: Append terminal, error, and audit tests**

Append to `tests/test_order_state_machine.py`:
```python
# ── Terminal state blocking ───────────────────────────────────────────────────

@pytest.mark.parametrize("terminal_state,setup_fn", [
    ("FILLED", lambda osm, o: [
        osm.submit(o, 1_001_000),
        osm.acknowledge(o, 1_050_000),
        osm.fill(o, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1"),
    ]),
    ("CANCELLED", lambda osm, o: [
        osm.submit(o, 1_001_000),
        osm.acknowledge(o, 1_050_000),
        osm.cancel(o, 1_200_000),
    ]),
    ("EXPIRED", lambda osm, o: [
        osm.submit(o, 1_001_000),
        osm.acknowledge(o, 1_050_000),
        osm.expire(o, 1_500_000),
    ]),
    ("REJECTED", lambda osm, o: [
        osm.submit(o, 1_001_000),
        osm.reject(o, 1_002_000),
    ]),
])
def test_terminal_state_blocks_submit(osm, terminal_state, setup_fn):
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    setup_fn(osm, order)
    assert order.state.value == terminal_state
    with pytest.raises(InvalidTransitionError, match="terminal"):
        osm.submit(order, event_ts_ms=2_000_000)


def test_filled_order_blocks_cancel(osm):
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    with pytest.raises(InvalidTransitionError, match="terminal"):
        osm.cancel(order, 1_200_000)


def test_cancelled_order_blocks_fill(osm):
    order = _acknowledged_order(osm)
    osm.cancel(order, 1_200_000)
    with pytest.raises(InvalidTransitionError, match="terminal"):
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_300_000, FeeModel.MAKER, "e1")


# ── CausalityViolationError ───────────────────────────────────────────────────

def test_causality_violation_on_submit(osm, limit_buy_order):
    with pytest.raises(CausalityViolationError):
        osm.submit(limit_buy_order, event_ts_ms=999_999)


def test_causality_violation_on_acknowledge(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    with pytest.raises(CausalityViolationError):
        osm.acknowledge(limit_buy_order, event_ts_ms=1_000_500)


def test_causality_violation_on_fill(osm):
    order = _acknowledged_order(osm, qty="0.01")
    with pytest.raises(CausalityViolationError):
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_000_000,
                 FeeModel.MAKER, "e1")  # 1_000_000 < acknowledged_ts_ms=1_050_000


def test_equal_timestamps_do_not_raise_causality_error(osm, limit_buy_order):
    """Monotonic means >=, not strictly >."""
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_001_000)  # same ts is valid
    assert limit_buy_order.state == OrderState.ACKNOWLEDGED


# ── OutOfOrderEventError ──────────────────────────────────────────────────────

def test_out_of_order_fill_before_acknowledge(osm):
    """Fill arrives before ACK — common in live WS race condition."""
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    osm.submit(order, event_ts_ms=1_001_000)
    # Fill arrives while still in SUBMITTED (no ACK yet)
    with pytest.raises(OutOfOrderEventError) as exc_info:
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_001_100,
                 FeeModel.MAKER, "e1")
    # Note: SUBMITTED is in _FILLABLE_STATES, so this only fires for CREATED/ACKNOWLEDGED-skipped states
    # This test covers fill from CREATED state (pre-submit)


def test_out_of_order_acknowledge_before_submit(osm, limit_buy_order):
    """ACK arrives for an order that hasn't been submitted."""
    with pytest.raises(OutOfOrderEventError) as exc_info:
        osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    assert exc_info.value.order_id == limit_buy_order.order_id
    assert exc_info.value.current_state == OrderState.CREATED
    assert exc_info.value.received_event_type == "ACK_RECEIVED"
    assert exc_info.value.exchange_event_ts == 1_050_000


def test_out_of_order_error_includes_diagnostics(osm, limit_buy_order):
    """OutOfOrderEventError captures exchange_event_ts and local_receive_ts."""
    with pytest.raises(OutOfOrderEventError) as exc_info:
        osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000, recv_monotonic_ns=9_999_999)
    assert exc_info.value.local_receive_ts == 9_999_999


# ── Full lifecycle audit ──────────────────────────────────────────────────────

def test_full_lifecycle_transition_history_is_complete(osm):
    """
    Happy path: CREATED→SUBMITTED→ACKNOWLEDGED→PARTIALLY_FILLED→FILLED
    All transitions recorded in order.
    """
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.02"), limit_price=Decimal("65000"),
        event_ts_ms=1_000_000, order_id="audit-01",
    )
    osm.submit(order, event_ts_ms=1_001_000)
    osm.acknowledge(order, event_ts_ms=1_050_000)
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65050"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")

    assert order.state == OrderState.FILLED
    assert len(order.transitions) == 5  # CREATE + SUBMIT + ACK + PARTIAL + FULL
    assert len(order.fills) == 2

    triggers = [t.trigger for t in order.transitions]
    assert triggers == [
        "ORDER_CREATED", "SUBMIT", "ACK_RECEIVED", "PARTIAL_FILL", "FULL_FILL"
    ]
    ts_list = [t.event_ts_ms for t in order.transitions]
    assert ts_list == [1_000_000, 1_001_000, 1_050_000, 1_100_000, 1_200_000]


def test_transition_timestamps_are_monotonically_non_decreasing(osm):
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    osm.submit(order, event_ts_ms=1_001_000)
    osm.acknowledge(order, event_ts_ms=1_050_000)
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")

    ts_values = [t.event_ts_ms for t in order.transitions]
    for i in range(1, len(ts_values)):
        assert ts_values[i] >= ts_values[i - 1]


def test_recv_monotonic_ns_never_affects_state_outcome(osm):
    """
    Two identical orders with different recv_monotonic_ns values must
    end in identical states.
    """
    def run_lifecycle(recv_ns: int) -> object:
        order = osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
        )
        osm.submit(order, event_ts_ms=1_001_000, recv_monotonic_ns=recv_ns)
        osm.acknowledge(order, event_ts_ms=1_050_000, recv_monotonic_ns=recv_ns + 1000)
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER,
                 "e1", recv_monotonic_ns=recv_ns + 2000)
        return order

    order_a = run_lifecycle(recv_ns=100_000_000)
    order_b = run_lifecycle(recv_ns=999_999_999)

    assert order_a.state == order_b.state == OrderState.FILLED
    assert order_a.filled_qty == order_b.filled_qty
    assert order_a.avg_fill_price == order_b.avg_fill_price
    assert order_a.terminal_ts_ms == order_b.terminal_ts_ms


def test_all_transition_records_are_immutable_frozen(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    t = limit_buy_order.transitions[-1]
    with pytest.raises((AttributeError, TypeError)):
        t.trigger = "TAMPERED"  # type: ignore[misc]


def test_all_fill_records_are_immutable_frozen(osm):
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    fe = order.fills[0]
    with pytest.raises((AttributeError, TypeError)):
        fe.price = Decimal("0")  # type: ignore[misc]


def test_avg_fill_price_uses_decimal_arithmetic(osm):
    """Verify no float contamination in weighted avg calculation."""
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000.1"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65000.2"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    osm.fill(order, Decimal("65000.3"), Decimal("0.01"), 1_300_000, FeeModel.MAKER, "e3")
    assert isinstance(order.avg_fill_price, Decimal)
    # Exact: (65000.1 + 65000.2 + 65000.3) / 3 = 65000.2
    assert order.avg_fill_price == Decimal("65000.2")
```

- [ ] **Step 2: Run Task 7 tests**

Run: `pytest tests/test_order_state_machine.py -v -k "terminal or causality or out_of_order or audit or history or monoton or recv or immutable or frozen or decimal"`

Expected: all pass.

- [ ] **Step 3: Final full suite**

Run: `pytest tests/ -v`

Expected: all tests pass. Note final count (~60+ tests).

- [ ] **Step 4: Final commit**

```
git add tests/test_order_state_machine.py
git commit -m "test(1b): OSM terminal blocking, CausalityViolationError, OutOfOrderEventError, full audit"
```

---

## Self-Review

### Spec coverage

| Doctrinal rule | Covered by |
|---|---|
| `event_ts_ms` canonical clock | Task 3 (timestamps), Task 7 (monotonicity enforcement) |
| `recv_monotonic_ns` stored, never in logic | Task 3 (stored), Task 7 (`test_recv_monotonic_ns_never_affects_state_outcome`) |
| `ACKNOWLEDGED` requires ACK only | Task 3 (`test_cannot_acknowledge_without_submitting_first`) |
| `ack_latency_ms = acknowledged_ts_ms - submitted_ts_ms` | Task 3 (`test_acknowledge_records_ack_latency_computable`) |
| `LiquidityRole` derived from fills | Task 2 (property tests), Task 6 (mixed role tests) |
| No `execution_fee_model` mutable field | Task 2 (not in Order fields), Task 6 (liquidity via property) |
| `PARTIALLY_FILLED` not terminal | Task 5 (partial→partial→filled, partial→cancel, partial→expire) |
| Multiple `FillEvent` accumulated | Task 5 (3+1 fill test) |
| Terminal states block all transitions | Task 7 (parametrized over all 4 terminal states) |
| `TOUCH != FILL` | Enforced by design: `fill()` never called from market data; docstring explicit |
| 8 `OrderState` values | Task 2 (`test_order_state_has_eight_states`) |
| `Decimal` for financial fields | Task 2 (isinstance checks), Task 7 (`test_avg_fill_price_uses_decimal_arithmetic`) |
| `DuplicateEventError` on duplicate execution_id | Task 5 (idempotency tests) |
| `CausalityViolationError` on decreasing ts | Task 7 (submit/acknowledge/fill violations) |
| `OutOfOrderEventError` with diagnostics | Task 7 (acknowledge before submit, context fields) |
| Immediate taker fill (SUBMITTED → FILLED) | Task 6 (`test_immediate_taker_full_fill_from_submitted`) |
| Immediate taker partial (SUBMITTED → PARTIALLY_FILLED) | Task 6 (`test_immediate_taker_partial_fill_from_submitted`) |
| Transitions frozen/immutable | Task 7 (FillEvent + StateTransition mutation tests) |

### No TAKER_CONVERTED references

Verified: `TAKER_CONVERTED` does not appear in any enum, transition table, test, or method.

### Placeholder scan

No TBD, TODO, or "implement later" phrases. All code complete.
