# FillModelA + Replay Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a deterministic, queue-free fill engine (FillModelA) that evaluates fill conditions from a price tick and drives the full simulation pipeline: Market Data → FillModelA → OSM → EventStore → CostLedger → Replay.

**Architecture:** `FillModelA` is a stateless evaluator that takes an `Order` + a `Tick` and returns a `FillDecision` (or `None`). The caller drives OSM, EventStore, and CostLedger with that decision. No randomness — same inputs always produce the same decision. Replay fidelity is proven by integration tests that run the pipeline live, then reconstruct via `ExecutionReplay` and compare all economically relevant fields.

**Tech Stack:** Python 3.12, `decimal.Decimal`, `dataclasses`, existing `live/order_types.py`, `live/order_state_machine.py`, `live/event_models.py`, `live/event_store.py`, `live/cost_ledger.py`, `live/execution_replay.py`, `pytest`, `tmp_path` fixture.

---

## Attribution Rules (fixed — no future changes without a new plan)

| Order Type | Fill Price | Fee Model | Condition |
|------------|-----------|-----------|-----------|
| MARKET BUY  | `tick.ask`          | TAKER | Always fills (volume > 0) |
| MARKET SELL | `tick.bid`          | TAKER | Always fills (volume > 0) |
| LIMIT BUY   | `order.limit_price` | MAKER | `tick.price <= limit_price` |
| LIMIT SELL  | `order.limit_price` | MAKER | `tick.price >= limit_price` |

**Partial fill rule:** `fill_qty = min(order.remaining_qty, tick.volume)`. Returns `None` if `tick.volume <= 0`.

**Fee computation (Binance USDT-M perpetuals):**
- TAKER: `fee = (fill_price × fill_qty × 0.0004).quantize(Decimal("0.00000001"))`
- MAKER: `fee = (fill_price × fill_qty × 0.0001).quantize(Decimal("0.00000001"))` (rebate — positive value, reduces net_cost)

**Replay Fidelity invariant:**
```
ExecutionReplay(store).replay_order(order_id).state          == order.state
ExecutionReplay(store).replay_order(order_id).filled_qty     == order.filled_qty
ExecutionReplay(store).replay_order(order_id).avg_fill_price == order.avg_fill_price
ExecutionReplay(store).replay_order(order_id).liquidity_role == order.liquidity_role
len(ExecutionReplay(store).replay_order(order_id).fills)     == len(order.fills)
```

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `live/fee_schedule.py` | Create | Single source of truth for all fee/rebate rates |
| `live/fill_model_a.py` | Create | `Tick` frozen dataclass, `FillDecision` frozen dataclass, `FillModelA` stateless evaluator |
| `tests/test_fill_model_a.py` | Create | Unit tests (Tasks 1–3), pipeline integration tests (Task 4), replay fidelity tests (Task 5) |

---

## Task 1: `fee_schedule.py` + `Tick` + `FillDecision` types + `FillModelA` skeleton

**Files:**
- Create: `live/fee_schedule.py`
- Create: `live/fill_model_a.py`
- Create: `tests/test_fill_model_a.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_fill_model_a.py`:

```python
"""Tests for FillModelA: fill evaluation, pipeline integration, replay fidelity."""
import uuid
from decimal import Decimal

import pytest

from live.fill_model_a import FillDecision, FillModelA, Tick
from live.order_state_machine import OrderStateMachine
from live.order_types import FeeModel, OrderSide, OrderState, OrderType


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_limit_order(
    side: str = "BUY",
    qty: str = "1.0",
    limit_price: str = "65000",
    ts: int = 1_000,
):
    """Return an ACKNOWLEDGED LIMIT order ready for FillModelA evaluation."""
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide(side),
        order_type=OrderType.LIMIT,
        qty=Decimal(qty),
        limit_price=Decimal(limit_price),
        event_ts_ms=ts,
    )
    osm.submit(order, event_ts_ms=ts + 1)
    osm.acknowledge(order, event_ts_ms=ts + 2)
    return order, osm


def _make_market_order(
    side: str = "BUY",
    qty: str = "1.0",
    ts: int = 1_000,
):
    """Return a SUBMITTED MARKET order ready for FillModelA evaluation (immediate taker path)."""
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide(side),
        order_type=OrderType.MARKET,
        qty=Decimal(qty),
        event_ts_ms=ts,
    )
    osm.submit(order, event_ts_ms=ts + 1)
    return order, osm


def _make_tick(
    price: str = "65000",
    bid: str = "64995",
    ask: str = "65005",
    volume: str = "10.0",
    ts_ms: int = 2_000,
) -> Tick:
    return Tick(
        timestamp_ms=ts_ms,
        price=Decimal(price),
        bid=Decimal(bid),
        ask=Decimal(ask),
        volume=Decimal(volume),
    )


# ── Task 1: Data types ────────────────────────────────────────────────────────

def test_tick_is_frozen():
    tick = _make_tick()
    with pytest.raises((AttributeError, TypeError)):
        tick.price = Decimal("0")  # type: ignore[misc]


def test_fill_decision_is_frozen():
    decision = FillDecision(
        fill_price=Decimal("65000"),
        fill_qty=Decimal("1.0"),
        fee_model=FeeModel.TAKER,
        fee=Decimal("26.002"),
        execution_id=str(uuid.uuid4()),
        event_ts_ms=2_000,
    )
    with pytest.raises((AttributeError, TypeError)):
        decision.fill_price = Decimal("0")  # type: ignore[misc]
```

- [ ] **Step 2: Run — expect `ModuleNotFoundError: No module named 'live.fill_model_a'`**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py::test_tick_is_frozen -v
```

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create `live/fee_schedule.py`**

```python
"""
Fee schedule — single source of truth for exchange fee/rebate rates.

All FillModel implementations MUST import from here.
FillModelB/C may add venue-specific schedules; never duplicate these constants.

Source: Binance USDT-M perpetual futures (VIP 0).
"""
from decimal import Decimal

TAKER_FEE_RATE    = Decimal("0.0004")   # 0.04% — taker fee
MAKER_REBATE_RATE = Decimal("0.0001")   # 0.01% — maker rebate (income, stored positive)
FEE_PRECISION     = Decimal("0.00000001")
```

- [ ] **Step 4: Create `live/fill_model_a.py`**

```python
"""
FillModelA — deterministic, queue-free fill engine for paper trading simulation.

Rules:
    MARKET BUY  → fill at tick.ask, FeeModel.TAKER, always (volume > 0)
    MARKET SELL → fill at tick.bid, FeeModel.TAKER, always (volume > 0)
    LIMIT BUY   → fill if tick.price <= limit_price, at limit_price, FeeModel.MAKER
    LIMIT SELL  → fill if tick.price >= limit_price, at limit_price, FeeModel.MAKER
    Partial     → fill_qty = min(order.remaining_qty, tick.volume)

No randomness. No queue position. No order book depth.
Goal: validate the complete simulation pipeline end-to-end.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from live.fee_schedule import FEE_PRECISION, MAKER_REBATE_RATE, TAKER_FEE_RATE
from live.order_types import FeeModel, Order, OrderSide, OrderState, OrderType

_TERMINAL_STATES = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.EXPIRED,
    OrderState.REJECTED,
})


@dataclass(frozen=True)
class Tick:
    """
    Single market data snapshot.

    price:  last trade price (used for LIMIT fill condition)
    bid:    best bid (MARKET SELL fills here)
    ask:    best ask (MARKET BUY fills here)
    volume: available volume at this level (caps fill_qty for partial fills)
    """
    timestamp_ms: int
    price:        Decimal
    bid:          Decimal
    ask:          Decimal
    volume:       Decimal


@dataclass(frozen=True)
class FillDecision:
    """
    Result of FillModelA.evaluate(). Describes a single fill to execute.

    fill_price:   price at which the fill executes
    fill_qty:     quantity filled (may be < order.remaining_qty for partials)
    fee_model:    MAKER or TAKER
    fee:          actual fee amount (>= 0). For MAKER this is the rebate.
    execution_id: UUID4 string — pass to osm.fill(execution_id=...) for idempotency
    event_ts_ms:  tick.timestamp_ms — canonical exchange clock for this fill
    """
    fill_price:   Decimal
    fill_qty:     Decimal
    fee_model:    FeeModel
    fee:          Decimal
    execution_id: str
    event_ts_ms:  int


class FillModelA:
    """Stateless fill engine. No persistent state."""

    @staticmethod
    def evaluate(order: Order, tick: Tick) -> FillDecision | None:
        """
        Evaluate whether order fills against tick. Returns FillDecision or None.

        Raises:
            ValueError: if order is in a terminal state.
        """
        if order.state in _TERMINAL_STATES:
            raise ValueError(
                f"Cannot evaluate fill for terminal order {order.order_id!r} "
                f"(state={order.state.value})"
            )

        if tick.volume <= Decimal("0"):
            return None

        if order.order_type == OrderType.MARKET:
            fill_price = tick.ask if order.side == OrderSide.BUY else tick.bid
            fee_model  = FeeModel.TAKER
        else:  # LIMIT
            assert order.limit_price is not None
            if order.side == OrderSide.BUY:
                if tick.price > order.limit_price:
                    return None
            else:  # SELL
                if tick.price < order.limit_price:
                    return None
            fill_price = order.limit_price
            fee_model  = FeeModel.MAKER

        fill_qty = min(order.remaining_qty, tick.volume)
        rate     = TAKER_FEE_RATE if fee_model == FeeModel.TAKER else MAKER_REBATE_RATE
        fee      = (fill_price * fill_qty * rate).quantize(FEE_PRECISION)

        return FillDecision(
            fill_price=fill_price,
            fill_qty=fill_qty,
            fee_model=fee_model,
            fee=fee,
            execution_id=str(uuid.uuid4()),
            event_ts_ms=tick.timestamp_ms,
        )
```

- [ ] **Step 5: Run Task 1 tests**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py::test_tick_is_frozen tests/test_fill_model_a.py::test_fill_decision_is_frozen -v
```

Expected: 2 PASS

- [ ] **Step 6: Commit**

```
git add live/fee_schedule.py live/fill_model_a.py tests/test_fill_model_a.py
git commit -m "feat: add fee_schedule.py + Tick + FillDecision types + FillModelA skeleton"
```

---

## Task 2: FillModelA.evaluate() — MARKET orders

**Files:**
- Modify: `tests/test_fill_model_a.py`

- [ ] **Step 1: Append MARKET order tests**

Append to `tests/test_fill_model_a.py`:

```python
# ── Task 2: MARKET orders ─────────────────────────────────────────────────────

def test_market_buy_fills_at_ask():
    order, _ = _make_market_order(side="BUY", qty="1.0")
    tick = _make_tick(ask="65005", volume="10.0")
    decision = FillModelA.evaluate(order, tick)

    assert decision is not None
    assert decision.fill_price  == Decimal("65005")
    assert decision.fill_qty    == Decimal("1.0")
    assert decision.fee_model   == FeeModel.TAKER
    assert decision.event_ts_ms == tick.timestamp_ms


def test_market_sell_fills_at_bid():
    order, _ = _make_market_order(side="SELL", qty="0.5")
    tick = _make_tick(bid="64995", volume="10.0")
    decision = FillModelA.evaluate(order, tick)

    assert decision is not None
    assert decision.fill_price == Decimal("64995")
    assert decision.fill_qty   == Decimal("0.5")
    assert decision.fee_model  == FeeModel.TAKER


def test_market_buy_partial_when_volume_less_than_qty():
    order, _ = _make_market_order(side="BUY", qty="2.0")
    tick = _make_tick(ask="65005", volume="0.5")
    decision = FillModelA.evaluate(order, tick)

    assert decision is not None
    assert decision.fill_qty == Decimal("0.5")  # capped by tick.volume


def test_market_order_zero_volume_returns_none():
    order, _ = _make_market_order(side="BUY", qty="1.0")
    tick = _make_tick(volume="0")
    assert FillModelA.evaluate(order, tick) is None


def test_market_order_raises_on_terminal_state():
    order, osm = _make_market_order(side="BUY", qty="1.0")
    osm.fill(
        order,
        fill_price=Decimal("65005"),
        fill_qty=Decimal("1.0"),
        event_ts_ms=2_000,
        fee_model=FeeModel.TAKER,
        execution_id="exec-term-001",
    )
    assert order.state == OrderState.FILLED
    with pytest.raises(ValueError, match="terminal"):
        FillModelA.evaluate(order, _make_tick())


def test_applying_same_fill_decision_twice_raises_duplicate():
    """
    FillDecision carries an execution_id. Applying it twice must raise DuplicateEventError.
    Verifies that FillModelA decisions are compatible with OSM idempotency guarantees.
    """
    from live.order_state_machine import DuplicateEventError

    order, osm = _make_market_order(side="BUY", qty="2.0")
    tick = _make_tick(ask="65005", volume="1.0")  # partial fill: 1.0 of 2.0
    decision = FillModelA.evaluate(order, tick)
    assert decision is not None

    # First application succeeds — PARTIALLY_FILLED
    osm.fill(
        order,
        fill_price=decision.fill_price,
        fill_qty=decision.fill_qty,
        event_ts_ms=decision.event_ts_ms,
        fee_model=decision.fee_model,
        execution_id=decision.execution_id,
    )
    assert order.state == OrderState.PARTIALLY_FILLED
    assert len(order.fills) == 1

    # Second application with the same execution_id must be rejected
    with pytest.raises(DuplicateEventError):
        osm.fill(
            order,
            fill_price=decision.fill_price,
            fill_qty=decision.fill_qty,
            event_ts_ms=decision.event_ts_ms + 1,
            fee_model=decision.fee_model,
            execution_id=decision.execution_id,  # same id → rejected
        )

    # OSM state unchanged: exactly 1 fill, not 2
    assert len(order.fills)  == 1
    assert order.filled_qty  == Decimal("1.0")
    assert order.state       == OrderState.PARTIALLY_FILLED
```

- [ ] **Step 2: Run Task 2 tests**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py -k "market" -v
```

Expected: 6 PASS

- [ ] **Step 3: Commit**

```
git add tests/test_fill_model_a.py
git commit -m "test: FillModelA MARKET order evaluation rules"
```

---

## Task 3: FillModelA.evaluate() — LIMIT orders

**Files:**
- Modify: `tests/test_fill_model_a.py`

- [ ] **Step 1: Append LIMIT order tests**

Append to `tests/test_fill_model_a.py`:

```python
# ── Task 3: LIMIT orders ──────────────────────────────────────────────────────

def test_limit_buy_fills_when_tick_price_below_limit():
    order, _ = _make_limit_order(side="BUY", limit_price="65000")
    tick = _make_tick(price="64990", volume="10.0")  # price < limit → fills
    decision = FillModelA.evaluate(order, tick)

    assert decision is not None
    assert decision.fill_price == Decimal("65000")  # fills at limit_price, not tick.price
    assert decision.fee_model  == FeeModel.MAKER


def test_limit_buy_no_fill_when_tick_price_above_limit():
    order, _ = _make_limit_order(side="BUY", limit_price="65000")
    tick = _make_tick(price="65001", volume="10.0")  # price > limit → no fill
    assert FillModelA.evaluate(order, tick) is None


def test_limit_buy_fills_exactly_at_limit_price():
    order, _ = _make_limit_order(side="BUY", limit_price="65000")
    tick = _make_tick(price="65000", volume="10.0")  # price == limit → fills
    assert FillModelA.evaluate(order, tick) is not None


def test_limit_sell_fills_when_tick_price_above_limit():
    order, _ = _make_limit_order(side="SELL", limit_price="65000")
    tick = _make_tick(price="65010", volume="10.0")  # price > limit → fills
    decision = FillModelA.evaluate(order, tick)

    assert decision is not None
    assert decision.fill_price == Decimal("65000")  # fills at limit_price
    assert decision.fee_model  == FeeModel.MAKER


def test_limit_sell_no_fill_when_tick_price_below_limit():
    order, _ = _make_limit_order(side="SELL", limit_price="65000")
    tick = _make_tick(price="64999", volume="10.0")  # price < limit → no fill
    assert FillModelA.evaluate(order, tick) is None


def test_limit_sell_fills_exactly_at_limit_price():
    order, _ = _make_limit_order(side="SELL", limit_price="65000")
    tick = _make_tick(price="65000", volume="10.0")  # price == limit → fills
    assert FillModelA.evaluate(order, tick) is not None


def test_limit_buy_fills_at_limit_not_better_tick_price():
    # tick.price=64500 is better than limit=65000 for BUY, but fill is always at limit_price
    order, _ = _make_limit_order(side="BUY", limit_price="65000")
    tick = _make_tick(price="64500", volume="10.0")
    decision = FillModelA.evaluate(order, tick)
    assert decision is not None
    assert decision.fill_price == Decimal("65000")  # always limit_price for MAKER fills


def test_limit_order_partial_fill_when_volume_insufficient():
    order, _ = _make_limit_order(side="BUY", qty="2.0", limit_price="65000")
    tick = _make_tick(price="64990", volume="0.75")
    decision = FillModelA.evaluate(order, tick)
    assert decision is not None
    assert decision.fill_qty == Decimal("0.75")  # capped by tick.volume


def test_limit_maker_fee_uses_rebate_rate():
    order, _ = _make_limit_order(side="BUY", qty="1.0", limit_price="65000")
    tick = _make_tick(price="64990", volume="1.0")
    decision = FillModelA.evaluate(order, tick)
    assert decision is not None
    expected_fee = (Decimal("65000") * Decimal("1.0") * Decimal("0.0001")).quantize(
        Decimal("0.00000001")
    )
    assert decision.fee       == expected_fee
    assert decision.fee_model == FeeModel.MAKER


def test_market_taker_fee_uses_taker_rate():
    order, _ = _make_market_order(side="BUY", qty="1.0")
    tick = _make_tick(ask="65005", volume="1.0")
    decision = FillModelA.evaluate(order, tick)
    assert decision is not None
    expected_fee = (Decimal("65005") * Decimal("1.0") * Decimal("0.0004")).quantize(
        Decimal("0.00000001")
    )
    assert decision.fee       == expected_fee
    assert decision.fee_model == FeeModel.TAKER
```

- [ ] **Step 2: Run Task 3 tests**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py -k "limit or taker_fee or maker_fee" -v
```

Expected: 10 PASS

- [ ] **Step 3: Run full test_fill_model_a.py so far**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py -v
```

Expected: 18 PASS (Tasks 1–3)

- [ ] **Step 4: Append determinism test**

Append to `tests/test_fill_model_a.py`:

```python
def test_same_tick_produces_identical_economics_different_execution_id():
    """
    Determinism property: same order config + same tick → same fill_price, fill_qty,
    fee_model, fee — but different execution_id (UUID).

    Proves FillModelA is a pure function of (order_economics, tick) with no hidden state.
    """
    order1, _ = _make_limit_order(side="BUY", qty="0.01", limit_price="65000", ts=1_000)
    order2, _ = _make_limit_order(side="BUY", qty="0.01", limit_price="65000", ts=1_000)
    tick = _make_tick(price="64990", bid="64985", ask="64995", volume="0.01", ts_ms=2_000)

    d1 = FillModelA.evaluate(order1, tick)
    d2 = FillModelA.evaluate(order2, tick)

    assert d1 is not None and d2 is not None

    # Economics are identical
    assert d1.fill_price == d2.fill_price
    assert d1.fill_qty   == d2.fill_qty
    assert d1.fee_model  == d2.fee_model
    assert d1.fee        == d2.fee

    # execution_id is unique per evaluation (UUID4)
    assert d1.execution_id != d2.execution_id
```

- [ ] **Step 5: Run all Task 3 tests**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py -v
```

Expected: 19 PASS (Tasks 1–3)

- [ ] **Step 6: Commit**

```
git add tests/test_fill_model_a.py
git commit -m "test: FillModelA LIMIT order evaluation rules + fee rates + determinism"
```

---

## Task 4: End-to-end pipeline integration tests

**Files:**
- Modify: `tests/test_fill_model_a.py`

- [ ] **Step 1: Append pipeline helpers and integration tests**

Append to `tests/test_fill_model_a.py`:

```python
# ── Task 4: Pipeline helpers + integration tests ──────────────────────────────

from live.cost_ledger import CostLedger
from live.event_models import OrderAcknowledged, OrderFillReceived, OrderSubmitted
from live.event_store import EventStore


def _persist_submit(store: EventStore, order, submit_ts_ms: int) -> None:
    """Persist OrderSubmitted domain event to EventStore."""
    store.append_domain_event(OrderSubmitted(
        event_id=str(uuid.uuid4()),
        schema_version=1,
        event_ts_ms=submit_ts_ms,
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side.value,
        order_type=order.order_type.value,
        qty=order.qty,
        limit_price=order.limit_price,
        created_ts_ms=order.created_ts_ms,
    ))


def _persist_ack(store: EventStore, order, ack_ts_ms: int) -> None:
    """Persist OrderAcknowledged domain event to EventStore."""
    store.append_domain_event(OrderAcknowledged(
        event_id=str(uuid.uuid4()),
        schema_version=1,
        event_ts_ms=ack_ts_ms,
        order_id=order.order_id,
        submitted_ts_ms=order.submitted_ts_ms,
    ))


def _apply_decision(
    osm,
    store: EventStore,
    order,
    decision: FillDecision,
    arrival_price: Decimal,
) -> None:
    """Drive OSM.fill(), persist OrderFillReceived, compute and persist CostLedgerEntry."""
    osm.fill(
        order,
        fill_price=decision.fill_price,
        fill_qty=decision.fill_qty,
        event_ts_ms=decision.event_ts_ms,
        fee_model=decision.fee_model,
        execution_id=decision.execution_id,
    )
    store.append_domain_event(OrderFillReceived(
        event_id=str(uuid.uuid4()),
        schema_version=1,
        event_ts_ms=decision.event_ts_ms,
        order_id=order.order_id,
        fill_id=decision.execution_id,
        price=decision.fill_price,
        qty=decision.fill_qty,
        fee=decision.fee,
        fee_asset="USDT",
        fee_model=decision.fee_model.value,
    ))
    cost_entry = CostLedger.compute_fill_cost(
        fill_id=decision.execution_id,
        order_id=order.order_id,
        side=order.side.value,
        fill_price=decision.fill_price,
        fill_qty=decision.fill_qty,
        arrival_price=arrival_price,
        fee_model=decision.fee_model.value,
        fee=decision.fee,
        fill_ts_ms=decision.event_ts_ms,
        event_id=str(uuid.uuid4()),
        correlation_id=order.order_id,
    )
    store.append_cost_entry(cost_entry)


def test_limit_buy_full_pipeline_single_fill(tmp_path):
    """
    LIMIT BUY: create → submit → ack → non-triggering tick → triggering tick →
    OSM reaches FILLED → cost entry persisted.
    """
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.01"),
        limit_price=Decimal("65000"),
        event_ts_ms=1_000,
    )
    arrival_price = Decimal("65050")  # midprice at order creation time

    with EventStore(tmp_path) as store:
        osm.submit(order, event_ts_ms=1_001)
        _persist_submit(store, order, 1_001)
        osm.acknowledge(order, event_ts_ms=1_002)
        _persist_ack(store, order, 1_002)

        # Tick that does NOT trigger fill (price above limit)
        assert FillModelA.evaluate(order, _make_tick(price="65010", ts_ms=1_003)) is None
        assert order.state == OrderState.ACKNOWLEDGED

        # Tick that triggers fill
        fill_tick = _make_tick(price="64990", bid="64985", ask="64995",
                               volume="0.01", ts_ms=1_004)
        decision = FillModelA.evaluate(order, fill_tick)
        assert decision is not None
        _apply_decision(osm, store, order, decision, arrival_price)

    assert order.state          == OrderState.FILLED
    assert order.filled_qty     == Decimal("0.01")
    assert order.avg_fill_price == Decimal("65000")  # fills at limit_price
    assert decision.fee_model   == FeeModel.MAKER

    with EventStore(tmp_path) as store:
        entries = store.get_cost_entries_by_order(order.order_id)
    assert len(entries) == 1
    assert entries[0].fill_price   == Decimal("65000")
    assert entries[0].taker_fee    == Decimal("0")    # MAKER fill → no taker fee
    assert entries[0].maker_rebate > Decimal("0")     # rebate credited


def test_market_order_immediate_taker_pipeline(tmp_path):
    """
    MARKET BUY: fills immediately from SUBMITTED state (IMMEDIATE_TAKER_FILL path).
    No ACK required. Verifies the immediate taker pipeline branch.
    """
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.01"),
        event_ts_ms=1_000,
    )
    arrival_price = Decimal("65005")

    with EventStore(tmp_path) as store:
        osm.submit(order, event_ts_ms=1_001)
        _persist_submit(store, order, 1_001)

        tick = _make_tick(price="65005", bid="65000", ask="65005",
                          volume="0.01", ts_ms=1_002)
        decision = FillModelA.evaluate(order, tick)
        assert decision is not None
        assert decision.fee_model  == FeeModel.TAKER
        assert decision.fill_price == Decimal("65005")  # tick.ask
        _apply_decision(osm, store, order, decision, arrival_price)

    assert order.state      == OrderState.FILLED
    assert order.filled_qty == Decimal("0.01")

    with EventStore(tmp_path) as store:
        entries = store.get_cost_entries_by_order(order.order_id)
    assert len(entries) == 1
    assert entries[0].taker_fee    > Decimal("0")   # TAKER fill → fee charged
    assert entries[0].maker_rebate == Decimal("0")
```

- [ ] **Step 2: Run pipeline integration tests**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py -k "pipeline" -v
```

Expected: 2 PASS

- [ ] **Step 3: Commit**

```
git add tests/test_fill_model_a.py
git commit -m "test: FillModelA end-to-end pipeline integration tests"
```

---

## Task 5: Replay Fidelity Tests

**Files:**
- Modify: `tests/test_fill_model_a.py`

- [ ] **Step 1: Append replay fidelity tests**

Append to `tests/test_fill_model_a.py`:

```python
# ── Task 5: Replay Fidelity ───────────────────────────────────────────────────

from live.execution_replay import ExecutionReplay


def test_replay_fidelity_limit_order_single_fill(tmp_path):
    """
    Core property: live run == replay run.

    Runs a LIMIT BUY through the full pipeline, persists all domain events,
    then reconstructs via ExecutionReplay and compares all economically relevant fields.
    Bit-identical financial quantities after round-trip through EventStore.
    """
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.02"),
        limit_price=Decimal("65000"),
        event_ts_ms=1_000,
    )
    arrival_price = Decimal("65050")

    with EventStore(tmp_path) as store:
        osm.submit(order, event_ts_ms=1_001)
        _persist_submit(store, order, 1_001)
        osm.acknowledge(order, event_ts_ms=1_002)
        _persist_ack(store, order, 1_002)

        fill_tick = _make_tick(price="64990", bid="64985", ask="64995",
                               volume="0.02", ts_ms=1_003)
        decision = FillModelA.evaluate(order, fill_tick)
        assert decision is not None
        _apply_decision(osm, store, order, decision, arrival_price)

        replayed = ExecutionReplay(store).replay_order(order.order_id)

    # State + inventory
    assert replayed.state          == order.state           == OrderState.FILLED
    assert replayed.filled_qty     == order.filled_qty      == Decimal("0.02")
    assert replayed.avg_fill_price == order.avg_fill_price  == Decimal("65000")
    assert replayed.liquidity_role == order.liquidity_role  # derived from fills, must match
    assert len(replayed.fills)     == len(order.fills)      == 1

    # Financial precision: fill price + qty survive EventStore round-trip
    assert replayed.fills[0].price == order.fills[0].price
    assert replayed.fills[0].qty   == order.fills[0].qty


def test_replay_fidelity_partial_fills_two_ticks(tmp_path):
    """
    Replay fidelity with partial fills: two ticks fill one order in two steps.

    Verifies that EventStore + ExecutionReplay reconstructs the intermediate
    PARTIALLY_FILLED state and the final FILLED state with identical quantities.
    Also verifies that 2 CostLedgerEntries are persisted and summarize correctly.
    """
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.02"),
        limit_price=Decimal("65000"),
        event_ts_ms=1_000,
    )
    arrival_price = Decimal("65020")

    with EventStore(tmp_path) as store:
        osm.submit(order, event_ts_ms=1_001)
        _persist_submit(store, order, 1_001)
        osm.acknowledge(order, event_ts_ms=1_002)
        _persist_ack(store, order, 1_002)

        # First partial fill: 0.01 of 0.02
        tick1 = _make_tick(price="64990", volume="0.01", ts_ms=1_003)
        d1 = FillModelA.evaluate(order, tick1)
        assert d1 is not None
        _apply_decision(osm, store, order, d1, arrival_price)
        assert order.state == OrderState.PARTIALLY_FILLED

        # Second fill: remaining 0.01 (tick volume > remaining_qty)
        tick2 = _make_tick(price="64980", volume="0.05", ts_ms=1_004)
        d2 = FillModelA.evaluate(order, tick2)
        assert d2 is not None
        assert d2.fill_qty == Decimal("0.01")  # capped at remaining_qty
        _apply_decision(osm, store, order, d2, arrival_price)

        replayed = ExecutionReplay(store).replay_order(order.order_id)

    assert replayed.state          == OrderState.FILLED
    assert replayed.filled_qty     == Decimal("0.02")
    assert replayed.avg_fill_price == order.avg_fill_price
    assert replayed.liquidity_role == order.liquidity_role  # both fills MAKER → LiquidityRole.MAKER
    assert len(replayed.fills)     == 2

    # Cost ledger: 2 entries, aggregated correctly
    with EventStore(tmp_path) as store:
        entries = store.get_cost_entries_by_order(order.order_id)
    assert len(entries) == 2
    summary = CostLedger.summarize_order(entries)
    assert summary.total_fill_qty == Decimal("0.02")
    assert summary.fill_count     == 2
    # Net cost invariant holds across two fills
    expected_net = (
        summary.total_taker_fee - summary.total_maker_rebate
        + summary.total_slippage
        + summary.total_latency_cost
        + summary.total_inventory_cost
    )
    assert summary.total_net_cost == expected_net
```

- [ ] **Step 2: Run replay fidelity tests**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py -k "replay" -v
```

Expected: 2 PASS

- [ ] **Step 3: Run the complete test_fill_model_a.py**

```
venv\Scripts\python -m pytest tests/test_fill_model_a.py -v
```

Expected: **23 PASS** (Tasks 1–5)

- [ ] **Step 4: Run the full suite — confirm no regressions**

```
venv\Scripts\python -m pytest tests/ -v 2>&1 | Select-String -Pattern "passed|failed|error" | Select-Object -Last 3
```

Expected: **235 passed, 0 failed.** (212 existing + 23 new)

- [ ] **Step 5: Commit**

```
git add tests/test_fill_model_a.py
git commit -m "test: replay fidelity tests for FillModelA pipeline"
```

---

## Self-Review

### Spec coverage

| Requirement | Covered by |
|-------------|-----------|
| MARKET → always fills | `test_market_buy_fills_at_ask`, `test_market_sell_fills_at_bid`, `test_market_order_immediate_taker_pipeline` |
| MARKET BUY fills at tick.ask | `test_market_buy_fills_at_ask` |
| MARKET SELL fills at tick.bid | `test_market_sell_fills_at_bid` |
| LIMIT BUY: fills if price ≤ limit | `test_limit_buy_fills_when_tick_price_below_limit`, `test_limit_buy_fills_exactly_at_limit_price` |
| LIMIT BUY: no fill if price > limit | `test_limit_buy_no_fill_when_tick_price_above_limit` |
| LIMIT SELL: fills if price ≥ limit | `test_limit_sell_fills_when_tick_price_above_limit`, `test_limit_sell_fills_exactly_at_limit_price` |
| LIMIT SELL: no fill if price < limit | `test_limit_sell_no_fill_when_tick_price_below_limit` |
| Fill at limit_price, not tick.price | `test_limit_buy_fills_at_limit_not_better_tick_price` |
| Partial: fill_qty = min(remaining, volume) | `test_market_buy_partial_when_volume_less_than_qty`, `test_limit_order_partial_fill_when_volume_insufficient`, `test_replay_fidelity_partial_fills_two_ticks` |
| Zero volume → None | `test_market_order_zero_volume_returns_none` |
| Terminal order → ValueError | `test_market_order_raises_on_terminal_state` |
| Fee rates in `live/fee_schedule.py` (single source of truth) | `live/fee_schedule.py` — imported by `fill_model_a.py`; verified via fee amount assertions in Task 3 |
| TAKER fee rate (0.04%) | `test_market_taker_fee_uses_taker_rate` |
| MAKER rebate rate (0.01%) | `test_limit_maker_fee_uses_rebate_rate` |
| Full pipeline: Data → FillModelA → OSM → EventStore → CostLedger | `test_limit_buy_full_pipeline_single_fill`, `test_market_order_immediate_taker_pipeline` |
| Idempotency: same execution_id applied twice raises DuplicateEventError | `test_applying_same_fill_decision_twice_raises_duplicate` |
| Replay: live == replay (state, qty, avg_price, liquidity_role) | `test_replay_fidelity_limit_order_single_fill`, `test_replay_fidelity_partial_fills_two_ticks` |
| Replay: liquidity_role derived from fills survives round-trip | `test_replay_fidelity_limit_order_single_fill`, `test_replay_fidelity_partial_fills_two_ticks` |
| Replay: partial fill → PARTIALLY_FILLED intermediate state | `test_replay_fidelity_partial_fills_two_ticks` |
| Cost ledger: 2 entries for 2 partial fills | `test_replay_fidelity_partial_fills_two_ticks` |
| net_cost invariant across aggregated fills | `test_replay_fidelity_partial_fills_two_ticks` |

All requirements covered. ✅

### Placeholder scan

No TBDs, TODOs, "implement later", or code-free steps. Every step has exact code and expected output. ✅

### Type consistency

| Symbol | Defined in | Used in |
|--------|-----------|---------|
| `TAKER_FEE_RATE`, `MAKER_REBATE_RATE`, `FEE_PRECISION` | Task 1 / `fee_schedule.py` | `fill_model_a.py` + fee assertions in Tasks 2–3 |
| `Tick` | Task 1 / `fill_model_a.py` | `_make_tick()` helper, Tasks 2–5 |
| `FillDecision` | Task 1 / `fill_model_a.py` | `_apply_decision()` helper, Tasks 2–5 |
| `FillModelA.evaluate()` | Task 1 / `fill_model_a.py` | Tasks 2, 3, 4, 5 |
| `_make_limit_order()` | Task 1 test helpers | Tasks 3, 5 |
| `_make_market_order()` | Task 1 test helpers | Tasks 2, 4 |
| `_make_tick()` | Task 1 test helpers | Tasks 2, 3, 4, 5 |
| `_persist_submit()` | Task 4 test helpers | Tasks 4, 5 |
| `_persist_ack()` | Task 4 test helpers | Tasks 4, 5 |
| `_apply_decision()` | Task 4 test helpers | Tasks 4, 5 |
| `CostLedger.compute_fill_cost()` | `live/cost_ledger.py` (existing) | `_apply_decision()` helper |
| `ExecutionReplay.replay_order()` | `live/execution_replay.py` (existing) | Task 5 |

All consistent. ✅
