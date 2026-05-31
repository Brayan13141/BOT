# FillModelB — Volume-Aware Fill Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement FillModelB, a volume-aware fill engine that constrains fill quantity by a configurable `participation_rate`, producing a more conservative and realistic simulation than FillModelA without introducing synthetic queue position data.

**Architecture:** `FillModelB` delegates to `FillModelA` via composition. The only logic it owns: create a `capped` tick with `volume = tick.volume * participation_rate`, then call `FillModelA.evaluate(order, capped)`. All fill conditions (price checks, bid/ask routing, fee model) remain in `FillModelA` — no duplication. An optional `execution_id_factory` parameter is threaded through to `FillModelA` to enable deterministic IDs for golden A/B/C comparison tests. This requires a one-line non-breaking change to `FillModelA.evaluate()` (Task 0).

**Tech Stack:** Python 3.12, `decimal.Decimal`, `uuid`, existing `live/fill_model_a.py` (for `Tick`, `FillDecision`), `live/order_types.py`, `live/fee_schedule.py`, `live/order_state_machine.py`, `live/event_store.py`, `live/cost_ledger.py`, `live/execution_replay.py`, `pytest`, `tmp_path` fixture.

---

## Key Design Decisions (fixed — no changes without new plan)

| Decision | Value |
|----------|-------|
| `participation_rate` default | `Decimal("0.01")` (1% — conservative retail) |
| `participation_rate` range | `(0, 1]` — strictly positive, at most 1.0 |
| `effective_liquidity` | `tick.volume * self.participation_rate` |
| `fill_qty` | `min(order.remaining_qty, effective_liquidity)` |
| Fill condition | identical to FillModelA |
| Zero `effective_liquidity` | returns `None` |
| RNG | none — fully deterministic |
| Reused types | `Tick`, `FillDecision` from `live/fill_model_a.py` |

**Core difference from FillModelA:**
```
order.qty = 1.0 BTC, tick.volume = 10.0 BTC, participation_rate = 0.01

FillModelA:  fill_qty = min(1.0, 10.0)          = 1.0   → FULL FILL
FillModelB:  fill_qty = min(1.0, 10.0 × 0.01)   = 0.1   → PARTIAL FILL
```

At `participation_rate=1.0`, FillModelB degenerates to FillModelA behaviour.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `live/fill_model_a.py` | Modify | Add optional `execution_id_factory` param to `evaluate()` (Task 0) |
| `tests/test_fill_model_a.py` | Modify | Add 1 injectable factory test (Task 0) |
| `live/fill_model_b.py` | Create | `FillModelB` class: `__init__` + `evaluate()` via composition |
| `tests/test_fill_model_b.py` | Create | All 24 tests (Tasks 1–5, incl. decimal precision) |

---

## Task 0: Prepare FillModelA — injectable execution_id_factory

**Files:**
- Modify: `live/fill_model_a.py`
- Modify: `tests/test_fill_model_a.py`

**Motivation:** FillModelB delegates to FillModelA. When comparing FillModelA vs
FillModelB vs FillModelC on the same order+tick sequence, `uuid4()` makes golden
tests impossible. An injectable factory is a non-breaking addition — the default
behaviour (uuid4) is unchanged and all 23 existing tests keep passing.

- [ ] **Step 1: Update `FillModelA.evaluate` in `live/fill_model_a.py`**

Add `from collections.abc import Callable` to imports. Add the optional parameter:

```python
from collections.abc import Callable

class FillModelA:
    """Stateless fill engine. No persistent state."""

    @staticmethod
    def evaluate(
        order: Order,
        tick: Tick,
        execution_id_factory: Callable[[], str] | None = None,
    ) -> FillDecision | None:
        """
        Evaluate whether order fills against tick. Returns FillDecision or None.

        execution_id_factory: optional callable that returns a string ID.
            Defaults to uuid4. Pass a deterministic factory for golden tests
            comparing FillModelA vs FillModelB vs FillModelC.

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
            if order.limit_price is None:
                raise ValueError(
                    f"LIMIT order {order.order_id!r} has no limit_price — OSM invariant violated."
                )
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

        id_fn = execution_id_factory if execution_id_factory is not None else (lambda: str(uuid.uuid4()))
        return FillDecision(
            fill_price=fill_price,
            fill_qty=fill_qty,
            fee_model=fee_model,
            fee=fee,
            execution_id=id_fn(),
            event_ts_ms=tick.timestamp_ms,
        )
```

- [ ] **Step 2: Run existing FillModelA tests — confirm no regressions**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_a.py -v 2>&1 | Select-Object -Last 5
```

Expected: **23 passed** (default factory still uses uuid4; all existing tests unchanged)

- [ ] **Step 3: Add injectable factory test to `tests/test_fill_model_a.py`**

Append to `tests/test_fill_model_a.py`:

```python
def test_evaluate_uses_injected_execution_id_factory():
    """Injectable factory enables deterministic IDs for golden A vs B vs C tests."""
    ids = iter(["exec-factory-001"])
    order, _ = _make_market_order(side="BUY", qty="0.01")
    tick = _make_tick(volume="10.0")
    decision = FillModelA.evaluate(order, tick, execution_id_factory=ids.__next__)
    assert decision.execution_id == "exec-factory-001"
```

- [ ] **Step 4: Run — expect 24 PASS**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_a.py -v 2>&1 | Select-Object -Last 5
```

Expected: **24 passed**

---

## Task 1: FillModelB constructor + participation_rate validation

**Files:**
- Create: `live/fill_model_b.py`
- Create: `tests/test_fill_model_b.py`

- [ ] **Step 1: Write failing constructor tests**

Create `tests/test_fill_model_b.py`:

```python
"""Tests for FillModelB: volume-aware fill engine with configurable participation_rate."""
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
    """Return an ACKNOWLEDGED LIMIT order ready for evaluation."""
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
    """Return a SUBMITTED MARKET order ready for evaluation (immediate taker path)."""
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


# ── Task 1: Constructor ───────────────────────────────────────────────────────

from live.fill_model_b import FillModelB


def test_fill_model_b_default_participation_rate():
    model = FillModelB()
    assert model.participation_rate == Decimal("0.01")


def test_fill_model_b_custom_participation_rate():
    model = FillModelB(participation_rate=Decimal("0.05"))
    assert model.participation_rate == Decimal("0.05")


def test_fill_model_b_participation_rate_one_is_valid():
    """participation_rate=1.0 is the upper bound — degenerates to FillModelA behaviour."""
    model = FillModelB(participation_rate=Decimal("1"))
    assert model.participation_rate == Decimal("1")


def test_fill_model_b_zero_participation_rate_raises():
    with pytest.raises(ValueError, match="participation_rate"):
        FillModelB(participation_rate=Decimal("0"))


def test_fill_model_b_negative_participation_rate_raises():
    with pytest.raises(ValueError, match="participation_rate"):
        FillModelB(participation_rate=Decimal("-0.01"))


def test_fill_model_b_above_one_participation_rate_raises():
    with pytest.raises(ValueError, match="participation_rate"):
        FillModelB(participation_rate=Decimal("1.001"))


def test_effective_liquidity_decimal_precision():
    """
    tick.volume * participation_rate can produce repeating Decimals.
    Verify no exception and that Decimal arithmetic preserves exact precision
    (no float rounding error).
    e.g. Decimal("0.33333333") * Decimal("0.01") = Decimal("0.0033333333") — exact.
    """
    order, _ = _make_market_order(side="BUY", qty="1.0")
    tick = _make_tick(ask="65005", volume="0.33333333")
    model = FillModelB(participation_rate=Decimal("0.01"))
    decision = model.evaluate(order, tick)
    assert decision is not None
    # effective_liquidity = Decimal("0.33333333") * Decimal("0.01") = Decimal("0.0033333333")
    # fill_qty = min(1.0, 0.0033333333) = 0.0033333333 — Decimal preserves all digits
    assert decision.fill_qty == Decimal("0.33333333") * Decimal("0.01")
```

- [ ] **Step 2: Run — expect `ModuleNotFoundError: No module named 'live.fill_model_b'`**

```
cd C:/Users/Lenovo/Documents/TRADING-BOT
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_b.py -v 2>&1 | Select-Object -First 10
```

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create `live/fill_model_b.py`**

```python
"""
FillModelB — Volume-Aware fill engine for paper trading simulation.

Extends FillModelA by limiting fill quantity to a configurable fraction
(participation_rate) of the available tick volume. Models that a retail
order competes with other participants for the same candle volume.

Design: delegates entirely to FillModelA via a capped tick.
    FillModelA: fill_qty = min(remaining_qty, tick.volume)
    FillModelB: fill_qty = min(remaining_qty, tick.volume * participation_rate)
              → achieved by passing tick with volume=effective_liquidity to FillModelA

All fill conditions (price checks, bid/ask routing, fee model) are maintained
in FillModelA. No logic duplication.
No RNG. No queue position. Deterministic given (order, tick, participation_rate).
At participation_rate=1.0, degenerates to FillModelA behaviour.
"""
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from live.fill_model_a import FillDecision, FillModelA, Tick
from live.order_types import Order


class FillModelB:
    """
    Volume-Aware fill engine. Configure via participation_rate.

    participation_rate: fraction of tick.volume available to this order.
    Default 0.01 (1%) — calibration parameter, not market-derived.
    Valid range: (0, 1]. At 1.0, equivalent to FillModelA.
    """

    def __init__(
        self,
        participation_rate: Decimal = Decimal("0.01"),  # calibration parameter — not market-derived
        execution_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if participation_rate <= Decimal("0") or participation_rate > Decimal("1"):
            raise ValueError(
                f"participation_rate must be in (0, 1], got {participation_rate}"
            )
        self.participation_rate = participation_rate
        self._execution_id_factory = execution_id_factory

    def evaluate(self, order: Order, tick: Tick) -> FillDecision | None:
        """
        Cap tick volume by participation_rate, then delegate to FillModelA.

        effective_liquidity = tick.volume * participation_rate
        All fill logic (price condition, bid/ask routing, fee) handled by FillModelA.

        Raises:
            ValueError: if order is in a terminal state (raised by FillModelA).
        """
        effective_liquidity = tick.volume * self.participation_rate
        if effective_liquidity <= Decimal("0"):
            return None
        capped = Tick(
            timestamp_ms=tick.timestamp_ms,
            price=tick.price,
            bid=tick.bid,
            ask=tick.ask,
            volume=effective_liquidity,
        )
        return FillModelA.evaluate(order, capped, execution_id_factory=self._execution_id_factory)
```

- [ ] **Step 4: Run Task 1 tests**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_b.py -v 2>&1 | Select-Object -Last 12
```

Expected: 7 PASS (6 constructor + 1 decimal precision)

- [ ] **Step 5: Commit**

```
git add live/fill_model_b.py tests/test_fill_model_b.py
git commit -m "feat: FillModelB skeleton + constructor + participation_rate validation"
```

---

## Task 2: MARKET order evaluation with effective_liquidity

**Files:**
- Modify: `tests/test_fill_model_b.py`

- [ ] **Step 1: Append MARKET order tests**

Append to `tests/test_fill_model_b.py`:

```python
# ── Task 2: MARKET orders ─────────────────────────────────────────────────────

def test_market_buy_fills_at_ask():
    order, _ = _make_market_order(side="BUY", qty="0.5")
    tick = _make_tick(ask="65005", volume="10.0")
    decision = FillModelB().evaluate(order, tick)

    assert decision is not None
    assert decision.fill_price == Decimal("65005")  # tick.ask
    assert decision.fee_model  == FeeModel.TAKER


def test_market_sell_fills_at_bid():
    order, _ = _make_market_order(side="SELL", qty="0.5")
    tick = _make_tick(bid="64995", volume="10.0")
    decision = FillModelB().evaluate(order, tick)

    assert decision is not None
    assert decision.fill_price == Decimal("64995")  # tick.bid
    assert decision.fee_model  == FeeModel.TAKER


def test_market_fill_qty_capped_by_effective_liquidity():
    """
    order.qty = 1.0, tick.volume = 10.0, participation_rate = 0.05
    effective_liquidity = 10.0 * 0.05 = 0.5
    fill_qty = min(1.0, 0.5) = 0.5  → partial fill
    """
    order, _ = _make_market_order(side="BUY", qty="1.0")
    tick = _make_tick(ask="65005", volume="10.0")
    decision = FillModelB(participation_rate=Decimal("0.05")).evaluate(order, tick)

    assert decision is not None
    assert decision.fill_qty == Decimal("0.5")  # 10.0 * 0.05


def test_market_full_fill_when_remaining_qty_less_than_effective_liquidity():
    """
    order.qty = 0.01, tick.volume = 10.0, participation_rate = 0.01
    effective_liquidity = 0.1 > 0.01 → order fills fully, capped by remaining_qty
    """
    order, _ = _make_market_order(side="BUY", qty="0.01")
    tick = _make_tick(ask="65005", volume="10.0")
    decision = FillModelB().evaluate(order, tick)

    assert decision is not None
    assert decision.fill_qty == Decimal("0.01")  # capped by remaining_qty


def test_market_zero_tick_volume_returns_none():
    order, _ = _make_market_order(side="BUY", qty="1.0")
    tick = _make_tick(volume="0")
    assert FillModelB().evaluate(order, tick) is None


def test_market_terminal_state_raises():
    order, osm = _make_market_order(side="BUY", qty="0.01")
    osm.fill(
        order,
        fill_price=Decimal("65005"),
        fill_qty=Decimal("0.01"),
        event_ts_ms=2_000,
        fee_model=FeeModel.TAKER,
        execution_id="exec-term-b-001",
    )
    assert order.state == OrderState.FILLED
    with pytest.raises(ValueError, match="terminal"):
        FillModelB().evaluate(order, _make_tick())
```

- [ ] **Step 2: Run Task 2 tests**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_b.py -k "market" -v
```

Expected: 6 PASS

- [ ] **Step 3: Commit**

```
git add tests/test_fill_model_b.py
git commit -m "test: FillModelB MARKET order evaluation with effective_liquidity"
```

---

## Task 3: LIMIT order evaluation with effective_liquidity

**Files:**
- Modify: `tests/test_fill_model_b.py`

- [ ] **Step 1: Append LIMIT order tests**

Append to `tests/test_fill_model_b.py`:

```python
# ── Task 3: LIMIT orders ──────────────────────────────────────────────────────

def test_limit_buy_no_fill_when_price_above_limit():
    order, _ = _make_limit_order(side="BUY", limit_price="65000")
    tick = _make_tick(price="65001", volume="10.0")  # price > limit → no fill
    assert FillModelB().evaluate(order, tick) is None


def test_limit_buy_fills_at_limit_when_price_touched():
    order, _ = _make_limit_order(side="BUY", limit_price="65000")
    tick = _make_tick(price="64990", volume="10.0")
    decision = FillModelB().evaluate(order, tick)

    assert decision is not None
    assert decision.fill_price == Decimal("65000")  # fills at limit_price, not tick.price
    assert decision.fee_model  == FeeModel.MAKER


def test_limit_buy_fills_exactly_at_limit():
    order, _ = _make_limit_order(side="BUY", limit_price="65000")
    tick = _make_tick(price="65000", volume="10.0")  # price == limit → fills
    assert FillModelB().evaluate(order, tick) is not None


def test_limit_sell_no_fill_when_price_below_limit():
    order, _ = _make_limit_order(side="SELL", limit_price="65000")
    tick = _make_tick(price="64999", volume="10.0")  # price < limit → no fill
    assert FillModelB().evaluate(order, tick) is None


def test_limit_sell_fills_at_limit_when_price_touched():
    order, _ = _make_limit_order(side="SELL", limit_price="65000")
    tick = _make_tick(price="65010", volume="10.0")
    decision = FillModelB().evaluate(order, tick)

    assert decision is not None
    assert decision.fill_price == Decimal("65000")
    assert decision.fee_model  == FeeModel.MAKER


def test_limit_fill_qty_capped_by_effective_liquidity():
    """
    order.qty = 1.0, tick.volume = 10.0, participation_rate = 0.02
    effective_liquidity = 10.0 * 0.02 = 0.2
    fill_qty = min(1.0, 0.2) = 0.2  → partial fill
    """
    order, _ = _make_limit_order(side="BUY", qty="1.0", limit_price="65000")
    tick = _make_tick(price="64990", volume="10.0")
    decision = FillModelB(participation_rate=Decimal("0.02")).evaluate(order, tick)

    assert decision is not None
    assert decision.fill_qty == Decimal("0.2")  # 10.0 * 0.02


def test_limit_fee_uses_maker_rebate_rate():
    """
    MAKER fee = fill_price × fill_qty × MAKER_REBATE_RATE (0.0001).
    Large volume → effective_liq=100*0.01=1.0 ≥ order.qty=1.0 → full fill.
    """
    order, _ = _make_limit_order(side="BUY", qty="1.0", limit_price="65000")
    tick = _make_tick(price="64990", volume="100.0")
    decision = FillModelB().evaluate(order, tick)  # effective_liq = 100 * 0.01 = 1.0

    assert decision is not None
    assert decision.fill_qty == Decimal("1.0")
    expected_fee = (Decimal("65000") * Decimal("1.0") * Decimal("0.0001")).quantize(
        Decimal("0.00000001")
    )
    assert decision.fee == expected_fee
```

- [ ] **Step 2: Run Task 3 tests**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_b.py -k "limit" -v
```

Expected: 7 PASS

- [ ] **Step 3: Run full test_fill_model_b.py so far**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_b.py -v
```

Expected: 20 PASS (Tasks 1–3)

- [ ] **Step 4: Commit**

```
git add tests/test_fill_model_b.py
git commit -m "test: FillModelB LIMIT order evaluation + fee calibration"
```

---

## Task 4: FillModelA vs FillModelB comparison — conservation property

**Files:**
- Modify: `tests/test_fill_model_b.py`

- [ ] **Step 1: Append comparison tests**

Append to `tests/test_fill_model_b.py`:

```python
# ── Task 4: FillModelA vs FillModelB comparison ───────────────────────────────

def test_fill_model_b_fills_less_than_fill_model_a_same_tick():
    """
    Core conservation property: FillModelB is strictly more conservative than
    FillModelA when effective_liquidity < order.remaining_qty.

    order.qty = 1.0 BTC, tick.volume = 10.0 BTC, participation_rate = 0.01:
        FillModelA: fill_qty = min(1.0, 10.0)        = 1.0  → FULL FILL
        FillModelB: fill_qty = min(1.0, 10.0 × 0.01) = 0.1  → PARTIAL FILL
    """
    osm = OrderStateMachine()

    def _ack_buy(order_id: str):
        o = osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("1.0"), limit_price=Decimal("65000"),
            event_ts_ms=1_000, order_id=order_id,
        )
        osm.submit(o, event_ts_ms=1_001)
        osm.acknowledge(o, event_ts_ms=1_002)
        return o

    tick = _make_tick(price="64990", bid="64985", ask="64995", volume="10.0")

    d_a = FillModelA.evaluate(_ack_buy("order-cmp-a"), tick)
    d_b = FillModelB(participation_rate=Decimal("0.01")).evaluate(_ack_buy("order-cmp-b"), tick)

    assert d_a is not None and d_b is not None
    assert d_a.fill_qty  == Decimal("1.0")   # full — tick.volume > order.qty
    assert d_b.fill_qty  == Decimal("0.1")   # 10.0 * 0.01 = 0.1 < 1.0 → partial
    assert d_b.fill_qty  <  d_a.fill_qty
    assert d_a.fill_price == d_b.fill_price  # both at limit_price
    assert d_a.fee_model  == d_b.fee_model   # both MAKER


def test_fill_model_b_at_rate_one_equals_fill_model_a():
    """
    participation_rate=1.0 → effective_liquidity = tick.volume → identical to FillModelA.
    Proves the two models share the same fill logic; only the liquidity cap differs.
    """
    osm = OrderStateMachine()

    def _ack_buy(order_id: str):
        o = osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("0.5"), limit_price=Decimal("65000"),
            event_ts_ms=1_000, order_id=order_id,
        )
        osm.submit(o, event_ts_ms=1_001)
        osm.acknowledge(o, event_ts_ms=1_002)
        return o

    tick = _make_tick(price="64990", volume="10.0")

    d_a = FillModelA.evaluate(_ack_buy("order-eq-a"), tick)
    d_b = FillModelB(participation_rate=Decimal("1")).evaluate(_ack_buy("order-eq-b"), tick)

    assert d_a is not None and d_b is not None
    assert d_a.fill_qty   == d_b.fill_qty    # both = min(0.5, 10.0) = 0.5
    assert d_a.fill_price == d_b.fill_price
    assert d_a.fee_model  == d_b.fee_model
```

- [ ] **Step 2: Run Task 4 tests**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_b.py -k "fill_model_a" -v
```

Expected: 2 PASS

- [ ] **Step 3: Commit**

```
git add tests/test_fill_model_b.py
git commit -m "test: FillModelA vs FillModelB conservation property"
```

---

## Task 5: Pipeline integration + replay fidelity

**Files:**
- Modify: `tests/test_fill_model_b.py`

- [ ] **Step 1: Append pipeline helpers and integration tests**

Append to `tests/test_fill_model_b.py`:

```python
# ── Task 5: Pipeline integration + replay fidelity ────────────────────────────

from live.cost_ledger import CostLedger
from live.event_models import OrderAcknowledged, OrderFillReceived, OrderSubmitted
from live.event_store import EventStore
from live.execution_replay import ExecutionReplay


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
    store.append_cost_entry(CostLedger.compute_fill_cost(
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
    ))


def test_fill_model_b_pipeline_partial_fill(tmp_path):
    """
    LIMIT BUY through FillModelB with participation_rate < 1:
    fill_qty capped by effective_liquidity → PARTIALLY_FILLED.
    Pipeline: FillModelB → OSM → EventStore → CostLedger.

    order.qty = 1.0, tick.volume = 10.0, participation_rate = 0.02
    effective_liq = 0.2 < 1.0 → partial fill of 0.2 BTC
    """
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("1.0"), limit_price=Decimal("65000"), event_ts_ms=1_000,
    )
    model = FillModelB(participation_rate=Decimal("0.02"))
    arrival_price = Decimal("65050")

    with EventStore(tmp_path) as store:
        osm.submit(order, event_ts_ms=1_001)
        _persist_submit(store, order, 1_001)
        osm.acknowledge(order, event_ts_ms=1_002)
        _persist_ack(store, order, 1_002)

        tick = _make_tick(price="64990", volume="10.0", ts_ms=1_003)
        decision = model.evaluate(order, tick)
        assert decision is not None
        assert decision.fill_qty == Decimal("0.2")  # 10.0 * 0.02
        _apply_decision(osm, store, order, decision, arrival_price)

    assert order.state      == OrderState.PARTIALLY_FILLED
    assert order.filled_qty == Decimal("0.2")

    with EventStore(tmp_path) as store:
        entries = store.get_cost_entries_by_order(order.order_id)
    assert len(entries) == 1
    assert entries[0].fill_qty     == Decimal("0.2")
    assert entries[0].maker_rebate  > Decimal("0")   # MAKER fill → rebate
    assert entries[0].taker_fee    == Decimal("0")


def test_fill_model_b_replay_fidelity(tmp_path):
    """
    Replay fidelity: live FillModelB run == ExecutionReplay reconstruction.

    order.qty = 0.02 BTC, tick.volume = 1.0, participation_rate = 0.5
    effective_liq = 0.5 > 0.02 → full fill at limit_price

    Verifies: state, filled_qty, avg_fill_price, liquidity_role after EventStore round-trip.
    """
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.02"), limit_price=Decimal("65000"), event_ts_ms=1_000,
    )
    model = FillModelB(participation_rate=Decimal("0.5"))
    arrival_price = Decimal("65020")

    with EventStore(tmp_path) as store:
        osm.submit(order, event_ts_ms=1_001)
        _persist_submit(store, order, 1_001)
        osm.acknowledge(order, event_ts_ms=1_002)
        _persist_ack(store, order, 1_002)

        # effective_liq = 1.0 * 0.5 = 0.5 > order.qty=0.02 → full fill
        tick = _make_tick(price="64990", volume="1.0", ts_ms=1_003)
        decision = model.evaluate(order, tick)
        assert decision is not None
        assert decision.fill_qty == Decimal("0.02")  # capped by remaining_qty
        _apply_decision(osm, store, order, decision, arrival_price)

        replayed = ExecutionReplay(store).replay_order(order.order_id)

    assert replayed.state          == order.state           == OrderState.FILLED
    assert replayed.filled_qty     == order.filled_qty      == Decimal("0.02")
    assert replayed.avg_fill_price == order.avg_fill_price  == Decimal("65000")
    assert replayed.liquidity_role == order.liquidity_role
    assert len(replayed.fills)     == 1
```

- [ ] **Step 2: Run Task 5 tests**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_b.py -k "pipeline or replay" -v
```

Expected: 2 PASS

- [ ] **Step 3: Run complete test_fill_model_b.py**

```
.\venv\Scripts\python.exe -m pytest tests/test_fill_model_b.py -v
```

Expected: **24 PASS**

- [ ] **Step 4: Run full suite — confirm no regressions**

```
.\venv\Scripts\python.exe -m pytest tests/ -v 2>&1 | Select-Object -Last 3
```

Expected: **238 passed, 0 failed.** (213 existing + 1 Task 0 FillModelA + 24 FillModelB)

- [ ] **Step 5: Commit**

```
git add tests/test_fill_model_b.py
git commit -m "test: FillModelB pipeline integration + replay fidelity"
```

---

## Self-Review

### Spec coverage

| Requirement | Covered by |
|-------------|-----------|
| `FillModelB(participation_rate=Decimal("0.01"))` default | `test_fill_model_b_default_participation_rate` |
| Custom `participation_rate` stored | `test_fill_model_b_custom_participation_rate` |
| Range `(0, 1]` — zero invalid | `test_fill_model_b_zero_participation_rate_raises` |
| Range `(0, 1]` — negative invalid | `test_fill_model_b_negative_participation_rate_raises` |
| Range `(0, 1]` — above 1 invalid | `test_fill_model_b_above_one_participation_rate_raises` |
| `participation_rate=1.0` valid | `test_fill_model_b_participation_rate_one_is_valid` |
| `effective_liquidity = tick.volume * participation_rate` | `test_market_fill_qty_capped_by_effective_liquidity`, `test_limit_fill_qty_capped_by_effective_liquidity` |
| `fill_qty = min(remaining_qty, effective_liquidity)` | `test_market_full_fill_when_remaining_qty_less_than_effective_liquidity` |
| MARKET BUY fills at `tick.ask` | `test_market_buy_fills_at_ask` |
| MARKET SELL fills at `tick.bid` | `test_market_sell_fills_at_bid` |
| LIMIT BUY: fills if `tick.price ≤ limit_price` | `test_limit_buy_fills_at_limit_when_price_touched`, `test_limit_buy_fills_exactly_at_limit` |
| LIMIT BUY: no fill if `tick.price > limit_price` | `test_limit_buy_no_fill_when_price_above_limit` |
| LIMIT SELL: fills if `tick.price ≥ limit_price` | `test_limit_sell_fills_at_limit_when_price_touched` |
| LIMIT SELL: no fill if `tick.price < limit_price` | `test_limit_sell_no_fill_when_price_below_limit` |
| Fill at `order.limit_price`, not `tick.price` | `test_limit_buy_fills_at_limit_when_price_touched` — asserts `fill_price == Decimal("65000")` |
| Zero `tick.volume` → `None` | `test_market_zero_tick_volume_returns_none` |
| Terminal order → `ValueError` | `test_market_terminal_state_raises` |
| Fee: MAKER rebate rate (0.01%) | `test_limit_fee_uses_maker_rebate_rate` |
| FillModelB more conservative than FillModelA | `test_fill_model_b_fills_less_than_fill_model_a_same_tick` |
| FillModelB at `rate=1.0` equals FillModelA | `test_fill_model_b_at_rate_one_equals_fill_model_a` |
| No RNG (deterministic) | Implicit: `fill_qty` depends only on `remaining_qty`, `tick.volume`, `participation_rate` |
| No logic duplication (delegates to FillModelA) | Task 1 `FillModelB.evaluate()` — creates capped tick, calls `FillModelA.evaluate()` |
| `execution_id_factory` injectable | Task 0 (FillModelA), Task 1 `FillModelB.__init__` |
| Decimal precision preserved (no float rounding) | `test_effective_liquidity_decimal_precision` |
| Pipeline: FillModelB → OSM → EventStore → CostLedger | `test_fill_model_b_pipeline_partial_fill` |
| Replay fidelity: live == replay | `test_fill_model_b_replay_fidelity` |

All requirements covered. ✅

### Placeholder scan

No TBDs, TODOs, or code-free steps. Every step contains exact code and expected output. ✅

### Type consistency

| Symbol | Defined in | Used in |
|--------|-----------|---------|
| `FillModelB` | Task 1 / `fill_model_b.py` | All tasks |
| `FillModelB.participation_rate` | Task 1 | Tasks 2, 3, 4, 5 |
| `FillModelB.evaluate()` | Task 1 | Tasks 2, 3, 4, 5 |
| `Tick` | `fill_model_a.py` (existing) | `_make_tick()` helper, Tasks 2–5 |
| `FillDecision` | `fill_model_a.py` (existing) | `_apply_decision()` helper, Task 5 |
| `FillModelA.evaluate()` | `fill_model_a.py` (existing) | Task 4 comparison tests |
| `_make_limit_order()` | Task 1 helpers | Tasks 3, 4, 5 |
| `_make_market_order()` | Task 1 helpers | Task 2 |
| `_make_tick()` | Task 1 helpers | Tasks 2, 3, 4, 5 |
| `_persist_submit()` | Task 5 helpers | Task 5 |
| `_persist_ack()` | Task 5 helpers | Task 5 |
| `_apply_decision()` | Task 5 helpers | Task 5 |
| `CostLedger.compute_fill_cost()` | `live/cost_ledger.py` (existing) | `_apply_decision()` |
| `ExecutionReplay.replay_order()` | `live/execution_replay.py` (existing) | Task 5 |

All consistent. ✅
