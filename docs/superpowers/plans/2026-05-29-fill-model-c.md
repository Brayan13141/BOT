# FillModelC — Queue-Aware Tape-Driven Fill Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Commits:** Per project convention, **Bryan makes all commits himself.** The commit steps below mark logical checkpoints — when executing, stage the files (`git add`) and **pause for Bryan to commit**. Do NOT run `git commit` yourself.

**Goal:** Implement FillModelC, a queue-aware fill engine that simulates order execution against the aggTrades stream, killing the `TOUCH != FILL` fantasy by requiring real opposing aggressor volume to trade through a resting level after an explicit, calibrated `queue_ahead`.

**Architecture:** A single class `FillModelC` with one public method `evaluate(order, trades, *, active_since_agg_trade_id) -> FillEvaluationResult`. It is a pure fold over the trade window: stateless between calls, stateful only within one call. Two internal fill paths — MAKER (opposite-side flow through the limit price, gated by `queue_ahead`) and TAKER (same-side print walk, no queue). The anti-look-ahead rule (`agg_trade_id > active_since_agg_trade_id`) is enforced inside the model. Maker fills always execute at the limit price (passive-price invariant); taker fills at the realized print price. No L2 reconstruction, no synthetic spread, no estimated queue.

**Tech Stack:** Python 3.12, `decimal.Decimal`, `uuid`, `pytest`, `tmp_path` fixture. Reuses `live/fill_model_a.py` (`FillDecision`), `live/fee_schedule.py` (`TAKER_FEE_RATE`, `MAKER_REBATE_RATE`, `FEE_PRECISION`), `live/order_types.py` (`Order`, `OrderSide`, `OrderType`, `OrderState`, `FeeModel`), `live/order_state_machine.py`, `live/event_store.py`, `live/event_models.py`, `live/cost_ledger.py`, `live/execution_replay.py`.

**Spec:** `OBSIDIAN/docs/superpowers/specs/2026-05-29-fill-model-c-design.md` (APPROVED rev 1).

---

## Key Design Decisions (fixed — no changes without a new spec)

| Decision | Value |
|----------|-------|
| `queue_ahead` | Constructor arg, **mandatory, no default**. Validation: `>= 0` (0 = optimistic baseline). No upper bound. |
| Interface | `evaluate(order, trades, *, active_since_agg_trade_id) -> FillEvaluationResult` |
| Anti-look-ahead | Enforced **inside** C: skip every `trade.agg_trade_id <= active_since_agg_trade_id` |
| MAKER fill price | **Always `order.limit_price`** (passive-price invariant), `FeeModel.MAKER` |
| MAKER qualifying print | BUY @ P: `SELL` print with `price <= P`. SELL @ P: `BUY` print with `price >= P`. |
| TAKER fill price | Realized `trade.price` (same-side walk), `FeeModel.TAKER` |
| TAKER aggressor side | BUY order consumes `BUY` prints; SELL order consumes `SELL` prints |
| `queue_ahead` for takers | Unused → `queue_consumed = 0`, `queue_remaining = queue_ahead` |
| `volume_through` for takers | `Decimal("0")` (defined only for the maker level) |
| Maker/taker routing | By `order_type` only: `MARKET`→taker walk, `LIMIT`→maker queue. No tape-based marketability inference (OSM decides execution nature via per-fill `fee_model`). |
| `prints_consumed` | Count of qualifying prints the fold processed (maker: every opposite print through P; taker: every same-side print walked) |
| Fee formula | `(fill_price * fill_qty * rate).quantize(FEE_PRECISION)` — identical to FillModelA |
| RNG | none — fully deterministic; `execution_id_factory` injectable (default `uuid4`) |
| Forbidden | No `book_depth`, `queue_position`, `queue_probability`, synthetic spread, estimated queue |

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `live/fill_model_c.py` | Create | `AggTrade`, `FillEvaluationResult`, `FillModelC` |
| `tests/test_fill_model_c.py` | Create | All tests (Tasks 1–5): constructor, taker walk, maker queue, classification, invariants + pipeline |
| `live/fill_model_a.py` | Reuse (unchanged) | `FillDecision` |
| `live/fee_schedule.py` | Reuse (unchanged) | `TAKER_FEE_RATE`, `MAKER_REBATE_RATE`, `FEE_PRECISION` |
| `live/order_types.py` | Reuse (unchanged) | `Order`, `OrderSide`, `OrderType`, `OrderState`, `FeeModel` |

`order_types.py` stays untouched: `active_since_agg_trade_id` is a keyword arg of `evaluate`, not an `Order` field. This keeps the shared OSM contract free of aggTrades-specific concepts.

---

## Task 1: Data contracts + constructor

**Files:**
- Create: `live/fill_model_c.py`
- Test: `tests/test_fill_model_c.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fill_model_c.py`:

```python
"""Tests for FillModelC: queue-aware, tape-driven fill engine over aggTrades."""
from decimal import Decimal

import pytest

from live.fill_model_a import FillDecision
from live.fill_model_c import AggTrade, FillEvaluationResult, FillModelC
from live.order_state_machine import OrderStateMachine
from live.order_types import FeeModel, OrderSide, OrderState, OrderType


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agg(agg_trade_id, price, qty, side, ts_ms=None):
    """Build an AggTrade. ts_ms defaults to agg_trade_id*1000 for readability."""
    return AggTrade(
        timestamp_ms=ts_ms if ts_ms is not None else agg_trade_id * 1_000,
        price=Decimal(price),
        qty=Decimal(qty),
        side=OrderSide(side),
        agg_trade_id=agg_trade_id,
    )


def _make_limit_order(side="BUY", qty="1.0", limit_price="65000", ts=1_000):
    """Return an ACKNOWLEDGED LIMIT order ready for evaluation."""
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide(side), order_type=OrderType.LIMIT,
        qty=Decimal(qty), limit_price=Decimal(limit_price), event_ts_ms=ts,
    )
    osm.submit(order, event_ts_ms=ts + 1)
    osm.acknowledge(order, event_ts_ms=ts + 2)
    return order, osm


def _make_market_order(side="BUY", qty="1.0", ts=1_000):
    """Return a SUBMITTED MARKET order ready for evaluation (immediate taker path)."""
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide(side), order_type=OrderType.MARKET,
        qty=Decimal(qty), event_ts_ms=ts,
    )
    osm.submit(order, event_ts_ms=ts + 1)
    return order, osm


def _counter_factory():
    """Deterministic execution_id factory for golden tests."""
    n = {"i": 0}
    def _next():
        n["i"] += 1
        return f"exec-{n['i']:04d}"
    return _next


# ── Task 1: Constructor + data contracts ──────────────────────────────────────

def test_aggtrade_is_frozen():
    t = _agg(1, "65000", "0.5", "BUY")
    assert t.agg_trade_id == 1
    assert t.side == OrderSide.BUY
    with pytest.raises(Exception):
        t.price = Decimal("1")  # frozen


def test_constructor_stores_queue_ahead():
    model = FillModelC(queue_ahead=Decimal("0.5"))
    assert model.queue_ahead == Decimal("0.5")


def test_constructor_queue_ahead_zero_is_valid():
    """queue_ahead=0 is the optimistic baseline (conceptual fill fantasy)."""
    model = FillModelC(queue_ahead=Decimal("0"))
    assert model.queue_ahead == Decimal("0")


def test_constructor_negative_queue_ahead_raises():
    with pytest.raises(ValueError, match="queue_ahead"):
        FillModelC(queue_ahead=Decimal("-0.1"))


def test_constructor_requires_queue_ahead():
    """queue_ahead has no default — calling without it is a TypeError."""
    with pytest.raises(TypeError):
        FillModelC()  # type: ignore[call-arg]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fill_model_c.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'live.fill_model_c'`.

- [ ] **Step 3: Write minimal implementation**

Create `live/fill_model_c.py`:

```python
"""
FillModelC — queue-aware, tape-driven fill engine for execution simulation.

Simulates fills against the aggTrades stream (ts, price, qty, side, agg_trade_id)
instead of OHLCV ticks. Kills the TOUCH != FILL fantasy that invalidated Exp08a:
a resting maker fills only when real opposing aggressor volume trades through its
level, after an explicit, calibrated queue_ahead.

Doctrine:
- The tape is ground truth. Only observed trades drive fills.
- queue_ahead is the only unobservable, made an explicit calibration knob.
- TOUCH != FILL. Maker fills require opposing volume through the level.
- No look-ahead: only trades with agg_trade_id > active_since_agg_trade_id.
- Passive-price invariant: maker fills always at limit price, never the print price.
- No L2 reconstruction, no synthetic spread, no estimated queue.

Stateless between calls; stateful only within a single evaluate() fold.
Deterministic given (order, trades, queue_ahead, active_since_agg_trade_id).
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal

from live.fee_schedule import FEE_PRECISION, MAKER_REBATE_RATE, TAKER_FEE_RATE
from live.fill_model_a import FillDecision
from live.order_types import FeeModel, Order, OrderSide, OrderState, OrderType

_TERMINAL_STATES = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.EXPIRED,
    OrderState.REJECTED,
})


@dataclass(frozen=True)
class AggTrade:
    """
    Single aggregated trade print from the exchange tape.

    side is the TAKER side: BUY = taker lifted the ask, SELL = taker hit the bid.
    agg_trade_id is strictly increasing (source guarantee) and anchors causality.
    """
    timestamp_ms: int
    price:        Decimal
    qty:          Decimal
    side:         OrderSide
    agg_trade_id: int


@dataclass(frozen=True)
class FillEvaluationResult:
    """
    Self-contained result of FillModelC.evaluate(). Contains zero L2-inference fields.

    volume_through:  total opposite-aggressor volume through P (maker); 0 for taker.
    queue_ahead:     the simulation hypothesis (model config), always present.
    queue_consumed:  how much of queue_ahead was eaten (<= queue_ahead); 0 for taker.
    queue_remaining: queue_ahead - queue_consumed.
    prints_consumed: qualifying prints the fold processed (maker + taker).
    """
    fills:                     list[FillDecision]
    active_since_agg_trade_id: int
    volume_through:            Decimal
    queue_ahead:               Decimal
    queue_consumed:            Decimal
    queue_remaining:           Decimal
    filled_qty:                Decimal
    remaining_qty:             Decimal
    fully_filled:              bool
    prints_consumed:           int


class FillModelC:
    """Queue-aware fill engine over aggTrades. Configure via queue_ahead (BTC)."""

    def __init__(
        self,
        queue_ahead: Decimal,                                    # REQUIRED, no default
        execution_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if queue_ahead < Decimal("0"):
            raise ValueError(f"queue_ahead must be >= 0, got {queue_ahead}")
        self.queue_ahead = queue_ahead
        self._execution_id_factory = execution_id_factory

    def _new_id(self) -> str:
        fn = self._execution_id_factory if self._execution_id_factory is not None else (lambda: str(uuid.uuid4()))
        return fn()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fill_model_c.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Stage for commit (Bryan commits)**

```bash
git add live/fill_model_c.py tests/test_fill_model_c.py
```
Then pause: tell Bryan the suggested message `feat: FillModelC data contracts + constructor`.

---

## Task 2: TAKER walk (MARKET orders)

**Files:**
- Modify: `live/fill_model_c.py`
- Test: `tests/test_fill_model_c.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fill_model_c.py`:

```python
# ── Task 2: TAKER walk (MARKET orders) ────────────────────────────────────────

def test_market_buy_fills_at_first_same_side_print():
    """MARKET BUY walks BUY-side prints; single print covers qty -> one fill."""
    order, _ = _make_market_order(side="BUY", qty="0.5")
    trades = [_agg(101, "65005", "1.0", "BUY")]
    result = FillModelC(Decimal("0.5")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert len(result.fills) == 1
    assert result.fills[0].fill_price == Decimal("65005")   # realized print price
    assert result.fills[0].fill_qty   == Decimal("0.5")     # capped by remaining_qty
    assert result.fills[0].fee_model  == FeeModel.TAKER
    assert result.fully_filled is True
    assert result.filled_qty == Decimal("0.5")
    assert result.volume_through == Decimal("0")             # taker: not defined
    assert result.queue_consumed == Decimal("0")
    assert result.prints_consumed == 1


def test_market_buy_walks_multiple_prints():
    """qty exceeds first print -> walk forward, accumulating honest slippage."""
    order, _ = _make_market_order(side="BUY", qty="0.5")
    trades = [
        _agg(101, "100000", "0.20", "BUY"),
        _agg(102, "100001", "0.15", "BUY"),
        _agg(103, "100002", "0.30", "BUY"),
    ]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert [f.fill_qty for f in result.fills] == [Decimal("0.20"), Decimal("0.15"), Decimal("0.15")]
    assert [f.fill_price for f in result.fills] == [Decimal("100000"), Decimal("100001"), Decimal("100002")]
    assert result.fully_filled is True
    assert result.prints_consumed == 3


def test_market_buy_ignores_opposite_side_prints():
    """MARKET BUY consumes only BUY prints; SELL prints are skipped."""
    order, _ = _make_market_order(side="BUY", qty="0.3")
    trades = [
        _agg(101, "100000", "5.0", "SELL"),   # opposite side -> ignored
        _agg(102, "100001", "0.3", "BUY"),
    ]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert len(result.fills) == 1
    assert result.fills[0].fill_price == Decimal("100001")
    assert result.prints_consumed == 1


def test_market_sell_walks_sell_prints():
    order, _ = _make_market_order(side="SELL", qty="0.4")
    trades = [_agg(101, "99999", "1.0", "SELL")]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert len(result.fills) == 1
    assert result.fills[0].fill_price == Decimal("99999")
    assert result.fills[0].fee_model  == FeeModel.TAKER


def test_taker_anti_lookahead_skips_trigger_and_earlier():
    """Trades with agg_trade_id <= active_since are never consumed."""
    order, _ = _make_market_order(side="BUY", qty="0.5")
    trades = [
        _agg(100, "100000", "5.0", "BUY"),   # the trigger trade -> excluded
        _agg(99,  "99999",  "5.0", "BUY"),   # earlier -> excluded
        _agg(101, "100001", "0.5", "BUY"),   # first valid
    ]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert len(result.fills) == 1
    assert result.fills[0].fill_price == Decimal("100001")
    assert result.active_since_agg_trade_id == 100


def test_taker_partial_when_insufficient_volume():
    order, _ = _make_market_order(side="BUY", qty="1.0")
    trades = [_agg(101, "100000", "0.3", "BUY")]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert result.filled_qty == Decimal("0.3")
    assert result.remaining_qty == Decimal("0.7")
    assert result.fully_filled is False


def test_taker_empty_window_returns_empty_result():
    order, _ = _make_market_order(side="BUY", qty="1.0")
    result = FillModelC(Decimal("0")).evaluate(order, [], active_since_agg_trade_id=100)

    assert result.fills == []
    assert result.filled_qty == Decimal("0")
    assert result.remaining_qty == Decimal("1.0")
    assert result.fully_filled is False


def test_taker_terminal_state_raises():
    order, osm = _make_market_order(side="BUY", qty="0.01")
    osm.fill(
        order, fill_price=Decimal("65005"), fill_qty=Decimal("0.01"),
        event_ts_ms=2_000, fee_model=FeeModel.TAKER, execution_id="exec-term-c-001",
    )
    assert order.state == OrderState.FILLED
    with pytest.raises(ValueError, match="terminal"):
        FillModelC(Decimal("0")).evaluate(order, [_agg(101, "65005", "1.0", "BUY")], active_since_agg_trade_id=100)


def test_taker_fee_uses_taker_rate():
    order, _ = _make_market_order(side="BUY", qty="1.0")
    trades = [_agg(101, "100000", "1.0", "BUY")]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    expected_fee = (Decimal("100000") * Decimal("1.0") * Decimal("0.0004")).quantize(Decimal("0.00000001"))
    assert result.fills[0].fee == expected_fee
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fill_model_c.py -k "taker or market" -v`
Expected: FAIL — `FillModelC` has no `evaluate` method (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

Add to `live/fill_model_c.py` inside `class FillModelC` (after `_new_id`):

```python
    def evaluate(
        self,
        order: Order,
        trades: Iterable[AggTrade],
        *,
        active_since_agg_trade_id: int,
    ) -> FillEvaluationResult:
        """
        Fold over trades with agg_trade_id > active_since_agg_trade_id and return fills.

        MARKET orders take the TAKER walk. (LIMIT routing added in later tasks.)

        Raises:
            ValueError: if order is in a terminal state.
        """
        if order.state in _TERMINAL_STATES:
            raise ValueError(
                f"Cannot evaluate fill for terminal order {order.order_id!r} "
                f"(state={order.state.value})"
            )

        window = [t for t in trades if t.agg_trade_id > active_since_agg_trade_id]

        if order.order_type == OrderType.MARKET:
            return self._taker_walk(order, window, active_since_agg_trade_id)
        # LIMIT routing is added in Task 3 / Task 4. Placeholder until then:
        return self._taker_walk(order, window, active_since_agg_trade_id)

    def _taker_walk(
        self,
        order: Order,
        window: list[AggTrade],
        active_since_agg_trade_id: int,
    ) -> FillEvaluationResult:
        """Same-side print walk. No queue. Fills at realized print price (TAKER)."""
        remaining = order.remaining_qty
        fills: list[FillDecision] = []
        prints_consumed = 0
        for t in window:
            if remaining <= Decimal("0"):
                break
            if t.side != order.side:          # same-side only: BUY order consumes BUY prints
                continue
            fill_qty = min(t.qty, remaining)
            fee = (t.price * fill_qty * TAKER_FEE_RATE).quantize(FEE_PRECISION)
            fills.append(FillDecision(
                fill_price=t.price,
                fill_qty=fill_qty,
                fee_model=FeeModel.TAKER,
                fee=fee,
                execution_id=self._new_id(),
                event_ts_ms=t.timestamp_ms,
            ))
            remaining -= fill_qty
            prints_consumed += 1

        filled = order.remaining_qty - remaining
        return FillEvaluationResult(
            fills=fills,
            active_since_agg_trade_id=active_since_agg_trade_id,
            volume_through=Decimal("0"),
            queue_ahead=self.queue_ahead,
            queue_consumed=Decimal("0"),
            queue_remaining=self.queue_ahead,
            filled_qty=filled,
            remaining_qty=remaining,
            fully_filled=(remaining == Decimal("0")),
            prints_consumed=prints_consumed,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fill_model_c.py -v`
Expected: all Task 1 + Task 2 tests PASS.

- [ ] **Step 5: Stage for commit (Bryan commits)**

```bash
git add live/fill_model_c.py tests/test_fill_model_c.py
```
Suggested message: `feat: FillModelC TAKER same-side print walk`.

---

## Task 3: MAKER queue (resting LIMIT)

**Files:**
- Modify: `live/fill_model_c.py`
- Test: `tests/test_fill_model_c.py`

For this task, resting LIMIT orders are routed to the maker queue. We make the orders
clearly non-marketable (limit far from the first print) so the Task-4 classifier (added
next) will keep routing them to MAKER.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fill_model_c.py`:

```python
# ── Task 3: MAKER queue (resting LIMIT) ───────────────────────────────────────

def test_maker_buy_no_fill_when_no_volume_through():
    """Resting BUY @ P: SELL prints above P do not reach us -> no fill."""
    order, _ = _make_limit_order(side="BUY", qty="1.0", limit_price="65000")
    trades = [_agg(101, "65010", "5.0", "SELL")]   # 65010 > 65000 -> not through P
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert result.fills == []
    assert result.volume_through == Decimal("0")
    assert result.prints_consumed == 0


def test_maker_buy_fills_at_limit_price_passive_invariant():
    """
    Passive-price invariant: a SELL print at 49990 fills our BUY limit @ 50000 AT 50000.
    queue_ahead=0 -> the through-volume fills us immediately.
    """
    order, _ = _make_limit_order(side="BUY", qty="0.3", limit_price="50000")
    trades = [_agg(101, "49990", "1.0", "SELL")]   # through P, qty 1.0 >= 0.3
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert len(result.fills) == 1
    assert result.fills[0].fill_price == Decimal("50000")   # NOT 49990
    assert result.fills[0].fee_model  == FeeModel.MAKER
    assert result.fully_filled is True
    assert result.volume_through == Decimal("1.0")
    assert result.prints_consumed == 1


def test_maker_queue_ahead_consumed_before_fill():
    """
    queue_ahead=2.0 BTC. First 2.0 BTC of through-volume is eaten by the queue
    (no fill); volume beyond fills us.
    """
    order, _ = _make_limit_order(side="BUY", qty="0.5", limit_price="50000")
    trades = [
        _agg(101, "49999", "1.5", "SELL"),   # eats 1.5 of queue, 0 fill
        _agg(102, "49998", "1.0", "SELL"),   # eats 0.5 queue, 0.5 available -> fills 0.5
    ]
    result = FillModelC(Decimal("2.0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert result.queue_consumed == Decimal("2.0")
    assert result.queue_remaining == Decimal("0")
    assert result.filled_qty == Decimal("0.5")
    assert result.fully_filled is True
    assert len(result.fills) == 1
    assert result.fills[0].fill_qty == Decimal("0.5")
    assert result.fills[0].event_ts_ms == 102_000      # stamped from the filling print
    assert result.volume_through == Decimal("2.5")
    assert result.prints_consumed == 2                 # both prints processed


def test_maker_partial_fill_when_through_volume_insufficient():
    """queue_ahead=0, through-volume < order.qty -> partial fill."""
    order, _ = _make_limit_order(side="BUY", qty="1.0", limit_price="50000")
    trades = [_agg(101, "49995", "0.4", "SELL")]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert result.filled_qty == Decimal("0.4")
    assert result.remaining_qty == Decimal("0.6")
    assert result.fully_filled is False


def test_maker_sell_fills_on_buy_through():
    """Resting SELL @ P fills on BUY prints with price >= P, at P."""
    order, _ = _make_limit_order(side="SELL", qty="0.3", limit_price="50000")
    trades = [_agg(101, "50010", "1.0", "BUY")]   # 50010 >= 50000 -> through P
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert len(result.fills) == 1
    assert result.fills[0].fill_price == Decimal("50000")
    assert result.fills[0].fee_model  == FeeModel.MAKER


def test_maker_ignores_same_side_prints():
    """Resting BUY is filled by SELL aggressors only; BUY prints are ignored."""
    order, _ = _make_limit_order(side="BUY", qty="0.3", limit_price="50000")
    trades = [
        _agg(101, "49990", "5.0", "BUY"),    # same side as a buyer aggressor -> ignored
        _agg(102, "49990", "0.3", "SELL"),   # opposite, through P -> fills
    ]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    assert len(result.fills) == 1
    assert result.prints_consumed == 1
    assert result.volume_through == Decimal("0.3")


def test_maker_fee_uses_maker_rebate_rate():
    order, _ = _make_limit_order(side="BUY", qty="1.0", limit_price="50000")
    trades = [_agg(101, "49990", "1.0", "SELL")]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)

    expected_fee = (Decimal("50000") * Decimal("1.0") * Decimal("0.0001")).quantize(Decimal("0.00000001"))
    assert result.fills[0].fee == expected_fee
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fill_model_c.py -k "maker" -v`
Expected: FAIL — LIMIT orders currently route to `_taker_walk` (placeholder), so prices/fees/volume_through are wrong.

- [ ] **Step 3: Write minimal implementation**

In `live/fill_model_c.py`, replace the LIMIT placeholder line in `evaluate` and add `_maker_queue` + `_qualifies_maker`.

Replace this block in `evaluate`:

```python
        # LIMIT routing is added in Task 3 / Task 4. Placeholder until then:
        return self._taker_walk(order, window, active_since_agg_trade_id)
```

with:

```python
        # LIMIT: maker queue. (Marketable-limit -> taker routing added in Task 4.)
        return self._maker_queue(order, window, active_since_agg_trade_id)
```

Add these methods to `class FillModelC`:

```python
    @staticmethod
    def _qualifies_maker(order: Order, trade: AggTrade, limit_price: Decimal) -> bool:
        """Opposite-side aggressor trading through the resting limit price."""
        if order.side == OrderSide.BUY:
            return trade.side == OrderSide.SELL and trade.price <= limit_price
        return trade.side == OrderSide.BUY and trade.price >= limit_price

    def _maker_queue(
        self,
        order: Order,
        window: list[AggTrade],
        active_since_agg_trade_id: int,
    ) -> FillEvaluationResult:
        """
        Opposite flow through P, gated by queue_ahead. Fills at limit price (passive).

        For each qualifying print: accumulate volume_through, consume queue_ahead first
        (does not fill), then fill from any remaining print volume up to remaining_qty.
        """
        if order.limit_price is None:
            raise ValueError(
                f"LIMIT order {order.order_id!r} has no limit_price — OSM invariant violated."
            )
        limit_price = order.limit_price
        remaining = order.remaining_qty
        fills: list[FillDecision] = []
        volume_through = Decimal("0")
        queue_consumed = Decimal("0")
        prints_consumed = 0

        for t in window:
            if not self._qualifies_maker(order, t, limit_price):
                continue
            volume_through += t.qty
            prints_consumed += 1                      # every qualifying print is processed

            # 1. consume the queue ahead of us first (does NOT fill us)
            if queue_consumed < self.queue_ahead:
                eaten = min(t.qty, self.queue_ahead - queue_consumed)
                queue_consumed += eaten
                available = t.qty - eaten
            else:
                available = t.qty

            # 2. volume beyond the queue fills us, at the limit price
            if available > Decimal("0") and remaining > Decimal("0"):
                fill_qty = min(available, remaining)
                fee = (limit_price * fill_qty * MAKER_REBATE_RATE).quantize(FEE_PRECISION)
                fills.append(FillDecision(
                    fill_price=limit_price,
                    fill_qty=fill_qty,
                    fee_model=FeeModel.MAKER,
                    fee=fee,
                    execution_id=self._new_id(),
                    event_ts_ms=t.timestamp_ms,
                ))
                remaining -= fill_qty

            if remaining <= Decimal("0"):
                break

        filled = order.remaining_qty - remaining
        return FillEvaluationResult(
            fills=fills,
            active_since_agg_trade_id=active_since_agg_trade_id,
            volume_through=volume_through,
            queue_ahead=self.queue_ahead,
            queue_consumed=queue_consumed,
            queue_remaining=self.queue_ahead - queue_consumed,
            filled_qty=filled,
            remaining_qty=remaining,
            fully_filled=(remaining == Decimal("0")),
            prints_consumed=prints_consumed,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fill_model_c.py -v`
Expected: all Task 1–3 tests PASS.

- [ ] **Step 5: Stage for commit (Bryan commits)**

```bash
git add live/fill_model_c.py tests/test_fill_model_c.py
```
Suggested message: `feat: FillModelC MAKER queue-aware fills (passive-price invariant)`.

---

## Task 4: LIMIT routing — no tape-based marketability inference

**Files:**
- Modify: `live/fill_model_c.py`
- Test: `tests/test_fill_model_c.py`

> **Redesigned during execution (2026-05-29).** The original plan routed LIMIT orders by a
> tape-derived `reference_price` heuristic (marketable iff the limit was at/through the first
> in-window print → TAKER, else MAKER). Implementation revealed this was **doctrinally wrong**
> (it re-derives execution intent the OSM already determines via per-fill `fee_model`,
> violating the layer separation and "do not infer the unobserved") and **fragile** (it
> misclassified canonical resting-maker fills, e.g. a BUY limit filled by SELL prints below it
> read as "marketable" and routed to a same-side taker walk that found no prints). It was
> removed. **C routes purely by `order_type`: `MARKET`→taker walk, `LIMIT`→maker queue.** A
> LIMIT reaching C is a resting maker; a marketable/taker-converted limit is handled upstream
> by the OSM (`IMMEDIATE_TAKER_FILL` from `SUBMITTED`), never arriving here as a resting limit.

Net effect on the code after Task 3: the `evaluate` LIMIT branch stays `return
self._maker_queue(...)` (no `_is_marketable`, no marketability branch). Only documentation
and one edge-case test are added.

- [ ] **Step 1: Confirm `evaluate` LIMIT routing is unconditional maker queue**

In `live/fill_model_c.py`, the LIMIT branch of `evaluate` must read exactly:

```python
        if order.order_type == OrderType.MARKET:
            return self._taker_walk(order, window, active_since_agg_trade_id)
        # LIMIT orders evaluated by C are resting makers. Maker-vs-taker is decided
        # upstream by the OSM (per-fill fee_model); C does not infer it from the tape.
        return self._maker_queue(order, window, active_since_agg_trade_id)
```

There is NO `_is_marketable` method and NO `reference_price` logic anywhere in the file.

- [ ] **Step 2: Add the LIMIT empty-window edge-case test**

Append to `tests/test_fill_model_c.py`:

```python
# ── LIMIT routing: a LIMIT reaching C is a resting maker ───────────────────────
# (maker-vs-taker is decided upstream by the OSM via per-fill fee_model; C does
#  not infer marketability from the tape — see 2026-05-29 design decision.)

def test_limit_empty_window_returns_empty_maker_result():
    """A resting LIMIT with no qualifying trades -> empty result (no fills)."""
    order, _ = _make_limit_order(side="BUY", qty="1.0", limit_price="64990")
    result = FillModelC(Decimal("0")).evaluate(order, [], active_since_agg_trade_id=100)

    assert result.fills == []
    assert result.fully_filled is False
    assert result.remaining_qty == Decimal("1.0")
```

- [ ] **Step 3: Run the full test file**

Run: `python -m pytest tests/test_fill_model_c.py -v`
Expected: all Task 1–3 tests + this one PASS (22 total: 5 + 9 + 7 + 1).

- [ ] **Step 4: Stage for commit (Bryan commits)**

```bash
git add live/fill_model_c.py tests/test_fill_model_c.py
```
Suggested message:
```
refactor: drop tape-based marketability inference from FillModelC
```

---

## Task 5: Invariants — determinism, idempotency, replay fidelity + pipeline

**Files:**
- Modify: `tests/test_fill_model_c.py`

This task adds no production code — it proves the three execution invariants and the
end-to-end pipeline (FillModelC → OSM → EventStore → CostLedger → ExecutionReplay), the
same integration pattern used by FillModelA/B.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fill_model_c.py`:

```python
# ── Task 5: Invariants + pipeline integration ─────────────────────────────────

import uuid

from live.cost_ledger import CostLedger
from live.event_models import OrderAcknowledged, OrderFillReceived, OrderSubmitted
from live.event_store import EventStore
from live.execution_replay import ExecutionReplay


def test_determinism_two_runs_identical_economics():
    """
    Golden re-run: two passes over the same inputs with the same deterministic
    factory produce identical fills (prices, qtys, fees, execution_ids).
    """
    trades = [
        _agg(101, "49999", "1.5", "SELL"),
        _agg(102, "49998", "1.0", "SELL"),
    ]

    order_a, _ = _make_limit_order(side="BUY", qty="0.5", limit_price="50000")
    order_b, _ = _make_limit_order(side="BUY", qty="0.5", limit_price="50000")

    r1 = FillModelC(Decimal("2.0"), execution_id_factory=_counter_factory()).evaluate(
        order_a, trades, active_since_agg_trade_id=100)
    r2 = FillModelC(Decimal("2.0"), execution_id_factory=_counter_factory()).evaluate(
        order_b, trades, active_since_agg_trade_id=100)

    assert [(f.fill_price, f.fill_qty, f.fee, f.execution_id, f.event_ts_ms) for f in r1.fills] \
        == [(f.fill_price, f.fill_qty, f.fee, f.execution_id, f.event_ts_ms) for f in r2.fills]
    assert r1.queue_consumed == r2.queue_consumed
    assert r1.volume_through == r2.volume_through


def test_default_factory_unique_execution_ids():
    """Without an injected factory, each fill gets a distinct uuid4 id."""
    order, _ = _make_market_order(side="BUY", qty="0.5")
    trades = [
        _agg(101, "100000", "0.2", "BUY"),
        _agg(102, "100001", "0.3", "BUY"),
    ]
    result = FillModelC(Decimal("0")).evaluate(order, trades, active_since_agg_trade_id=100)
    ids = [f.execution_id for f in result.fills]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def _persist_submit(store, order, ts):
    store.append_domain_event(OrderSubmitted(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=ts,
        order_id=order.order_id, symbol=order.symbol, side=order.side.value,
        order_type=order.order_type.value, qty=order.qty,
        limit_price=order.limit_price, created_ts_ms=order.created_ts_ms,
    ))


def _persist_ack(store, order, ts):
    store.append_domain_event(OrderAcknowledged(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=ts,
        order_id=order.order_id, submitted_ts_ms=order.submitted_ts_ms,
    ))


def _apply_decision(osm, store, order, decision, arrival_price):
    osm.fill(
        order, fill_price=decision.fill_price, fill_qty=decision.fill_qty,
        event_ts_ms=decision.event_ts_ms, fee_model=decision.fee_model,
        execution_id=decision.execution_id,
    )
    store.append_domain_event(OrderFillReceived(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=decision.event_ts_ms,
        order_id=order.order_id, fill_id=decision.execution_id,
        price=decision.fill_price, qty=decision.fill_qty, fee=decision.fee,
        fee_asset="USDT", fee_model=decision.fee_model.value,
    ))
    store.append_cost_entry(CostLedger.compute_fill_cost(
        fill_id=decision.execution_id, order_id=order.order_id, side=order.side.value,
        fill_price=decision.fill_price, fill_qty=decision.fill_qty,
        arrival_price=arrival_price, fee_model=decision.fee_model.value,
        fee=decision.fee, fill_ts_ms=decision.event_ts_ms,
        event_id=str(uuid.uuid4()), correlation_id=order.order_id,
    ))


def test_pipeline_maker_multi_partial_then_replay(tmp_path):
    """
    Full pipeline with a multi-print maker fill, then replay fidelity.
    queue_ahead=0; two SELL prints through P each fill part of the order.
    """
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.5"), limit_price=Decimal("50000"), event_ts_ms=1_000,
    )
    model = FillModelC(Decimal("0"), execution_id_factory=_counter_factory())
    arrival_price = Decimal("50050")
    trades = [
        _agg(101, "49995", "0.2", "SELL", ts_ms=1_003),
        _agg(102, "49994", "0.3", "SELL", ts_ms=1_004),
    ]

    with EventStore(tmp_path) as store:
        osm.submit(order, event_ts_ms=1_001)
        _persist_submit(store, order, 1_001)
        osm.acknowledge(order, event_ts_ms=1_002)
        _persist_ack(store, order, 1_002)

        result = model.evaluate(order, trades, active_since_agg_trade_id=100)
        assert len(result.fills) == 2
        assert result.fully_filled is True
        for decision in result.fills:
            _apply_decision(osm, store, order, decision, arrival_price)

        replayed = ExecutionReplay(store).replay_order(order.order_id)

    assert order.state == OrderState.FILLED
    assert order.filled_qty == Decimal("0.5")
    assert order.avg_fill_price == Decimal("50000")        # both fills at limit price
    assert replayed.state          == order.state
    assert replayed.filled_qty     == order.filled_qty
    assert replayed.avg_fill_price == order.avg_fill_price
    assert replayed.liquidity_role == order.liquidity_role
    assert len(replayed.fills)     == 2


def test_pipeline_cost_ledger_records_maker_rebate(tmp_path):
    osm = OrderStateMachine()
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.5"), limit_price=Decimal("50000"), event_ts_ms=1_000,
    )
    model = FillModelC(Decimal("0"), execution_id_factory=_counter_factory())
    trades = [_agg(101, "49995", "0.5", "SELL", ts_ms=1_003)]

    with EventStore(tmp_path) as store:
        osm.submit(order, event_ts_ms=1_001)
        _persist_submit(store, order, 1_001)
        osm.acknowledge(order, event_ts_ms=1_002)
        _persist_ack(store, order, 1_002)
        result = model.evaluate(order, trades, active_since_agg_trade_id=100)
        for decision in result.fills:
            _apply_decision(osm, store, order, decision, Decimal("50050"))

    with EventStore(tmp_path) as store:
        entries = store.get_cost_entries_by_order(order.order_id)
    assert len(entries) == 1
    assert entries[0].maker_rebate > Decimal("0")
    assert entries[0].taker_fee   == Decimal("0")
```

- [ ] **Step 2: Run tests to verify they fail (or error)**

Run: `python -m pytest tests/test_fill_model_c.py -k "determinism or pipeline or factory" -v`
Expected: PASS if production code is complete. If any import name (`CostLedger.compute_fill_cost`, `OrderFillReceived` fields, `get_cost_entries_by_order`) differs in the current codebase, the test will error — fix the test to match the real signatures (verify against `live/cost_ledger.py`, `live/event_models.py`, `live/event_store.py`). Do NOT change production behavior to satisfy a mis-written test.

- [ ] **Step 3: Reconcile test helpers with real signatures (if needed)**

If Step 2 errored on signatures, open `live/cost_ledger.py`, `live/event_models.py`, and
`live/event_store.py`, and align `_apply_decision` / `_persist_*` exactly with the existing
FillModelB pipeline tests in `tests/test_fill_model_b.py` (the canonical reference). No
production code changes.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: previous total (238) + all new FillModelC tests PASS, 0 regressions.

- [ ] **Step 5: Stage for commit (Bryan commits)**

```bash
git add tests/test_fill_model_c.py
```
Suggested message: `test: FillModelC determinism, idempotency, replay fidelity + pipeline`.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Purpose / kill fill-fantasy → maker queue requires through-volume (Task 3). ✓
- Construction, `queue_ahead` mandatory no default → Task 1. ✓
- Interface (per-slice fold, stateless between calls) → Task 2 `evaluate`. ✓
- Anti-look-ahead enforced inside model → Task 2 window filter + `test_taker_anti_lookahead_*`. ✓
- Maker/taker routing by order_type (no marketability inference) → Task 4 (redesigned as removal). ✓
- MAKER rule (opposite flow, queue_ahead, passive-price) → Task 3. ✓
- TAKER rule (same-side walk, no queue) → Task 2. ✓
- `FillEvaluationResult` schema (incl. `active_since_agg_trade_id`, `prints_consumed`) → Task 1. ✓
- Invariants (determinism, idempotency, replay) → Task 5. ✓
- Passive-price invariant → `test_maker_buy_fills_at_limit_price_passive_invariant`. ✓
- `volume_through=0` and `queue_consumed=0` for taker → Task 2 asserts. ✓

**Out of scope (correctly absent):** HET sweep notebook, `FillModelCLive`, ETH dataset.

**Placeholder scan:** The Task-2 `evaluate` intentionally contains a temporary LIMIT
placeholder that is replaced in Task 3 then Task 4 — each replacement is shown in full. No
unresolved placeholders remain after Task 4.

**Type consistency:** `FillDecision` fields (`fill_price`, `fill_qty`, `fee_model`, `fee`,
`execution_id`, `event_ts_ms`) match `live/fill_model_a.py`. `FillEvaluationResult` fields are
identical across Task 1 definition and Task 2–5 usage. `evaluate` signature is stable from
Task 2 onward; only the LIMIT branch body changes.
```
