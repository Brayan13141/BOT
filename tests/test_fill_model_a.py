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
    """Return a SUBMITTED (not ACKNOWLEDGED) MARKET order for immediate taker path evaluation."""
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
