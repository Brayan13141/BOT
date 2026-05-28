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
