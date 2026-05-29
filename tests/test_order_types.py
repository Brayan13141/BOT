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
