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
