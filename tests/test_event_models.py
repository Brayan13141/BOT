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
