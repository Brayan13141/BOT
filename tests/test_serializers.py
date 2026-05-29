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


def test_unknown_event_type_raises_value_error():
    """Deserializing an unknown event type must fail fast with informative message."""
    import json
    bad_json = json.dumps({"__event_type__": "UnknownEvent", "event_id": "x",
                           "schema_version": 1, "event_ts_ms": 1_000_000})
    with pytest.raises(ValueError, match="Unknown event type"):
        json_to_event(bad_json)
