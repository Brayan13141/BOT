"""
Tests for ExecutionReplay — deterministic Order reconstruction from domain events.

Key invariant under test:
    Replaying the same sequence of domain events through a fresh OSM always
    produces the same Order state (same state, filled_qty, avg_fill_price,
    transition triggers, fill execution_ids).

This is the central guarantee of the execution event sourcing system.
"""
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from live.event_models import (
    OrderAcknowledged,
    OrderCancelled,
    OrderExpired,
    OrderFillReceived,
    OrderRejected,
    OrderSubmitted,
)
from live.event_store import EventStore
from live.execution_replay import ExecutionReplay
from live.order_types import OrderState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(logs_dir=tmp_path)


@pytest.fixture
def replay(store: EventStore) -> ExecutionReplay:
    return ExecutionReplay(event_store=store)


def _evt(ts: int = 1_000_000) -> dict:
    return {"event_id": str(uuid.uuid4()), "schema_version": 1, "event_ts_ms": ts}


# ── Helper to store a complete lifecycle ──────────────────────────────────────

def _store_full_lifecycle(store: EventStore, order_id: str = "replay-01"):
    """Store: submitted → acknowledged → partial fill → full fill."""
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=order_id,
        order_id=order_id, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.02"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(1_050_000), correlation_id=order_id,
        order_id=order_id, submitted_ts_ms=1_000_000,
    ))
    store.append_domain_event(OrderFillReceived(
        **_evt(1_100_000), correlation_id=order_id,
        order_id=order_id, fill_id="fill-01",
        price=Decimal("65100"), qty=Decimal("0.01"),
        fee=Decimal("0.1302"), fee_asset="USDT", fee_model="MAKER",
    ))
    store.append_domain_event(OrderFillReceived(
        **_evt(1_200_000), correlation_id=order_id,
        order_id=order_id, fill_id="fill-02",
        price=Decimal("65200"), qty=Decimal("0.01"),
        fee=Decimal("0.1304"), fee_asset="USDT", fee_model="MAKER",
    ))


# ── Basic replay paths ────────────────────────────────────────────────────────

def test_replay_submitted_only(store: EventStore, replay: ExecutionReplay):
    """Replay an order that only reached SUBMITTED state."""
    oid = "r-submit"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.SUBMITTED
    assert order.order_id == oid
    assert order.qty == Decimal("0.01")


def test_replay_acknowledged(store: EventStore, replay: ExecutionReplay):
    oid = "r-ack"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(1_050_000), correlation_id=oid,
        order_id=oid, submitted_ts_ms=1_000_000,
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.ACKNOWLEDGED
    assert order.acknowledged_ts_ms == 1_050_000


def test_replay_full_fill(store: EventStore, replay: ExecutionReplay):
    oid = "r-filled"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    assert order.state == OrderState.FILLED
    assert order.filled_qty == Decimal("0.02")
    assert order.remaining_qty == Decimal("0")


def test_replay_partial_fill_then_cancel(store: EventStore, replay: ExecutionReplay):
    oid = "r-partial-cancel"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.03"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(1_050_000), correlation_id=oid,
        order_id=oid, submitted_ts_ms=1_000_000,
    ))
    store.append_domain_event(OrderFillReceived(
        **_evt(1_100_000), correlation_id=oid,
        order_id=oid, fill_id="f1",
        price=Decimal("65000"), qty=Decimal("0.01"),
        fee=Decimal("0.13"), fee_asset="USDT", fee_model="MAKER",
    ))
    store.append_domain_event(OrderCancelled(
        **_evt(1_200_000), correlation_id=oid,
        order_id=oid, trigger="USER_CANCEL",
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.CANCELLED
    assert order.filled_qty == Decimal("0.01")


def test_replay_expired(store: EventStore, replay: ExecutionReplay):
    oid = "r-expired"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(1_050_000), correlation_id=oid,
        order_id=oid, submitted_ts_ms=1_000_000,
    ))
    store.append_domain_event(OrderExpired(
        **_evt(1_500_000), correlation_id=oid, order_id=oid,
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.EXPIRED


def test_replay_rejected(store: EventStore, replay: ExecutionReplay):
    oid = "r-rejected"
    store.append_domain_event(OrderSubmitted(
        **_evt(1_000_000), correlation_id=oid,
        order_id=oid, symbol="BTCUSDT", side="BUY",
        order_type="LIMIT", qty=Decimal("0.01"),
        created_ts_ms=999_000, limit_price=Decimal("65000"),
    ))
    store.append_domain_event(OrderRejected(
        **_evt(1_001_000), correlation_id=oid,
        order_id=oid, reason="INSUFFICIENT_MARGIN",
    ))
    order = replay.replay_order(oid)
    assert order.state == OrderState.REJECTED


def test_replay_nonexistent_order_raises(replay: ExecutionReplay):
    with pytest.raises(ValueError, match="No domain events"):
        replay.replay_order("ghost-order-id")


# ── Deterministic replay — core guarantee ─────────────────────────────────────

def test_replay_avg_fill_price_matches_live(store: EventStore, replay: ExecutionReplay):
    """Weighted avg fill price reconstructed exactly with Decimal arithmetic."""
    oid = "r-avg"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    # avg = (65100 * 0.01 + 65200 * 0.01) / 0.02 = 65150
    expected_avg = (Decimal("65100") * Decimal("0.01") +
                    Decimal("65200") * Decimal("0.01")) / Decimal("0.02")
    assert order.avg_fill_price == expected_avg
    assert isinstance(order.avg_fill_price, Decimal)


def test_replay_fill_count_matches(store: EventStore, replay: ExecutionReplay):
    oid = "r-fills"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    assert len(order.fills) == 2


def test_replay_fill_execution_ids_preserved(store: EventStore, replay: ExecutionReplay):
    oid = "r-exec-ids"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    execution_ids = {f.execution_id for f in order.fills}
    assert execution_ids == {"fill-01", "fill-02"}


def test_replay_transition_triggers_match_live(store: EventStore, replay: ExecutionReplay):
    """Transition trigger sequence must be identical in live vs replay."""
    oid = "r-triggers"
    _store_full_lifecycle(store, oid)
    order = replay.replay_order(oid)
    triggers = [t.trigger for t in order.transitions]
    assert triggers == [
        "ORDER_CREATED", "SUBMIT", "ACK_RECEIVED", "PARTIAL_FILL", "FULL_FILL"
    ]


def test_replay_is_deterministic_across_two_runs(store: EventStore, replay: ExecutionReplay):
    """Two replay runs on the same events produce identical Order state."""
    oid = "r-det"
    _store_full_lifecycle(store, oid)
    order_a = replay.replay_order(oid)
    order_b = replay.replay_order(oid)
    assert order_a.state == order_b.state
    assert order_a.filled_qty == order_b.filled_qty
    assert order_a.avg_fill_price == order_b.avg_fill_price
    assert order_a.terminal_ts_ms == order_b.terminal_ts_ms
    assert [t.trigger for t in order_a.transitions] == [t.trigger for t in order_b.transitions]


def test_two_independent_orders_replay_correctly(store: EventStore, replay: ExecutionReplay):
    """Events for different orders don't interfere."""
    _store_full_lifecycle(store, "order-A")
    store.append_domain_event(OrderSubmitted(
        **_evt(2_000_000), correlation_id="order-B",
        order_id="order-B", symbol="ETHUSDT", side="SELL",
        order_type="LIMIT", qty=Decimal("1.0"),
        created_ts_ms=1_999_000, limit_price=Decimal("3000"),
    ))
    store.append_domain_event(OrderAcknowledged(
        **_evt(2_050_000), correlation_id="order-B",
        order_id="order-B", submitted_ts_ms=2_000_000,
    ))
    order_a = replay.replay_order("order-A")
    order_b = replay.replay_order("order-B")
    assert order_a.state == OrderState.FILLED
    assert order_b.state == OrderState.ACKNOWLEDGED
    assert order_a.symbol == "BTCUSDT"
    assert order_b.symbol == "ETHUSDT"


# ── Integration: live OSM calls → EventStore → replay → identical state ──────

def test_live_vs_replay_full_lifecycle(tmp_path: Path):
    """
    CRITICAL DETERMINISM TEST.

    Process a complete order lifecycle using OSM directly.
    Log all domain events to EventStore.
    Replay from EventStore.
    Assert final state is byte-for-byte identical to live processing.
    """
    from live.order_state_machine import OrderStateMachine
    from live.order_types import FeeModel, OrderSide, OrderState, OrderType

    store = EventStore(logs_dir=tmp_path / "live_test")
    osm = OrderStateMachine()

    order_id = "integration-001"
    created_ts = 1_000_000
    submit_ts  = 1_001_000
    ack_ts     = 1_050_000
    fill1_ts   = 1_100_000
    fill2_ts   = 1_200_000

    # ── Live OSM processing ──────────────────────────────────────────────────
    live_order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.02"),
        limit_price=Decimal("65000"),
        event_ts_ms=created_ts,
        order_id=order_id,
    )

    # Log OrderSubmitted domain event
    store.append_domain_event(OrderSubmitted(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=submit_ts,
        correlation_id=order_id, order_id=order_id,
        symbol="BTCUSDT", side="BUY", order_type="LIMIT",
        qty=Decimal("0.02"), created_ts_ms=created_ts,
        limit_price=Decimal("65000"),
    ))
    osm.submit(live_order, event_ts_ms=submit_ts)

    # Log OrderAcknowledged
    store.append_domain_event(OrderAcknowledged(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=ack_ts,
        correlation_id=order_id, order_id=order_id, submitted_ts_ms=submit_ts,
    ))
    osm.acknowledge(live_order, event_ts_ms=ack_ts)

    # Log first partial fill
    store.append_domain_event(OrderFillReceived(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=fill1_ts,
        correlation_id=order_id, order_id=order_id, fill_id="exec-001",
        price=Decimal("65100"), qty=Decimal("0.01"),
        fee=Decimal("0.1302"), fee_asset="USDT", fee_model="MAKER",
    ))
    osm.fill(live_order, fill_price=Decimal("65100"), fill_qty=Decimal("0.01"),
             event_ts_ms=fill1_ts, fee_model=FeeModel.MAKER, execution_id="exec-001")

    # Log second (final) fill
    store.append_domain_event(OrderFillReceived(
        event_id=str(uuid.uuid4()), schema_version=1, event_ts_ms=fill2_ts,
        correlation_id=order_id, order_id=order_id, fill_id="exec-002",
        price=Decimal("65200"), qty=Decimal("0.01"),
        fee=Decimal("0.1304"), fee_asset="USDT", fee_model="MAKER",
    ))
    osm.fill(live_order, fill_price=Decimal("65200"), fill_qty=Decimal("0.01"),
             event_ts_ms=fill2_ts, fee_model=FeeModel.MAKER, execution_id="exec-002")

    # ── Replay ───────────────────────────────────────────────────────────────
    replay = ExecutionReplay(event_store=store)
    replayed_order = replay.replay_order(order_id)

    # ── Assert identical final state ─────────────────────────────────────────
    assert live_order.state == replayed_order.state == OrderState.FILLED
    assert live_order.filled_qty == replayed_order.filled_qty == Decimal("0.02")
    assert live_order.remaining_qty == replayed_order.remaining_qty == Decimal("0")

    # Decimal-exact average price
    expected_avg = (Decimal("65100") * Decimal("0.01") +
                    Decimal("65200") * Decimal("0.01")) / Decimal("0.02")
    assert live_order.avg_fill_price == replayed_order.avg_fill_price == expected_avg

    # Timestamps
    assert live_order.terminal_ts_ms == replayed_order.terminal_ts_ms == fill2_ts

    # Transition sequence
    live_triggers = [t.trigger for t in live_order.transitions]
    replay_triggers = [t.trigger for t in replayed_order.transitions]
    assert live_triggers == replay_triggers == [
        "ORDER_CREATED", "SUBMIT", "ACK_RECEIVED", "PARTIAL_FILL", "FULL_FILL"
    ]

    # Fill execution IDs
    live_exec_ids = {f.execution_id for f in live_order.fills}
    replay_exec_ids = {f.execution_id for f in replayed_order.fills}
    assert live_exec_ids == replay_exec_ids == {"exec-001", "exec-002"}

    # Liquidity role
    assert live_order.liquidity_role == replayed_order.liquidity_role

    store.close()
