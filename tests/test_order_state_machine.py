"""
Tests for OrderStateMachine — all valid and invalid transitions.

Doctrinal rules under test:
- event_ts_ms is the only clock used in logic
- recv_monotonic_ns stored but never affects any assertion on state
- ACKNOWLEDGED requires ACK event, never market data inference
- All transitions recorded in order.transitions
- Terminal states block further transitions
- Decimal for all financial quantities
- Idempotency: duplicate execution_ids raise DuplicateEventError
- Causal ordering: decreasing timestamps raise CausalityViolationError
"""
from decimal import Decimal

import pytest

from live.order_types import FeeModel, LiquidityRole, OrderSide, OrderState, OrderType
from live.order_state_machine import (
    CausalityViolationError,
    DuplicateEventError,
    InvalidTransitionError,
    OrderStateMachine,
    OutOfOrderEventError,
)


# ── create_order ─────────────────────────────────────────────────────────────

def test_create_order_initial_state_is_created(osm):
    order = osm.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.01"),
        limit_price=Decimal("65000"),
        event_ts_ms=1_000_000,
        order_id="o1",
    )
    assert order.state == OrderState.CREATED
    assert order.order_id == "o1"
    assert order.qty == Decimal("0.01")
    assert order.remaining_qty == Decimal("0.01")
    assert order.filled_qty == Decimal("0")
    assert order.created_ts_ms == 1_000_000


def test_create_order_records_created_transition(osm):
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    assert len(order.transitions) == 1
    t = order.transitions[0]
    assert t.from_state == OrderState.CREATED
    assert t.to_state == OrderState.CREATED
    assert t.trigger == "ORDER_CREATED"
    assert t.event_ts_ms == 1_000_000


def test_create_limit_order_without_price_raises(osm):
    with pytest.raises(ValueError, match="limit_price"):
        osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("0.01"), event_ts_ms=1_000_000,
        )


def test_create_market_order_with_price_raises(osm):
    with pytest.raises(ValueError, match="MARKET"):
        osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
        )


def test_create_order_zero_qty_raises(osm):
    with pytest.raises(ValueError, match="qty"):
        osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("0"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
        )


def test_create_order_negative_qty_raises(osm):
    with pytest.raises(ValueError, match="qty"):
        osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("-0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
        )


def test_create_order_auto_generates_order_id_if_none(osm):
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    assert order.order_id is not None
    assert len(order.order_id) > 0


# ── submit ────────────────────────────────────────────────────────────────────

def test_submit_transitions_to_submitted(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    assert limit_buy_order.state == OrderState.SUBMITTED
    assert limit_buy_order.submitted_ts_ms == 1_001_000


def test_submit_appends_transition(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.CREATED
    assert last.to_state == OrderState.SUBMITTED
    assert last.trigger == "SUBMIT"
    assert last.event_ts_ms == 1_001_000


def test_submit_stores_recv_monotonic_ns_without_affecting_state(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000, recv_monotonic_ns=5_000_000_000)
    assert limit_buy_order.state == OrderState.SUBMITTED
    assert limit_buy_order.submitted_ts_ms == 1_001_000
    last = limit_buy_order.transitions[-1]
    assert last.recv_monotonic_ns == 5_000_000_000


def test_cannot_submit_twice(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    with pytest.raises(InvalidTransitionError):
        osm.submit(limit_buy_order, event_ts_ms=1_002_000)


def test_submit_with_earlier_timestamp_raises_causality_error(osm, limit_buy_order):
    """Causal ordering enforced at runtime — not just in tests."""
    with pytest.raises(CausalityViolationError):
        osm.submit(limit_buy_order, event_ts_ms=999_999)  # before created_ts_ms=1_000_000


# ── acknowledge (ACKNOWLEDGED) ────────────────────────────────────────────────

def test_acknowledge_transitions_to_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    assert limit_buy_order.state == OrderState.ACKNOWLEDGED
    assert limit_buy_order.acknowledged_ts_ms == 1_050_000


def test_acknowledge_records_ack_latency_computable(osm, limit_buy_order):
    """ack_latency_ms = acknowledged_ts_ms - submitted_ts_ms."""
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_051_000)
    ack_latency_ms = limit_buy_order.acknowledged_ts_ms - limit_buy_order.submitted_ts_ms
    assert ack_latency_ms == 50_000


def test_acknowledge_appends_transition(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.SUBMITTED
    assert last.to_state == OrderState.ACKNOWLEDGED
    assert last.trigger == "ACK_RECEIVED"
    assert last.event_ts_ms == 1_050_000


def test_cannot_acknowledge_without_submitting_first(osm, limit_buy_order):
    """ACKNOWLEDGED requires prior SUBMITTED state."""
    with pytest.raises(OutOfOrderEventError):
        osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)


def test_acknowledge_stores_recv_monotonic_ns_without_affecting_state(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000, recv_monotonic_ns=9_999_999)
    assert limit_buy_order.state == OrderState.ACKNOWLEDGED
    assert limit_buy_order.acknowledged_ts_ms == 1_050_000
    last = limit_buy_order.transitions[-1]
    assert last.recv_monotonic_ns == 9_999_999


# ── reject ────────────────────────────────────────────────────────────────────

def test_reject_transitions_submitted_to_rejected(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.reject(limit_buy_order, event_ts_ms=1_002_000)
    assert limit_buy_order.state == OrderState.REJECTED
    assert limit_buy_order.terminal_ts_ms == 1_002_000


def test_reject_appends_transition(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.reject(limit_buy_order, event_ts_ms=1_002_000)
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.SUBMITTED
    assert last.to_state == OrderState.REJECTED
    assert last.trigger == "EXCHANGE_REJECT"


def test_cannot_reject_from_created(osm, limit_buy_order):
    with pytest.raises(InvalidTransitionError):
        osm.reject(limit_buy_order, event_ts_ms=1_002_000)


def test_cannot_reject_from_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    with pytest.raises(InvalidTransitionError):
        osm.reject(limit_buy_order, event_ts_ms=1_060_000)


# ── cancel ────────────────────────────────────────────────────────────────────

def test_user_cancel_from_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.cancel(limit_buy_order, event_ts_ms=1_200_000, trigger="USER_CANCEL")
    assert limit_buy_order.state == OrderState.CANCELLED
    assert limit_buy_order.terminal_ts_ms == 1_200_000


def test_system_cancel_from_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.cancel(limit_buy_order, event_ts_ms=1_200_000, trigger="SYSTEM_CANCEL")
    assert limit_buy_order.state == OrderState.CANCELLED


def test_cancel_records_correct_from_state(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.cancel(limit_buy_order, event_ts_ms=1_200_000, trigger="USER_CANCEL")
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.ACKNOWLEDGED
    assert last.to_state == OrderState.CANCELLED
    assert last.trigger == "USER_CANCEL"


def test_invalid_cancel_trigger_raises_value_error(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    with pytest.raises(ValueError, match="USER_CANCEL"):
        osm.cancel(limit_buy_order, event_ts_ms=1_200_000, trigger="FORCE_CANCEL")


def test_cannot_cancel_from_submitted(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    with pytest.raises(InvalidTransitionError):
        osm.cancel(limit_buy_order, event_ts_ms=1_100_000)


# ── expire ────────────────────────────────────────────────────────────────────

def test_expire_from_acknowledged(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.expire(limit_buy_order, event_ts_ms=1_500_000)
    assert limit_buy_order.state == OrderState.EXPIRED
    assert limit_buy_order.terminal_ts_ms == 1_500_000


def test_expire_appends_transition(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    osm.expire(limit_buy_order, event_ts_ms=1_500_000)
    last = limit_buy_order.transitions[-1]
    assert last.from_state == OrderState.ACKNOWLEDGED
    assert last.to_state == OrderState.EXPIRED
    assert last.trigger == "TTL_EXPIRED"


def test_cannot_expire_from_created(osm, limit_buy_order):
    with pytest.raises(InvalidTransitionError):
        osm.expire(limit_buy_order, event_ts_ms=1_500_000)


# ── fill — helpers ────────────────────────────────────────────────────────────

def _acknowledged_order(osm, qty: str = "0.03", price: str = "65000") -> object:
    """Create and advance an order to ACKNOWLEDGED state."""
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal(qty), limit_price=Decimal(price), event_ts_ms=1_000_000,
    )
    osm.submit(order, event_ts_ms=1_001_000)
    osm.acknowledge(order, event_ts_ms=1_050_000)
    return order


# ── fill — single partial fill ────────────────────────────────────────────────

def test_partial_fill_transitions_to_partially_filled(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
             event_ts_ms=1_100_000, fee_model=FeeModel.MAKER, execution_id="e1")
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.filled_qty == Decimal("0.01")
    assert order.remaining_qty == Decimal("0.02")


def test_partial_fill_sets_first_fill_ts(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
             event_ts_ms=1_100_000, fee_model=FeeModel.MAKER, execution_id="e1")
    assert order.first_fill_ts_ms == 1_100_000
    assert order.last_fill_ts_ms == 1_100_000


def test_partial_fill_appends_fill_event(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, fill_price=Decimal("65100"), fill_qty=Decimal("0.01"),
             event_ts_ms=1_100_000, fee_model=FeeModel.MAKER, execution_id="e1")
    assert len(order.fills) == 1
    fe = order.fills[0]
    assert fe.price == Decimal("65100")
    assert fe.qty == Decimal("0.01")
    assert fe.fee_model == FeeModel.MAKER
    assert fe.execution_id == "e1"
    assert fe.order_id == order.order_id


def test_partial_fill_sets_avg_fill_price(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
             event_ts_ms=1_100_000, fee_model=FeeModel.MAKER, execution_id="e1")
    assert order.avg_fill_price == Decimal("65000")


# ── fill — accumulation ────────────────────────────────────────────────────────

def test_two_partial_fills_accumulate_inventory(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65200"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.filled_qty == Decimal("0.02")
    assert order.remaining_qty == Decimal("0.01")
    assert order.first_fill_ts_ms == 1_100_000
    assert order.last_fill_ts_ms == 1_200_000
    assert len(order.fills) == 2


def test_two_partial_fills_weighted_avg_price(osm):
    """avg = (65000 * 0.01 + 65200 * 0.01) / 0.02 = 65100"""
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65200"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    assert order.avg_fill_price == Decimal("65100")


def test_three_partial_fills_then_full_fill(osm):
    order = _acknowledged_order(osm, qty="0.04")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65100"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    osm.fill(order, Decimal("65200"), Decimal("0.01"), 1_300_000, FeeModel.MAKER, "e3")
    assert order.state == OrderState.PARTIALLY_FILLED
    osm.fill(order, Decimal("65300"), Decimal("0.01"), 1_400_000, FeeModel.MAKER, "e4")
    assert order.state == OrderState.FILLED
    assert order.filled_qty == Decimal("0.04")
    assert order.remaining_qty == Decimal("0")
    assert order.terminal_ts_ms == 1_400_000
    assert len(order.fills) == 4


def test_full_fill_in_one_shot(osm):
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.TAKER, "e1")
    assert order.state == OrderState.FILLED
    assert order.remaining_qty == Decimal("0")
    last = order.transitions[-1]
    assert last.trigger == "FULL_FILL"


def test_fill_exceeding_remaining_qty_raises(osm):
    order = _acknowledged_order(osm, qty="0.01")
    with pytest.raises(ValueError, match="remaining_qty"):
        osm.fill(order, Decimal("65000"), Decimal("0.02"), 1_100_000, FeeModel.MAKER, "e1")


def test_fill_zero_qty_raises(osm):
    order = _acknowledged_order(osm)
    with pytest.raises(ValueError, match="fill_qty"):
        osm.fill(order, Decimal("65000"), Decimal("0"), 1_100_000, FeeModel.MAKER, "e1")


def test_fill_records_bid_ask_snapshot_in_transition(osm):
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1",
             bid_price=Decimal("64999"), ask_price=Decimal("65001"))
    last = order.transitions[-1]
    assert last.bid_price == Decimal("64999")
    assert last.ask_price == Decimal("65001")


def test_partially_filled_can_be_cancelled(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    assert order.state == OrderState.PARTIALLY_FILLED
    osm.cancel(order, event_ts_ms=1_200_000, trigger="USER_CANCEL")
    assert order.state == OrderState.CANCELLED
    last = order.transitions[-1]
    assert last.from_state == OrderState.PARTIALLY_FILLED


def test_partially_filled_can_expire(osm):
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.expire(order, event_ts_ms=1_500_000)
    assert order.state == OrderState.EXPIRED


# ── idempotency ────────────────────────────────────────────────────────────────

def test_duplicate_execution_id_raises_duplicate_event_error(osm):
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    with pytest.raises(DuplicateEventError):
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e1")


def test_duplicate_execution_id_does_not_mutate_state(osm):
    """State is unchanged when DuplicateEventError is raised."""
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    state_before = order.state
    filled_before = order.filled_qty
    try:
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e1")
    except DuplicateEventError:
        pass
    assert order.state == state_before
    assert order.filled_qty == filled_before


def test_different_execution_ids_both_processed(osm):
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65100"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    assert "e1" in order.processed_execution_ids
    assert "e2" in order.processed_execution_ids
    assert order.state == OrderState.FILLED


# ── Immediate taker fill (SUBMITTED → FILLED) ─────────────────────────────────

def test_immediate_taker_full_fill_from_submitted(osm, limit_buy_order):
    """
    Limit order crosses spread on submission — executes immediately as taker.
    No ACKNOWLEDGED state: SUBMITTED → FILLED in one fill event.
    """
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.fill(
        limit_buy_order,
        fill_price=Decimal("65000"),
        fill_qty=Decimal("0.01"),
        event_ts_ms=1_001_050,
        fee_model=FeeModel.TAKER,
        execution_id="e-imm-1",
    )
    assert limit_buy_order.state == OrderState.FILLED
    assert limit_buy_order.remaining_qty == Decimal("0")
    assert limit_buy_order.terminal_ts_ms == 1_001_050
    last = limit_buy_order.transitions[-1]
    assert last.trigger == "IMMEDIATE_TAKER_FILL"
    assert last.from_state == OrderState.SUBMITTED
    assert last.to_state == OrderState.FILLED


def test_immediate_taker_full_fill_liquidity_role_is_taker(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.fill(limit_buy_order, Decimal("65000"), Decimal("0.01"),
             1_001_050, FeeModel.TAKER, "e-imm-1")
    assert limit_buy_order.liquidity_role.value == "TAKER"


def test_immediate_taker_partial_fill_from_submitted(osm):
    """
    Immediate partial: SUBMITTED → PARTIALLY_FILLED.
    Remaining qty will rest in the book (ACKNOWLEDGED on next ACK).
    """
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.03"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    osm.submit(order, event_ts_ms=1_001_000)
    osm.fill(order, Decimal("65000"), Decimal("0.01"),
             1_001_050, FeeModel.TAKER, "e-imm-1")
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.remaining_qty == Decimal("0.02")
    last = order.transitions[-1]
    assert last.trigger == "IMMEDIATE_TAKER_PARTIAL_FILL"


# ── Mixed liquidity role (reprice path) ────────────────────────────────────────

def test_maker_fill_then_taker_fill_produces_mixed_role(osm):
    """
    Order rests as maker, then repriced → next fill is taker.
    liquidity_role = MIXED.
    """
    order = _acknowledged_order(osm, qty="0.02")
    # First fill: resting maker fill
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    assert order.liquidity_role.value == "MAKER"
    # Reprice converts remaining → taker fill
    osm.fill(order, Decimal("65050"), Decimal("0.01"), 1_200_000, FeeModel.TAKER, "e2")
    assert order.state == OrderState.FILLED
    assert order.liquidity_role.value == "MIXED"


def test_all_taker_fills_liquidity_role(osm):
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.TAKER, "e1")
    osm.fill(order, Decimal("65100"), Decimal("0.01"), 1_200_000, FeeModel.TAKER, "e2")
    assert order.liquidity_role.value == "TAKER"


def test_all_maker_fills_liquidity_role(osm):
    order = _acknowledged_order(osm, qty="0.02")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65100"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    assert order.liquidity_role.value == "MAKER"


def test_liquidity_role_not_affected_by_state_only_by_fills(osm):
    """
    liquidity_role is a property of fills, not of OrderState.
    A FILLED order with all MAKER fills still shows MAKER.
    """
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    assert order.state == OrderState.FILLED
    assert order.liquidity_role.value == "MAKER"


# ── Terminal state blocking ───────────────────────────────────────────────────

@pytest.mark.parametrize("terminal_state,setup_fn", [
    ("FILLED", lambda osm, o: [
        osm.submit(o, 1_001_000),
        osm.acknowledge(o, 1_050_000),
        osm.fill(o, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1"),
    ]),
    ("CANCELLED", lambda osm, o: [
        osm.submit(o, 1_001_000),
        osm.acknowledge(o, 1_050_000),
        osm.cancel(o, 1_200_000),
    ]),
    ("EXPIRED", lambda osm, o: [
        osm.submit(o, 1_001_000),
        osm.acknowledge(o, 1_050_000),
        osm.expire(o, 1_500_000),
    ]),
    ("REJECTED", lambda osm, o: [
        osm.submit(o, 1_001_000),
        osm.reject(o, 1_002_000),
    ]),
])
def test_terminal_state_blocks_submit(osm, terminal_state, setup_fn):
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    setup_fn(osm, order)
    assert order.state.value == terminal_state
    with pytest.raises(InvalidTransitionError, match="terminal"):
        osm.submit(order, event_ts_ms=2_000_000)


def test_filled_order_blocks_cancel(osm):
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    with pytest.raises(InvalidTransitionError, match="terminal"):
        osm.cancel(order, 1_200_000)


def test_cancelled_order_blocks_fill(osm):
    order = _acknowledged_order(osm)
    osm.cancel(order, 1_200_000)
    with pytest.raises(InvalidTransitionError, match="terminal"):
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_300_000, FeeModel.MAKER, "e1")


# ── CausalityViolationError ───────────────────────────────────────────────────

def test_causality_violation_on_submit(osm, limit_buy_order):
    with pytest.raises(CausalityViolationError):
        osm.submit(limit_buy_order, event_ts_ms=999_999)


def test_causality_violation_on_acknowledge(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    with pytest.raises(CausalityViolationError):
        osm.acknowledge(limit_buy_order, event_ts_ms=1_000_500)


def test_causality_violation_on_fill(osm):
    order = _acknowledged_order(osm, qty="0.01")
    with pytest.raises(CausalityViolationError):
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_000_000,
                 FeeModel.MAKER, "e1")  # 1_000_000 < acknowledged_ts_ms=1_050_000


def test_equal_timestamps_do_not_raise_causality_error(osm, limit_buy_order):
    """Monotonic means >=, not strictly >."""
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    osm.acknowledge(limit_buy_order, event_ts_ms=1_001_000)  # same ts is valid
    assert limit_buy_order.state == OrderState.ACKNOWLEDGED


# ── OutOfOrderEventError ──────────────────────────────────────────────────────

def test_out_of_order_fill_before_submit(osm):
    """Fill arrives for an order that hasn't been submitted yet (CREATED state)."""
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    # No submit — order is still in CREATED state
    with pytest.raises(OutOfOrderEventError):
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_001_100,
                 FeeModel.MAKER, "e1")


def test_out_of_order_acknowledge_before_submit(osm, limit_buy_order):
    """ACK arrives for an order that hasn't been submitted."""
    with pytest.raises(OutOfOrderEventError) as exc_info:
        osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000)
    assert exc_info.value.order_id == limit_buy_order.order_id
    assert exc_info.value.current_state == OrderState.CREATED
    assert exc_info.value.received_event_type == "ACK_RECEIVED"
    assert exc_info.value.exchange_event_ts == 1_050_000


def test_out_of_order_error_includes_diagnostics(osm, limit_buy_order):
    """OutOfOrderEventError captures exchange_event_ts and local_receive_ts."""
    with pytest.raises(OutOfOrderEventError) as exc_info:
        osm.acknowledge(limit_buy_order, event_ts_ms=1_050_000, recv_monotonic_ns=9_999_999)
    assert exc_info.value.local_receive_ts == 9_999_999


# ── Full lifecycle audit ──────────────────────────────────────────────────────

def test_full_lifecycle_transition_history_is_complete(osm):
    """
    Happy path: CREATED→SUBMITTED→ACKNOWLEDGED→PARTIALLY_FILLED→FILLED
    All transitions recorded in order.
    """
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.02"), limit_price=Decimal("65000"),
        event_ts_ms=1_000_000, order_id="audit-01",
    )
    osm.submit(order, event_ts_ms=1_001_000)
    osm.acknowledge(order, event_ts_ms=1_050_000)
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65050"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")

    assert order.state == OrderState.FILLED
    assert len(order.transitions) == 5  # CREATE + SUBMIT + ACK + PARTIAL + FULL
    assert len(order.fills) == 2

    triggers = [t.trigger for t in order.transitions]
    assert triggers == [
        "ORDER_CREATED", "SUBMIT", "ACK_RECEIVED", "PARTIAL_FILL", "FULL_FILL"
    ]
    ts_list = [t.event_ts_ms for t in order.transitions]
    assert ts_list == [1_000_000, 1_001_000, 1_050_000, 1_100_000, 1_200_000]


def test_transition_timestamps_are_monotonically_non_decreasing(osm):
    order = osm.create_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
    )
    osm.submit(order, event_ts_ms=1_001_000)
    osm.acknowledge(order, event_ts_ms=1_050_000)
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")

    ts_values = [t.event_ts_ms for t in order.transitions]
    for i in range(1, len(ts_values)):
        assert ts_values[i] >= ts_values[i - 1]


def test_recv_monotonic_ns_never_affects_state_outcome(osm):
    """
    Two identical orders with different recv_monotonic_ns values must
    end in identical states.
    """
    def run_lifecycle(recv_ns: int) -> object:
        order = osm.create_order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=Decimal("0.01"), limit_price=Decimal("65000"), event_ts_ms=1_000_000,
        )
        osm.submit(order, event_ts_ms=1_001_000, recv_monotonic_ns=recv_ns)
        osm.acknowledge(order, event_ts_ms=1_050_000, recv_monotonic_ns=recv_ns + 1000)
        osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER,
                 "e1", recv_monotonic_ns=recv_ns + 2000)
        return order

    order_a = run_lifecycle(recv_ns=100_000_000)
    order_b = run_lifecycle(recv_ns=999_999_999)

    assert order_a.state == order_b.state == OrderState.FILLED
    assert order_a.filled_qty == order_b.filled_qty
    assert order_a.avg_fill_price == order_b.avg_fill_price
    assert order_a.terminal_ts_ms == order_b.terminal_ts_ms


def test_all_transition_records_are_immutable_frozen(osm, limit_buy_order):
    osm.submit(limit_buy_order, event_ts_ms=1_001_000)
    t = limit_buy_order.transitions[-1]
    with pytest.raises((AttributeError, TypeError)):
        t.trigger = "TAMPERED"  # type: ignore[misc]


def test_all_fill_records_are_immutable_frozen(osm):
    order = _acknowledged_order(osm, qty="0.01")
    osm.fill(order, Decimal("65000"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    fe = order.fills[0]
    with pytest.raises((AttributeError, TypeError)):
        fe.price = Decimal("0")  # type: ignore[misc]


def test_avg_fill_price_uses_decimal_arithmetic(osm):
    """Verify no float contamination in weighted avg calculation."""
    order = _acknowledged_order(osm, qty="0.03")
    osm.fill(order, Decimal("65000.1"), Decimal("0.01"), 1_100_000, FeeModel.MAKER, "e1")
    osm.fill(order, Decimal("65000.2"), Decimal("0.01"), 1_200_000, FeeModel.MAKER, "e2")
    osm.fill(order, Decimal("65000.3"), Decimal("0.01"), 1_300_000, FeeModel.MAKER, "e3")
    assert isinstance(order.avg_fill_price, Decimal)
    # Exact: (65000.1 + 65000.2 + 65000.3) / 3 = 65000.2
    assert order.avg_fill_price == Decimal("65000.2")
