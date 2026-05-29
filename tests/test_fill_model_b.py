"""Tests for FillModelB: volume-aware fill engine with configurable participation_rate."""
import uuid
from decimal import Decimal

import pytest

from live.fill_model_a import FillDecision, FillModelA, Tick
from live.fill_model_b import FillModelB
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
    fill_qty = min(1.0, 0.5) = 0.5  -> partial fill
    """
    order, _ = _make_market_order(side="BUY", qty="1.0")
    tick = _make_tick(ask="65005", volume="10.0")
    decision = FillModelB(participation_rate=Decimal("0.05")).evaluate(order, tick)

    assert decision is not None
    assert decision.fill_qty == Decimal("0.5")  # 10.0 * 0.05


def test_market_full_fill_when_remaining_qty_less_than_effective_liquidity():
    """
    order.qty = 0.01, tick.volume = 10.0, participation_rate = 0.01
    effective_liquidity = 0.1 > 0.01 -> order fills fully, capped by remaining_qty
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


# ── Task 3: LIMIT orders ──────────────────────────────────────────────────────

def test_limit_buy_no_fill_when_price_above_limit():
    order, _ = _make_limit_order(side="BUY", limit_price="65000")
    tick = _make_tick(price="65001", volume="10.0")  # price > limit -> no fill
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
    tick = _make_tick(price="65000", volume="10.0")  # price == limit -> fills
    assert FillModelB().evaluate(order, tick) is not None


def test_limit_sell_no_fill_when_price_below_limit():
    order, _ = _make_limit_order(side="SELL", limit_price="65000")
    tick = _make_tick(price="64999", volume="10.0")  # price < limit -> no fill
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
    fill_qty = min(1.0, 0.2) = 0.2  -> partial fill
    """
    order, _ = _make_limit_order(side="BUY", qty="1.0", limit_price="65000")
    tick = _make_tick(price="64990", volume="10.0")
    decision = FillModelB(participation_rate=Decimal("0.02")).evaluate(order, tick)

    assert decision is not None
    assert decision.fill_qty == Decimal("0.2")  # 10.0 * 0.02


def test_limit_fee_uses_maker_rebate_rate():
    """
    MAKER fee = fill_price x fill_qty x MAKER_REBATE_RATE (0.0001).
    Large volume -> effective_liq=100*0.01=1.0 >= order.qty=1.0 -> full fill.
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


# ── Task 4: FillModelA vs FillModelB comparison ───────────────────────────────

def test_fill_model_b_fills_less_than_fill_model_a_same_tick():
    """
    Core conservation property: FillModelB is strictly more conservative than
    FillModelA when effective_liquidity < order.remaining_qty.

    order.qty = 1.0 BTC, tick.volume = 10.0 BTC, participation_rate = 0.01:
        FillModelA: fill_qty = min(1.0, 10.0)        = 1.0  -> FULL FILL
        FillModelB: fill_qty = min(1.0, 10.0 x 0.01) = 0.1  -> PARTIAL FILL
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
    assert d_a.fill_qty  == Decimal("1.0")   # full -- tick.volume > order.qty
    assert d_b.fill_qty  == Decimal("0.1")   # 10.0 * 0.01 = 0.1 < 1.0 -> partial
    assert d_b.fill_qty  <  d_a.fill_qty
    assert d_a.fill_price == d_b.fill_price  # both at limit_price
    assert d_a.fee_model  == d_b.fee_model   # both MAKER


def test_fill_model_b_at_rate_one_equals_fill_model_a():
    """
    participation_rate=1.0 -> effective_liquidity = tick.volume -> identical to FillModelA.
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
    fill_qty capped by effective_liquidity -> PARTIALLY_FILLED.
    Pipeline: FillModelB -> OSM -> EventStore -> CostLedger.

    order.qty = 1.0, tick.volume = 10.0, participation_rate = 0.02
    effective_liq = 0.2 < 1.0 -> partial fill of 0.2 BTC
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
    assert entries[0].maker_rebate  > Decimal("0")   # MAKER fill -> rebate
    assert entries[0].taker_fee    == Decimal("0")


def test_fill_model_b_replay_fidelity(tmp_path):
    """
    Replay fidelity: live FillModelB run == ExecutionReplay reconstruction.

    order.qty = 0.02 BTC, tick.volume = 1.0, participation_rate = 0.5
    effective_liq = 0.5 > 0.02 -> full fill at limit_price

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

        # effective_liq = 1.0 * 0.5 = 0.5 > order.qty=0.02 -> full fill
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
