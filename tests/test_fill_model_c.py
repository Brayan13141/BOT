"""Tests for FillModelC: queue-aware, tape-driven fill engine over aggTrades."""
from dataclasses import FrozenInstanceError
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
    with pytest.raises(FrozenInstanceError):
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
