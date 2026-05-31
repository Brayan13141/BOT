"""Tests for the HET sweep harness (S28A). Small synthetic tapes; no dataset needed."""
from decimal import Decimal

import numpy as np
import pytest

from research.experiments.het_sweep import (
    Tape, build_window, load_canonical_tape, generate_passive_orders,
    GeneratedOrder, DELTA, ORDER_QTY,
    wilson_ci, conditional_median_time_to_fill, through_volume_percentiles,
    measure_through_volume, acceptance_verdict, PROBE_QUEUE_AHEAD,
    run_queue_ahead_sweep, summarize_sweep,
)
from live.fill_model_c import AggTrade, FillModelC
from live.order_state_machine import OrderStateMachine
from live.order_types import OrderSide, OrderType, OrderState


def _tape(rows):
    """rows = list of (ts, price, qty, side, agg_id)."""
    return Tape(
        ts=np.array([r[0] for r in rows], dtype=np.int64),
        agg_id=np.array([r[4] for r in rows], dtype=np.int64),
        price=[r[1] for r in rows],
        qty=[r[2] for r in rows],
        side=[r[3] for r in rows],
    )


def test_build_window_slices_after_reference_within_ttl():
    tape = _tape([
        (1000, "100.0", "1", "SELL", 10),
        (1500, "100.1", "1", "BUY", 11),   # reference (last ts <= 1500)
        (1800, "100.2", "1", "SELL", 12),  # in window
        (2400, "100.3", "1", "BUY", 13),   # in window (<= 1500+1000)
        (2600, "100.4", "1", "SELL", 14),  # out (> 2500)
    ])
    active_since, window = build_window(tape, ref_index=1, arrival_ts=1500, ttl_ms=1000)
    assert active_since == 11
    assert [t.agg_trade_id for t in window] == [12, 13]
    assert window[0].price == Decimal("100.2")
    assert window[0].side == OrderSide.SELL


def test_build_window_empty_when_no_prints_in_ttl():
    tape = _tape([(1000, "100", "1", "BUY", 10), (5000, "100", "1", "SELL", 11)])
    active_since, window = build_window(tape, ref_index=0, arrival_ts=1000, ttl_ms=500)
    assert active_since == 10
    assert window == []


def test_load_canonical_tape_reads_columns_and_checks_monotonic(tmp_path):
    p = tmp_path / "mini.csv"
    p.write_text(
        "timestamp_ms,price,qty,side,agg_trade_id\n"
        "1000,100.0,0.5,SELL,10\n"
        "1000,100.1,0.2,BUY,11\n"
        "1003,100.2,0.3,SELL,12\n",
        encoding="utf-8",
    )
    tape = load_canonical_tape(p)
    assert list(tape.ts) == [1000, 1000, 1003]
    assert list(tape.agg_id) == [10, 11, 12]
    assert tape.price == ["100.0", "100.1", "100.2"]
    assert tape.side == ["SELL", "BUY", "SELL"]


def test_load_canonical_tape_rejects_non_monotonic(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "timestamp_ms,price,qty,side,agg_trade_id\n"
        "1000,100.0,0.5,SELL,10\n"
        "1001,100.1,0.2,BUY,9\n",   # agg_id goes backward -> not canonical
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not strictly increasing"):
        load_canonical_tape(p)


def _big_tape(n=2000, step_ms=1000):
    rows = [(1000 + i * step_ms, "100.0", "1", "BUY" if i % 2 else "SELL", 10 + i) for i in range(n)]
    return _tape(rows)


def test_generator_is_deterministic_under_seed():
    tape = _big_tape()
    a = generate_passive_orders(tape, OrderSide.BUY, lam_per_sec=1/10, ttl_ms=5000, seed=[28, 0])
    b = generate_passive_orders(tape, OrderSide.BUY, lam_per_sec=1/10, ttl_ms=5000, seed=[28, 0])
    assert [g.arrival_ts for g in a] == [g.arrival_ts for g in b]
    assert [str(g.order.limit_price) for g in a] == [str(g.order.limit_price) for g in b]


def test_generator_places_buy_below_and_sell_above_reference():
    tape = _big_tape()
    buys = generate_passive_orders(tape, OrderSide.BUY, lam_per_sec=1/10, ttl_ms=5000, seed=[28, 0])
    sells = generate_passive_orders(tape, OrderSide.SELL, lam_per_sec=1/10, ttl_ms=5000, seed=[28, 1])
    g = buys[0]
    ref = Decimal(tape.price[g.ref_index])
    assert g.order.limit_price == ref - DELTA
    assert g.order.side == OrderSide.BUY
    assert g.order.order_type == OrderType.LIMIT
    assert g.order.state == OrderState.ACKNOWLEDGED
    assert g.order.qty == ORDER_QTY
    s = sells[0]
    assert s.order.limit_price == Decimal(tape.price[s.ref_index]) + DELTA


def test_generator_respects_edge_censoring():
    tape = _big_tape()
    ttl = 5000
    orders = generate_passive_orders(tape, OrderSide.BUY, lam_per_sec=1/2, ttl_ms=ttl, seed=[28, 0])
    t_end = int(tape.ts[-1]) - ttl
    assert all(g.arrival_ts <= t_end for g in orders)
    assert len(orders) > 0


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(successes=50, n=100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61   # ~ [0.404, 0.596]
    assert wilson_ci(0, 0) == (0.0, 0.0)            # empty guard


def test_conditional_median_ignores_unfilled():
    rows = [
        {"filled": True,  "time_to_fill_ms": 100},
        {"filled": True,  "time_to_fill_ms": 300},
        {"filled": False, "time_to_fill_ms": None},   # censored -> excluded
    ]
    assert conditional_median_time_to_fill(rows) == 200.0


def test_conditional_median_none_when_no_fills():
    rows = [{"filled": False, "time_to_fill_ms": None}]
    assert conditional_median_time_to_fill(rows) is None


def test_through_volume_percentiles():
    vols = [Decimal(str(v)) for v in range(1, 101)]   # 1..100
    pct = through_volume_percentiles(vols, [50, 90, 95, 99])
    assert pct[50] == pytest.approx(50.5, abs=1.0)
    assert pct[99] == pytest.approx(99.0, abs=1.5)


def generate_one_buy(tape, ref_index, arrival_ts):
    """Build a single ACKNOWLEDGED passive BUY GeneratedOrder for probe/sweep tests."""
    ref_price = Decimal(tape.price[ref_index])
    osm = OrderStateMachine()
    order = osm.create_order(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                             qty=ORDER_QTY, limit_price=ref_price - DELTA, event_ts_ms=arrival_ts)
    osm.submit(order, event_ts_ms=arrival_ts + 1)
    osm.acknowledge(order, event_ts_ms=arrival_ts + 2)
    return GeneratedOrder(order=order, arrival_ts=arrival_ts,
                          active_since=int(tape.agg_id[ref_index]), ref_index=ref_index)


def test_probe_accumulates_full_through_volume_without_filling():
    # Resting BUY @ 99.5 (ref 100.0 - DELTA 0.5). Two SELL prints through it.
    tape = _tape([
        (1000, "100.0", "1", "BUY",  10),   # reference
        (1500, "99.4",  "2", "SELL", 11),   # through 99.5
        (2000, "99.3",  "3", "SELL", 12),   # through 99.5
    ])
    g = generate_one_buy(tape, ref_index=0, arrival_ts=1000)
    rows = measure_through_volume(tape, [g], ttl_ms=5000)
    assert rows[0]["through_volume"] == Decimal("5")   # 2 + 3, NOT capped by fill
    assert rows[0]["reached"] is True


def test_probe_reach_false_when_no_through_prints():
    tape = _tape([
        (1000, "100.0", "1", "BUY",  10),
        (1500, "100.9", "9", "SELL", 11),   # 100.9 > 99.5 -> not through a BUY @ 99.5
    ])
    g = generate_one_buy(tape, ref_index=0, arrival_ts=1000)
    rows = measure_through_volume(tape, [g], ttl_ms=5000)
    assert rows[0]["through_volume"] == Decimal("0")
    assert rows[0]["reached"] is False


def test_probe_queue_ahead_never_fills_even_with_enormous_volume():
    """Verified property (not intuition): PROBE_QUEUE_AHEAD never fills, regardless of volume."""
    order, _osm = (lambda g: (g.order, None))(generate_one_buy(
        _tape([(1000, "100.0", "1", "BUY", 10)]), ref_index=0, arrival_ts=1000))
    huge = AggTrade(timestamp_ms=1500, price=Decimal("99.0"), qty=Decimal("1000000000"),
                    side=OrderSide.SELL, agg_trade_id=11)
    result = FillModelC(PROBE_QUEUE_AHEAD).evaluate(order, [huge], active_since_agg_trade_id=10)
    assert result.fully_filled is False
    assert result.filled_qty == Decimal("0")
    assert result.fills == []
    assert result.volume_through == Decimal("1000000000")   # accumulated, never consumed by a fill


def test_acceptance_verdict_bands():
    assert acceptance_verdict(v50=0.1, g_max=5.0) == "WIDEN_TTL_OR_REDUCE_GRID"   # 0.1 < 2.5
    assert acceptance_verdict(v50=80.0, g_max=5.0) == "REDUCE_TTL_OR_WIDEN_GRID"  # 80 > 50
    assert acceptance_verdict(v50=5.0, g_max=5.0) == "ACCEPT"                     # 2.5 <= 5 <= 50


def test_sweep_window_built_once_swept_across_grid():
    # BUY @ 99.5. One SELL of 3.0 through it. queue_ahead 0 fills; 5.0 does not.
    tape = _tape([
        (1000, "100.0", "1", "BUY",  10),
        (1500, "99.4",  "3", "SELL", 11),
    ])
    g = generate_one_buy(tape, ref_index=0, arrival_ts=1000)
    rows = run_queue_ahead_sweep(tape, [g], ttl_ms=5000,
                                 grid=[Decimal("0"), Decimal("5.0")])
    by_q = {r["queue_ahead"]: r for r in rows}
    assert by_q[Decimal("0")]["filled"] is True
    assert by_q[Decimal("0")]["fill_price"] == Decimal("99.5")     # passive-price invariant
    assert by_q[Decimal("0")]["time_to_fill_ms"] == 500            # 1500 - 1000
    assert by_q[Decimal("5.0")]["filled"] is False                 # queue 5 > through 3
    assert by_q[Decimal("5.0")]["queue_consumed"] == Decimal("3")


def test_summarize_sweep_reports_fill_rate_and_ci():
    rows = [
        {"queue_ahead": Decimal("0"), "filled": True,  "time_to_fill_ms": 100, "reached": True,
         "queue_consumed": Decimal("0"), "queue_remaining": Decimal("0"), "through_volume": Decimal("1")},
        {"queue_ahead": Decimal("0"), "filled": False, "time_to_fill_ms": None, "reached": True,
         "queue_consumed": Decimal("0"), "queue_remaining": Decimal("0"), "through_volume": Decimal("0")},
    ]
    summ = summarize_sweep(rows, grid=[Decimal("0")])
    s0 = summ[Decimal("0")]
    assert s0["n"] == 2
    assert s0["fill_rate"] == 0.5
    assert s0["reach_rate"] == 1.0
    assert s0["median_time_to_fill_ms"] == 100.0
    assert 0.0 <= s0["fill_rate_ci"][0] <= s0["fill_rate_ci"][1] <= 1.0
