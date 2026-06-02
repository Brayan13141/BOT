"""
HET sweep harness (S28A). Consumes the frozen FillModelC over the canonical
BTCUSDT aggTrades tape. All deterministic logic lives here; the notebooks are thin.

Spec: OBSIDIAN/docs/superpowers/specs/2026-05-30-het-sweep-s28a-design.md
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

from live.fill_model_c import AggTrade, FillModelC
from live.order_state_machine import OrderStateMachine
from live.order_types import Order, OrderSide, OrderType

TICK            = Decimal("0.10")                 # BTCUSDT perp price tick
DELTA_TICKS     = 5
DELTA           = TICK * DELTA_TICKS              # 0.50 USDT passive offset (P1)
ORDER_QTY       = Decimal("0.01")                 # micro-scalp clip
SEED_EXPERIMENT = 28
LAMBDA_PER_SEC  = 1.0 / 60.0                      # Poisson rate: 1 order/min per side (SAMPLING param)
PROBE_QUEUE_AHEAD = Decimal("1000000000000")      # 1e12 BTC: unreachable -> never fills -> full through-volume
# --- S28A-0 calibration inputs (a-priori candidate grid evaluated against through-volume) ---
QUEUE_AHEAD_GRID  = [Decimal("0"), Decimal("0.1"), Decimal("0.5"), Decimal("1.0"), Decimal("5.0")]  # candidate; S28A-0 found it mis-scaled at TTL>=300s (Caso B). Consumed only by the (frozen) calibration notebook.
TTL_CANDIDATES_MS = [60_000, 300_000, 900_000]    # S28A-0 candidates
# --- S28A sweep config (FROZEN by S28A-0 verdict, 2026-05-31) ---
TTL_FROZEN_MS     = 60_000                         # 60s straddles median through-volume (BTC P50=4.879); 300s/900s saturate the grid (Caso B). recommended_ttl=300s from the calib notebook is an artifact (min|t-300000|), ignored.
QUEUE_AHEAD_GRID_FROZEN = [Decimal("0"), Decimal("0.5"), Decimal("2"), Decimal("5"), Decimal("15"), Decimal("50")]  # spans P0->P90 @ 60s: maps decay (q->5~=median) AND plateau (q->50~=P90=55.860)
SIDE_SEED = {OrderSide.BUY: 0, OrderSide.SELL: 1}  # independent streams under SEED_EXPERIMENT
# --- S28B depth sweep config (FROZEN by 2026-06-01-s28b-depth-sweep-design.md) ---
DELTA_GRID_TICKS  = [1, 5, 20]                     # swept placement depth; 5t = shared anchor with S28A
QUEUE_FIXED_S28B  = Decimal("5")                   # fixed queue_ahead (median through-volume scale; S28A anchor)


@dataclass
class Tape:
    """Canonical aggTrades in column arrays. ts non-decreasing, agg_id strictly increasing."""
    ts:     np.ndarray   # int64
    agg_id: np.ndarray   # int64
    price:  list[str]    # raw decimal strings (converted to Decimal per window)
    qty:    list[str]
    side:   list[str]    # "BUY" / "SELL"


def build_window(tape: Tape, ref_index: int, arrival_ts: int, ttl_ms: int) -> tuple[int, list[AggTrade]]:
    """
    Return (active_since_agg_trade_id, window). The window is every print strictly
    after ref_index with ts <= arrival_ts + ttl_ms, converted to AggTrade (Decimal).
    Precondition: tape is canonical (strictly increasing agg_id, non-decreasing ts).
    """
    active_since = int(tape.agg_id[ref_index])
    end = int(np.searchsorted(tape.ts, arrival_ts + ttl_ms, side="right"))
    window = [
        AggTrade(
            timestamp_ms=int(tape.ts[i]),
            price=Decimal(tape.price[i]),
            qty=Decimal(tape.qty[i]),
            side=OrderSide(tape.side[i]),
            agg_trade_id=int(tape.agg_id[i]),
        )
        for i in range(ref_index + 1, end)
    ]
    return active_since, window


def load_canonical_tape(path: str | Path) -> Tape:
    """Load a canonical aggTrades CSV. Asserts strictly increasing agg_trade_id (fail-fast)."""
    ts: list[int] = []
    agg: list[int] = []
    price: list[str] = []
    qty: list[str] = []
    side: list[str] = []
    last_id = -1
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = int(row["agg_trade_id"])
            if aid <= last_id:
                raise ValueError(f"agg_trade_id not strictly increasing ({aid} <= {last_id})")
            last_id = aid
            ts.append(int(row["timestamp_ms"]))
            agg.append(aid)
            price.append(row["price"])
            qty.append(row["qty"])
            side.append(row["side"])
    return Tape(ts=np.array(ts, dtype=np.int64), agg_id=np.array(agg, dtype=np.int64),
                price=price, qty=qty, side=side)


@dataclass
class GeneratedOrder:
    order:        Order
    arrival_ts:   int
    active_since: int
    ref_index:    int


def generate_passive_orders(tape: Tape, side: OrderSide, lam_per_sec: float,
                            ttl_ms: int, seed, delta: Decimal = DELTA) -> list[GeneratedOrder]:
    """
    Poisson arrivals (exponential inter-arrival), P1 passive placement off the tape
    reference (last print with ts <= arrival). Edge-censored: only arrivals with a full
    TTL of tape ahead. A fresh OrderStateMachine per order avoids cross-order timestamp
    coupling. `seed` is a sampling parameter (reproducibility only), not a phenomenon param.
    `delta` is the placement-depth phenomenon parameter (offset of L from the reference);
    it defaults to the module DELTA (non-breaking) and is swept by the S28B depth sweep.
    """
    rng = np.random.default_rng(seed)
    t0 = int(tape.ts[0])
    t_end = int(tape.ts[-1]) - ttl_ms
    out: list[GeneratedOrder] = []
    t = float(t0)
    while True:
        t += rng.exponential(1.0 / lam_per_sec) * 1000.0   # seconds -> ms
        arrival_ts = int(t)
        if arrival_ts > t_end:
            break
        ref_index = int(np.searchsorted(tape.ts, arrival_ts, side="right")) - 1
        if ref_index < 0:
            continue
        ref_price = Decimal(tape.price[ref_index])
        limit_price = ref_price - delta if side == OrderSide.BUY else ref_price + delta
        osm = OrderStateMachine()
        order = osm.create_order(symbol="BTCUSDT", side=side, order_type=OrderType.LIMIT,
                                 qty=ORDER_QTY, limit_price=limit_price, event_ts_ms=arrival_ts)
        osm.submit(order, event_ts_ms=arrival_ts + 1)
        osm.acknowledge(order, event_ts_ms=arrival_ts + 2)
        out.append(GeneratedOrder(order=order, arrival_ts=arrival_ts,
                                  active_since=int(tape.agg_id[ref_index]), ref_index=ref_index))
    return out


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. (0.0, 0.0) for n == 0."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - margin, center + margin)


def conditional_median_time_to_fill(rows: list[dict]) -> float | None:
    """Median time_to_fill_ms over FILLED rows only (right-censored: unfilled excluded)."""
    vals = sorted(r["time_to_fill_ms"] for r in rows if r["filled"])
    if not vals:
        return None
    return float(np.median(vals))


def through_volume_percentiles(volumes: list[Decimal], pcts: list[int]) -> dict[int, float]:
    """Percentiles of through-volume (Decimal -> float for numpy). Zeros for empty input."""
    if not volumes:
        return {p: 0.0 for p in pcts}
    arr = np.array([float(v) for v in volumes])
    return {p: float(np.percentile(arr, p)) for p in pcts}


def measure_through_volume(tape: Tape, gen_orders: list[GeneratedOrder], ttl_ms: int) -> list[dict]:
    """
    S28A-0: per order, run FillModelC with an unreachable queue_ahead so it NEVER fills
    and `volume_through` accumulates over the entire window. Returns one row per order.
    """
    model = FillModelC(PROBE_QUEUE_AHEAD)
    rows: list[dict] = []
    for g in gen_orders:
        _, window = build_window(tape, g.ref_index, g.arrival_ts, ttl_ms)
        r = model.evaluate(g.order, window, active_since_agg_trade_id=g.active_since)
        rows.append({
            "arrival_ts": g.arrival_ts,
            "ttl_ms": ttl_ms,
            "through_volume": r.volume_through,
            "through_rate": float(r.volume_through) / (ttl_ms / 1000.0),
            "reached": r.prints_consumed > 0,
        })
    return rows


def acceptance_verdict(v50: float, g_max: float) -> str:
    """A-priori grid-scale rule (spec). v50 = P50(through_volume @ TTL); g_max = max(grid)."""
    if v50 < 0.5 * g_max:
        return "WIDEN_TTL_OR_REDUCE_GRID"
    if v50 > 10.0 * g_max:
        return "REDUCE_TTL_OR_WIDEN_GRID"
    return "ACCEPT"


def run_queue_ahead_sweep(tape: Tape, gen_orders: list[GeneratedOrder], ttl_ms: int,
                          grid: list[Decimal]) -> list[dict]:
    """
    S28A: per order, build the window ONCE, then evaluate FillModelC for every queue_ahead
    in the grid (the window is independent of queue_ahead). FillModelC.evaluate() does not
    mutate the order, so reuse across the grid is safe. One row per (order, queue_ahead).
    """
    models = {q: FillModelC(q) for q in grid}
    rows: list[dict] = []
    for g in gen_orders:
        _, window = build_window(tape, g.ref_index, g.arrival_ts, ttl_ms)
        for q in grid:
            r = models[q].evaluate(g.order, window, active_since_agg_trade_id=g.active_since)
            ttf = (r.fills[-1].event_ts_ms - g.arrival_ts) if r.fully_filled else None
            rows.append({
                "queue_ahead": q,
                "arrival_ts": g.arrival_ts,
                "filled": r.fully_filled,
                "filled_qty": r.filled_qty,
                "fill_price": r.fills[-1].fill_price if r.fills else None,
                "time_to_fill_ms": ttf,
                "reached": r.prints_consumed > 0,
                "queue_consumed": r.queue_consumed,
                "queue_remaining": r.queue_remaining,
                "through_volume": r.volume_through,
            })
    return rows


def run_depth_sweep(tape: Tape, delta_ticks_list: list[int], q_fixed: Decimal, ttl_ms: int,
                    lam_per_sec: float = LAMBDA_PER_SEC, seed_experiment: int = SEED_EXPERIMENT,
                    side_seed: dict = SIDE_SEED) -> list[dict]:
    """
    S28B: queue_ahead is FIXED at q_fixed; the swept axis is placement depth Δ (in ticks).
    Δ changes the order's resting level (L = ref ∓ Δ·TICK), so orders and windows are
    regenerated per Δ. Arrivals are paired across Δ (same seed => same arrival times).
    One row per (Δ, order). FillModelC is consumed unmodified at the single fixed queue.
    """
    model = FillModelC(q_fixed)
    rows: list[dict] = []
    for dticks in delta_ticks_list:
        delta = TICK * dticks
        gen: list[GeneratedOrder] = []
        for side in (OrderSide.BUY, OrderSide.SELL):
            gen += generate_passive_orders(tape, side, lam_per_sec, ttl_ms,
                                           seed=[seed_experiment, side_seed[side]], delta=delta)
        for g in gen:
            _, window = build_window(tape, g.ref_index, g.arrival_ts, ttl_ms)
            r = model.evaluate(g.order, window, active_since_agg_trade_id=g.active_since)
            ttf = (r.fills[-1].event_ts_ms - g.arrival_ts) if r.fully_filled else None
            rows.append({
                "delta_ticks": dticks,
                "arrival_ts": g.arrival_ts,
                "filled": r.fully_filled,
                "time_to_fill_ms": ttf,
                "reached": r.prints_consumed > 0,
                "queue_consumed": r.queue_consumed,
                "queue_remaining": r.queue_remaining,
                "through_volume": r.volume_through,
            })
    return rows


def summarize_sweep(rows: list[dict], grid: list[Decimal]) -> dict[Decimal, dict]:
    """Aggregate per queue_ahead: fill_rate (+Wilson CI), reach_rate, conditional median TTF."""
    summary: dict[Decimal, dict] = {}
    for q in grid:
        qr = [r for r in rows if r["queue_ahead"] == q]
        n = len(qr)
        fills = sum(1 for r in qr if r["filled"])
        reaches = sum(1 for r in qr if r["reached"])
        summary[q] = {
            "queue_ahead": float(q),
            "n": n,
            "fill_rate": (fills / n) if n else 0.0,
            "fill_rate_ci": wilson_ci(fills, n),
            "reach_rate": (reaches / n) if n else 0.0,
            "median_time_to_fill_ms": conditional_median_time_to_fill(qr),
            "queue_consumed_mean": float(np.mean([float(r["queue_consumed"]) for r in qr])) if n else 0.0,
            "queue_remaining_mean": float(np.mean([float(r["queue_remaining"]) for r in qr])) if n else 0.0,
        }
    return summary


def summarize_depth_sweep(rows: list[dict], delta_ticks_list: list[int]) -> dict[int, dict]:
    """
    Aggregate per Δ, DECOMPOSED: reach(Δ) × cond_fill(Δ|reached) = fill_rate(Δ).
    cond_fill is the conditional proportion fills/reaches (Wilson CI over reaches);
    fill_rate is fills/n (Wilson CI over n). The identity fill_rate == reach × cond_fill
    holds by construction. median time-to-fill is conditional on fill (right-censored).
    """
    summary: dict[int, dict] = {}
    for d in delta_ticks_list:
        dr = [r for r in rows if r["delta_ticks"] == d]
        n = len(dr)
        reaches = sum(1 for r in dr if r["reached"])
        fills = sum(1 for r in dr if r["filled"])
        summary[d] = {
            "delta_ticks": d,
            "n": n,
            "reach_rate": (reaches / n) if n else 0.0,
            "reach_ci": wilson_ci(reaches, n),
            "cond_fill": (fills / reaches) if reaches else 0.0,
            "cond_fill_ci": wilson_ci(fills, reaches),
            "fill_rate": (fills / n) if n else 0.0,
            "fill_rate_ci": wilson_ci(fills, n),
            "median_time_to_fill_ms": conditional_median_time_to_fill(dr),
        }
    return summary


def reach_monotonicity_ok(summary: dict[int, dict], delta_ticks_list: list[int]) -> bool:
    """
    Sanity expectation (NOT a PASS/FAIL acceptance criterion): a deeper level is harder to
    reach, so reach_rate should be non-increasing in Δ. False => raise a visible alarm
    (a bug, or an extremely interesting finding). Reported, not gating.
    """
    reaches = [summary[d]["reach_rate"] for d in delta_ticks_list]
    return all(reaches[i] >= reaches[i + 1] for i in range(len(reaches) - 1))
