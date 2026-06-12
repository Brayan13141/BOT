"""Tests for Exp10 proxy — M1: side_to_sign + compute_bin_stats (15) | M2: detect_events (10)."""
import math

import numpy as np
import pandas as pd
import pytest

from research.experiments.exp10_proxy import (
    side_to_sign,
    compute_bin_stats,
    detect_events,
    CONFIG,
    BIN_SIZE_MS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trades(*rows):
    """rows = (timestamp_ms, price, qty, side)."""
    return pd.DataFrame(rows, columns=["timestamp_ms", "price", "qty", "side"])


# ---------------------------------------------------------------------------
# side_to_sign — 4 tests
# ---------------------------------------------------------------------------

def test_side_to_sign_buy_returns_positive_one():
    assert side_to_sign("BUY") == 1


def test_side_to_sign_sell_returns_negative_one():
    assert side_to_sign("SELL") == -1


def test_side_to_sign_unknown_raises_value_error():
    with pytest.raises(ValueError, match="BUY.*SELL|buy"):
        side_to_sign("buy")


def test_side_to_sign_empty_string_raises_value_error():
    with pytest.raises(ValueError):
        side_to_sign("")


# ---------------------------------------------------------------------------
# compute_bin_stats — 11 tests
# ---------------------------------------------------------------------------

def test_compute_bin_stats_empty_returns_empty_dataframe():
    result = compute_bin_stats(pd.DataFrame(columns=["timestamp_ms", "price", "qty", "side"]))
    assert result.empty
    assert list(result.columns) == [
        "t_bin", "net_quote_W", "total_quote_W", "dominance_ratio",
        "sign_W", "sigma_W", "t0_ms", "p0",
    ]


def test_compute_bin_stats_output_columns_exact():
    df = _trades((1000, 100.0, 1.0, "BUY"))
    result = compute_bin_stats(df)
    assert list(result.columns) == [
        "t_bin", "net_quote_W", "total_quote_W", "dominance_ratio",
        "sign_W", "sigma_W", "t0_ms", "p0",
    ]


def test_compute_bin_stats_sigma_W_nan_for_single_trade():
    df = _trades((1000, 100.0, 1.0, "BUY"))
    result = compute_bin_stats(df)
    assert result.shape[0] == 1
    assert math.isnan(result["sigma_W"].iloc[0])


def test_compute_bin_stats_sigma_W_nan_for_two_trades():
    # 2 trades → 1 log return → std(ddof=1) with n=1 → NaN
    df = _trades(
        (1000, 100.0, 1.0, "BUY"),
        (2000, 100.1, 1.0, "BUY"),
    )
    result = compute_bin_stats(df)
    assert result.shape[0] == 1
    assert math.isnan(result["sigma_W"].iloc[0])


def test_compute_bin_stats_sigma_W_float64_for_three_trades():
    df = _trades(
        (1000, 100.0, 1.0, "BUY"),
        (2000, 100.1, 1.0, "BUY"),
        (3000, 100.2, 1.0, "BUY"),
    )
    result = compute_bin_stats(df)
    sigma = result["sigma_W"].iloc[0]
    assert not math.isnan(sigma)
    assert isinstance(sigma, float)
    assert sigma >= 0.0


def test_compute_bin_stats_all_buy_single_bin():
    df = _trades(
        (1000, 100.0, 2.0, "BUY"),
        (2000, 101.0, 1.0, "BUY"),
    )
    row = compute_bin_stats(df).iloc[0]
    assert row["net_quote_W"] > 0
    assert abs(row["dominance_ratio"] - 1.0) < 1e-10
    assert row["sign_W"] == 1


def test_compute_bin_stats_all_sell_single_bin():
    df = _trades(
        (1000, 100.0, 2.0, "SELL"),
        (2000, 101.0, 1.0, "SELL"),
    )
    row = compute_bin_stats(df).iloc[0]
    assert row["net_quote_W"] < 0
    assert abs(row["dominance_ratio"] - 1.0) < 1e-10
    assert row["sign_W"] == -1


def test_compute_bin_stats_net_and_total_quote_W():
    # BUY 100*2=200 (net +200), SELL 101*1=101 (net -101)
    # net = 200 - 101 = 99, total = 200 + 101 = 301
    df = _trades(
        (1000, 100.0, 2.0, "BUY"),
        (2000, 101.0, 1.0, "SELL"),
    )
    row = compute_bin_stats(df).iloc[0]
    assert abs(row["net_quote_W"] - 99.0) < 1e-9
    assert abs(row["total_quote_W"] - 301.0) < 1e-9
    assert abs(row["dominance_ratio"] - 99.0 / 301.0) < 1e-9


def test_compute_bin_stats_t0_ms_and_p0_are_from_last_trade():
    # last trade by timestamp: ts=5000, price=102.0
    df = _trades(
        (1000, 100.0, 1.0, "BUY"),
        (5000, 102.0, 1.0, "SELL"),
        (3000, 101.0, 1.0, "BUY"),
    )
    row = compute_bin_stats(df).iloc[0]
    assert row["t0_ms"] == 5000
    assert abs(row["p0"] - 102.0) < 1e-10


def test_compute_bin_stats_bins_are_30s_utc_anchored():
    # ts=0 and ts=29_999 → bin=0; ts=30_000 → bin=30_000
    df = _trades(
        (0, 100.0, 1.0, "BUY"),
        (29_999, 100.1, 1.0, "BUY"),
        (30_000, 100.2, 1.0, "SELL"),
    )
    result = compute_bin_stats(df)
    assert result.shape[0] == 2
    bins = sorted(result["t_bin"].tolist())
    assert bins == [0, 30_000]


def test_compute_bin_stats_multiple_bins_one_row_each():
    # 3 distinct 30s bins
    df = _trades(
        (0, 100.0, 1.0, "BUY"),
        (30_000, 101.0, 1.0, "SELL"),
        (60_000, 102.0, 1.0, "BUY"),
    )
    result = compute_bin_stats(df)
    assert result.shape[0] == 3
    assert sorted(result["t_bin"].tolist()) == [0, 30_000, 60_000]


# ---------------------------------------------------------------------------
# Módulo 2 — detect_events (10 tests)
# ---------------------------------------------------------------------------

_WARMUP_MS: int = CONFIG["threshold_window_ms"]  # 86_400_000 ms
_N_HISTORY: int = _WARMUP_MS // BIN_SIZE_MS       # 2880 bins

_BIN_COLS_M2 = [
    "t_bin", "net_quote_W", "total_quote_W", "dominance_ratio",
    "sign_W", "sigma_W", "t0_ms", "p0",
]
_EVENT_COLS_M2 = _BIN_COLS_M2 + ["threshold"]


def _make_bins_df(spikes: list) -> pd.DataFrame:
    """
    2880 history bins (|net_quote_W|=1.0, dominance=0.3) at t_bin=0,30s,...,WARMUP-30s
    followed by spike bins specified as (bin_step_offset, net_quote_W, dominance_ratio).
    Spike t_bin = WARMUP + offset * BIN_SIZE_MS.
    History dominance=0.3 keeps history bins below the 0.75 gate so they never qualify.
    """
    W = _WARMUP_MS
    B = BIN_SIZE_MS
    rows = [
        {
            "t_bin": i * B, "net_quote_W": 1.0, "total_quote_W": 1.0,
            "dominance_ratio": 0.3, "sign_W": 1, "sigma_W": float("nan"),
            "t0_ms": i * B + 1, "p0": 100.0,
        }
        for i in range(_N_HISTORY)
    ]
    for offset, nq, dom in spikes:
        t = W + offset * B
        rows.append({
            "t_bin": t, "net_quote_W": nq, "total_quote_W": abs(nq),
            "dominance_ratio": dom, "sign_W": 1 if nq >= 0 else -1,
            "sigma_W": float("nan"), "t0_ms": t + 1, "p0": 100.0,
        })
    return pd.DataFrame(rows, columns=_BIN_COLS_M2)


# ── Test M2-1: empty input ──────────────────────────────────────────────────

def test_detect_events_empty_returns_empty_with_correct_columns():
    result = detect_events(pd.DataFrame(columns=_BIN_COLS_M2))
    assert result.empty
    assert list(result.columns) == _EVENT_COLS_M2


# ── Test M2-2: rolling threshold excludes current bin (closed='left') ───────

def test_detect_events_rolling_threshold_excludes_current_bin():
    """
    Empirical proof that closed='left' is in effect.
    All 2880 history bins have |net_quote_W|=1.0 → threshold(WARMUP)=1.0.
    Spike at WARMUP has |net_quote_W|=999.0.

    If closed='right': 999.0 enters its own threshold → threshold≈999 → |999|>999 False → 0 events.
    If closed='left':  threshold=1.0 → |999|>1.0 True → 1 event with threshold==1.0.
    """
    bins_df = _make_bins_df([(0, 999.0, 1.0)])
    events = detect_events(bins_df)
    assert len(events) == 1, (
        "Spike not detected. Likely closed='right' — current bin entered its own threshold."
    )
    assert abs(events["threshold"].iloc[0] - 1.0) < 1e-9, (
        f"threshold should be 1.0 (history only), got {events['threshold'].iloc[0]}"
    )


# ── Test M2-3: warmup excludes bins before tape_start + 24h ─────────────────

def test_detect_events_warmup_excludes_pre_warmup_bins():
    W = _WARMUP_MS
    B = BIN_SIZE_MS
    # Spike at WARMUP-B (1 step before warmup boundary) → must not produce event
    # Spike at WARMUP (exactly at boundary) → must produce event
    rows_before = [
        {
            "t_bin": i * B, "net_quote_W": 1.0, "total_quote_W": 1.0,
            "dominance_ratio": 0.3, "sign_W": 1, "sigma_W": float("nan"),
            "t0_ms": i * B + 1, "p0": 100.0,
        }
        for i in range(_N_HISTORY)
    ]
    # pre-warmup spike: t_bin = WARMUP - B (should be excluded)
    pre = {
        "t_bin": W - B, "net_quote_W": 999.0, "total_quote_W": 999.0,
        "dominance_ratio": 1.0, "sign_W": 1, "sigma_W": float("nan"),
        "t0_ms": W - B + 1, "p0": 100.0,
    }
    # post-warmup spike: t_bin = WARMUP (should be included)
    post = {
        "t_bin": W, "net_quote_W": 999.0, "total_quote_W": 999.0,
        "dominance_ratio": 1.0, "sign_W": 1, "sigma_W": float("nan"),
        "t0_ms": W + 1, "p0": 100.0,
    }
    bins_df = pd.DataFrame(rows_before + [pre, post], columns=_BIN_COLS_M2)
    events = detect_events(bins_df)
    assert len(events) == 1
    assert events.iloc[0]["t_bin"] == W


# ── Test M2-4: dominance gate ────────────────────────────────────────────────

def test_detect_events_dominance_gate_rejects_low_dominance():
    # net_quote_W=999 but dominance=0.74 < 0.75 → no event
    bins_df = _make_bins_df([(0, 999.0, 0.74)])
    events = detect_events(bins_df)
    assert len(events) == 0


def test_detect_events_dominance_gate_accepts_threshold_dominance():
    # dominance exactly 0.75 → event should be detected
    bins_df = _make_bins_df([(0, 999.0, 0.75)])
    events = detect_events(bins_df)
    assert len(events) == 1


# ── Test M2-5: single isolated bin above threshold → 1 event ────────────────

def test_detect_events_single_qualifying_bin_produces_one_event():
    bins_df = _make_bins_df([(0, 500.0, 1.0)])
    events = detect_events(bins_df)
    assert len(events) == 1
    assert list(events.columns) == _EVENT_COLS_M2
    assert events.iloc[0]["t_bin"] == _WARMUP_MS


# ── Test M2-6: cluster peak is the bin with max |net_quote_W| ────────────────

def test_detect_events_cluster_peak_is_max_abs_net_quote():
    # Three consecutive bins (one cluster) — middle has the largest |net_quote_W|
    bins_df = _make_bins_df([
        (0, 100.0, 1.0),   # t = WARMUP
        (1, 500.0, 1.0),   # t = WARMUP+30s — this is the peak
        (2, 200.0, 1.0),   # t = WARMUP+60s
    ])
    events = detect_events(bins_df)
    assert len(events) == 1
    assert events.iloc[0]["t_bin"] == _WARMUP_MS + BIN_SIZE_MS


# ── Test M2-7: cluster peak tie-break — first bin wins ──────────────────────

def test_detect_events_cluster_peak_tiebreak_selects_first():
    # Two consecutive bins with identical |net_quote_W| → first (smaller t_bin) wins
    bins_df = _make_bins_df([
        (0, 300.0, 1.0),   # t = WARMUP — should win tie-break
        (1, 300.0, 1.0),   # t = WARMUP+30s
    ])
    events = detect_events(bins_df)
    assert len(events) == 1
    assert events.iloc[0]["t_bin"] == _WARMUP_MS


# ── Test M2-8: cooldown — two distant events both retained ───────────────────

def test_detect_events_cooldown_two_distant_events_both_retained():
    # Two isolated spikes separated by 11 bins = 330s > 300s cooldown
    bins_df = _make_bins_df([
        (0,  200.0, 1.0),   # t = WARMUP
        (11, 300.0, 1.0),   # t = WARMUP + 330s — no conflict
    ])
    events = detect_events(bins_df)
    assert len(events) == 2
    assert events.iloc[0]["t_bin"] == _WARMUP_MS
    assert events.iloc[1]["t_bin"] == _WARMUP_MS + 11 * BIN_SIZE_MS


# ── Test M2-9: cooldown chain A→B→C — Bryan's edge case ─────────────────────

def test_detect_events_cooldown_chain_abc_retains_max():
    """
    A at WARMUP+0s, B at WARMUP+240s (8 bins), C at WARMUP+360s (12 bins).
    B-A=240s < 300s, C-B=120s < 300s → one conflict chain → keep C (largest |nq|).
    Verifies _apply_cooldown resolves the full chain, not just pairwise last-retained.
    """
    bins_df = _make_bins_df([
        (0,  100.0, 1.0),   # A: t = WARMUP
        (8,  200.0, 1.0),   # B: t = WARMUP+240s, gap from A = 240s < 300s
        (12, 300.0, 1.0),   # C: t = WARMUP+360s, gap from B = 120s < 300s
    ])
    events = detect_events(bins_df)
    assert len(events) == 1, f"Expected 1 event (C retained), got {len(events)}"
    assert events.iloc[0]["t_bin"] == _WARMUP_MS + 12 * BIN_SIZE_MS
    assert abs(events.iloc[0]["net_quote_W"] - 300.0) < 1e-9


# ── Test M2-10: output columns always include threshold ──────────────────────

def test_detect_events_output_always_includes_threshold_column():
    bins_df = _make_bins_df([(0, 999.0, 1.0)])
    events = detect_events(bins_df)
    assert "threshold" in events.columns
    assert events["threshold"].notna().all()
