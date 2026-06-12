"""Tests for Exp10 proxy — M1: side_to_sign + compute_bin_stats (15) | M2: detect_events (10) | M3: match_controls (13) | M4: compute_directed_returns (11) | M5: run_inference (12)."""
import math

import numpy as np
import pandas as pd
import pytest

from research.experiments.exp10_proxy import (
    side_to_sign,
    compute_bin_stats,
    detect_events,
    match_controls,
    compute_directed_returns,
    paired_permutation_test,
    paired_bootstrap_ci,
    classify_outcome,
    run_inference,
    _MATCHED_PAIR_COLS,
    _DIRECTED_RET_COLS,
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


# ---------------------------------------------------------------------------
# Módulo 3 — match_controls (13 tests)
# ---------------------------------------------------------------------------

_COOLDOWN_MS = CONFIG["cooldown_ms"]   # 300_000
_T_ASIA   = 1_000_000                  # ~16 min into day, Asia session
_T_LONDON = 29_800_000                 # 8h + ~16 min, London session

_BIN_COLS_M3 = [
    "t_bin", "net_quote_W", "total_quote_W", "dominance_ratio",
    "sign_W", "sigma_W", "t0_ms", "p0",
]
_EVENT_COLS_M3 = _BIN_COLS_M3 + ["threshold"]


def _mk_event(t_bin, nq, sigma_W=0.001, total_quote_W=None, dom=0.9, sign_W=None):
    """Single-row events_df."""
    if total_quote_W is None:
        total_quote_W = abs(nq)
    if sign_W is None:
        sign_W = 1 if nq >= 0 else -1
    return pd.DataFrame([{
        "t_bin": t_bin, "net_quote_W": nq, "total_quote_W": total_quote_W,
        "dominance_ratio": dom, "sign_W": sign_W, "sigma_W": sigma_W,
        "t0_ms": t_bin + 1, "p0": 100.0, "threshold": 5.0,
    }], columns=_EVENT_COLS_M3)


def _mk_bin(t_bin, nq, dom, sigma_W=0.001, total_quote_W=None, sign_W=None):
    """Single bin dict for bins_df construction."""
    if total_quote_W is None:
        total_quote_W = abs(nq)
    if sign_W is None:
        sign_W = 1 if nq >= 0 else -1
    return {
        "t_bin": t_bin, "net_quote_W": nq, "total_quote_W": total_quote_W,
        "dominance_ratio": dom, "sign_W": sign_W, "sigma_W": sigma_W,
        "t0_ms": t_bin + 1, "p0": 100.0,
    }


def _base_bins(extra: list) -> pd.DataFrame:
    """
    5 background bins at t=0..4*BIN_SIZE_MS (dom=0.1, sigma_W=0.001, quote=1000)
    anchoring refs: sigma_ref=0.001, total_quote_ref=1000.  dom=0.1 < 0.20 so these
    bins are always excluded from the control pool.
    """
    rows = [
        _mk_bin(i * BIN_SIZE_MS, 1000.0, 0.1, sigma_W=0.001, total_quote_W=1000.0)
        for i in range(5)
    ]
    rows.extend(extra)
    return pd.DataFrame(rows, columns=_BIN_COLS_M3)


# ── M3-1: empty events → empty output with correct columns ──────────────────

def test_match_controls_empty_events_returns_empty_with_correct_columns():
    events_df = pd.DataFrame(columns=_EVENT_COLS_M3)
    bins_df = _base_bins([])
    result = match_controls(events_df, bins_df)
    assert result.empty
    assert list(result.columns) == list(_MATCHED_PAIR_COLS)


# ── M3-2: output columns exact + match_distance=0 for perfect match ─────────

def test_match_controls_output_columns_exact_and_zero_distance():
    ctrl_t = _T_ASIA + _COOLDOWN_MS + BIN_SIZE_MS
    events_df = _mk_event(_T_ASIA, 100.0, sigma_W=0.001, total_quote_W=1000.0)
    bins_df = _base_bins([_mk_bin(ctrl_t, 50.0, 0.35, sigma_W=0.001, total_quote_W=1000.0)])
    result = match_controls(events_df, bins_df)
    assert len(result) == 1
    assert list(result.columns) == list(_MATCHED_PAIR_COLS)
    assert result.iloc[0]["match_distance"] == pytest.approx(0.0)


# ── M3-3: event with sigma_W=NaN is discarded ───────────────────────────────

def test_match_controls_event_nan_sigma_discarded():
    ctrl_t = _T_ASIA + _COOLDOWN_MS + BIN_SIZE_MS
    events_df = _mk_event(_T_ASIA, 100.0, sigma_W=float("nan"))
    bins_df = _base_bins([_mk_bin(ctrl_t, 50.0, 0.35)])
    result = match_controls(events_df, bins_df)
    assert result.empty


# ── M3-4: control with sigma_W=NaN excluded from pool ───────────────────────

def test_match_controls_control_nan_sigma_excluded():
    ctrl_t = _T_ASIA + _COOLDOWN_MS + BIN_SIZE_MS
    events_df = _mk_event(_T_ASIA, 100.0)
    bins_df = _base_bins([_mk_bin(ctrl_t, 50.0, 0.35, sigma_W=float("nan"))])
    result = match_controls(events_df, bins_df)
    assert result.empty


# ── M3-5: sign filter — wrong-sign control excluded ─────────────────────────

def test_match_controls_sign_filter_excludes_wrong_sign():
    ctrl_t = _T_ASIA + _COOLDOWN_MS + BIN_SIZE_MS
    events_df = _mk_event(_T_ASIA, 100.0, sign_W=1)         # BUY event
    bins_df = _base_bins([_mk_bin(ctrl_t, -50.0, 0.35, sign_W=-1)])  # SELL control
    result = match_controls(events_df, bins_df)
    assert result.empty


# ── M3-6: session filter — different-session control excluded ────────────────

def test_match_controls_session_filter_excludes_different_session():
    # Event in Asia; control in London (different session)
    ctrl_t = _T_LONDON + _COOLDOWN_MS + BIN_SIZE_MS
    events_df = _mk_event(_T_ASIA, 100.0)
    bins_df = _base_bins([_mk_bin(ctrl_t, 50.0, 0.35)])
    result = match_controls(events_df, bins_df)
    assert result.empty


# ── M3-7a: dominance > 0.50 excluded ────────────────────────────────────────

def test_match_controls_dominance_above_max_excluded():
    ctrl_t = _T_ASIA + _COOLDOWN_MS + BIN_SIZE_MS
    events_df = _mk_event(_T_ASIA, 100.0)
    bins_df = _base_bins([_mk_bin(ctrl_t, 50.0, 0.51)])   # 0.51 > 0.50
    result = match_controls(events_df, bins_df)
    assert result.empty


# ── M3-7b: dominance exactly 0.50 included ──────────────────────────────────

def test_match_controls_dominance_boundary_050_included():
    ctrl_t = _T_ASIA + _COOLDOWN_MS + BIN_SIZE_MS
    events_df = _mk_event(_T_ASIA, 100.0)
    bins_df = _base_bins([_mk_bin(ctrl_t, 50.0, 0.50)])
    result = match_controls(events_df, bins_df)
    assert len(result) == 1


# ── M3-8: event bin excluded from pool (condition 4) ────────────────────────

def test_match_controls_event_bin_excluded_from_pool():
    # The event's own t_bin is added to bins_df with dom=0.35 (would qualify otherwise).
    # Condition 4 must exclude it; with no other valid control, event is discarded.
    events_df = _mk_event(_T_ASIA, 100.0)
    bins_df = _base_bins([_mk_bin(_T_ASIA, 80.0, 0.35)])
    result = match_controls(events_df, bins_df)
    assert result.empty


# ── M3-9: control within cooldown of event excluded ─────────────────────────

def test_match_controls_control_within_cooldown_excluded():
    near_t = _T_ASIA + 200_000    # 200 s < 300 s → excluded
    far_t  = _T_ASIA + 600_000    # 600 s > 300 s → included
    events_df = _mk_event(_T_ASIA, 100.0)
    bins_df = _base_bins([
        _mk_bin(near_t, 50.0, 0.35),
        _mk_bin(far_t,  50.0, 0.35),
    ])
    result = match_controls(events_df, bins_df)
    assert len(result) == 1
    assert int(result.iloc[0]["control_t_bin"]) == far_t


# ── M3-10: greedy min-distance, no replacement ──────────────────────────────

def test_match_controls_greedy_min_distance_no_replacement():
    """
    Both events prefer Control X (d=0).  Event A (|nq|=200) has priority → gets X.
    Event B (|nq|=100) falls back to Control Y.
    """
    t_ctrl_x = _T_ASIA + _COOLDOWN_MS + BIN_SIZE_MS
    t_ctrl_y = _T_ASIA + _COOLDOWN_MS + 2 * BIN_SIZE_MS
    ev_a = _mk_event(_T_ASIA,              200.0, sigma_W=0.001, total_quote_W=1000.0)
    ev_b = _mk_event(_T_ASIA + BIN_SIZE_MS, 100.0, sigma_W=0.001, total_quote_W=1000.0)
    events_df = pd.concat([ev_a, ev_b], ignore_index=True)
    bins_df = _base_bins([
        _mk_bin(t_ctrl_x, 50.0, 0.35, sigma_W=0.001, total_quote_W=1000.0),
        _mk_bin(t_ctrl_y, 50.0, 0.35, sigma_W=0.005, total_quote_W=5000.0),
    ])
    result = match_controls(events_df, bins_df)
    assert len(result) == 2
    row_a = result[result["event_t_bin"] == _T_ASIA].iloc[0]
    row_b = result[result["event_t_bin"] == _T_ASIA + BIN_SIZE_MS].iloc[0]
    assert int(row_a["control_t_bin"]) == t_ctrl_x
    assert int(row_b["control_t_bin"]) == t_ctrl_y


# ── M3-11: event without match is discarded ─────────────────────────────────

def test_match_controls_event_without_match_discarded():
    """Two events, one control → only larger-|nq| event gets matched."""
    ctrl_t = _T_ASIA + _COOLDOWN_MS + BIN_SIZE_MS
    ev_a = _mk_event(_T_ASIA,              200.0)
    ev_b = _mk_event(_T_ASIA + BIN_SIZE_MS, 100.0)
    events_df = pd.concat([ev_a, ev_b], ignore_index=True)
    bins_df = _base_bins([_mk_bin(ctrl_t, 50.0, 0.35)])
    result = match_controls(events_df, bins_df)
    assert len(result) == 1
    assert abs(float(result.iloc[0]["event_net_quote_W"])) == pytest.approx(200.0)


# ── M3-12: sigma_ref NaN raises ValueError ──────────────────────────────────

def test_match_controls_sigma_ref_nan_raises():
    events_df = _mk_event(_T_ASIA, 100.0)
    # All bins have sigma_W=NaN → median=NaN → ValueError
    bins_df = pd.DataFrame(
        [_mk_bin(_T_ASIA + 600_000, 50.0, 0.35, sigma_W=float("nan"))],
        columns=_BIN_COLS_M3,
    )
    with pytest.raises(ValueError, match="sigma_ref"):
        match_controls(events_df, bins_df)


# ── M3-13: total_quote_ref = 0 raises ValueError ────────────────────────────

def test_match_controls_quote_ref_zero_raises():
    events_df = _mk_event(_T_ASIA, 100.0)
    bins_df = pd.DataFrame(
        [_mk_bin(_T_ASIA + 600_000, 50.0, 0.35, sigma_W=0.001, total_quote_W=0.0)],
        columns=_BIN_COLS_M3,
    )
    with pytest.raises(ValueError, match="total_quote_ref"):
        match_controls(events_df, bins_df)


# ---------------------------------------------------------------------------
# Módulo 4 — compute_directed_returns (11 tests)
# ---------------------------------------------------------------------------

def _mk_pair_m4(
    ev_sign: int,
    ev_t0_ms: int,
    ev_p0: float,
    ctrl_t0_ms: int,
    ctrl_p0: float,
) -> pd.DataFrame:
    """Minimal single-row matched_pairs_df for M4 tests."""
    return pd.DataFrame([{
        "event_t_bin":           0,
        "event_net_quote_W":     100.0 * ev_sign,
        "event_sign_W":          ev_sign,
        "event_t0_ms":           ev_t0_ms,
        "event_p0":              ev_p0,
        "event_sigma_W":         0.001,
        "event_total_quote_W":   1000.0,
        "event_dominance_ratio": 0.9,
        "event_threshold":       5.0,
        "control_t_bin":         600_000,
        "control_t0_ms":         ctrl_t0_ms,
        "control_p0":            ctrl_p0,
        "control_sigma_W":       0.001,
        "control_total_quote_W": 1000.0,
        "control_dominance_ratio": 0.35,
        "match_distance":        0.0,
    }], columns=_MATCHED_PAIR_COLS)


def _mk_trades_m4(*rows) -> pd.DataFrame:
    """rows = (timestamp_ms, price). Must be in ascending timestamp order."""
    return pd.DataFrame(rows, columns=["timestamp_ms", "price"])


# ── M4-1: BUY reversal → directed_ret > 0 ───────────────────────────────────

def test_compute_directed_returns_buy_reversal_positive():
    # sign=+1, p0=100, PH=99 → −1×log(99/100) > 0
    pair = _mk_pair_m4(ev_sign=1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=0, ctrl_p0=100.0)
    trades = _mk_trades_m4((500, 99.0))
    result, n_dropped = compute_directed_returns(pair, trades, horizon_ms=500)
    assert n_dropped == 0
    assert float(result.iloc[0]["event_directed_ret"]) > 0.0


# ── M4-2: BUY continuation → directed_ret < 0 ───────────────────────────────

def test_compute_directed_returns_buy_continuation_negative():
    # sign=+1, p0=100, PH=101 → −1×log(101/100) < 0
    pair = _mk_pair_m4(ev_sign=1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=0, ctrl_p0=100.0)
    trades = _mk_trades_m4((500, 101.0))
    result, n_dropped = compute_directed_returns(pair, trades, horizon_ms=500)
    assert n_dropped == 0
    assert float(result.iloc[0]["event_directed_ret"]) < 0.0


# ── M4-3: SELL reversal → directed_ret > 0 ──────────────────────────────────

def test_compute_directed_returns_sell_reversal_positive():
    # sign=−1, p0=100, PH=101 → +1×log(101/100) > 0
    pair = _mk_pair_m4(ev_sign=-1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=0, ctrl_p0=100.0)
    trades = _mk_trades_m4((500, 101.0))
    result, n_dropped = compute_directed_returns(pair, trades, horizon_ms=500)
    assert n_dropped == 0
    assert float(result.iloc[0]["event_directed_ret"]) > 0.0


# ── M4-4: exact boundary timestamp — searchsorted(side="left") ───────────────

def test_compute_directed_returns_exact_boundary_timestamp_selected():
    # target_ms = t0_ms + horizon = 0 + 1000 = 1000
    # tape has trade exactly at 1000 → must be selected (not skipped)
    pair = _mk_pair_m4(ev_sign=1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=0, ctrl_p0=100.0)
    trades = _mk_trades_m4((1000, 99.0), (2000, 101.0))
    result, n_dropped = compute_directed_returns(pair, trades, horizon_ms=1000)
    assert n_dropped == 0
    expected = -1.0 * math.log(99.0 / 100.0)
    assert float(result.iloc[0]["event_directed_ret"]) == pytest.approx(expected)


# ── M4-5: event lookup beyond tape end → pair discarded, n_dropped=1 ─────────

def test_compute_directed_returns_event_beyond_tape_dropped():
    # tape ends at t=500; event target = 0 + 1000 = 1000 → beyond tape
    pair = _mk_pair_m4(ev_sign=1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=0, ctrl_p0=100.0)
    trades = _mk_trades_m4((500, 99.0))
    result, n_dropped = compute_directed_returns(pair, trades, horizon_ms=1000)
    assert result.empty
    assert n_dropped == 1


# ── M4-6: control lookup beyond tape end → pair discarded, n_dropped=1 ───────

def test_compute_directed_returns_control_beyond_tape_dropped():
    # event target=500 ✓ (trade exists); control target=1+1000=1001 beyond tape
    pair = _mk_pair_m4(ev_sign=1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=1, ctrl_p0=100.0)
    trades = _mk_trades_m4((500, 99.0))   # only one trade, at t=500
    result, n_dropped = compute_directed_returns(pair, trades, horizon_ms=500)
    assert result.empty
    assert n_dropped == 1


# ── M4-7: control directed_ret uses event_sign_W, not control's own sign ─────

def test_compute_directed_returns_control_uses_event_sign_w():
    """
    Event sign=−1 (SELL).  Control forward price goes UP (+1%).
    correct:  control_directed_ret = −(−1)×log(101/100) > 0
    bug (sign=+1): control_directed_ret = −(+1)×log(101/100) < 0
    """
    pair = _mk_pair_m4(ev_sign=-1, ev_t0_ms=0,    ev_p0=100.0,
                       ctrl_t0_ms=1000, ctrl_p0=100.0)
    # trade at t=500 for event (PH=99, direction irrelevant here)
    # trade at t=1500 for control (PH=101, price went UP)
    trades = _mk_trades_m4((500, 99.0), (1500, 101.0))
    result, n_dropped = compute_directed_returns(pair, trades, horizon_ms=500)
    assert n_dropped == 0
    assert float(result.iloc[0]["control_directed_ret"]) > 0.0


# ── M4-8: empty input → empty output, 0 dropped ─────────────────────────────

def test_compute_directed_returns_empty_input_returns_empty():
    pairs = pd.DataFrame(columns=_MATCHED_PAIR_COLS)
    trades = _mk_trades_m4((500, 99.0))
    result, n_dropped = compute_directed_returns(pairs, trades, horizon_ms=500)
    assert result.empty
    assert list(result.columns) == list(_DIRECTED_RET_COLS)
    assert n_dropped == 0


# ── M4-9: output columns are exactly _DIRECTED_RET_COLS (18) ─────────────────

def test_compute_directed_returns_output_columns_exact():
    pair = _mk_pair_m4(ev_sign=1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=0, ctrl_p0=100.0)
    trades = _mk_trades_m4((500, 99.0))
    result, _ = compute_directed_returns(pair, trades, horizon_ms=500)
    assert list(result.columns) == list(_DIRECTED_RET_COLS)
    assert len(_DIRECTED_RET_COLS) == 18


# ── M4-10: unsorted tape raises ValueError ───────────────────────────────────

def test_compute_directed_returns_unsorted_tape_raises():
    pair = _mk_pair_m4(ev_sign=1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=0, ctrl_p0=100.0)
    trades = _mk_trades_m4((2000, 100.0), (1000, 99.0))   # descending → unsorted
    with pytest.raises(ValueError, match="sorted"):
        compute_directed_returns(pair, trades, horizon_ms=500)


# ── M4-11: horizon_ms is respected — different horizons, different results ────

def test_compute_directed_returns_horizon_ms_respected():
    # At H=500ms: price=99 → reversal (ret>0)
    # At H=1000ms: price=101 → continuation (ret<0)
    pair = _mk_pair_m4(ev_sign=1, ev_t0_ms=0, ev_p0=100.0,
                       ctrl_t0_ms=0, ctrl_p0=100.0)
    trades = _mk_trades_m4((500, 99.0), (1000, 101.0))
    res_500,  _ = compute_directed_returns(pair, trades, horizon_ms=500)
    res_1000, _ = compute_directed_returns(pair, trades, horizon_ms=1000)
    assert float(res_500.iloc[0]["event_directed_ret"])  > 0.0
    assert float(res_1000.iloc[0]["event_directed_ret"]) < 0.0


# ===========================================================================
# M5: run_inference — paired permutation + bootstrap
# ===========================================================================

def _mk_returns_df(ev_drs, ctrl_drs) -> pd.DataFrame:
    """Minimal returns_df with the two columns run_inference needs."""
    return pd.DataFrame({
        "event_directed_ret":   list(ev_drs),
        "control_directed_ret": list(ctrl_drs),
    })


# ── M5-1: result has required keys ──────────────────────────────────────────

def test_run_inference_result_keys():
    df = _mk_returns_df([0.001], [-0.001])
    result = run_inference(df, n_perm=50, n_boot=50, rng_seed=0)
    assert {"delta_obs", "p_value_perm", "boot_ci_95", "n_pairs", "outcome"} <= set(result)


# ── M5-2: delta_obs = mean(event_dr) − mean(control_dr) ─────────────────────

def test_run_inference_delta_obs():
    ev   = [0.002, 0.004, 0.006]
    ctrl = [0.001, 0.001, 0.001]
    df = _mk_returns_df(ev, ctrl)
    result = run_inference(df, n_perm=100, n_boot=100, rng_seed=0)
    expected = np.mean(ev) - np.mean(ctrl)
    assert abs(result["delta_obs"] - expected) < 1e-12


# ── M5-3: p_perm ∈ [0, 1] ────────────────────────────────────────────────────

def test_run_inference_p_perm_bounded():
    df = _mk_returns_df([0.001, 0.002], [-0.001, 0.001])
    result = run_inference(df, n_perm=500, n_boot=100, rng_seed=42)
    assert 0.0 <= result["p_value_perm"] <= 1.0


# ── M5-4: n_pairs == len(returns_df) ────────────────────────────────────────

def test_run_inference_n_pairs():
    df = _mk_returns_df([0.001] * 20, [0.0] * 20)
    result = run_inference(df, n_perm=100, n_boot=100, rng_seed=0)
    assert result["n_pairs"] == 20


# ── M5-5: boot_ci_95 is (lo, hi) with lo ≤ hi ───────────────────────────────

def test_run_inference_ci_structure():
    df = _mk_returns_df([0.001, 0.002, 0.003], [0.0, 0.0, 0.0])
    result = run_inference(df, n_perm=100, n_boot=500, rng_seed=0)
    ci_lo, ci_hi = result["boot_ci_95"]
    assert ci_lo <= ci_hi


# ── M5-6: same rng_seed → identical result ───────────────────────────────────

def test_run_inference_determinism():
    df = _mk_returns_df([0.001, -0.002, 0.003], [0.0, 0.001, -0.001])
    r1 = run_inference(df, n_perm=200, n_boot=100, rng_seed=7)
    r2 = run_inference(df, n_perm=200, n_boot=100, rng_seed=7)
    assert r1["p_value_perm"] == r2["p_value_perm"]
    assert r1["boot_ci_95"]   == r2["boot_ci_95"]


# ── M5-7: outcome NULO when delta_obs ≤ 0 ───────────────────────────────────

def test_run_inference_nulo_negative_delta():
    df = _mk_returns_df([0.001] * 50, [0.002] * 50)
    result = run_inference(df, n_perm=500, n_boot=100, rng_seed=0)
    assert result["delta_obs"] < 0
    assert result["outcome"] == "NULO"


# ── M5-8: outcome FUERTE when diffs strongly positive ───────────────────────

def test_run_inference_outcome_fuerte():
    # 50 pairs with diffs = 0.01 >> 0 → p_perm ≈ 0, CI strictly > 0
    df = _mk_returns_df([0.01] * 50, [0.0] * 50)
    result = run_inference(df, n_perm=2_000, n_boot=500, rng_seed=0)
    assert result["outcome"] == "FUERTE"
    assert result["delta_obs"] > 0
    assert result["p_value_perm"] < 0.05
    ci_lo, _ = result["boot_ci_95"]
    assert ci_lo > 0


# ── M5-9: outcome DÉBIL — N=3, all diffs equal → p_perm ≈ 1/8 = 0.125 ──────
# With N=3 identical diffs d>0: P(Δ_perm ≥ Δ_obs) = P(all signs=+1) = (1/2)^3 = 0.125

def test_run_inference_outcome_debil():
    df = _mk_returns_df([0.001, 0.001, 0.001], [0.0, 0.0, 0.0])
    result = run_inference(df, n_perm=10_000, n_boot=500, rng_seed=0)
    assert result["outcome"] == "DÉBIL"
    assert result["delta_obs"] > 0
    assert 0.05 <= result["p_value_perm"] < 0.20


# ── M5-10: outcome NULO via p_perm ≥ 0.20 — N=2 → p_perm = 0.25 ────────────

def test_run_inference_nulo_large_pvalue():
    # N=2, all diffs equal → P(Δ_perm ≥ Δ_obs) = (1/2)^2 = 0.25 ≥ 0.20 → NULO
    df = _mk_returns_df([0.001, 0.001], [0.0, 0.0])
    result = run_inference(df, n_perm=10_000, n_boot=500, rng_seed=0)
    assert result["delta_obs"] > 0
    assert result["p_value_perm"] >= 0.20
    assert result["outcome"] == "NULO"


# ── M5-11: all diffs = 0 → p_perm = 1.0 (every permutation ties) ────────────

def test_run_inference_all_zero_diffs():
    df = _mk_returns_df([0.001, 0.001], [0.001, 0.001])
    result = run_inference(df, n_perm=200, n_boot=100, rng_seed=0)
    assert result["delta_obs"] == pytest.approx(0.0)
    assert result["p_value_perm"] == pytest.approx(1.0)
    assert result["outcome"] == "NULO"


# ── M5-12: empty DataFrame raises ValueError ─────────────────────────────────

def test_run_inference_empty_raises():
    df = _mk_returns_df([], [])
    with pytest.raises(ValueError, match="empty"):
        run_inference(df)


# ===========================================================================
# M5 — paired_permutation_test (funciones individuales)
# ===========================================================================

# ── PPT-1: Δ = 0 → p ≈ 0.5 ──────────────────────────────────────────────────
# 20 pairs with distinct diffs summing to 0. All diffs distinct → P(Δ_perm = 0) ≈ 0,
# so p_perm ≈ 0.5 (permutation distribution is symmetric around 0).

def test_ppt_delta_zero_p_near_half():
    ev   = [0.001 * (i + 1) for i in range(10)] + [0.0] * 10
    ctrl = [0.0] * 10 + [0.001 * (i + 1) for i in range(10)]
    df = _mk_returns_df(ev, ctrl)
    delta, p = paired_permutation_test(df, n_permutations=10_000, seed=0)
    assert abs(delta) < 1e-12
    assert 0.40 <= p <= 0.60


# ── PPT-2: ev >> ctrl → p small (< 0.05) ────────────────────────────────────

def test_ppt_strong_positive_p_small():
    df = _mk_returns_df([0.01] * 50, [0.0] * 50)
    delta, p = paired_permutation_test(df, n_permutations=2_000, seed=0)
    assert delta > 0
    assert p < 0.05


# ── PPT-3: ev << ctrl → p near 1 (> 0.95) ───────────────────────────────────

def test_ppt_strong_negative_p_near_one():
    df = _mk_returns_df([0.0] * 50, [0.01] * 50)
    delta, p = paired_permutation_test(df, n_permutations=2_000, seed=0)
    assert delta < 0
    assert p > 0.95


# ── PPT-4: same seed → identical result ─────────────────────────────────────

def test_ppt_reproducibility():
    df = _mk_returns_df([0.001, -0.002, 0.003], [0.0, 0.001, -0.001])
    r1 = paired_permutation_test(df, n_permutations=200, seed=9)
    r2 = paired_permutation_test(df, n_permutations=200, seed=9)
    assert r1 == r2


# ===========================================================================
# M5 — paired_bootstrap_ci (funciones individuales)
# ===========================================================================

# ── PBC-1: CI is ordered: ci_low ≤ ci_high ───────────────────────────────────

def test_pbc_ci_ordered():
    df = _mk_returns_df([0.001, 0.002, 0.003], [0.0, 0.0, 0.0])
    ci_lo, ci_hi = paired_bootstrap_ci(df, n_resamples=500, seed=0)
    assert ci_lo <= ci_hi


# ── PBC-2: same seed → identical result ─────────────────────────────────────

def test_pbc_reproducibility():
    df = _mk_returns_df([0.001, -0.002, 0.003], [0.0, 0.001, -0.001])
    r1 = paired_bootstrap_ci(df, n_resamples=200, seed=3)
    r2 = paired_bootstrap_ci(df, n_resamples=200, seed=3)
    assert r1 == r2


# ── PBC-3: N=1 does not crash — returns degenerate CI ───────────────────────

def test_pbc_n1_no_crash():
    df = _mk_returns_df([0.005], [0.002])
    ci_lo, ci_hi = paired_bootstrap_ci(df, n_resamples=200, seed=0)
    assert ci_lo <= ci_hi
    assert ci_lo == pytest.approx(0.003) and ci_hi == pytest.approx(0.003)


# ── PBC-4: identical diffs → degenerate CI = Δ_obs ──────────────────────────

def test_pbc_degenerate_identical_diffs():
    df = _mk_returns_df([0.007] * 10, [0.003] * 10)   # all diffs = 0.004
    ci_lo, ci_hi = paired_bootstrap_ci(df, n_resamples=500, seed=0)
    assert ci_lo == pytest.approx(0.004)
    assert ci_hi == pytest.approx(0.004)


# ===========================================================================
# M5 — classify_outcome (función pura)
# ===========================================================================

# ── CO-1: FUERTE ─────────────────────────────────────────────────────────────

def test_classify_outcome_fuerte():
    assert classify_outcome(delta=0.01, p_perm=0.01, ci_low=0.002, ci_high=0.020) == "FUERTE"


# ── CO-2: DÉBIL ──────────────────────────────────────────────────────────────

def test_classify_outcome_debil():
    assert classify_outcome(delta=0.01, p_perm=0.10, ci_low=-0.001, ci_high=0.020) == "DÉBIL"
    assert classify_outcome(delta=0.01, p_perm=0.05, ci_low=0.002,  ci_high=0.020) == "DÉBIL"
    assert classify_outcome(delta=0.01, p_perm=0.199, ci_low=0.002, ci_high=0.020) == "DÉBIL"


# ── CO-3: NULO ───────────────────────────────────────────────────────────────

def test_classify_outcome_nulo():
    assert classify_outcome(delta=-0.001, p_perm=0.01,  ci_low=-0.01, ci_high=-0.001) == "NULO"
    assert classify_outcome(delta=0.0,    p_perm=0.40,  ci_low=-0.01, ci_high=0.01)   == "NULO"
    assert classify_outcome(delta=0.01,   p_perm=0.20,  ci_low=0.001, ci_high=0.02)   == "NULO"
    assert classify_outcome(delta=0.01,   p_perm=0.99,  ci_low=0.001, ci_high=0.02)   == "NULO"
    # FUERTE requires ci_low > 0; p<0.05 but ci_low≤0 → not FUERTE → DÉBIL? No, p<0.05 and ci_low≤0
    # The spec says FUERTE: p<0.05 AND ci_low>0 AND delta>0. If ci_low≤0: falls to DÉBIL check.
    # p<0.05 is not in [0.05, 0.20), so → NULO.
    assert classify_outcome(delta=0.01, p_perm=0.01, ci_low=-0.001, ci_high=0.020) == "NULO"
