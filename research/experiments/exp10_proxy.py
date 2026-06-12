"""
Exp10 proxy test: reversal after unilateral aggressive flow burst.
Spec: OBSIDIAN/docs/superpowers/specs/2026-06-11-exp10-proxy-spec-v1.md
SPEC_HASH = 7a809aaca7f3  — asserted in notebook cell-1 via sha256(CONFIG, sort_keys=True)[:12].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_SIZE_MS: int = 30_000  # 30 s UTC-anchored bins

CONFIG: dict = {
    # --- identity ---
    "spec_name": "Exp10_proxy_v1",

    # --- binning ---
    "bin_size_ms": 30000,
    "sign_convention": "BUY_pos1_SELL_neg1",
    "input_side_column": "side",
    "price_qty_type": "Decimal",
    "sigma_log_return_type": "float64",
    "empty_bin_policy": "skip_no_fill",

    # --- event detection ---
    "threshold_quantile": 0.995,
    "threshold_window_ms": 86400000,
    "warmup_ms": 86400000,
    "dominance_threshold": 0.75,
    "cluster_event_selection": "argmax_abs_net_quote_tiebreak_first",
    "cooldown_ms": 300000,

    # --- control matching ---
    "control_sign": "same_as_event",
    "control_session": "same_as_event",
    "control_not_event": True,
    "control_outside_cooldown": True,
    "control_dominance_min": 0.20,
    "control_dominance_max": 0.50,
    "control_matching_vars": ["sigma_W", "total_quote_W"],
    "control_sigma_ref": "P50_sigma_W_month",
    "control_quote_ref": "P50_total_quote_W_month",
    "control_matching_order": "greedy_min_euclidean_distance",
    "control_match_with_replacement": False,
    "matching_sigma_epsilon": 1e-08,

    # --- return computation ---
    "directed_ret_formula": "neg_sign_event_times_log_P_H_over_P0",
    "forward_price_method": "first_agg_trade_at_or_after_target_ms",
    "horizon_primary_ms": 300000,

    # --- inference ---
    "test_primary": "paired_permutation",
    "test_secondary": "paired_bootstrap",
    "bootstrap_n_resamples": 2000,
    "bootstrap_ci_level": 0.95,
    "statistic": "mean_directed_ret_event_minus_control",

    # --- sessions (UTC hours, inclusive start, exclusive end) ---
    "session_asia_utc_hours": [0, 8],
    "session_london_utc_hours": [8, 16],
    "session_ny_utc_hours": [16, 24],

    # --- segmentations (pre-registered) ---
    "segmentations": [
        "S1_direction",
        "S2_regime_conditional",
        "S3_session_utc",
        "S4_local_intensity_rolling",
    ],

    # --- falsifiers (pre-registered) ---
    "falsifiers": [
        "F1_null_vs_control",
        "F2_mechanical_only_1m",
        "F3_control_equals_if_E_positive",
        "F4_no_monotonicity_WEAK",
        "F5_directional_asymmetry_informative",
    ],
}

_SIDE_MAP: dict[str, int] = {"BUY": 1, "SELL": -1}

_BIN_COLS: list[str] = [
    "t_bin", "net_quote_W", "total_quote_W", "dominance_ratio",
    "sign_W", "sigma_W", "t0_ms", "p0",
]


def side_to_sign(side: str) -> int:
    """BUY → +1, SELL → −1. Raises ValueError for any other value."""
    try:
        return _SIDE_MAP[side]
    except KeyError:
        raise ValueError(
            f"Unknown side {side!r}. Expected 'BUY' or 'SELL'."
        )


def compute_bin_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate aggTrades into 30 s UTC-anchored bins.

    Input columns : timestamp_ms (int64), price (float64), qty (float64), side (str)
    Output columns: t_bin, net_quote_W, total_quote_W, dominance_ratio,
                    sign_W, sigma_W, t0_ms, p0

    empty_bin_policy=skip_no_fill — bins with no trades are never emitted.
    sigma_W — std of intrabin log returns (float64, NaN when n < 3 trades).
    t0_ms, p0 — timestamp and price of the last aggTrade in the bin.
    """
    if trades.empty:
        return pd.DataFrame(columns=_BIN_COLS)

    df = trades.copy()
    df["_sign"] = df["side"].map(_SIDE_MAP)
    unknown = df["_sign"].isna()
    if unknown.any():
        bad = df.loc[unknown, "side"].unique().tolist()
        raise ValueError(f"Unknown side values: {bad}. Expected 'BUY' or 'SELL'.")

    df["_quote"] = df["price"] * df["qty"]
    df["_signed_quote"] = df["_quote"] * df["_sign"]
    df["t_bin"] = (df["timestamp_ms"] // BIN_SIZE_MS) * BIN_SIZE_MS
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    records: list[dict] = []
    for t_bin, g in df.groupby("t_bin", sort=True):
        g = g.sort_values("timestamp_ms")

        net_quote_W = float(g["_signed_quote"].sum())
        total_quote_W = float(g["_quote"].sum())
        dominance_ratio = (
            abs(net_quote_W) / total_quote_W if total_quote_W > 0.0 else float("nan")
        )
        sign_W = int(np.sign(net_quote_W))

        prices = g["price"].to_numpy(dtype=np.float64)
        if len(prices) >= 3:
            log_rets = np.diff(np.log(prices))
            sigma_W = float(np.std(log_rets, ddof=1))
        else:
            sigma_W = float("nan")

        records.append({
            "t_bin": int(t_bin),
            "net_quote_W": net_quote_W,
            "total_quote_W": total_quote_W,
            "dominance_ratio": dominance_ratio,
            "sign_W": sign_W,
            "sigma_W": sigma_W,
            "t0_ms": int(g["timestamp_ms"].iloc[-1]),
            "p0": float(g["price"].iloc[-1]),
        })

    return pd.DataFrame(records, columns=_BIN_COLS)


# ---------------------------------------------------------------------------
# Módulo 2 — Event detection
# ---------------------------------------------------------------------------

_EVENT_COLS: list[str] = _BIN_COLS + ["threshold"]


def _apply_cooldown(peaks: pd.DataFrame, cooldown_ms: int) -> pd.DataFrame:
    """
    Group cluster peaks into conflict chains (consecutive gap < cooldown_ms).
    Retain the peak with max |net_quote_W| per chain; tie-break = first (min t_bin).
    Handles A→B→C chains correctly: compares each candidate against the last
    element in the current chain, not against the last retained winner.
    """
    if len(peaks) <= 1:
        return peaks.reset_index(drop=True)

    t_arr = peaks["t_bin"].values
    abs_nq = peaks["net_quote_W"].abs().values

    groups: list[list[int]] = []
    current: list[int] = [0]
    for i in range(1, len(peaks)):
        if t_arr[i] - t_arr[current[-1]] < cooldown_ms:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    retained = [
        max(g, key=lambda idx, _a=abs_nq, _t=t_arr: (_a[idx], -_t[idx]))
        for g in groups
    ]
    return peaks.iloc[retained].reset_index(drop=True)


def detect_events(bins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect unilateral flow burst events from bin stats.

    Input:  bins_df — output of compute_bin_stats
    Output: events_df — rows from bins_df that are cluster-peak events,
            with 'threshold' column (rolling P99.5 of |net_quote_W| at detection time).

    Pipeline: rolling_threshold → eligibility_gate → clusters → cluster_peaks → cooldown
    """
    if bins_df.empty:
        return pd.DataFrame(columns=_EVENT_COLS)

    df = bins_df.sort_values("t_bin").reset_index(drop=True).copy()
    tape_start_ms = int(df["t_bin"].iloc[0])

    # ── rolling P99.5 — window [T − threshold_window_ms, T), excludes current bin
    idx_dt = pd.to_datetime(df["t_bin"], unit="ms", utc=True)
    abs_nq_s = pd.Series(df["net_quote_W"].abs().values, index=idx_dt)
    df["threshold"] = (
        abs_nq_s
        .rolling(
            window=pd.Timedelta(milliseconds=CONFIG["threshold_window_ms"]),
            closed="left",
            min_periods=1,
        )
        .quantile(CONFIG["threshold_quantile"])
        .values
    )

    # ── eligibility gate: warmup + threshold exceeded + dominance
    warmup_end_ms = tape_start_ms + CONFIG["warmup_ms"]
    df["_eligible"] = (
        (df["t_bin"] >= warmup_end_ms)
        & df["threshold"].notna()
        & (df["net_quote_W"].abs() > df["threshold"])
        & (df["dominance_ratio"] >= CONFIG["dominance_threshold"])
    )

    eligible = df[df["_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=_EVENT_COLS)

    # ── cluster formation — maximal consecutive runs (adjacent gap == BIN_SIZE_MS)
    t_arr = eligible["t_bin"].values
    is_consecutive = np.concatenate([[False], np.diff(t_arr) == BIN_SIZE_MS])
    eligible["_cluster"] = np.cumsum(~is_consecutive) - 1
    eligible["_abs_nq"] = eligible["net_quote_W"].abs()

    # ── cluster peak: argmax |net_quote_W|, tie-break = first (smallest t_bin)
    peaks = (
        eligible
        .sort_values(["_cluster", "_abs_nq", "t_bin"], ascending=[True, False, True])
        .groupby("_cluster", sort=False)
        .first()
        .reset_index(drop=True)
        .sort_values("t_bin")
        .reset_index(drop=True)
    )

    # ── cooldown: conflict chains → retain max per chain
    retained = _apply_cooldown(peaks, CONFIG["cooldown_ms"])

    drop_cols = [c for c in ("_eligible", "_abs_nq") if c in retained.columns]
    return retained.drop(columns=drop_cols)[_EVENT_COLS].reset_index(drop=True)
