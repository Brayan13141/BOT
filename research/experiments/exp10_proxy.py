"""
Exp10 proxy test: reversal after unilateral aggressive flow burst.
Spec: OBSIDIAN/docs/superpowers/specs/2026-06-11-exp10-proxy-spec-v1.md
SPEC_HASH = 7a809aaca7f3  — asserted in notebook cell-1 via sha256(CONFIG, sort_keys=True)[:12].
"""
from __future__ import annotations

import math

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


# ---------------------------------------------------------------------------
# Módulo 3 — Control matching
# ---------------------------------------------------------------------------

_MATCHED_PAIR_COLS: list[str] = [
    "event_t_bin", "event_net_quote_W", "event_sign_W",
    "event_t0_ms", "event_p0", "event_sigma_W",
    "event_total_quote_W", "event_dominance_ratio", "event_threshold",
    "control_t_bin", "control_t0_ms", "control_p0",
    "control_sigma_W", "control_total_quote_W", "control_dominance_ratio",
    "match_distance",
]

_DAY_MS: int = 86_400_000
_HOUR_MS: int = 3_600_000


def _session(t_bin: int) -> int:
    """Session index: 0=Asia [0,8)h UTC, 1=London [8,16)h, 2=NY [16,24)h."""
    return (int(t_bin) % _DAY_MS) // _HOUR_MS // 8


def match_controls(
    events_df: pd.DataFrame,
    bins_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Greedy min-Euclidean-distance control matching (Exp10 spec v1, Module 3).

    events_df : output of detect_events (includes 'threshold' column)
    bins_df   : output of compute_bin_stats (full tape)
    Returns   : matched_pairs_df — one row per (event, control) pair.
                Events without an available match are discarded.
    """
    if events_df.empty:
        return pd.DataFrame(columns=_MATCHED_PAIR_COLS)

    # ── normalisation references: P50 over the full tape (NaN-skipping)
    sigma_ref = float(bins_df["sigma_W"].median())
    quote_ref = float(bins_df["total_quote_W"].median())

    if np.isnan(sigma_ref) or sigma_ref < CONFIG["matching_sigma_epsilon"]:
        raise ValueError(
            f"sigma_ref={sigma_ref!r} is NaN or below "
            f"matching_sigma_epsilon={CONFIG['matching_sigma_epsilon']}."
        )
    if np.isnan(quote_ref) or quote_ref <= 0.0:
        raise ValueError(
            f"total_quote_ref={quote_ref!r} is NaN or ≤ 0."
        )

    # ── pre-filter events: discard rows where sigma_W is NaN
    events = events_df.dropna(subset=["sigma_W"]).copy()
    if events.empty:
        return pd.DataFrame(columns=_MATCHED_PAIR_COLS)

    # ── global control pool: conditions 3, 4, 5 + NaN sigma_W excluded
    # Use full events_df (before NaN drop) for event_t_bins — conditions 4 & 5
    event_t_arr = np.array(
        sorted(int(t) for t in events_df["t_bin"]), dtype=np.int64
    )
    pool_t = bins_df["t_bin"].to_numpy(dtype=np.int64)
    cooldown_ms = int(CONFIG["cooldown_ms"])

    # vectorised: min distance from each pool bin to any event t_bin
    min_dist = np.min(np.abs(pool_t[:, None] - event_t_arr[None, :]), axis=1)
    outside_cooldown = min_dist >= cooldown_ms

    pool_mask = (
        (bins_df["dominance_ratio"] >= CONFIG["control_dominance_min"])
        & (bins_df["dominance_ratio"] <= CONFIG["control_dominance_max"])
        & ~bins_df["t_bin"].isin(set(event_t_arr.tolist()))
        & bins_df["sigma_W"].notna()
        & outside_cooldown
    )
    pool = bins_df[pool_mask].copy().reset_index(drop=True)

    if pool.empty:
        return pd.DataFrame(columns=_MATCHED_PAIR_COLS)

    pool["_session"] = pool["t_bin"].apply(lambda t: _session(int(t)))

    # ── greedy matching: events ordered by |net_quote_W| descending
    events = events.copy()
    events["_abs_nq"] = events["net_quote_W"].abs()
    events["_session"] = events["t_bin"].apply(lambda t: _session(int(t)))
    events = events.sort_values("_abs_nq", ascending=False).reset_index(drop=True)

    pool_sigma = pool["sigma_W"].to_numpy(dtype=np.float64)
    pool_quote = pool["total_quote_W"].to_numpy(dtype=np.float64)
    pool_sign  = pool["sign_W"].to_numpy()
    pool_sess  = pool["_session"].to_numpy()
    available  = np.ones(len(pool), dtype=bool)

    records: list[dict] = []
    for _, ev in events.iterrows():
        ev_sigma = float(ev["sigma_W"])
        ev_quote = float(ev["total_quote_W"])
        ev_sign  = int(ev["sign_W"])
        ev_sess  = int(ev["_session"])

        cand = available & (pool_sign == ev_sign) & (pool_sess == ev_sess)
        if not cand.any():
            continue

        cand_idx = np.where(cand)[0]
        d = np.sqrt(
            ((pool_sigma[cand_idx] - ev_sigma) / sigma_ref) ** 2
            + ((pool_quote[cand_idx] - ev_quote) / quote_ref) ** 2
        )
        best_local = int(np.argmin(d))
        best_i     = int(cand_idx[best_local])
        available[best_i] = False
        ctrl = pool.iloc[best_i]

        records.append({
            "event_t_bin":           int(ev["t_bin"]),
            "event_net_quote_W":     float(ev["net_quote_W"]),
            "event_sign_W":          int(ev["sign_W"]),
            "event_t0_ms":           int(ev["t0_ms"]),
            "event_p0":              float(ev["p0"]),
            "event_sigma_W":         ev_sigma,
            "event_total_quote_W":   ev_quote,
            "event_dominance_ratio": float(ev["dominance_ratio"]),
            "event_threshold":       float(ev["threshold"]),
            "control_t_bin":         int(ctrl["t_bin"]),
            "control_t0_ms":         int(ctrl["t0_ms"]),
            "control_p0":            float(ctrl["p0"]),
            "control_sigma_W":       float(ctrl["sigma_W"]),
            "control_total_quote_W": float(ctrl["total_quote_W"]),
            "control_dominance_ratio": float(ctrl["dominance_ratio"]),
            "match_distance":        float(d[best_local]),
        })

    return pd.DataFrame(records, columns=_MATCHED_PAIR_COLS)


# ---------------------------------------------------------------------------
# Módulo 4 — Directed returns
# ---------------------------------------------------------------------------

_DIRECTED_RET_COLS: list[str] = _MATCHED_PAIR_COLS + [
    "event_directed_ret",
    "control_directed_ret",
]


def compute_directed_returns(
    matched_pairs_df: pd.DataFrame,
    aggtrades_df: pd.DataFrame,
    *,
    horizon_ms: int,
) -> tuple[pd.DataFrame, int]:
    """
    Compute directed returns for each (event, control) pair.

    directed_ret = −sign_W_event × log( P(t0 + H) / p0 )
    P(t0 + H) = price of the first aggTrade with timestamp_ms >= t0_ms + horizon_ms.

    Precondition: aggtrades_df must be sorted by timestamp_ms (ascending).

    Returns
    -------
    (pairs_with_returns_df, n_dropped)
        pairs_with_returns_df : matched_pairs_df + event_directed_ret + control_directed_ret
        n_dropped             : number of pairs dropped because a forward-price lookup
                                failed (no trade at or after target timestamp).
    """
    if matched_pairs_df.empty:
        return pd.DataFrame(columns=_DIRECTED_RET_COLS), 0

    # ── A1: tape must be sorted
    ts_arr = aggtrades_df["timestamp_ms"].to_numpy(dtype=np.int64)
    px_arr = aggtrades_df["price"].to_numpy(dtype=np.float64)
    n_tape = len(ts_arr)

    if n_tape > 1 and not bool(np.all(ts_arr[1:] >= ts_arr[:-1])):
        raise ValueError(
            "aggtrades_df is not sorted by timestamp_ms. "
            "Sort before calling compute_directed_returns."
        )

    def _lookup_price(t0_ms: int) -> float | None:
        target = int(t0_ms) + horizon_ms
        idx = int(np.searchsorted(ts_arr, target, side="left"))
        return float(px_arr[idx]) if idx < n_tape else None

    records: list[dict] = []
    n_dropped = 0
    _dropped_event = 0    # internal diagnostics
    _dropped_control = 0

    for _, row in matched_pairs_df.iterrows():
        # ── A2: p0 must be positive
        ev_p0   = float(row["event_p0"])
        ctrl_p0 = float(row["control_p0"])
        if ev_p0 <= 0.0:
            raise ValueError(f"event_p0={ev_p0!r} <= 0 at event_t_bin={row['event_t_bin']}.")
        if ctrl_p0 <= 0.0:
            raise ValueError(f"control_p0={ctrl_p0!r} <= 0 at control_t_bin={row['control_t_bin']}.")

        p_H_event = _lookup_price(int(row["event_t0_ms"]))
        if p_H_event is None:
            _dropped_event += 1
            n_dropped += 1
            continue

        p_H_ctrl = _lookup_price(int(row["control_t0_ms"]))
        if p_H_ctrl is None:
            _dropped_control += 1
            n_dropped += 1
            continue

        sign_e = int(row["event_sign_W"])
        ev_dr   = -sign_e * math.log(p_H_event / ev_p0)
        ctrl_dr = -sign_e * math.log(p_H_ctrl  / ctrl_p0)

        rec = row.to_dict()
        rec["event_directed_ret"]   = ev_dr
        rec["control_directed_ret"] = ctrl_dr
        records.append(rec)

    result = (
        pd.DataFrame(records, columns=_DIRECTED_RET_COLS)
        if records
        else pd.DataFrame(columns=_DIRECTED_RET_COLS)
    )
    return result, n_dropped


def paired_permutation_test(
    returns_df: pd.DataFrame,
    *,
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Paired permutation test for Δ = mean(event_dr − control_dr).

    Each permutation independently flips the event/control label within each pair.
    Returns (delta_observed, p_perm) where p_perm = P(Δ_perm ≥ Δ_obs) — one-sided H1: Δ > 0.
    """
    if returns_df.empty:
        raise ValueError("returns_df is empty.")

    diffs = (returns_df["event_directed_ret"].to_numpy(dtype=np.float64)
             - returns_df["control_directed_ret"].to_numpy(dtype=np.float64))
    n = len(diffs)
    delta_obs = float(diffs.mean())

    rng   = np.random.default_rng(seed)
    bits  = rng.integers(0, 2, size=(n_permutations, n))
    signs = (2 * bits - 1).astype(np.float64)
    perm_deltas = (diffs[np.newaxis, :] * signs).mean(axis=1)
    p_perm = float((perm_deltas >= delta_obs).mean())

    return delta_obs, p_perm


def paired_bootstrap_ci(
    returns_df: pd.DataFrame,
    *,
    n_resamples: int = CONFIG["bootstrap_n_resamples"],
    ci_level: float = CONFIG["bootstrap_ci_level"],
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Paired bootstrap CI for Δ = mean(event_dr − control_dr).
    Returns (ci_low, ci_high) at the requested ci_level.
    """
    if returns_df.empty:
        raise ValueError("returns_df is empty.")

    diffs = (returns_df["event_directed_ret"].to_numpy(dtype=np.float64)
             - returns_df["control_directed_ret"].to_numpy(dtype=np.float64))
    n = len(diffs)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_deltas = diffs[idx].mean(axis=1)

    ci_half = (1.0 - ci_level) / 2.0
    ci_low  = float(np.percentile(boot_deltas, ci_half * 100))
    ci_high = float(np.percentile(boot_deltas, (1.0 - ci_half) * 100))
    return ci_low, ci_high


def classify_outcome(
    delta: float,
    p_perm: float,
    ci_low: float,
    ci_high: float,
) -> str:
    """
    Classify experiment outcome per spec v1 falsifiers:
      FUERTE : p_perm < 0.05  AND  ci_low > 0  AND  delta > 0
      DÉBIL  : delta > 0  AND  p_perm ∈ [0.05, 0.20)
      NULO   : delta ≤ 0  OR  p_perm ≥ 0.20
    """
    if delta > 0 and p_perm < 0.05 and ci_low > 0:
        return "FUERTE"
    if delta > 0 and 0.05 <= p_perm < 0.20:
        return "DÉBIL"
    return "NULO"


def run_inference(
    returns_df: pd.DataFrame,
    *,
    n_perm: int = 10_000,
    n_boot: int = CONFIG["bootstrap_n_resamples"],
    ci_level: float = CONFIG["bootstrap_ci_level"],
    rng_seed: int = 0,
) -> dict:
    """
    Convenience wrapper: paired_permutation_test + paired_bootstrap_ci + classify_outcome.
    Returns dict with keys: delta_obs, p_value_perm, boot_ci_95, n_pairs, outcome.
    """
    if returns_df.empty:
        raise ValueError("returns_df is empty — no pairs to analyze.")

    delta_obs, p_perm = paired_permutation_test(
        returns_df, n_permutations=n_perm, seed=rng_seed
    )
    ci_low, ci_high = paired_bootstrap_ci(
        returns_df, n_resamples=n_boot, ci_level=ci_level, seed=rng_seed
    )
    outcome = classify_outcome(delta_obs, p_perm, ci_low, ci_high)

    return {
        "delta_obs":    delta_obs,
        "p_value_perm": p_perm,
        "boot_ci_95":   (ci_low, ci_high),
        "n_pairs":      len(returns_df),
        "outcome":      outcome,
    }
