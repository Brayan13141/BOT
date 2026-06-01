# S28B Depth Sensitivity Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a depth-sensitivity sweep (`Δ ∈ {1,5,20}` ticks at fixed `queue_ahead=5 BTC`, `TTL=60s`) to the existing HET harness, reporting `reach × cond_fill = fill_rate` per `Δ`, to answer whether placement depth moves fill comparably to the queue (S28A).

**Architecture:** Extend `research/experiments/het_sweep.py` (FillModelC consumed unmodified). Parametrize `Δ` in the existing order generator (non-breaking default), add a depth-sweep driver + a decomposing summary + a `reach`-monotonicity sanity helper, then author a thin notebook (`S28B_depth_sweep.ipynb`) that Bryan runs.

**Tech Stack:** Python, NumPy, Decimal, pytest, Jupyter (nbformat 4.5). Branch: `feature/event-sourced-logging` (same as S28A; no worktree).

**Spec:** `OBSIDIAN/docs/superpowers/specs/2026-06-01-s28b-depth-sweep-design.md`

---

## File Structure

- **Modify** `research/experiments/het_sweep.py`:
  - `generate_passive_orders(...)` — add `delta: Decimal = DELTA` parameter (non-breaking).
  - New constants `DELTA_GRID_TICKS`, `QUEUE_FIXED_S28B`.
  - New `run_depth_sweep(...)` driver.
  - New `summarize_depth_sweep(...)` (decomposition + Wilson CIs).
  - New `reach_monotonicity_ok(...)` sanity helper.
- **Modify** `tests/test_het_sweep.py` — append S28B tests (same file: 1:1 module↔test convention).
- **Create** `research/experiments/S28B_depth_sweep.ipynb` — thin notebook; Bryan runs it.

All new public symbols are added to `het_sweep.py`; the notebook only imports and orchestrates.

---

### Task 1: Parametrize `Δ` in `generate_passive_orders` (non-breaking)

**Files:**
- Modify: `research/experiments/het_sweep.py:100-130`
- Test: `tests/test_het_sweep.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_het_sweep.py`:

```python
def test_generator_accepts_custom_delta():
    tape = _big_tape()
    # default delta == module DELTA (non-breaking)
    d_default = generate_passive_orders(tape, OrderSide.BUY, lam_per_sec=1/10, ttl_ms=5000, seed=[28, 0])
    # explicit delta = DELTA reproduces the default placement exactly
    d_same = generate_passive_orders(tape, OrderSide.BUY, lam_per_sec=1/10, ttl_ms=5000, seed=[28, 0],
                                     delta=DELTA)
    assert [str(g.order.limit_price) for g in d_default] == [str(g.order.limit_price) for g in d_same]
    # a deeper delta moves a BUY limit further BELOW the reference
    deep = generate_passive_orders(tape, OrderSide.BUY, lam_per_sec=1/10, ttl_ms=5000, seed=[28, 0],
                                   delta=Decimal("2.0"))
    g0, deep0 = d_default[0], deep[0]
    ref = Decimal(tape.price[g0.ref_index])
    assert g0.order.limit_price == ref - DELTA          # 0.50 below
    assert deep0.order.limit_price == ref - Decimal("2.0")  # 2.00 below
    assert deep0.arrival_ts == g0.arrival_ts            # same arrivals (delta does not touch the RNG)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_generator_accepts_custom_delta -v`
Expected: FAIL — `TypeError: generate_passive_orders() got an unexpected keyword argument 'delta'`.

- [ ] **Step 3: Write minimal implementation**

In `research/experiments/het_sweep.py`, change the signature (line ~100) and the placement line (~122). Replace:

```python
def generate_passive_orders(tape: Tape, side: OrderSide, lam_per_sec: float,
                            ttl_ms: int, seed) -> list[GeneratedOrder]:
```

with:

```python
def generate_passive_orders(tape: Tape, side: OrderSide, lam_per_sec: float,
                            ttl_ms: int, seed, delta: Decimal = DELTA) -> list[GeneratedOrder]:
```

and replace:

```python
        limit_price = ref_price - DELTA if side == OrderSide.BUY else ref_price + DELTA
```

with:

```python
        limit_price = ref_price - delta if side == OrderSide.BUY else ref_price + delta
```

(Also update the docstring line referencing "P1 passive placement" to note `delta` is the placement-depth phenomenon parameter.)

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_generator_accepts_custom_delta -v`
Expected: PASS.

- [ ] **Step 5: Verify S28A tests still pass (non-breaking)**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py -v`
Expected: all PASS (the 17 S28A tests + the new one). `delta` defaults to `DELTA`, so existing placement tests are unaffected.

- [ ] **Step 6: Commit**

```bash
git add research/experiments/het_sweep.py tests/test_het_sweep.py
git commit -m "feat: parametrize placement depth (delta) in generate_passive_orders

Non-breaking: delta defaults to module DELTA. Enables the S28B depth sweep.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Add S28B constants

**Files:**
- Modify: `research/experiments/het_sweep.py:34` (after `SIDE_SEED`)
- Test: `tests/test_het_sweep.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_het_sweep.py` (add the two names to the existing `from research.experiments.het_sweep import (...)` block as you write each task; this test imports them):

```python
def test_s28b_constants():
    from research.experiments.het_sweep import DELTA_GRID_TICKS, QUEUE_FIXED_S28B
    assert DELTA_GRID_TICKS == [1, 5, 20]
    assert QUEUE_FIXED_S28B == Decimal("5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_s28b_constants -v`
Expected: FAIL — `ImportError: cannot import name 'DELTA_GRID_TICKS'`.

- [ ] **Step 3: Write minimal implementation**

In `research/experiments/het_sweep.py`, immediately after the `SIDE_SEED = {...}` line (~34), add:

```python
# --- S28B depth sweep config (FROZEN by 2026-06-01-s28b-depth-sweep-design.md) ---
DELTA_GRID_TICKS  = [1, 5, 20]                     # swept placement depth; 5t = shared anchor with S28A
QUEUE_FIXED_S28B  = Decimal("5")                   # fixed queue_ahead (median through-volume scale; S28A anchor)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_s28b_constants -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research/experiments/het_sweep.py tests/test_het_sweep.py
git commit -m "feat: add S28B depth-sweep constants (DELTA_GRID_TICKS, QUEUE_FIXED_S28B)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `run_depth_sweep` driver

**Files:**
- Modify: `research/experiments/het_sweep.py` (append after `run_queue_ahead_sweep`)
- Test: `tests/test_het_sweep.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_het_sweep.py`:

```python
def test_depth_sweep_arrivals_paired_across_delta():
    """Same seed => identical arrival stream at every depth (paired design)."""
    tape = _big_tape()
    rows = run_depth_sweep(tape, [1, 20], q_fixed=Decimal("5"), ttl_ms=5000,
                           lam_per_sec=1/10, seed_experiment=28, side_seed=SIDE_SEED)
    a1 = sorted(r["arrival_ts"] for r in rows if r["delta_ticks"] == 1)
    a20 = sorted(r["arrival_ts"] for r in rows if r["delta_ticks"] == 20)
    assert len(a1) > 0
    assert a1 == a20
    assert {r["delta_ticks"] for r in rows} == {1, 20}


def test_depth_sweep_matches_manual_composition():
    """Driver wiring: run_depth_sweep([5]) == manual generate(delta=DELTA)+window+evaluate."""
    tape = _big_tape()
    q = Decimal("0")
    rows = run_depth_sweep(tape, [5], q_fixed=q, ttl_ms=5000,
                           lam_per_sec=1/10, seed_experiment=28, side_seed=SIDE_SEED)
    got = [(r["arrival_ts"], r["filled"], r["reached"]) for r in rows]

    delta = TICK * 5
    model = FillModelC(q)
    expected = []
    for side in (OrderSide.BUY, OrderSide.SELL):
        gen = generate_passive_orders(tape, side, 1/10, 5000, seed=[28, SIDE_SEED[side]], delta=delta)
        for g in gen:
            _, w = build_window(tape, g.ref_index, g.arrival_ts, 5000)
            r = model.evaluate(g.order, w, active_since_agg_trade_id=g.active_since)
            expected.append((g.arrival_ts, r.fully_filled, r.prints_consumed > 0))
    assert got == expected
```

Update the import block at the top of `tests/test_het_sweep.py` to include `run_depth_sweep`, `TICK`, `SIDE_SEED`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_depth_sweep_matches_manual_composition -v`
Expected: FAIL — `ImportError: cannot import name 'run_depth_sweep'`.

- [ ] **Step 3: Write minimal implementation**

In `research/experiments/het_sweep.py`, append after `run_queue_ahead_sweep` (before `summarize_sweep`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_depth_sweep_arrivals_paired_across_delta tests/test_het_sweep.py::test_depth_sweep_matches_manual_composition -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add research/experiments/het_sweep.py tests/test_het_sweep.py
git commit -m "feat: add run_depth_sweep driver (S28B, queue fixed, sweep delta)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `summarize_depth_sweep` (decomposition + Wilson CIs)

**Files:**
- Modify: `research/experiments/het_sweep.py` (append after `summarize_sweep`)
- Test: `tests/test_het_sweep.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_het_sweep.py`:

```python
def test_summarize_depth_sweep_decomposes_reach_and_conditional():
    rows = [
        {"delta_ticks": 5, "filled": True,  "reached": True,  "time_to_fill_ms": 100},
        {"delta_ticks": 5, "filled": False, "reached": True,  "time_to_fill_ms": None},  # reached, queue too deep
        {"delta_ticks": 5, "filled": False, "reached": False, "time_to_fill_ms": None},  # never reached
        {"delta_ticks": 5, "filled": True,  "reached": True,  "time_to_fill_ms": 300},
    ]
    summ = summarize_depth_sweep(rows, [5])
    s = summ[5]
    assert s["n"] == 4
    assert s["reach_rate"] == 0.75                       # 3 reached / 4
    assert s["cond_fill"] == pytest.approx(2 / 3)        # 2 fills / 3 reached
    assert s["fill_rate"] == 0.5                         # 2 fills / 4
    assert s["fill_rate"] == pytest.approx(s["reach_rate"] * s["cond_fill"])  # decomposition identity
    assert s["median_time_to_fill_ms"] == 200.0
    assert 0.0 <= s["reach_ci"][0] <= s["reach_ci"][1] <= 1.0
    assert 0.0 <= s["cond_fill_ci"][0] <= s["cond_fill_ci"][1] <= 1.0
    assert 0.0 <= s["fill_rate_ci"][0] <= s["fill_rate_ci"][1] <= 1.0


def test_summarize_depth_sweep_handles_zero_reach():
    rows = [{"delta_ticks": 20, "filled": False, "reached": False, "time_to_fill_ms": None}]
    s = summarize_depth_sweep(rows, [20])[20]
    assert s["reach_rate"] == 0.0
    assert s["cond_fill"] == 0.0          # guard: no reached orders -> 0, not ZeroDivisionError
    assert s["fill_rate"] == 0.0
    assert s["median_time_to_fill_ms"] is None
```

Update the import block to include `summarize_depth_sweep`.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_summarize_depth_sweep_decomposes_reach_and_conditional -v`
Expected: FAIL — `ImportError: cannot import name 'summarize_depth_sweep'`.

- [ ] **Step 3: Write minimal implementation**

In `research/experiments/het_sweep.py`, append after `summarize_sweep`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_summarize_depth_sweep_decomposes_reach_and_conditional tests/test_het_sweep.py::test_summarize_depth_sweep_handles_zero_reach -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add research/experiments/het_sweep.py tests/test_het_sweep.py
git commit -m "feat: add summarize_depth_sweep (reach x cond_fill = fill_rate, Wilson CIs)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `reach_monotonicity_ok` sanity helper

**Files:**
- Modify: `research/experiments/het_sweep.py` (append after `summarize_depth_sweep`)
- Test: `tests/test_het_sweep.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_het_sweep.py`:

```python
def test_reach_monotonicity_ok_flags_violation():
    good = {1: {"reach_rate": 0.9}, 5: {"reach_rate": 0.7}, 20: {"reach_rate": 0.3}}
    bad  = {1: {"reach_rate": 0.7}, 5: {"reach_rate": 0.9}, 20: {"reach_rate": 0.3}}  # 5 > 1 bump
    assert reach_monotonicity_ok(good, [1, 5, 20]) is True
    assert reach_monotonicity_ok(bad, [1, 5, 20]) is False
    assert reach_monotonicity_ok({5: {"reach_rate": 0.5}}, [5]) is True  # single point trivially ok
```

Update the import block to include `reach_monotonicity_ok`.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_reach_monotonicity_ok_flags_violation -v`
Expected: FAIL — `ImportError: cannot import name 'reach_monotonicity_ok'`.

- [ ] **Step 3: Write minimal implementation**

In `research/experiments/het_sweep.py`, append after `summarize_depth_sweep`:

```python
def reach_monotonicity_ok(summary: dict[int, dict], delta_ticks_list: list[int]) -> bool:
    """
    Sanity expectation (NOT a PASS/FAIL acceptance criterion): a deeper level is harder to
    reach, so reach_rate should be non-increasing in Δ. False => raise a visible alarm
    (a bug, or an extremely interesting finding). Reported, not gating.
    """
    reaches = [summary[d]["reach_rate"] for d in delta_ticks_list]
    return all(reaches[i] >= reaches[i + 1] for i in range(len(reaches) - 1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_het_sweep.py::test_reach_monotonicity_ok_flags_violation -v`
Expected: PASS.

- [ ] **Step 5: Run the FULL suite (no regressions)**

Run: `venv\Scripts\python.exe -m pytest -q`
Expected: PASS — 286 prior + 7 new S28B tests = 293 passed, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add research/experiments/het_sweep.py tests/test_het_sweep.py
git commit -m "feat: add reach_monotonicity_ok sanity helper for S28B

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Author the `S28B_depth_sweep.ipynb` notebook (thin)

**Files:**
- Create: `research/experiments/S28B_depth_sweep.ipynb`

The notebook only imports `het_sweep` and orchestrates. Bryan runs it; do NOT execute it. Build it with valid nbformat 4.5 JSON. Use these exact cells (in order).

- [ ] **Step 1: Create the notebook with the cells below**

**Cell 1 (markdown):**
```
# S28B — Depth Sensitivity Sweep (execution sensitivity to placement depth Δ)
Single axis: Δ ∈ {1,5,20} ticks. queue_ahead, TTL, qty, lambda frozen. Layer A only (no PnL/economics).
Spec: OBSIDIAN/docs/superpowers/specs/2026-06-01-s28b-depth-sweep-design.md
```

**Cell 2 (code) — repo-root bootstrap (identical to the S28A notebook):**
```python
# --- repo-root bootstrap ---
import sys, os
from pathlib import Path

_root = Path.cwd()
while not (_root / "research" / "__init__.py").exists() and _root != _root.parent:
    _root = _root.parent
if not (_root / "research" / "__init__.py").exists():
    raise RuntimeError("repo root not found: launch Jupyter from within the TRADING-BOT repo")
os.chdir(_root)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
print("repo root:", _root)
```

**Cell 3 (code) — imports, config, SPEC_HASH:**
```python
import json, hashlib
from pathlib import Path
from decimal import Decimal
import numpy as np, matplotlib.pyplot as plt
from research.experiments.het_sweep import (
    load_canonical_tape, run_depth_sweep, summarize_depth_sweep, reach_monotonicity_ok,
    DELTA_GRID_TICKS, QUEUE_FIXED_S28B, TTL_FROZEN_MS, LAMBDA_PER_SEC, SEED_EXPERIMENT, SIDE_SEED,
)

TAPE_PATH = "data/raw/BTCUSDT_AGGTRADES.csv"          # canonical tape (gate PASS, 41,544,041 rows)
OUT = Path("research/experiments/outputs"); OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {"delta_ticks": DELTA_GRID_TICKS, "queue_fixed": str(QUEUE_FIXED_S28B),
          "order_qty": "0.01", "lambda_per_sec": LAMBDA_PER_SEC, "seed": SEED_EXPERIMENT,
          "ttl_ms": TTL_FROZEN_MS}
SPEC_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]
print("SPEC_HASH:", SPEC_HASH)
```

**Cell 4 (code) — run sweep + print decomposed table:**
```python
tape = load_canonical_tape(TAPE_PATH)
rows = run_depth_sweep(tape, DELTA_GRID_TICKS, QUEUE_FIXED_S28B, TTL_FROZEN_MS)
summary = summarize_depth_sweep(rows, DELTA_GRID_TICKS)

print(f"queue_ahead fixed = {QUEUE_FIXED_S28B} BTC   TTL = {TTL_FROZEN_MS/1000:.0f}s   orders/Δ ~ {len(rows)//len(DELTA_GRID_TICKS):,}")
print(f"{'Δ(t)':>5} {'reach':>7} {'cond_fill':>10} {'fill':>7} {'med_ttf(ms)':>12}")
for d in DELTA_GRID_TICKS:
    s = summary[d]
    print(f"{d:>5} {s['reach_rate']:>7.4f} {s['cond_fill']:>10.4f} {s['fill_rate']:>7.4f} {str(s['median_time_to_fill_ms']):>12}")
```

**Cell 5 (code) — cross-experiment replication (triple) + monotonicity alarm:**
```python
# (1) Replication invariant: the (Δ=5t, q=5) cell must reproduce S28A's TRIPLE (within tolerance).
S28A_REF = {"reach": 0.753, "fill": 0.5613, "ttf_ms": 18626.0}
TOL = {"reach": 0.01, "fill": 0.01, "ttf_ms": 1000.0}
s5 = summary[5]
checks = {
    "reach": abs(s5["reach_rate"] - S28A_REF["reach"]) <= TOL["reach"],
    "fill":  abs(s5["fill_rate"] - S28A_REF["fill"])   <= TOL["fill"],
    "ttf_ms": (s5["median_time_to_fill_ms"] is not None
               and abs(s5["median_time_to_fill_ms"] - S28A_REF["ttf_ms"]) <= TOL["ttf_ms"]),
}
print("REPLICATION (Δ=5t, q=5) vs S28A:", checks)
assert all(checks.values()), f"REPLICATION FAILED — likely a bug in per-Δ regeneration: {checks}"

# (2) reach monotonicity — reported alarm, NOT a PASS/FAIL gate.
if reach_monotonicity_ok(summary, DELTA_GRID_TICKS):
    print("SANITY ok: reach(Δ) is non-increasing in depth.")
else:
    print("⚠️ ALARM: reach(Δ) is NOT monotone non-increasing — bug OR an extremely interesting finding. Investigate.")
```

**Cell 6 (code) — plot reach / cond_fill / fill vs Δ:**
```python
ds = DELTA_GRID_TICKS
reach = [summary[d]["reach_rate"] for d in ds]
cond  = [summary[d]["cond_fill"] for d in ds]
fill  = [summary[d]["fill_rate"] for d in ds]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ds, reach, marker="o", label="reach(Δ)  — does price arrive?")
ax.plot(ds, cond,  marker="s", label="cond_fill(Δ|reached) — queue survival")
ax.plot(ds, fill,  marker="^", label="fill_rate(Δ) = reach × cond_fill")
ax.set_xlabel("Δ (ticks)"); ax.set_ylabel("rate"); ax.set_ylim(0, 1)
ax.set_title(f"S28B depth sweep (queue_ahead={QUEUE_FIXED_S28B} BTC, TTL={TTL_FROZEN_MS/1000:.0f}s)")
ax.legend(); ax.grid(alpha=0.3)
fig.savefig(OUT / f"S28B_depth_curves_{SPEC_HASH}.png", dpi=120, bbox_inches="tight")
plt.show()
```

**Cell 7 (code) — write FROZEN artifacts:**
```python
import csv
serial = {str(d): summary[d] for d in DELTA_GRID_TICKS}
for v in serial.values():
    for k in ("reach_ci", "cond_fill_ci", "fill_rate_ci"):
        v[k] = list(v[k])
(OUT / f"S28B_summary_{SPEC_HASH}.json").write_text(
    json.dumps({"config": CONFIG, "spec_hash": SPEC_HASH, "summary": serial}, indent=2), encoding="utf-8")

with open(OUT / f"S28B_rows_{SPEC_HASH}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["delta_ticks", "arrival_ts", "filled", "time_to_fill_ms", "reached",
                "queue_consumed", "queue_remaining", "through_volume"])
    for r in rows:
        w.writerow([r["delta_ticks"], r["arrival_ts"], r["filled"], r["time_to_fill_ms"],
                    r["reached"], str(r["queue_consumed"]), str(r["queue_remaining"]), str(r["through_volume"])])
print("FROZEN artifacts written with SPEC_HASH", SPEC_HASH)
```

**Cell 8 (markdown):**
```
Read the table directly (the verdict — elasticity is secondary color):
- If fill_rate(Δ) drops from Δ=5→20 comparably-or-more than fill dropped from queue=5→50 in S28A (0.561→0.140), depth is at least as material as the queue.
- Decomposition localizes the Hidden Execution Tax: reach↓ = "never arrives" (depth axis); cond_fill↓ = "queue too deep" (queue axis).
After S28B: EVALUATION PAUSE before S28C (A+B may already cover a v1 operational model). NO PnL/economics here (Layer B, deferred).
```

- [ ] **Step 2: Validate the notebook is well-formed nbformat**

Run: `venv\Scripts\python.exe -c "import nbformat; nbformat.validate(nbformat.read(open('research/experiments/S28B_depth_sweep.ipynb'), as_version=4)); print('nbformat OK')"`
Expected: `nbformat OK` (no validation error).

- [ ] **Step 3: Commit the authored notebook**

```bash
git add research/experiments/S28B_depth_sweep.ipynb
git commit -m "feat: author S28B depth-sweep notebook (thin; Bryan runs it)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Final verification + handoff to Bryan

**Files:** none (verification only)

- [ ] **Step 1: Full suite green**

Run: `venv\Scripts\python.exe -m pytest -q`
Expected: 293 passed (286 prior + 7 new), 0 failures.

- [ ] **Step 2: Confirm FillModelC untouched**

Run: `git diff --stat s28-fillmodelc-baseline -- live/fill_model_c.py`
Expected: no output (no changes to `live/fill_model_c.py` since the baseline tag).

- [ ] **Step 3: Hand off to Bryan**

Tell Bryan: the S28B framework is committed and green. He should run `S28B_depth_sweep.ipynb`, confirm `SPEC_HASH` printed in Cell 3, confirm the Cell 5 replication assertion PASSES (the (Δ=5t, q=5) triple reproduces S28A), watch for the reach-monotonicity alarm, then bring back `outputs/S28B_summary_<hash>.json` + the curves PNG for analysis. The sweep artifacts get committed after he runs it (as with S28A).

---

## Self-Review

**1. Spec coverage:**
- Swept axis Δ∈{1,5,20}, single → Task 2 (`DELTA_GRID_TICKS`), Task 3 (driver loops it). ✓
- Frozen queue=5, TTL=60s, qty, λ, seed → Task 2 (`QUEUE_FIXED_S28B`), Task 3 (defaults from module). ✓
- Paired arrivals across Δ → Task 3 (`test_depth_sweep_arrivals_paired_across_delta`). ✓
- Metrics decomposed reach × cond_fill = fill_rate + median_ttf + Wilson CI → Task 4. ✓
- Verdict direct from table; elasticity secondary → notebook Cell 4 (table) + Cell 8 (reading); no elasticity computed in code (secondary/manual). ✓
- Strengthened replication triple (reach 0.753 / fill 0.561 / ttf 18.6s) → notebook Cell 5 assertion. ✓
- reach monotonicity sanity (reported, not PASS/FAIL) → Task 5 + notebook Cell 5 alarm. ✓
- FillModelC unmodified → no task touches `live/fill_model_c.py`; Task 7 Step 2 verifies. ✓
- Deterministic SPEC_HASH, FROZEN artifacts, Bryan runs notebook → Task 6 Cells 3/7. ✓
- Δ parametrized non-breaking → Task 1 (default `DELTA`, S28A tests still pass). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has expected output. ✓

**3. Type consistency:** `delta: Decimal`; `delta_ticks: int` (grid is ints, converted via `TICK * dticks`); row key `"delta_ticks"` used identically in `run_depth_sweep`, `summarize_depth_sweep`, tests, and notebook; `summarize_depth_sweep` returns `dict[int, dict]` keyed by int Δ, consumed as `summary[d]` (int) in the notebook and `reach_monotonicity_ok`. Consistent. ✓
