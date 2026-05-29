"""
Create 05_session_breakout_fade.ipynb from 04_session_breakout.ipynb.
Single change: swap signal_long/signal_short (continuation -> fade).
Everything else frozen.
"""
import json
import copy
import sys

sys.stdout.reconfigure(encoding="utf-8")

SRC = "C:/Users/Lenovo/Documents/TRADING-BOT/research/experiments/04_session_breakout.ipynb"
DST = "C:/Users/Lenovo/Documents/TRADING-BOT/research/experiments/05_session_breakout_fade.ipynb"


def patch(cell, old: str, new: str) -> bool:
    src = cell["source"]
    if isinstance(src, list):
        joined = "".join(src)
        patched = joined.replace(old, new)
        if patched == joined:
            return False
        cell["source"] = patched.splitlines(keepends=True)
        return True
    if old in src:
        cell["source"] = src.replace(old, new)
        return True
    return False


with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)

# ── Cell 0: description / spec_version ────────────────────────────────────────
c0 = nb["cells"][0]
patch(c0,
      "# ── FROZEN_PARAMS — Exp 04: Session Breakout Continuation",
      "# ── FROZEN_PARAMS — Exp 05A: Session Breakout Fade (anti-continuation)")
patch(c0,
      "# Hypothesis: Asia range (00-08 UTC) defines liquidity. London break (08-12 UTC)\n"
      "# with expansion candle signals institutional displacement. Retest of the Asia\n"
      "# level at session open (12-16 UTC) within 60min = continuation entry.\n"
      "#\n"
      "# These exits are NOT inherited from Exp 01-03. Calibrated fresh for M5 geometry.\n"
      "# SL=0.15% chosen to survive M5 noise (Exp03 avg_MAE=0.164%). TP=0.35% at R:R≈2.3.\n"
      "# Break-even WR ≈ 46% — achievable vs stochastic family's 30-35%.",
      "# Hypothesis: Asia range (00-08 UTC) -> London break (08-12 UTC, expansion candle)\n"
      "# -> NY retest of Asia level (12-16 UTC, max 60min) = EXHAUSTION / REVERSAL entry.\n"
      "#\n"
      "# SINGLE CHANGE vs Exp04: signal direction inverted (fade, not continuation).\n"
      "# Exp04 WR=28.2% (below random). Falsification test: does anti-edge flip sign?\n"
      "# TP/SL/windows/costs FROZEN — isolate the direction hypothesis.\n"
      "# Break-even WR = 46% (unchanged).")
patch(c0, '"spec_version":    "S6-r1"', '"spec_version":    "S7-r1"')
patch(c0, '"experiment":      "04-session-breakout"',
      '"experiment":      "05-session-breakout-fade"')
patch(c0, 'print("Cell 1 OK — Exp 04: Session Breakout Continuation")',
      'print("Cell 1 OK — Exp 05A: Session Breakout Fade (direction inverted vs Exp04)")')
patch(c0,
      'print(f"  Break-even WR = {be_wr*100:.1f}%  (vs stochastic family 30-35%)")',
      'print(f"  Break-even WR = {be_wr*100:.1f}%  (Exp04 fade hypothesis — single direction change)")')

# ── Cell 3: asia range png ────────────────────────────────────────────────────
patch(nb["cells"][3], "'04_asia_range.png'", "'05_asia_range.png'")
patch(nb["cells"][3], '"Saved: 04_asia_range.png"', '"Saved: 05_asia_range.png"')

# ── Cell 5: THE KEY CHANGE — swap signal_long / signal_short ─────────────────
c5 = nb["cells"][5]
patch(c5,
      "# ── Retest conditions ─────────────────────────────────────────────────────────\n"
      "# LONG retest: price pulls back to Asia high (support test) after LONG break.\n"
      "#   low touched the level (within tolerance above) + close held above.\n"
      "# SHORT retest: price rallies back to Asia low (resistance test) after SHORT break.\n"
      "#   high touched the level (within tolerance below) + close held below.",
      "# ── Retest conditions (same detection logic as Exp04) ────────────────────────\n"
      "# retest_long  = long break day  + price touched Asia HIGH (Exp04 continuation long setup)\n"
      "# retest_short = short break day + price touched Asia LOW  (Exp04 continuation short setup)\n"
      "# Exp05A FADE: long_break+retest_high -> SHORT; short_break+retest_low -> LONG")
# THE ACTUAL SIGNAL SWAP
patch(c5,
      "df['signal_long']  = df.groupby('date')['retest_long'].shift(1).fillna(False)\n"
      "df['signal_short'] = df.groupby('date')['retest_short'].shift(1).fillna(False)",
      "# EXP05A: INVERT DIRECTION — fade the breakout\n"
      "# LONG after a SHORT break retest (fade down-move, expect reversal up)\n"
      "# SHORT after a LONG break retest (fade up-move, expect reversal down)\n"
      "df['signal_long']  = df.groupby('date')['retest_short'].shift(1).fillna(False)\n"
      "df['signal_short'] = df.groupby('date')['retest_long'].shift(1).fillna(False)")
patch(c5, 'print("Cell 6 OK")', 'print("Cell 6 OK — FADE signals: signal_long<-retest_short, signal_short<-retest_long")')

# ── Cell 9: equity csv ────────────────────────────────────────────────────────
patch(nb["cells"][9], "OUTPUT_DIR / '04_equity.csv'", "OUTPUT_DIR / '05_equity.csv'")
patch(nb["cells"][9], '"\\nSaved: outputs/04_equity.csv"', '"\\nSaved: outputs/05_equity.csv"')

# ── Cell 12: baseline csv ─────────────────────────────────────────────────────
patch(nb["cells"][12], "OUTPUT_DIR / '04_baseline_dist.csv'", "OUTPUT_DIR / '05_baseline_dist.csv'")
patch(nb["cells"][12], '"Saved: outputs/04_baseline_dist.csv"', '"Saved: outputs/05_baseline_dist.csv"')

# ── Cell 14: viz title + png ──────────────────────────────────────────────────
patch(nb["cells"][14],
      "ax1.set_title('Equity Curve — Exp 04 Session Breakout Continuation', fontweight='bold')",
      "ax1.set_title('Equity Curve — Exp 05A Session Breakout Fade (anti-continuation)', fontweight='bold')")
patch(nb["cells"][14], "OUTPUT_DIR / '04_visualizations.png'", "OUTPUT_DIR / '05_visualizations.png'")
patch(nb["cells"][14], '"Saved: outputs/04_visualizations.png"', '"Saved: outputs/05_visualizations.png"')

# ── Cell 15: verdict ──────────────────────────────────────────────────────────
c15 = nb["cells"][15]
patch(c15,
      'print(f"  EXPERIMENT 04 — SESSION BREAKOUT CONTINUATION")',
      'print(f"  EXPERIMENT 05A — SESSION BREAKOUT FADE (anti-continuation)")')
# Progression table: shrink to 3-col comparison
patch(c15,
      "print(f\"    {'Metric':<26} {'Exp01 M1':>9} {'Exp02A M1':>9} {'Exp03 M5':>9} {'Exp04 M5':>9}\")",
      "print(f\"    {'Metric':<26} {'Exp03 M5':>9} {'Exp04 M5':>9} {'Exp05A M5':>9}\")")
patch(c15,
      "    metrics_hist = [\n"
      "        ('E_net %',     -0.0743, -0.0695, -0.0628, compute_expectancy(holdout_pnl)*100),\n"
      "        ('Win Rate',     0.29,    0.3167,  0.3517,  (holdout_pnl > 0).mean()),\n"
      "        ('avg_MAE %',   0.0900,  0.1017,  0.1644,  h['mae_pct'].mean()*100),\n"
      "        ('avg_MFE %',   0.1100,  0.1376,  0.1849,  h['mfe_pct'].mean()*100),\n"
      "        ('avg_hold(c)', 5.8,     5.8,     1.3,     h['holding_candles'].mean()),\n"
      "        ('n_trades',    2169,    60,      290,     n_hold),\n"
      "    ]",
      "    metrics_hist = [\n"
      "        ('E_net %',     -0.0628, -0.0924, compute_expectancy(holdout_pnl)*100),\n"
      "        ('Win Rate',     0.3517,  0.2823,  (holdout_pnl > 0).mean()),\n"
      "        ('avg_MAE %',   0.1644,  0.1934,  h['mae_pct'].mean()*100),\n"
      "        ('avg_MFE %',   0.1849,  0.2087,  h['mfe_pct'].mean()*100),\n"
      "        ('avg_hold(c)', 1.3,     5.181,   h['holding_candles'].mean()),\n"
      "        ('n_trades',    290,     248,     n_hold),\n"
      "    ]")
patch(c15,
      "    for name, e01, e02, e03, e04 in metrics_hist:\n"
      "        print(f\"    {name:<26} {e01:>9.3f} {e02:>9.3f} {e03:>9.3f} {e04:>9.3f}\")",
      "    for name, e03, e04, e05 in metrics_hist:\n"
      "        print(f\"    {name:<26} {e03:>9.3f} {e04:>9.3f} {e05:>9.3f}\")")
# Artifact filenames
for fn in ["04_trades.csv", "04_equity.csv", "04_metrics.json",
           "04_baseline_dist.csv", "04_asia_range.png", "04_visualizations.png"]:
    patch(c15, fn, fn.replace("04_", "05_"))
patch(c15, "OUTPUT_DIR / '04_trades.csv'", "OUTPUT_DIR / '05_trades.csv'")
patch(c15, "OUTPUT_DIR / '04_metrics.json'", "OUTPUT_DIR / '05_metrics.json'")

# ── Clear all outputs ─────────────────────────────────────────────────────────
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("OK:", DST)
