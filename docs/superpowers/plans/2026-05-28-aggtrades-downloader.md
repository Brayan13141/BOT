# aggTrades Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Independence:** This plan is fully independent of `2026-05-28-fill-model-b.md`. Execute in parallel or in any order.

**Goal:** Download Binance USDT-M perpetual aggTrades historical data for BTCUSDT and ETHUSDT, store as CSV, and validate the dataset for use in FillModelC.

**Architecture:** A single script (`scripts/download_aggtrades.py`) pages through the Binance `/fapi/v1/aggTrades` REST endpoint using `fromId` pagination. The initial request uses `startTime` to anchor; all subsequent pages use `fromId = last_agg_trade_id + 1`. This correctly handles BTCUSDT's density of 20,000–100,000+ aggTrades per hour, where a single `limit=1000` request per time window would lose 95%+ of the data. Resume support tracks the last `agg_trade_id`, not just the last timestamp. A validation script (`scripts/validate_aggtrades.py`) checks schema, date coverage, and side distribution. No new tests — the output is validated by running the scripts.

**Tech Stack:** Python 3.12, `httpx` (already in project for scraping), `csv`, `pathlib`, Binance USDT-M Futures REST API (no auth required for public market data).

---

## Binance aggTrades API Reference

**Endpoint:** `GET https://fapi.binance.com/fapi/v1/aggTrades`

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `symbol` | str | e.g. `BTCUSDT` |
| `startTime` | int | Unix ms (inclusive) |
| `endTime` | int | Unix ms (exclusive) |
| `limit` | int | Max 1000 (default 500) |

**Response (one record):**

```json
{
  "a": 26129,          // agg_trade_id
  "p": "0.01633102",   // price
  "q": "4.70443515",   // qty (base asset)
  "f": 27781,          // first_trade_id
  "l": 27781,          // last_trade_id
  "T": 1498793709153,  // timestamp_ms
  "m": true,           // is_buyer_maker (true = sell aggressor = bearish)
  "M": true            // best_price_match (ignore)
}
```

**`is_buyer_maker` convention:**
- `true` → buyer is market maker → SELL order was aggressive → bearish taker flow
- `false` → buyer is market taker → BUY order was aggressive → bullish taker flow

**CSV `side` column mapping:**
- `is_buyer_maker=true` → `side="SELL"` (taker was selling)
- `is_buyer_maker=false` → `side="BUY"` (taker was buying)

**Rate limits:** 2400 weight/minute. Each aggTrades request costs 20 weight. Max ~120 requests/minute → safe at 1 request/second with sleep.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/download_aggtrades.py` | Create | Page through Binance API, write CSV, support resume |
| `scripts/validate_aggtrades.py` | Create | Check schema, coverage, side distribution |
| `data/raw/BTCUSDT_AGGTRADES.csv` | Output | Downloaded data (not committed to git) |
| `data/raw/ETHUSDT_AGGTRADES.csv` | Output | Downloaded data (not committed to git) |

**CSV schema:**
```
timestamp_ms,price,qty,side,agg_trade_id
1498793709153,65123.40,0.002,BUY,26129
```

---

## Task 1: Download script

**Files:**
- Create: `scripts/download_aggtrades.py`

- [ ] **Step 1: Create `scripts/download_aggtrades.py`**

```python
"""
Download Binance USDT-M perpetual aggTrades historical data.

Usage:
    python scripts/download_aggtrades.py --symbol BTCUSDT --start 2024-01-01 --end 2025-01-01
    python scripts/download_aggtrades.py --symbol ETHUSDT --start 2024-01-01 --end 2025-01-01

Output: data/raw/{SYMBOL}_AGGTRADES.csv
    Columns: timestamp_ms, price, qty, side, agg_trade_id

Pagination: uses fromId after the first startTime anchor request.
    BTCUSDT can exceed 100,000 aggTrades/hour — a single limit=1000
    request per time window would lose 99%+ of the data.

Resume: tracks last agg_trade_id. Resumes from that ID + 1 on next run.

Rate limit: 1 request/second (well within Binance 2400 weight/min limit).
"""
import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/aggTrades"
LIMIT     = 1000            # max records per request
SLEEP_S   = 1.0             # seconds between requests
DATA_DIR  = Path(__file__).parent.parent / "data" / "raw"

FIELDNAMES = ["timestamp_ms", "price", "qty", "side", "agg_trade_id"]


def _parse_ts(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to Unix milliseconds (UTC midnight)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _last_record_in_file(path: Path) -> tuple[int | None, int | None]:
    """Return (last_timestamp_ms, last_agg_trade_id) from CSV, or (None, None) if absent/empty."""
    if not path.exists():
        return None, None
    last_ts = last_id = None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_ts = int(row["timestamp_ms"])
            last_id = int(row["agg_trade_id"])
    return last_ts, last_id


def _fetch_batch(
    client: httpx.Client,
    symbol: str,
    *,
    from_id: int | None = None,
    start_ms: int | None = None,
) -> list[dict]:
    """
    Fetch up to LIMIT aggTrades.
    Use from_id for all pages after the first request.
    Use start_ms only for the initial anchor (first request, no resume).
    """
    params: dict = {"symbol": symbol, "limit": LIMIT}
    if from_id is not None:
        params["fromId"] = from_id
    else:
        assert start_ms is not None, "Either from_id or start_ms required"
        params["startTime"] = start_ms
    resp = client.get(BASE_URL + ENDPOINT, params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def _to_row(record: dict) -> dict:
    side = "SELL" if record["m"] else "BUY"
    return {
        "timestamp_ms":  record["T"],
        "price":         record["p"],
        "qty":           record["q"],
        "side":          side,
        "agg_trade_id":  record["a"],
    }


def download(symbol: str, start_ms: int, end_ms: int, out_path: Path) -> None:
    _, resume_id = _last_record_in_file(out_path)
    if resume_id is not None:
        print(f"Resuming from agg_trade_id {resume_id + 1} ({out_path.name})")

    mode          = "a" if resume_id is not None else "w"
    written       = 0
    next_from_id: int | None = (resume_id + 1) if resume_id is not None else None

    with (
        open(out_path, mode, newline="", encoding="utf-8") as f,
        httpx.Client() as client,
    ):
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()

        while True:
            if next_from_id is not None:
                batch = _fetch_batch(client, symbol, from_id=next_from_id)
            else:
                # First request only: anchor by startTime
                batch = _fetch_batch(client, symbol, start_ms=start_ms)

            if not batch:
                break

            done = False
            for rec in batch:
                if rec["T"] >= end_ms:
                    done = True
                    break
                writer.writerow(_to_row(rec))
                written += 1

            last = batch[-1]
            next_from_id = last["a"] + 1
            print(
                f"  {symbol} | "
                f"{datetime.fromtimestamp(last['T'] / 1000, tz=timezone.utc).date()} | "
                f"{written:,} trades written"
            )

            if done or len(batch) < LIMIT:
                break

            time.sleep(SLEEP_S)

    print(f"Done. {written:,} trades written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance aggTrades historical data")
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument("--start",  required=True, help="Start date YYYY-MM-DD (UTC)")
    parser.add_argument("--end",    required=True, help="End date YYYY-MM-DD (UTC, exclusive)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{args.symbol}_AGGTRADES.csv"

    start_ms = _parse_ts(args.start)
    end_ms   = _parse_ts(args.end)

    print(f"Downloading {args.symbol} aggTrades: {args.start} → {args.end}")
    print(f"Output: {out_path}")
    download(args.symbol, start_ms, end_ms, out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses without errors**

```
.\venv\Scripts\python.exe scripts/download_aggtrades.py --help
```

Expected output:
```
usage: download_aggtrades.py [-h] --symbol SYMBOL --start START --end END
...
```

- [ ] **Step 3: Test with a 1-hour sample (BTCUSDT, recent date)**

```
.\venv\Scripts\python.exe scripts/download_aggtrades.py --symbol BTCUSDT --start 2025-01-01 --end 2025-01-02
```

Expected: progress lines printed, `data/raw/BTCUSDT_AGGTRADES.csv` created.

Verify first and last lines:
```
Get-Content data/raw/BTCUSDT_AGGTRADES.csv | Select-Object -First 3
Get-Content data/raw/BTCUSDT_AGGTRADES.csv | Select-Object -Last 2
```

Expected: header + rows with 5 fields each.

- [ ] **Step 4: Test resume (run same command again)**

Run the same command as Step 3 again. Expected output: `Resuming from timestamp ...` and no duplicate rows added (script skips to after last recorded timestamp).

- [ ] **Step 5: Commit**

```
git add scripts/download_aggtrades.py
git commit -m "feat: Binance aggTrades historical downloader with resume support"
```

---

## Task 2: Validation script

**Files:**
- Create: `scripts/validate_aggtrades.py`

- [ ] **Step 1: Create `scripts/validate_aggtrades.py`**

```python
"""
Validate a downloaded aggTrades CSV file.

Usage:
    python scripts/validate_aggtrades.py data/raw/BTCUSDT_AGGTRADES.csv

Checks:
    1. Schema: correct column names and types
    2. agg_trade_id strictly increasing (primary monotonicity guarantee)
    3. Timestamp non-decreasing (same-ms timestamps are valid — multiple trades
       can share a millisecond; only regression is an error)
    4. Side distribution: BUY and SELL both present (sanity check)
    5. Price and qty are positive Decimal-parseable values
    6. Date coverage summary: first/last timestamp, total trade count
"""
import csv
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


FIELDNAMES = {"timestamp_ms", "price", "qty", "side", "agg_trade_id"}
VALID_SIDES = {"BUY", "SELL"}


def validate(path: Path) -> bool:
    ok = True

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Schema check
        if set(reader.fieldnames or []) != FIELDNAMES:
            print(f"[FAIL] Schema mismatch. Got: {reader.fieldnames}")
            return False
        print("[PASS] Schema OK")

        count               = 0
        buy_count           = 0
        sell_count          = 0
        last_ts             = -1
        last_agg_id         = -1
        first_ts: int | None = None

        for i, row in enumerate(reader, start=2):   # line 1 = header
            # timestamp_ms: integer
            try:
                ts = int(row["timestamp_ms"])
            except ValueError:
                print(f"[FAIL] Row {i}: timestamp_ms not int: {row['timestamp_ms']!r}")
                ok = False
                continue

            # Timestamp regression check (same ms is valid — multiple trades per ms are normal)
            if ts < last_ts:
                print(f"[FAIL] Row {i}: timestamp decreased ({ts} < {last_ts})")
                ok = False
            last_ts = ts

            if first_ts is None:
                first_ts = ts

            # agg_trade_id: strictly increasing — primary monotonicity guarantee
            try:
                agg_id = int(row["agg_trade_id"])
            except ValueError:
                print(f"[FAIL] Row {i}: agg_trade_id not int: {row['agg_trade_id']!r}")
                ok = False
                agg_id = last_agg_id  # skip monotonicity check for this row
            else:
                if agg_id <= last_agg_id:
                    print(
                        f"[FAIL] Row {i}: agg_trade_id not strictly increasing "
                        f"({agg_id} <= {last_agg_id})"
                    )
                    ok = False
                last_agg_id = agg_id

            # price and qty: positive Decimal
            for field in ("price", "qty"):
                try:
                    val = Decimal(row[field])
                    if val <= 0:
                        print(f"[FAIL] Row {i}: {field}={val} is not positive")
                        ok = False
                except InvalidOperation:
                    print(f"[FAIL] Row {i}: {field} not Decimal: {row[field]!r}")
                    ok = False

            # side
            if row["side"] not in VALID_SIDES:
                print(f"[FAIL] Row {i}: side={row['side']!r} not in {VALID_SIDES}")
                ok = False
            elif row["side"] == "BUY":
                buy_count += 1
            else:
                sell_count += 1

            count += 1

    if ok:
        print("[PASS] agg_trade_id strictly increasing")
        print("[PASS] Timestamps non-decreasing (same-ms allowed)")
        print("[PASS] Prices and quantities positive")

    if buy_count == 0 or sell_count == 0:
        print(f"[WARN] Unexpected side distribution: BUY={buy_count}, SELL={sell_count}")
        ok = False
    else:
        buy_pct  = 100 * buy_count  / count
        sell_pct = 100 * sell_count / count
        print(f"[PASS] Side distribution: BUY={buy_count:,} ({buy_pct:.1f}%), SELL={sell_count:,} ({sell_pct:.1f}%)")

    if first_ts and last_ts > 0:
        first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
        last_dt  = datetime.fromtimestamp(last_ts  / 1000, tz=timezone.utc)
        print(f"\nCoverage: {first_dt.date()} → {last_dt.date()}")
        print(f"Total trades: {count:,}")

    return ok


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python validate_aggtrades.py <path_to_csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    passed = validate(path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validator on the sample downloaded in Task 1**

```
.\venv\Scripts\python.exe scripts/validate_aggtrades.py data/raw/BTCUSDT_AGGTRADES.csv
```

Expected output (all PASS):
```
[PASS] Schema OK
[PASS] Timestamps monotonically increasing
[PASS] Prices and quantities positive
[PASS] Side distribution: BUY=... (...%), SELL=... (...%)

Coverage: 2025-01-01 → 2025-01-01
Total trades: ...
```

- [ ] **Step 3: Commit**

```
git add scripts/validate_aggtrades.py
git commit -m "feat: aggTrades CSV validator — schema, monotonicity, side distribution"
```

---

## Task 3: Download full historical dataset

**Files:** (no code changes — running the scripts)

- [ ] **Step 1: Download BTCUSDT (2023-01-01 → 2025-06-01)**

```
.\venv\Scripts\python.exe scripts/download_aggtrades.py --symbol BTCUSDT --start 2023-01-01 --end 2025-06-01
```

Expected: runs for ~15–30 minutes. Progress printed per hour window. Output: `data/raw/BTCUSDT_AGGTRADES.csv`.

- [ ] **Step 2: Download ETHUSDT (2023-01-01 → 2025-06-01)**

```
.\venv\Scripts\python.exe scripts/download_aggtrades.py --symbol ETHUSDT --start 2023-01-01 --end 2025-06-01
```

- [ ] **Step 3: Validate both files**

```
.\venv\Scripts\python.exe scripts/validate_aggtrades.py data/raw/BTCUSDT_AGGTRADES.csv
.\venv\Scripts\python.exe scripts/validate_aggtrades.py data/raw/ETHUSDT_AGGTRADES.csv
```

Expected: all PASS for both files.

- [ ] **Step 4: Verify `.gitignore` excludes raw data**

```
Get-Content .gitignore | Select-String "AGGTRADES\|data/"
```

If `data/raw/` is not excluded, add it:
```
Add-Content .gitignore "`ndata/raw/"
git add .gitignore
git commit -m "chore: exclude data/raw/ from git"
```

---

## Self-Review

### Spec coverage

| Requirement | Covered by |
|-------------|-----------|
| Download BTCUSDT aggTrades | Task 3 Step 1 |
| Download ETHUSDT aggTrades | Task 3 Step 2 |
| CSV schema: `timestamp_ms,price,qty,side,agg_trade_id` | Task 1 `_to_row()`, `FIELDNAMES` |
| `is_buyer_maker → side` mapping | Task 1 `_to_row()` |
| fromId pagination (handles 100k+ trades/hour) | Task 1 `_fetch_batch()` + `fromId` loop in `download()` |
| Resume support | Task 1 `_last_record_in_file()` → `mode="a"`, `next_from_id = resume_id + 1` |
| Rate limiting (1 req/s) | Task 1 `SLEEP_S = 1.0` |
| Schema validation | Task 2 header check |
| agg_trade_id strictly increasing (primary monotonicity) | Task 2 `last_agg_id` check |
| Timestamp non-decreasing (same-ms allowed) | Task 2 `ts < last_ts` check |
| Positive price/qty | Task 2 Decimal > 0 check |
| Side distribution check | Task 2 buy/sell counts |
| `.gitignore` for large CSVs | Task 3 Step 4 |

All requirements covered. ✅

### Placeholder scan

No TBDs or incomplete steps. Task 3 involves running existing scripts — no code to add. ✅

### Type consistency

No cross-file type dependencies (pure scripts, no imports from `live/`). ✅
