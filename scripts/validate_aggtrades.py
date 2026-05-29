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

        count                = 0
        buy_count            = 0
        sell_count           = 0
        last_ts              = -1
        last_agg_id          = -1
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
        print(
            f"[PASS] Side distribution: "
            f"BUY={buy_count:,} ({buy_pct:.1f}%), SELL={sell_count:,} ({sell_pct:.1f}%)"
        )

    if first_ts and last_ts > 0:
        first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
        last_dt  = datetime.fromtimestamp(last_ts  / 1000, tz=timezone.utc)
        print(f"\nCoverage: {first_dt.date()} -> {last_dt.date()}")
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
