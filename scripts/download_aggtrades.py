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

MAX_RETRIES    = 6          # attempts per batch before giving up
BACKOFF_BASE_S = 2.0        # exponential backoff: 2,4,8,16,32s across 5 waits
# Status codes worth retrying: rate limit (429), IP ban warning (418), 5xx server.
# Any other 4xx is a real client error and must fail fast.
_RETRY_STATUS  = {429, 418, 500, 502, 503, 504}

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
    Use from_id for pagination after the first request.
    Use start_ms only for the initial anchor request.
    """
    params: dict = {"symbol": symbol, "limit": LIMIT}
    if from_id is not None:
        params["fromId"] = from_id
    else:
        assert start_ms is not None, "Either from_id or start_ms required"
        params["startTime"] = start_ms

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(BASE_URL + ENDPOINT, params=params, timeout=30.0)
        except httpx.TransportError as exc:
            # Connection-level failure (disconnect, timeout, network). Transient.
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE_S * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{MAX_RETRIES}] {type(exc).__name__}: {exc} -> waiting {wait:.0f}s")
            time.sleep(wait)
            continue

        if resp.status_code in _RETRY_STATUS:
            if attempt == MAX_RETRIES:
                resp.raise_for_status()
            # Honor Retry-After when Binance sends it (429/418), else exp. backoff.
            wait = float(resp.headers.get("Retry-After", BACKOFF_BASE_S * (2 ** (attempt - 1))))
            print(f"  [retry {attempt}/{MAX_RETRIES}] HTTP {resp.status_code} -> waiting {wait:.0f}s")
            time.sleep(wait)
            continue

        resp.raise_for_status()  # any other 4xx is a real error -> fail fast
        return resp.json()

    raise RuntimeError("unreachable: retry loop exhausted without return or raise")


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
                # Guard: if the API returns data beyond our window, the requested
                # range is outside the API's retention window (~12 months).
                if batch and batch[0]["T"] >= end_ms:
                    earliest = datetime.fromtimestamp(batch[0]["T"] / 1000, tz=timezone.utc)
                    print(
                        f"[WARN] No data available for {symbol} in the requested range.\n"
                        f"       API earliest available: {earliest}\n"
                        f"       Requested end:          {datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)}\n"
                        f"       Binance aggTrades API retains ~12 months of history."
                    )
                    return

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

    print(f"Downloading {args.symbol} aggTrades: {args.start} -> {args.end}")
    print(f"Output: {out_path}")
    download(symbol=args.symbol, start_ms=start_ms, end_ms=end_ms, out_path=out_path)


if __name__ == "__main__":
    main()
