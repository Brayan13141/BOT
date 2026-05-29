"""
Download historical OHLCV candles from Binance public API.
Saves to data/raw/<SYMBOL>_<INTERVAL_LABEL>.csv

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --interval 5m --start-date 2019-01-01
    python scripts/download_data.py --interval 1m --months 12
    python scripts/download_data.py --interval 5m --symbol BTCUSDT --start-date 2019-01-01
"""
import argparse
import csv
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
LIMIT = 1000  # Binance max per request

INTERVAL_LABELS = {
    "1m": "M1", "3m": "M3", "5m": "M5", "15m": "M15",
    "30m": "M30", "1h": "H1", "4h": "H4", "1d": "D1",
}

CANDLES_PER_DAY = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
    "30m": 48,  "1h": 24,  "4h": 6,   "1d": 1,
}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     LIMIT,
    }
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def download_symbol(symbol: str, interval: str, start_dt: datetime,
                    end_dt: datetime, out_dir: Path) -> Path:
    label    = INTERVAL_LABELS.get(interval, interval.upper())
    out_path = out_dir / f"{symbol}_{label}.csv"

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    days_total     = (end_dt - start_dt).days
    est_candles    = days_total * CANDLES_PER_DAY.get(interval, 288)
    total_candles  = 0
    batch          = 0

    print(f"[{symbol}] Downloading {interval} data ({label})...")
    print(f"  From: {start_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  To:   {end_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Est. candles: ~{est_candles:,}  |  Est. requests: ~{est_candles // LIMIT + 1:,}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)

        cursor_ms = start_ms
        while cursor_ms < end_ms:
            rows = fetch_klines(symbol, interval, cursor_ms, end_ms)
            if not rows:
                break

            writer.writerows(rows)
            total_candles += len(rows)
            batch += 1

            cursor_ms = rows[-1][6] + 1  # close_time + 1ms

            if batch % 50 == 0:
                last_dt = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc)
                pct = (cursor_ms - start_ms) / (end_ms - start_ms) * 100
                print(f"  [{pct:5.1f}%] {total_candles:,} candles | last: {last_dt.strftime('%Y-%m-%d %H:%M')}")

            time.sleep(0.05)

    print(f"  Done: {total_candles:,} candles -> {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Binance OHLCV data")
    parser.add_argument("--symbol",     default=None,  help="Single symbol (e.g. BTCUSDT). Default: BTCUSDT + ETHUSDT.")
    parser.add_argument("--interval",   default="1m",  help="Candle interval: 1m, 5m, 15m, 1h, etc. (default: 1m)")
    parser.add_argument("--start-date", default=None,  help="Start date YYYY-MM-DD (UTC). Takes priority over --months.")
    parser.add_argument("--months",     type=int, default=12, help="Months of history if --start-date not given (default: 12)")
    args = parser.parse_args()

    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=30 * args.months)

    out_dir = Path(__file__).parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [args.symbol] if args.symbol else ["BTCUSDT", "ETHUSDT"]

    for sym in symbols:
        try:
            download_symbol(sym, args.interval, start_dt, end_dt, out_dir)
        except requests.RequestException as e:
            print(f"  ERROR [{sym}]: {e}")
        print()

    print("All downloads complete.")


if __name__ == "__main__":
    main()
