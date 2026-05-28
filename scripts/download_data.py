"""
Download historical OHLCV M1 candles from Binance public API.
Saves to data/raw/<SYMBOL>_M1.csv

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --months 12
    python scripts/download_data.py --symbol BTCUSDT --months 6
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


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def download_symbol(symbol: str, months: int, out_dir: Path) -> Path:
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=30 * months)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    out_path = out_dir / f"{symbol}_M1.csv"
    total_candles = 0
    batch = 0

    print(f"[{symbol}] Downloading {months} months of M1 data...")
    print(f"  From: {start_dt.isoformat()}")
    print(f"  To:   {end_dt.isoformat()}")
    print(f"  Est. candles: ~{months * 30 * 24 * 60:,}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)

        cursor_ms = start_ms
        while cursor_ms < end_ms:
            rows = fetch_klines(symbol, cursor_ms, end_ms)
            if not rows:
                break

            writer.writerows(rows)
            total_candles += len(rows)
            batch += 1

            # Advance cursor past last candle close time
            cursor_ms = rows[-1][6] + 1  # close_time + 1ms

            if batch % 50 == 0:
                last_dt = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc)
                pct = (cursor_ms - start_ms) / (end_ms - start_ms) * 100
                print(f"  [{pct:5.1f}%] {total_candles:,} candles | last: {last_dt.strftime('%Y-%m-%d %H:%M')}")

            # Respect rate limits (1200 req/min → 1 req every 50ms is safe)
            time.sleep(0.05)

    print(f"  Done: {total_candles:,} candles → {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Binance M1 OHLCV data")
    parser.add_argument("--symbol", default=None, help="Single symbol (e.g. BTCUSDT). Default: all configured symbols.")
    parser.add_argument("--months", type=int, default=12, help="Months of history to download (default: 12)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [args.symbol] if args.symbol else ["BTCUSDT", "ETHUSDT"]

    for sym in symbols:
        try:
            download_symbol(sym, args.months, out_dir)
        except requests.RequestException as e:
            print(f"  ERROR [{sym}]: {e}")
        print()

    print("All downloads complete.")


if __name__ == "__main__":
    main()
