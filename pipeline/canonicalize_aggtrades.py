"""
Canonicalize a downloaded aggTrades CSV by removing resume-overlap duplicates.

WHY
    The raw download can contain resume-overlap "rewind" points: on restart, the
    `fromId` resume re-fetches the last batch and appends agg_trade_ids that were
    already written. The result is a sequence that increases except for a few
    backward jumps which re-introduce already-seen ids (no NEW out-of-order ids).

METHOD — streaming running-max filter
    Single pass; keep a row iff `agg_trade_id > max_seen`. This is:
      - lossless (each agg_trade_id is immutable — a re-fetched id is identical data),
      - O(1) memory, O(n) time (no 1.9 GB load, no global sort),
      - order-preserving (original temporal order of the kept rows is untouched).
    The output is strictly increasing BY CONSTRUCTION.

INTEGRITY PROOF (the gate)
    Strictly increasing + `clean_count == max_id - min_id + 1` ⟹ the output is
    exactly the contiguous range [min_id, max_id] with no gaps and no duplicates.
    If `clean_count < expected`, legitimate agg_trade_ids are MISSING (gaps): the
    streaming filter is not at fault, the raw data is incomplete. The pipeline STOPS
    (exit 1) and promotes nothing — gaps must be analyzed / re-fetched first.

Usage:
    python pipeline/canonicalize_aggtrades.py data/raw/BTCUSDT_AGGTRADES.csv

Outputs (on PASS):
    <stem>.raw.csv          # original, preserved untouched (atomic rename)
    <stem>.csv              # canonical: strictly increasing, gap-free
    <stem>.validation.json  # integrity report (always written)
On FAIL (gaps detected):
    <stem>.validation.json  # report with canonicalization_passed=false
    original left intact; no canonical file promoted; exit 1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HEADER = "timestamp_ms,price,qty,side,agg_trade_id"


def canonicalize(input_path: Path) -> dict:
    stem = input_path.stem                      # e.g. BTCUSDT_AGGTRADES
    parent = input_path.parent
    raw_path = parent / f"{stem}.raw.csv"
    canonical_path = parent / f"{stem}.csv"     # == input_path
    tmp_path = parent / f"{stem}.canon.tmp.csv"
    report_path = parent / f"{stem}.validation.json"

    rows_raw = 0
    kept = 0
    duplicates_removed = 0
    rewind_points = 0
    min_id: int | None = None
    max_seen = -1
    prev_raw_id: int | None = None

    with open(input_path, "r", encoding="utf-8", newline="") as fin, \
         open(tmp_path, "w", encoding="utf-8", newline="") as fout:

        header = fin.readline().rstrip("\n").rstrip("\r")
        if set(header.split(",")) != set(HEADER.split(",")):
            raise SystemExit(f"[FAIL] Schema mismatch. Got header: {header!r}")
        fout.write(header + "\n")

        for line in fin:
            stripped = line.rstrip("\n").rstrip("\r")
            if not stripped:
                continue
            rows_raw += 1
            # agg_trade_id is the last column; preserve the line verbatim otherwise.
            agg_id = int(stripped.rsplit(",", 1)[1])

            if prev_raw_id is not None and agg_id < prev_raw_id:
                rewind_points += 1
            prev_raw_id = agg_id

            if agg_id > max_seen:
                fout.write(line if line.endswith("\n") else stripped + "\n")
                kept += 1
                max_seen = agg_id
                if min_id is None:
                    min_id = agg_id
            else:
                duplicates_removed += 1

    max_id = max_seen
    expected_contiguous_count = (max_id - min_id + 1) if min_id is not None else 0
    gap_free = (kept == expected_contiguous_count)
    passed = gap_free  # output is strictly increasing by construction

    report = {
        "rows_raw": rows_raw,
        "rows_clean": kept,
        "duplicates_removed": duplicates_removed,
        "rewind_points": rewind_points,
        "min_agg_trade_id": min_id,
        "max_agg_trade_id": max_id,
        "expected_contiguous_count": expected_contiguous_count,
        "gap_free": gap_free,
        "canonicalization_passed": passed,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if passed:
        # Atomic promotion: original -> .raw.csv, tmp -> canonical (.csv) name.
        os.replace(input_path, raw_path)
        os.replace(tmp_path, canonical_path)
        print("[PASS] Canonicalization: strictly increasing + contiguous (gap-free)")
        print(f"       raw preserved at:  {raw_path}")
        print(f"       canonical written: {canonical_path}")
    else:
        # Do NOT promote a gap dataset. Keep tmp for analysis, leave original intact.
        gap_count = expected_contiguous_count - kept
        os.replace(tmp_path, parent / f"{stem}.canon.FAILED.csv")
        print("[FAIL] Missing agg_trade_ids detected — pipeline STOPS.")
        print(f"       clean_count={kept:,} != expected={expected_contiguous_count:,} "
              f"(missing {gap_count:,} ids). Original left intact; nothing promoted.")

    print(f"\nReport: {report_path}")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python pipeline/canonicalize_aggtrades.py <path_to_raw_csv>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    report = canonicalize(path)
    sys.exit(0 if report["canonicalization_passed"] else 1)


if __name__ == "__main__":
    main()
