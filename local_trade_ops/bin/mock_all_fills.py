#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from apply_manual_fills import load_env, main as apply_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock all planned manual orders as fully filled.")
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--fill-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env()
    fill_dir = Path(args.fill_dir or env["LIVE_FILL_DIR"]).expanduser().resolve()
    fill_csv = fill_dir / f"manual_fill_template_{args.execution_date}.csv"
    if not fill_csv.exists():
        raise FileNotFoundError(f"Manual fill template not found: {fill_csv}")

    fills = pd.read_csv(fill_csv)
    required = {"shares", "estimated_price"}
    missing = sorted(required - set(fills.columns))
    if missing:
        raise ValueError(f"Fill template missing columns: {missing}")
    fills["fill_shares"] = pd.to_numeric(fills["shares"], errors="coerce").fillna(0).astype(int)
    fills["fill_price"] = pd.to_numeric(fills["estimated_price"], errors="coerce").fillna(0.0).astype(float)
    fills["fill_status"] = fills["fill_shares"].apply(lambda shares: "FILLED" if shares > 0 else "UNFILLED")
    fills["operator_note"] = "mock_all_filled_at_estimated_price"
    fills.to_csv(fill_csv, index=False)

    # Reuse the normal accounting path so mock and manual fill books stay comparable.
    import sys

    sys.argv = [
        "apply_manual_fills.py",
        "--execution-date",
        args.execution_date,
        "--fills-csv",
        str(fill_csv),
    ]
    apply_main()


if __name__ == "__main__":
    main()
