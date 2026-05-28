from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import qlib
from qlib.constant import REG_CN
from qlib.data import D


FIELDS = ["$open", "$high", "$low", "$close", "$volume"]


def init_qlib(provider_uri: str) -> None:
    qlib.init(provider_uri=str(Path(provider_uri).expanduser().resolve()), region=REG_CN)


def load_calendar(provider_uri: str, start: str, end: str) -> pd.DatetimeIndex:
    init_qlib(provider_uri)
    cal = D.calendar(start_time=start, end_time=end, freq="day")
    return pd.DatetimeIndex(cal)


def load_symbol_frame(provider_uri: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    init_qlib(provider_uri)
    df = D.features([symbol], FIELDS, start_time=start, end_time=end, freq="day")
    if df is None or df.empty:
        return pd.DataFrame(columns=FIELDS)
    df = df.reset_index().drop(columns=["instrument"])
    df = df.rename(columns={"datetime": "date"})
    return df.set_index("date").sort_index()


def summarize_calendar(old_cal: pd.DatetimeIndex, new_cal: pd.DatetimeIndex) -> dict[str, object]:
    overlap = old_cal.intersection(new_cal)
    old_only = old_cal.difference(new_cal)
    new_only = new_cal.difference(old_cal)
    return {
        "old_count": len(old_cal),
        "new_count": len(new_cal),
        "overlap_count": len(overlap),
        "old_only_count": len(old_only),
        "new_only_count": len(new_only),
        "overlap_start": overlap.min() if len(overlap) else None,
        "overlap_end": overlap.max() if len(overlap) else None,
    }


def summarize_symbol(old_df: pd.DataFrame, new_df: pd.DataFrame, symbol: str) -> list[dict[str, object]]:
    merged = old_df.join(new_df, how="inner", lsuffix="_old", rsuffix="_new")
    rows: list[dict[str, object]] = []

    for field in FIELDS:
        old_col = f"{field}_old"
        new_col = f"{field}_new"
        if old_col not in merged.columns or new_col not in merged.columns:
            continue

        pair = merged[[old_col, new_col]].dropna()
        if pair.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "field": field,
                    "rows": 0,
                    "corr": None,
                    "mean_abs_diff": None,
                    "mean_abs_pct_diff": None,
                }
            )
            continue

        corr = pair[old_col].corr(pair[new_col])
        abs_diff = (pair[old_col] - pair[new_col]).abs()
        denom = pair[old_col].abs().replace(0, pd.NA)
        abs_pct = (abs_diff / denom).dropna()

        rows.append(
            {
                "symbol": symbol,
                "field": field,
                "rows": int(len(pair)),
                "corr": None if pd.isna(corr) else float(corr),
                "mean_abs_diff": float(abs_diff.mean()),
                "mean_abs_pct_diff": None if abs_pct.empty else float(abs_pct.mean()),
            }
        )

    return rows


def format_ts(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate overlap quality between two Qlib daily datasets.")
    parser.add_argument("--old-provider-uri", required=True, help="Existing trusted qlib data directory.")
    parser.add_argument("--new-provider-uri", required=True, help="New yahoo-built qlib data directory.")
    parser.add_argument("--start", default="2019-01-01", help="Overlap validation start date.")
    parser.add_argument("--end", default="2020-09-25", help="Overlap validation end date.")
    parser.add_argument(
        "--symbols",
        default="SH600000,SZ000001,SH600519,SZ300750",
        help="Comma-separated symbols used for OHLCV spot checks.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/qlib_data_overlap_validation.md"),
        help="Markdown report output path.",
    )
    args = parser.parse_args()

    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    output_path = Path(args.output).expanduser().resolve()

    old_cal = load_calendar(args.old_provider_uri, args.start, args.end)
    new_cal = load_calendar(args.new_provider_uri, args.start, args.end)
    cal_summary = summarize_calendar(old_cal, new_cal)

    rows: list[dict[str, object]] = []
    for symbol in symbols:
        old_df = load_symbol_frame(args.old_provider_uri, symbol, args.start, args.end)
        new_df = load_symbol_frame(args.new_provider_uri, symbol, args.start, args.end)
        rows.extend(summarize_symbol(old_df, new_df, symbol))

    compare_df = pd.DataFrame(rows)

    report = f"""# Qlib Data Overlap Validation

## Setup

- Old provider: `{Path(args.old_provider_uri).expanduser().resolve()}`
- New provider: `{Path(args.new_provider_uri).expanduser().resolve()}`
- Validation range: `{args.start}` to `{args.end}`
- Symbols: `{", ".join(symbols)}`

## Calendar Summary

| Metric | Value |
| --- | ---: |
| old_count | {cal_summary["old_count"]} |
| new_count | {cal_summary["new_count"]} |
| overlap_count | {cal_summary["overlap_count"]} |
| old_only_count | {cal_summary["old_only_count"]} |
| new_only_count | {cal_summary["new_only_count"]} |
| overlap_start | {format_ts(cal_summary["overlap_start"])} |
| overlap_end | {format_ts(cal_summary["overlap_end"])} |

## Symbol OHLCV Comparison

{compare_df.to_markdown(index=False)}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("Qlib overlap validation complete.")
    print(f"- output: {output_path}")
    print()
    print("Calendar summary:")
    print(cal_summary)
    print()
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
