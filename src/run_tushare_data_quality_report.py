from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


BASE_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
LABEL = "Ref($close, -2) / Ref($close, -1) - 1"


def _pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.4f}%"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, float):
        if pd.isna(value):
            return "n/a"
        return f"{value:.6g}"
    return str(value)


def _safe_markdown(df: pd.DataFrame, index: bool = False) -> str:
    if df.empty:
        return "_No rows._"
    return df.to_markdown(index=index)


def init_qlib(provider_uri: str) -> Path:
    provider = Path(provider_uri).expanduser().resolve()
    qlib.init(provider_uri=str(provider), region=REG_CN)
    return provider


def summarize_calendar(start: str, end: str) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    cal = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day"))
    summary = {
        "start": cal.min() if len(cal) else None,
        "end": cal.max() if len(cal) else None,
        "trading_days": len(cal),
        "duplicate_days": int(cal.duplicated().sum()),
    }
    return cal, summary


def summarize_instruments(market: str, cal: pd.DatetimeIndex) -> tuple[pd.Series, dict[str, Any]]:
    counts: list[tuple[pd.Timestamp, int]] = []
    for dt in cal:
        d = dt.strftime("%Y-%m-%d")
        inst = D.list_instruments(D.instruments(market), start_time=d, end_time=d, freq="day", as_list=True)
        counts.append((dt, len(inst)))

    s = pd.Series(dict(counts), name="count").sort_index()
    non_500 = s[s != 500]
    summary = {
        "min": int(s.min()) if len(s) else None,
        "median": float(s.median()) if len(s) else None,
        "max": int(s.max()) if len(s) else None,
        "days": int(len(s)),
        "days_not_500": int(len(non_500)),
        "first_non_500": non_500.index.min() if len(non_500) else None,
        "last_non_500": non_500.index.max() if len(non_500) else None,
    }
    return s, summary


def load_market_features(market: str, start: str, end: str) -> pd.DataFrame:
    inst_conf = D.instruments(market)
    return D.features(inst_conf, BASE_FIELDS + [LABEL], start_time=start, end_time=end, freq="day")


def summarize_feature_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    n = len(df)
    for field in BASE_FIELDS + [LABEL]:
        s = df[field]
        rows.append(
            {
                "field": field,
                "rows": n,
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_rate": _pct(float(s.isna().mean()) if n else np.nan),
                "zero_or_negative": int((s <= 0).sum()) if field != LABEL else "n/a",
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            }
        )

    daily = df.groupby(level="datetime").agg(
        rows=("$close", "size"),
        close_missing=("$close", lambda x: int(x.isna().sum())),
        volume_missing=("$volume", lambda x: int(x.isna().sum())),
        factor_missing=("$factor", lambda x: int(x.isna().sum())),
        label_missing=(LABEL, lambda x: int(x.isna().sum())),
    )
    daily["close_missing_rate"] = daily["close_missing"] / daily["rows"]
    worst_daily = daily.sort_values(["close_missing_rate", "close_missing"], ascending=False).head(20)
    worst_daily = worst_daily.reset_index()
    worst_daily["datetime"] = worst_daily["datetime"].dt.strftime("%Y-%m-%d")
    worst_daily["close_missing_rate"] = worst_daily["close_missing_rate"].map(_pct)

    return pd.DataFrame(rows), worst_daily


def summarize_ohlcv_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["$open"]
    high = df["$high"]
    low = df["$low"]
    close = df["$close"]
    volume = df["$volume"]
    factor = df["$factor"]

    checks = {
        "price_non_positive": ((open_ <= 0) | (high <= 0) | (low <= 0) | (close <= 0)),
        "volume_negative": volume < 0,
        "factor_non_positive": factor <= 0,
        "high_below_low": high < low,
        "high_below_open_or_close": high < pd.concat([open_, close], axis=1).max(axis=1),
        "low_above_open_or_close": low > pd.concat([open_, close], axis=1).min(axis=1),
    }
    rows = []
    for name, mask in checks.items():
        mask = mask.fillna(False)
        rows.append({"check": name, "count": int(mask.sum()), "rate": _pct(mask.mean() if len(mask) else np.nan)})

    factor_jump = (
        factor.groupby(level="instrument")
        .pct_change(fill_method=None)
        .abs()
        .replace([np.inf, -np.inf], np.nan)
    )
    jump_mask = factor_jump > 0.5
    rows.append({"check": "factor_abs_pct_jump_gt_50pct", "count": int(jump_mask.sum()), "rate": _pct(jump_mask.mean())})
    return pd.DataFrame(rows)


def summarize_benchmark(benchmark: str, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    bench = D.features([benchmark], ["$close"], start_time=start, end_time=end, freq="day")
    if bench.empty:
        return bench, {"rows": 0, "missing_close": None, "start": None, "end": None}

    close = bench["$close"]
    ret = close.groupby(level="instrument").pct_change(fill_method=None)
    summary = {
        "rows": int(len(bench)),
        "start": bench.index.get_level_values("datetime").min(),
        "end": bench.index.get_level_values("datetime").max(),
        "missing_close": int(close.isna().sum()),
        "missing_return": int(ret.isna().sum()),
        "return_min": float(ret.min(skipna=True)) if ret.notna().any() else np.nan,
        "return_max": float(ret.max(skipna=True)) if ret.notna().any() else np.nan,
    }
    return bench, summary


def sample_pool_feature_check(market: str, dates: list[str]) -> pd.DataFrame:
    rows = []
    for d in dates:
        inst = D.list_instruments(D.instruments(market), start_time=d, end_time=d, freq="day", as_list=True)
        feat = D.features(inst, ["$close", "$volume", "$factor"], start_time=d, end_time=d, freq="day") if inst else pd.DataFrame()
        rows.append(
            {
                "date": d,
                "pool_count": len(inst),
                "feature_rows": int(len(feat)),
                "close_missing": int(feat["$close"].isna().sum()) if not feat.empty else "n/a",
                "volume_missing": int(feat["$volume"].isna().sum()) if not feat.empty else "n/a",
                "factor_missing": int(feat["$factor"].isna().sum()) if not feat.empty else "n/a",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Tushare Qlib data quality report.")
    parser.add_argument("--provider-uri", required=True, help="Qlib data provider directory.")
    parser.add_argument("--market", default="csi500", help="Instrument universe, e.g. csi500.")
    parser.add_argument("--benchmark", default="SH000905", help="Benchmark instrument.")
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports/tushare_data_quality_report.md"))
    parser.add_argument(
        "--sample-dates",
        default="2017-01-03,2019-01-02,2020-07-31,2021-12-30",
        help="Comma-separated dates for spot checks.",
    )
    args = parser.parse_args()

    provider = init_qlib(args.provider_uri)
    output = Path(args.output).expanduser().resolve()
    sample_dates = [x.strip() for x in args.sample_dates.split(",") if x.strip()]

    cal, calendar_summary = summarize_calendar(args.start, args.end)
    counts, instrument_summary = summarize_instruments(args.market, cal)
    market_df = load_market_features(args.market, args.start, args.end)
    feature_summary, worst_daily = summarize_feature_quality(market_df)
    anomaly_summary = summarize_ohlcv_anomalies(market_df)
    _, benchmark_summary = summarize_benchmark(args.benchmark, args.start, args.end)
    sample_summary = sample_pool_feature_check(args.market, sample_dates)

    count_desc = counts.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("value").reset_index()
    count_desc["value"] = count_desc["value"].map(_fmt)

    report = f"""# Tushare Qlib Data Quality Report

## Setup

- Provider: `{provider}`
- Market: `{args.market}`
- Benchmark: `{args.benchmark}`
- Range: `{args.start}` to `{args.end}`
- Generated rows loaded for market features: `{len(market_df)}`

## Calendar

| Metric | Value |
| --- | ---: |
| start | {_fmt(calendar_summary["start"])} |
| end | {_fmt(calendar_summary["end"])} |
| trading_days | {calendar_summary["trading_days"]} |
| duplicate_days | {calendar_summary["duplicate_days"]} |

## Instrument Pool Counts

| Metric | Value |
| --- | ---: |
| min | {_fmt(instrument_summary["min"])} |
| median | {_fmt(instrument_summary["median"])} |
| max | {_fmt(instrument_summary["max"])} |
| days | {instrument_summary["days"]} |
| days_not_500 | {instrument_summary["days_not_500"]} |
| first_non_500 | {_fmt(instrument_summary["first_non_500"])} |
| last_non_500 | {_fmt(instrument_summary["last_non_500"])} |

{_safe_markdown(count_desc)}

## Sample Date Pool Feature Checks

{_safe_markdown(sample_summary)}

## Feature And Label Coverage

{_safe_markdown(feature_summary)}

## Worst Daily Close Missing Rates

{_safe_markdown(worst_daily)}

## OHLCV And Factor Anomalies

{_safe_markdown(anomaly_summary)}

## Benchmark Coverage

| Metric | Value |
| --- | ---: |
| rows | {benchmark_summary["rows"]} |
| start | {_fmt(benchmark_summary["start"])} |
| end | {_fmt(benchmark_summary["end"])} |
| missing_close | {_fmt(benchmark_summary["missing_close"])} |
| missing_return | {_fmt(benchmark_summary["missing_return"])} |
| return_min | {_fmt(benchmark_summary["return_min"])} |
| return_max | {_fmt(benchmark_summary["return_max"])} |

## Interpretation Checklist

- `days_not_500` should be explained by index inception/history boundaries, not random gaps in the research/test window.
- `$close`, `$volume`, and `$factor` missing rates should be very low for active CSI500 rows.
- OHLC consistency checks should be zero or explainable by missing data.
- Benchmark rows should cover the whole backtest range.
- This report validates research-readiness; production use still needs external vendor reconciliation and automated daily monitoring.
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print("Tushare Qlib data quality report complete.")
    print(f"- output: {output}")
    print(f"- market rows: {len(market_df)}")
    print("- calendar:", calendar_summary)
    print("- instrument counts:", instrument_summary)
    print("- benchmark:", benchmark_summary)
    print()
    print("Feature summary:")
    print(feature_summary.to_string(index=False))
    print()
    print("Anomaly summary:")
    print(anomaly_summary.to_string(index=False))


if __name__ == "__main__":
    main()
