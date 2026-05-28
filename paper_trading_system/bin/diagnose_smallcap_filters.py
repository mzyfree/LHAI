from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eval_smallcap_models import load_list_dates, load_market_data, load_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose which small-cap filter rules reject the most candidates.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model names, e.g. lgb...,xgb...")
    parser.add_argument("--weights", required=True, help="Comma-separated weights, e.g. 1,2")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="2026-05-18")
    parser.add_argument("--topn", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=3e7)
    parser.add_argument("--min-list-days", type=int, default=120)
    parser.add_argument("--max-open-gap", type=float, default=0.05)
    parser.add_argument("--max-volatility", type=float, default=0.08)
    parser.add_argument("--max-order-amount-pct", type=float, default=0.005)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--topk", type=int, default=7)
    parser.add_argument("--list-date-csv", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def daily_zscore(score: pd.Series) -> pd.Series:
    def _z(group: pd.Series) -> pd.Series:
        std = group.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return group * 0.0
        return (group - group.mean()) / std

    return score.groupby(level="datetime", group_keys=False).apply(_z)


def fuse_predictions(pred_dir: Path, models: list[str], weights: list[float]) -> pd.DataFrame:
    denom = sum(weights)
    pieces = []
    for model, weight in zip(models, weights, strict=True):
        pred = load_pred(pred_dir / f"{model}.pkl")
        pieces.append(daily_zscore(pred["score"]) * (weight / denom))
    return sum(pieces).to_frame("score").sort_index()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    if len(models) != len(weights):
        raise ValueError("--models and --weights lengths must match")

    pred = fuse_predictions(Path(args.pred_dir), models, weights)
    dates = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    dates = dates[(dates >= pd.Timestamp(args.start_date)) & (dates <= pd.Timestamp(args.end_date))]
    instruments = sorted(pred.index.get_level_values("instrument").unique())
    open_px, close_px, amount, _bench = load_market_data(
        args.provider_uri,
        instruments,
        str(dates[0].date()),
        str(dates[-1].date()),
        "SH000852",
    )
    list_dates = load_list_dates(args.list_date_csv)
    target_value = args.capital * 0.98 / args.topk

    rows = []
    for i, signal_date in enumerate(dates[:-1]):
        execution_date = dates[i + 1]
        day = (
            pred.xs(signal_date, level="datetime")
            .reset_index()
            .sort_values("score", ascending=False)
            .head(args.topn)
            .reset_index(drop=True)
        )
        hist = close_px.loc[:signal_date].tail(args.lookback + 1)
        ret = hist.pct_change(fill_method=None)
        vol = ret.tail(args.lookback).std(axis=0, skipna=True)
        avg_amount = amount.iloc[max(0, i - args.lookback + 1) : i + 1].mean(axis=0, skipna=True)
        close_row = close_px.loc[signal_date]
        open_row = open_px.loc[execution_date]

        for _, row in day.iterrows():
            inst = str(row["instrument"])
            close_price = float(close_row.get(inst, np.nan))
            open_price = float(open_row.get(inst, np.nan))
            amt = float(avg_amount.get(inst, np.nan))
            list_date = list_dates.get(inst)
            listed_days = (signal_date - list_date).days if list_date is not None else np.nan
            gap = open_price / close_price - 1.0 if np.isfinite(close_price) and np.isfinite(open_price) and close_price > 0 else np.nan
            inst_vol = float(vol.get(inst, np.nan))
            rows.append(
                {
                    "date": signal_date.date(),
                    "instrument": inst,
                    "score": row["score"],
                    "price_ok": np.isfinite(close_price) and args.min_price <= close_price <= args.max_price,
                    "tradable_ok": np.isfinite(close_price) and np.isfinite(open_price) and close_price > 0 and open_price > 0,
                    "amount_ok": np.isfinite(amt) and amt >= args.min_amount,
                    "listed_ok": np.isfinite(listed_days) and listed_days >= args.min_list_days,
                    "gap_ok": np.isfinite(gap) and gap <= args.max_open_gap,
                    "vol_ok": np.isfinite(inst_vol) and inst_vol <= args.max_volatility,
                    "capacity_ok": np.isfinite(amt) and target_value <= amt * args.max_order_amount_pct,
                    "close": close_price,
                    "next_open": open_price,
                    "avg_amount": amt,
                    "listed_days": listed_days,
                    "gap": gap,
                    "vol20": inst_vol,
                }
            )

    out = pd.DataFrame(rows)
    check_cols = ["price_ok", "tradable_ok", "amount_ok", "listed_ok", "gap_ok", "vol_ok", "capacity_ok"]
    out["all_ok"] = out[check_cols].all(axis=1)

    print("===== pass rate by rule =====")
    print(out[check_cols].mean().sort_values().to_string())
    print("\n===== fail count by rule =====")
    print((~out[check_cols]).sum().sort_values(ascending=False).to_string())
    print("\n===== all_ok per day =====")
    daily = out.groupby("date").agg(topn=("instrument", "count"), pass_all=("all_ok", "sum"))
    print(daily.describe().to_string())
    print(daily.tail(20).to_string())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()
