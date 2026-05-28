#!/usr/bin/env python3
"""Mark same-day PnL for a pending order file using Qlib raw prices."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def init_qlib(provider_uri: str) -> None:
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=provider_uri, region=REG_CN)


def load_prices(provider_uri: str, instruments: list[str], date: str) -> pd.DataFrame:
    from qlib.data import D

    init_qlib(provider_uri)
    data = D.features(instruments, ["$open", "$close", "$factor"], start_time=date, end_time=date, freq="day")
    if data is None or data.empty:
        raise SystemExit(f"No Qlib price data found for {date}.")

    data = data.rename(columns={"$open": "open_adj", "$close": "close_adj", "$factor": "factor"}).reset_index()
    data["open"] = data["open_adj"]
    data["close"] = data["close_adj"]
    valid_factor = data["factor"].notna() & (data["factor"] > 0)
    data.loc[valid_factor, "open"] = data.loc[valid_factor, "open_adj"] / data.loc[valid_factor, "factor"]
    data.loc[valid_factor, "close"] = data.loc[valid_factor, "close_adj"] / data.loc[valid_factor, "factor"]
    return data.loc[:, ["instrument", "open", "close", "factor"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--orders", required=True)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--buy-cost-rate", type=float, default=0.0003)
    parser.add_argument("--sell-cost-rate", type=float, default=0.0008)
    parser.add_argument("--min-cost", type=float, default=5.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    orders_path = Path(args.orders).expanduser().resolve()
    orders = pd.read_csv(orders_path)
    orders = orders[orders["action"].isin(["BUY", "SELL"])].copy()
    if orders.empty:
        raise SystemExit(f"No BUY/SELL orders in {orders_path}")

    execution_dates = sorted(orders["execution_date"].astype(str).unique())
    if len(execution_dates) != 1:
        raise SystemExit(f"Expected one execution_date, got {execution_dates}")
    execution_date = execution_dates[0]

    instruments = sorted(orders["instrument"].astype(str).unique())
    prices = load_prices(args.provider_uri, instruments, execution_date)
    df = orders.merge(prices, on="instrument", how="left")
    if df[["open", "close"]].isna().any().any():
        missing = df[df[["open", "close"]].isna().any(axis=1)]["instrument"].tolist()
        raise SystemExit(f"Missing open/close prices: {missing}")

    sign = df["action"].map({"BUY": 1.0, "SELL": -1.0})
    fee_rate = df["action"].map({"BUY": args.buy_cost_rate, "SELL": args.sell_cost_rate})
    df["open_value"] = df["shares"] * df["open"]
    df["close_value"] = df["shares"] * df["close"]
    df["fee"] = (df["open_value"] * fee_rate).clip(lower=args.min_cost)
    df["pnl_gross"] = sign * (df["close_value"] - df["open_value"])
    df["pnl_net"] = df["pnl_gross"] - df["fee"]
    df["return_pct_on_open_value"] = df["pnl_net"] / df["open_value"] * 100.0

    cash_after_open = args.capital - (sign * df["open_value"]).sum() - df["fee"].sum()
    # For this diagnostic we mark the orders at close; starting from cash only is
    # the current live_10w_eastmoney_manual state before 2026-05-25 execution.
    nav_at_close = cash_after_open + (sign.clip(lower=0) * df["close_value"]).sum()

    detail_cols = [
        "instrument",
        "action",
        "shares",
        "open",
        "close",
        "open_value",
        "close_value",
        "fee",
        "pnl_gross",
        "pnl_net",
        "return_pct_on_open_value",
    ]
    summary = {
        "execution_date": execution_date,
        "capital": args.capital,
        "open_value": float((sign.clip(lower=0) * df["open_value"]).sum()),
        "fees": float(df["fee"].sum()),
        "cash_after_open": float(cash_after_open),
        "close_value": float((sign.clip(lower=0) * df["close_value"]).sum()),
        "gross_pnl": float(df["pnl_gross"].sum()),
        "net_pnl": float(df["pnl_net"].sum()),
        "nav_at_close": float(nav_at_close),
        "return_pct_on_capital": float((nav_at_close / args.capital - 1.0) * 100.0),
        "return_pct_on_deployed": float(df["pnl_net"].sum() / df["open_value"].sum() * 100.0),
    }

    print("===== detail =====")
    print(df.loc[:, detail_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n===== summary =====")
    for key, value in summary.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        df.loc[:, detail_cols].to_csv(output, index=False)
        print(f"\nWrote: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
