from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch A-share listing dates from Tushare and cache them as local metadata.")
    parser.add_argument("--output", default="data/meta/list_dates.csv")
    parser.add_argument("--token-env", default="TUSHARE_TOKEN")
    return parser.parse_args()


def to_instrument(ts_code: str) -> str | None:
    code, _, exchange = str(ts_code).partition(".")
    exchange = exchange.upper()
    if exchange == "SH":
        return f"SH{code}"
    if exchange == "SZ":
        return f"SZ{code}"
    return None


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Tushare token. Please set ${args.token_env}.")

    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,list_date",
    )
    if df.empty:
        raise RuntimeError("Tushare returned empty stock_basic result.")

    df["instrument"] = df["ts_code"].map(to_instrument)
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["instrument", "list_date"]).sort_values("instrument")

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["instrument", "list_date", "name", "industry", "area", "ts_code"]].to_csv(out, index=False)
    print(f"Wrote: {out}")
    print(f"Rows: {len(df)}")
    print(df[["instrument", "list_date", "name", "industry"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
