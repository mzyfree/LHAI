#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare short-hold v2 and v3 daily backtest results.")
    parser.add_argument("--v2-daily", required=True, help="Existing v2 daily result CSV.")
    parser.add_argument("--v3-daily", required=True, help="V3 daily result CSV.")
    parser.add_argument("--output", required=True, help="Comparison CSV output path.")
    return parser.parse_args()


def _daily_return_series(df: pd.DataFrame) -> pd.Series:
    if "daily_return" in df.columns:
        return pd.to_numeric(df["daily_return"], errors="coerce").fillna(0.0)
    if "nav" in df.columns:
        nav = pd.to_numeric(df["nav"], errors="coerce").ffill()
        return nav.pct_change().fillna(0.0)
    raise ValueError("daily result CSV must contain either daily_return or nav")


def summarize(df: pd.DataFrame, label: str) -> dict[str, object]:
    returns = _daily_return_series(df)
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return {
        "label": label,
        "days": int(len(df)),
        "cum_return": float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
        "avg_daily_return": float(returns.mean()) if len(returns) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
    }


def main() -> int:
    args = parse_args()
    v2 = pd.read_csv(args.v2_daily)
    v3 = pd.read_csv(args.v3_daily)
    result = pd.DataFrame([summarize(v2, "v2"), summarize(v3, "v3")])
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
