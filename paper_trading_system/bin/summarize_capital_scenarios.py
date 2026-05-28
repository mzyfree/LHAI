#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def load_benchmark(provider_uri: Path, dates: pd.Series) -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    start = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
    bench = D.features(["SH000905"], ["$open"], start_time=start, end_time=end, freq="day")
    bench = bench.rename(columns={"$open": "benchmark_open"}).reset_index()
    bench["datetime"] = pd.to_datetime(bench["datetime"])
    bench = bench.loc[:, ["datetime", "benchmark_open"]].dropna()
    if bench.empty:
        return pd.DataFrame(columns=["datetime", "benchmark_cum_return", "benchmark_max_drawdown"])
    base = float(bench.iloc[0]["benchmark_open"])
    bench["benchmark_cum_return"] = bench["benchmark_open"] / base - 1
    wealth = bench["benchmark_open"] / base
    bench["benchmark_drawdown"] = wealth / wealth.cummax() - 1
    return bench


def max_drawdown(nav: pd.Series) -> float:
    wealth = nav / float(nav.iloc[0])
    return float((wealth / wealth.cummax() - 1).min())


def main() -> None:
    paper_home = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path.cwd()
    provider_uri = paper_home / "data/current"
    cases = [
        ("10w", 100000, paper_home / "state_top20_drop2_cap10w_ytd"),
        ("20w", 200000, paper_home / "state_top20_drop2_cap20w_ytd"),
        ("100w", 1000000, paper_home / "state_top20_drop2_cap100w_ytd"),
    ]
    rows = []
    nav_frames = []
    for name, capital, state_dir in cases:
        nav_path = state_dir / "paper_nav.csv"
        pos_path = state_dir / "paper_positions.csv"
        if not nav_path.exists():
            continue
        nav = pd.read_csv(nav_path, parse_dates=["datetime"])
        pos = pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame()
        trade_frames = [pd.read_csv(p) for p in sorted(state_dir.glob("paper_trades_*.csv"))]
        trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        nav["case"] = name
        nav["cum_return"] = nav["nav"] / capital - 1
        nav["position_ratio"] = nav["position_value"] / nav["nav"]
        nav_frames.append(nav)
        benchmark = load_benchmark(provider_uri, nav["datetime"])
        benchmark_end_return = float(benchmark.iloc[-1]["benchmark_cum_return"]) if not benchmark.empty else float("nan")
        benchmark_mdd = float(benchmark["benchmark_drawdown"].min()) if not benchmark.empty else float("nan")
        rows.append(
            {
                "case": name,
                "capital": capital,
                "start_date": nav["datetime"].min().date(),
                "end_date": nav["datetime"].max().date(),
                "last_nav": nav.iloc[-1]["nav"],
                "pnl": nav.iloc[-1]["nav"] - capital,
                "cum_return": nav.iloc[-1]["cum_return"],
                "benchmark_return": benchmark_end_return,
                "excess_return": nav.iloc[-1]["cum_return"] - benchmark_end_return,
                "min_return": nav["cum_return"].min(),
                "max_drawdown": max_drawdown(nav["nav"]),
                "benchmark_max_drawdown": benchmark_mdd,
                "last_position_ratio": nav.iloc[-1]["position_ratio"],
                "avg_position_ratio": nav["position_ratio"].mean(),
                "final_holdings": len(pos),
                "trade_cost_sum": nav["trade_cost"].sum(),
                "buy_trades": int((trades["action"] == "BUY").sum()) if not trades.empty else 0,
                "sell_trades": int((trades["action"] == "SELL").sum()) if not trades.empty else 0,
            }
        )

    if not rows:
        raise SystemExit("No scenario NAV files found.")

    summary = pd.DataFrame(rows)
    out_dir = paper_home / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "capital_scenarios_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("===== summary =====")
    print(
        summary.to_string(
            index=False,
            formatters={
                "last_nav": "{:.2f}".format,
                "pnl": "{:.2f}".format,
                "cum_return": "{:.2%}".format,
                "benchmark_return": "{:.2%}".format,
                "excess_return": "{:.2%}".format,
                "min_return": "{:.2%}".format,
                "max_drawdown": "{:.2%}".format,
                "benchmark_max_drawdown": "{:.2%}".format,
                "last_position_ratio": "{:.2%}".format,
                "avg_position_ratio": "{:.2%}".format,
                "trade_cost_sum": "{:.2f}".format,
            },
        )
    )

    nav_all = pd.concat(nav_frames, ignore_index=True)
    pivot_path = out_dir / "capital_scenarios_nav.csv"
    nav_all.to_csv(pivot_path, index=False)
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {pivot_path}")


if __name__ == "__main__":
    main()
