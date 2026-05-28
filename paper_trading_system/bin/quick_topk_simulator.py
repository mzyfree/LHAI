from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper_trading_daily import calc_cost, fuse_three, load_pred, planned_buy_shares


@dataclass(frozen=True)
class Scenario:
    name: str
    capital: float
    topk: int
    n_drop: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast in-memory topK/drop paper simulator.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--pred-a", required=True)
    parser.add_argument("--pred-b", required=True)
    parser.add_argument("--pred-c", required=True)
    parser.add_argument("--weights", default="10,1,1")
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="2026-05-18")
    parser.add_argument("--output", default="reports/high_confidence_quick_summary.csv")
    return parser.parse_args()


def init_qlib(provider_uri: str) -> None:
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=provider_uri, region=REG_CN)


def raw_price_frames(provider_uri: str, instruments: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    from qlib.data import D

    init_qlib(provider_uri)
    data = D.features(instruments, ["$open", "$close", "$factor"], start_time=start, end_time=end, freq="day")
    data = data.rename(columns={"$open": "open_adj", "$close": "close_adj", "$factor": "factor"}).reset_index()
    valid = data["factor"].notna() & (data["factor"] > 0)
    data["open"] = data["open_adj"]
    data["close"] = data["close_adj"]
    data.loc[valid, "open"] = data.loc[valid, "open_adj"] / data.loc[valid, "factor"]
    data.loc[valid, "close"] = data.loc[valid, "close_adj"] / data.loc[valid, "factor"]
    data["datetime"] = pd.to_datetime(data["datetime"])
    close = data.pivot(index="datetime", columns="instrument", values="close").sort_index()
    open_ = data.pivot(index="datetime", columns="instrument", values="open").sort_index()
    return open_, close


def benchmark_open(provider_uri: str, dates: pd.DatetimeIndex) -> pd.Series:
    from qlib.data import D

    init_qlib(provider_uri)
    data = D.features(["SH000905"], ["$open"], start_time=dates[0], end_time=dates[-1], freq="day")
    data = data.rename(columns={"$open": "open"}).reset_index()
    data["datetime"] = pd.to_datetime(data["datetime"])
    return data.set_index("datetime")["open"].reindex(dates).dropna()


def max_drawdown(nav: pd.Series) -> float:
    wealth = nav / nav.iloc[0]
    return float((wealth / wealth.cummax() - 1.0).min())


def safe_price(row: pd.Series, instrument: str) -> float:
    value = row.get(instrument, 0.0)
    try:
        price = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(price) or price <= 0:
        return 0.0
    return price


def run_scenario(
    scenario: Scenario,
    scores_by_date: dict[pd.Timestamp, pd.DataFrame],
    dates: pd.DatetimeIndex,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
) -> dict:
    cash = float(scenario.capital)
    positions: dict[str, int] = {}
    total_cost = 0.0
    nav_rows: list[dict] = []
    buy_trades = 0
    sell_trades = 0

    for i, signal_date in enumerate(dates[:-1]):
        execution_date = dates[i + 1]
        signal = scores_by_date[signal_date]
        rank_map = dict(zip(signal["instrument"], signal["rank"], strict=False))
        close_row = close_px.loc[signal_date]

        marked_value = sum(shares * safe_price(close_row, inst) for inst, shares in positions.items())
        equity = cash + marked_value
        target_value = min(equity * 0.98 / scenario.topk, equity * 0.12)

        def rank_or_inf(inst: str) -> float:
            return float(rank_map.get(inst, 10**9))

        sell_list = sorted([inst for inst in positions if rank_or_inf(inst) > scenario.topk], key=rank_or_inf, reverse=True)[
            : scenario.n_drop
        ]
        kept = [inst for inst in positions if inst not in set(sell_list)]
        buy_slots = max(scenario.topk - len(kept), 0)
        if positions and sell_list:
            buy_slots = min(buy_slots, scenario.n_drop)

        available_cash = cash + sum(positions.get(inst, 0) * safe_price(close_row, inst) for inst in sell_list)
        reserved = 0.0
        buy_orders: list[tuple[str, int]] = []
        excluded = set(kept) | set(sell_list)

        for inst in signal.head(scenario.topk)["instrument"]:
            if len(buy_orders) >= buy_slots:
                break
            if inst in excluded:
                continue
            price = safe_price(close_row, inst)
            shares, _ = planned_buy_shares(target_value, equity, price, 100, 0.12, True)
            if shares <= 0:
                continue
            value = shares * price
            if value + reserved > available_cash:
                continue
            buy_orders.append((inst, shares))
            reserved += value

        open_row = open_px.loc[execution_date]
        for inst in sell_list:
            shares = positions.get(inst, 0)
            price = safe_price(open_row, inst)
            if shares <= 0 or price <= 0:
                continue
            value = shares * price
            fee = calc_cost(value, "SELL", type("Cost", (), {"buy_rate": 0.0003, "sell_rate": 0.0008, "min_cost": 5.0})())
            cash += value - fee
            total_cost += fee
            sell_trades += 1
            positions.pop(inst, None)

        for inst, shares in buy_orders:
            price = safe_price(open_row, inst)
            while shares > 0 and price > 0:
                value = shares * price
                fee = calc_cost(value, "BUY", type("Cost", (), {"buy_rate": 0.0003, "sell_rate": 0.0008, "min_cost": 5.0})())
                if value + fee <= cash:
                    break
                shares -= 100
            if shares <= 0 or price <= 0:
                continue
            value = shares * price
            fee = calc_cost(value, "BUY", type("Cost", (), {"buy_rate": 0.0003, "sell_rate": 0.0008, "min_cost": 5.0})())
            cash -= value + fee
            total_cost += fee
            buy_trades += 1
            positions[inst] = positions.get(inst, 0) + shares

        position_value = sum(shares * safe_price(open_row, inst) for inst, shares in positions.items())
        nav_rows.append(
            {
                "datetime": execution_date,
                "nav": cash + position_value,
                "cash": cash,
                "position_value": position_value,
                "holdings": len(positions),
            }
        )

    nav = pd.DataFrame(nav_rows)
    ret = float(nav.iloc[-1]["nav"] / scenario.capital - 1.0)
    pos_ratio = nav["position_value"] / nav["nav"]
    return {
        "case": scenario.name,
        "capital": scenario.capital,
        "topk": scenario.topk,
        "n_drop": scenario.n_drop,
        "last_nav": float(nav.iloc[-1]["nav"]),
        "strategy_return": ret,
        "max_drawdown": max_drawdown(nav["nav"]),
        "avg_position_ratio": float(pos_ratio.mean()),
        "last_position_ratio": float(pos_ratio.iloc[-1]),
        "final_holdings": int(nav.iloc[-1]["holdings"]),
        "trade_cost_sum": total_cost,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
    }


def main() -> None:
    args = parse_args()
    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) != 3:
        raise ValueError("--weights must be like 10,1,1")

    pred = fuse_three(
        load_pred(Path(args.pred_a)),
        load_pred(Path(args.pred_b)),
        load_pred(Path(args.pred_c)),
        weights,  # type: ignore[arg-type]
    )
    dates = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    dates = dates[(dates >= pd.Timestamp(args.start_date)) & (dates <= pd.Timestamp(args.end_date))]
    instruments = sorted(pred.index.get_level_values("instrument").unique())
    open_px, close_px = raw_price_frames(args.provider_uri, instruments, dates[0], dates[-1])
    bench = benchmark_open(args.provider_uri, dates[1:])
    bench_return = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
    bench_mdd = max_drawdown(bench)

    scores_by_date = {}
    for date in dates[:-1]:
        day = pred.xs(date, level="datetime").reset_index().sort_values("score", ascending=False).reset_index(drop=True)
        day["rank"] = day.index + 1
        scores_by_date[date] = day.loc[:, ["rank", "instrument", "score"]]

    scenarios = [
        Scenario("10w_top5_drop1", 100000.0, 5, 1),
        Scenario("20w_top5_drop1", 200000.0, 5, 1),
        Scenario("10w_top8_drop1", 100000.0, 8, 1),
        Scenario("20w_top8_drop1", 200000.0, 8, 1),
        Scenario("10w_top20_drop2", 100000.0, 20, 2),
        Scenario("20w_top20_drop2", 200000.0, 20, 2),
        Scenario("100w_top20_drop2", 1000000.0, 20, 2),
    ]
    rows = []
    for scenario in scenarios:
        row = run_scenario(scenario, scores_by_date, dates, open_px, close_px)
        row["benchmark_return"] = bench_return
        row["excess_return"] = row["strategy_return"] - bench_return
        row["benchmark_max_drawdown"] = bench_mdd
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("excess_return", ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(
        out.to_string(
            index=False,
            formatters={
                "last_nav": "{:.2f}".format,
                "strategy_return": "{:.2%}".format,
                "benchmark_return": "{:.2%}".format,
                "excess_return": "{:.2%}".format,
                "max_drawdown": "{:.2%}".format,
                "benchmark_max_drawdown": "{:.2%}".format,
                "avg_position_ratio": "{:.2%}".format,
                "last_position_ratio": "{:.2%}".format,
                "trade_cost_sum": "{:.2f}".format,
            },
        )
    )
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
