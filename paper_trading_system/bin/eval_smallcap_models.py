from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate small-cap model predictions with raw and JoinQuant-style filters.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--benchmark", default="SH000852")
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="2026-05-18")
    parser.add_argument("--topk-pairs", default="7:1,10:1,15:1,20:2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--capital-output", required=True)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=3e7)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-list-days", type=int, default=120)
    parser.add_argument("--max-open-gap", type=float, default=0.05)
    parser.add_argument("--max-volatility", type=float, default=0.08)
    parser.add_argument("--max-order-amount-pct", type=float, default=0.005)
    parser.add_argument("--list-date-csv", default="")
    return parser.parse_args()


def init_qlib(provider_uri: str) -> None:
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=provider_uri, region=REG_CN)


def load_pred(path: Path) -> pd.DataFrame:
    pred = pd.read_pickle(path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    pred = pred[["score"]].copy()
    pred.index = pred.index.set_names(["datetime", "instrument"])
    return pred.sort_index()


def model_name(path: Path, market: str) -> str:
    suffix = f"_{market}_prod2026"
    name = path.stem
    return name[: -len(suffix)] if name.endswith(suffix) else name


def load_market_data(provider_uri: str, instruments: list[str], start: str, end: str, benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    from qlib.data import D

    init_qlib(provider_uri)
    fields = ["$open", "$close", "$factor", "$volume"]
    data = D.features(instruments, fields, start_time=start, end_time=end, freq="day")
    data = data.rename(columns={"$open": "open_adj", "$close": "close_adj", "$factor": "factor", "$volume": "volume"}).reset_index()
    data["datetime"] = pd.to_datetime(data["datetime"])
    valid = data["factor"].notna() & (data["factor"] > 0)
    data["open"] = data["open_adj"]
    data["close"] = data["close_adj"]
    data.loc[valid, "open"] = data.loc[valid, "open_adj"] / data.loc[valid, "factor"]
    data.loc[valid, "close"] = data.loc[valid, "close_adj"] / data.loc[valid, "factor"]
    data["amount"] = data["close"] * data["volume"]
    open_px = data.pivot(index="datetime", columns="instrument", values="open").sort_index()
    close_px = data.pivot(index="datetime", columns="instrument", values="close").sort_index()
    amount = data.pivot(index="datetime", columns="instrument", values="amount").sort_index()

    bench = D.features([benchmark], ["$open"], start_time=start, end_time=end, freq="day")
    bench = bench.rename(columns={"$open": "open"}).reset_index()
    bench["datetime"] = pd.to_datetime(bench["datetime"])
    bench_open = bench.set_index("datetime")["open"].sort_index()
    return open_px, close_px, amount, bench_open


def max_drawdown(nav: pd.Series) -> float:
    wealth = nav / nav.iloc[0]
    return float((wealth / wealth.cummax() - 1).min())


def safe_price(row: pd.Series, inst: str) -> float:
    try:
        price = float(row.get(inst, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return price if math.isfinite(price) and price > 0 else 0.0


def filter_signal(signal: pd.DataFrame, close_row: pd.Series, amount_window: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if amount_window.empty:
        avg_amount = pd.Series(dtype=float)
    else:
        avg_amount = amount_window.mean(axis=0, skipna=True)

    keep = []
    for inst in signal["instrument"].astype(str):
        price = safe_price(close_row, inst)
        amt = float(avg_amount.get(inst, 0.0) or 0.0)
        keep.append(args.min_price <= price <= args.max_price and amt >= args.min_amount)
    return signal.loc[keep].reset_index(drop=True)


def first_seen_dates(pred: pd.DataFrame) -> dict[str, pd.Timestamp]:
    idx = pred.index.to_frame(index=False)
    return idx.groupby("instrument")["datetime"].min().to_dict()


def load_list_dates(path: str) -> dict[str, pd.Timestamp]:
    if not path:
        return {}
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"--list-date-csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "instrument" not in df.columns or "list_date" not in df.columns:
        raise ValueError("--list-date-csv must contain columns: instrument,list_date")
    df = df[["instrument", "list_date"]].copy()
    df["instrument"] = df["instrument"].astype(str)
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df = df.dropna(subset=["instrument", "list_date"])
    return dict(zip(df["instrument"], df["list_date"], strict=False))


def filter_signal_v1(
    signal: pd.DataFrame,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    open_row: pd.Series,
    close_px: pd.DataFrame,
    amount_window: pd.DataFrame,
    first_seen: dict[str, pd.Timestamp],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if amount_window.empty:
        avg_amount = pd.Series(dtype=float)
    else:
        avg_amount = amount_window.mean(axis=0, skipna=True)

    hist = close_px.loc[:signal_date].tail(args.lookback + 1)
    ret = hist.pct_change(fill_method=None)
    vol = ret.tail(args.lookback).std(axis=0, skipna=True)
    close_row = close_px.loc[signal_date]

    keep = []
    for inst in signal["instrument"].astype(str):
        close_price = safe_price(close_row, inst)
        open_price = safe_price(open_row, inst)
        amt = float(avg_amount.get(inst, 0.0) or 0.0)
        listed_at = first_seen.get(inst)
        listed_days = (signal_date - listed_at).days if listed_at is not None else 0
        gap = open_price / close_price - 1.0 if close_price > 0 and open_price > 0 else np.inf
        inst_vol = float(vol.get(inst, np.inf))
        keep.append(
            args.min_price <= close_price <= args.max_price
            and close_price > 0
            and open_price > 0
            and amt >= args.min_amount
            and listed_days >= args.min_list_days
            and gap <= args.max_open_gap
            and np.isfinite(inst_vol)
            and inst_vol <= args.max_volatility
        )
    return signal.loc[keep].reset_index(drop=True)


def simulate_topdrop(
    pred: pd.DataFrame,
    dates: pd.DatetimeIndex,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    amount: pd.DataFrame,
    topk: int,
    n_drop: int,
    use_filter: bool,
    args: argparse.Namespace,
    filter_mode: str = "raw",
    first_seen: dict[str, pd.Timestamp] | None = None,
) -> dict:
    positions: set[str] = set()
    daily = []
    turnover = []
    holdings = []

    for i, signal_date in enumerate(dates[:-1]):
        execution_date = dates[i + 1]
        day = pred.xs(signal_date, level="datetime").reset_index().sort_values("score", ascending=False).reset_index(drop=True)
        day["rank"] = day.index + 1
        if filter_mode == "jq_filter":
            start_i = max(0, i - args.lookback + 1)
            day = filter_signal(day, close_px.loc[signal_date], amount.iloc[start_i : i + 1], args)
            day["rank"] = day.index + 1
        elif filter_mode == "jq_filter_v1":
            start_i = max(0, i - args.lookback + 1)
            day = filter_signal_v1(
                day,
                signal_date,
                execution_date,
                open_px.loc[execution_date],
                close_px,
                amount.iloc[start_i : i + 1],
                first_seen or {},
                args,
            )
            day["rank"] = day.index + 1

        ranked = day["instrument"].astype(str).tolist()
        rank_map = dict(zip(ranked, day["rank"], strict=False))
        sell = sorted([inst for inst in positions if rank_map.get(inst, 10**9) > topk], key=lambda inst: rank_map.get(inst, 10**9), reverse=True)[
            :n_drop
        ]
        kept = [inst for inst in positions if inst not in set(sell)]
        buy_slots = max(topk - len(kept), 0)
        if positions and sell:
            buy_slots = min(buy_slots, n_drop)
        buy = []
        excluded = set(kept) | set(sell)
        for inst in ranked:
            if len(buy) >= buy_slots:
                break
            if inst not in excluded:
                buy.append(inst)
        new_positions = set(kept) | set(buy)

        if new_positions:
            prev = close_px.loc[signal_date, list(new_positions)].replace([np.inf, -np.inf], np.nan)
            nxt = open_px.loc[execution_date, list(new_positions)].replace([np.inf, -np.inf], np.nan)
            ret = (nxt / prev - 1.0).dropna()
            daily_ret = float(ret.mean()) if not ret.empty else 0.0
        else:
            daily_ret = 0.0
        cost = 0.0003 * len(buy) / max(topk, 1) + 0.0008 * len(sell) / max(topk, 1)
        daily.append(daily_ret - cost)
        turnover.append((len(buy) + len(sell)) / max(topk, 1))
        holdings.append(len(new_positions))
        positions = new_positions

    ret = pd.Series(daily, index=dates[1:])
    nav = (1.0 + ret).cumprod()
    return {
        "cum_return": float(nav.iloc[-1] - 1.0),
        "ann_return": float((1.0 + ret.mean()) ** 252 - 1.0),
        "ir": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else float("nan"),
        "max_drawdown": max_drawdown(nav),
        "avg_turnover": float(np.mean(turnover)),
        "avg_holdings": float(np.mean(holdings)),
    }


def capital_sim(
    pred: pd.DataFrame,
    dates: pd.DatetimeIndex,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    amount: pd.DataFrame,
    topk: int,
    n_drop: int,
    capital: float,
    use_filter: bool,
    args: argparse.Namespace,
    filter_mode: str = "raw",
    first_seen: dict[str, pd.Timestamp] | None = None,
) -> dict:
    cash = float(capital)
    positions: dict[str, int] = {}
    nav_rows = []
    total_cost = 0.0

    for i, signal_date in enumerate(dates[:-1]):
        execution_date = dates[i + 1]
        day = pred.xs(signal_date, level="datetime").reset_index().sort_values("score", ascending=False).reset_index(drop=True)
        day["rank"] = day.index + 1
        if filter_mode == "jq_filter":
            start_i = max(0, i - args.lookback + 1)
            day = filter_signal(day, close_px.loc[signal_date], amount.iloc[start_i : i + 1], args)
            day["rank"] = day.index + 1
        elif filter_mode == "jq_filter_v1":
            start_i = max(0, i - args.lookback + 1)
            day = filter_signal_v1(
                day,
                signal_date,
                execution_date,
                open_px.loc[execution_date],
                close_px,
                amount.iloc[start_i : i + 1],
                first_seen or {},
                args,
            )
            day["rank"] = day.index + 1
        ranked = day["instrument"].astype(str).tolist()
        rank_map = dict(zip(ranked, day["rank"], strict=False))

        close_row = close_px.loc[signal_date]
        equity = cash + sum(shares * safe_price(close_row, inst) for inst, shares in positions.items())
        target_value = min(equity * 0.98 / topk, equity * 0.12)

        sell = sorted([inst for inst in positions if rank_map.get(inst, 10**9) > topk], key=lambda inst: rank_map.get(inst, 10**9), reverse=True)[
            :n_drop
        ]
        kept = [inst for inst in positions if inst not in set(sell)]
        buy_slots = max(topk - len(kept), 0)
        if positions and sell:
            buy_slots = min(buy_slots, n_drop)

        buy = []
        reserved = 0.0
        available_cash = cash + sum(positions.get(inst, 0) * safe_price(close_row, inst) for inst in sell)
        excluded = set(kept) | set(sell)
        for inst in ranked:
            if len(buy) >= buy_slots:
                break
            if inst in excluded:
                continue
            price = safe_price(close_row, inst)
            if price <= 0:
                continue
            if filter_mode == "jq_filter_v1":
                avg_amount = amount.iloc[max(0, i - args.lookback + 1) : i + 1].mean(axis=0, skipna=True)
                if target_value > float(avg_amount.get(inst, 0.0) or 0.0) * args.max_order_amount_pct:
                    continue
            shares = int(math.floor(target_value / (price * 100)) * 100)
            if shares <= 0 and price * 100 <= equity * 0.12:
                shares = 100
            value = shares * price
            if shares > 0 and value + reserved <= available_cash:
                buy.append((inst, shares))
                reserved += value

        open_row = open_px.loc[execution_date]
        for inst in sell:
            shares = positions.get(inst, 0)
            price = safe_price(open_row, inst)
            if shares <= 0 or price <= 0:
                continue
            value = shares * price
            fee = max(value * 0.0008, 5.0)
            cash += value - fee
            total_cost += fee
            positions.pop(inst, None)
        for inst, shares in buy:
            price = safe_price(open_row, inst)
            while shares > 0 and price > 0:
                value = shares * price
                fee = max(value * 0.0003, 5.0)
                if value + fee <= cash:
                    break
                shares -= 100
            if shares <= 0 or price <= 0:
                continue
            value = shares * price
            fee = max(value * 0.0003, 5.0)
            cash -= value + fee
            total_cost += fee
            positions[inst] = positions.get(inst, 0) + shares
        position_value = sum(shares * safe_price(open_row, inst) for inst, shares in positions.items())
        nav_rows.append({"datetime": execution_date, "nav": cash + position_value, "position_value": position_value, "cash": cash, "holdings": len(positions)})

    nav = pd.DataFrame(nav_rows)
    pos_ratio = nav["position_value"] / nav["nav"]
    return {
        "capital": capital,
        "last_nav": float(nav.iloc[-1]["nav"]),
        "cum_return": float(nav.iloc[-1]["nav"] / capital - 1.0),
        "max_drawdown": max_drawdown(nav["nav"]),
        "avg_position_ratio": float(pos_ratio.mean()),
        "last_position_ratio": float(pos_ratio.iloc[-1]),
        "final_holdings": int(nav.iloc[-1]["holdings"]),
        "total_cost": total_cost,
    }


def main() -> None:
    args = parse_args()
    pred_paths = sorted(Path(args.pred_dir).glob("*.pkl"))
    if not pred_paths:
        raise FileNotFoundError(f"No predictions in {args.pred_dir}")
    preds = {model_name(p, args.market): load_pred(p) for p in pred_paths}
    first = next(iter(preds.values()))
    first_seen = load_list_dates(args.list_date_csv) or first_seen_dates(first)
    dates = pd.DatetimeIndex(sorted(first.index.get_level_values("datetime").unique()))
    dates = dates[(dates >= pd.Timestamp(args.start_date)) & (dates <= pd.Timestamp(args.end_date))]
    instruments = sorted(first.index.get_level_values("instrument").unique())
    open_px, close_px, amount, bench_open = load_market_data(args.provider_uri, instruments, str(dates[0].date()), str(dates[-1].date()), args.benchmark)
    bench = bench_open.reindex(dates[1:]).dropna()
    bench_return = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
    pairs = [tuple(int(x) for x in item.split(":")) for item in args.topk_pairs.split(",") if item.strip()]

    rows = []
    for name, pred in preds.items():
        for topk, n_drop in pairs:
            for filter_name, use_filter in [("raw", False), ("jq_filter", True), ("jq_filter_v1", True)]:
                result = simulate_topdrop(pred, dates, open_px, close_px, amount, topk, n_drop, use_filter, args, filter_name, first_seen)
                rows.append({"model": name, "filter": filter_name, "topk": topk, "n_drop": n_drop, **result, "benchmark_return": bench_return})
    out = pd.DataFrame(rows)
    out["excess_return"] = out["cum_return"] - out["benchmark_return"]
    out = out.sort_values(["excess_return", "ir"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("===== top model results =====")
    print(out.head(40).to_string(index=False))

    cap_rows = []
    for _, row in out.head(20).iterrows():
        pred = preds[row["model"]]
        for capital in [100000.0, 200000.0, 1000000.0]:
            result = capital_sim(
                pred,
                dates,
                open_px,
                close_px,
                amount,
                int(row["topk"]),
                int(row["n_drop"]),
                capital,
                row["filter"] != "raw",
                args,
                row["filter"],
                first_seen,
            )
            cap_rows.append({"model": row["model"], "filter": row["filter"], "topk": row["topk"], "n_drop": row["n_drop"], **result, "benchmark_return": bench_return})
    cap = pd.DataFrame(cap_rows)
    cap["excess_return"] = cap["cum_return"] - cap["benchmark_return"]
    cap = cap.sort_values(["excess_return", "capital"], ascending=[False, True])
    cap.to_csv(args.capital_output, index=False)
    print("===== top capital results =====")
    print(cap.head(60).to_string(index=False))
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.capital_output}")


if __name__ == "__main__":
    main()
