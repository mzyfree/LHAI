#!/usr/bin/env python3
"""Online-style strict intraday TOPK simulation from packaged Qlib models.

This runner intentionally favors correctness over speed.  For each signal date
it creates a single-day inference dataset, predicts with the packaged models,
then simulates buying the next trading day's open and selling that day's close.

Trading timeline:
    signal_date close data is available after market close
    entry_date open buys are decided from signal_date predictions
    exit_date close exits positions
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from strict_intraday_cached_runner import (  # noqa: E402
    DEFAULT_MODELS,
    daily_zscore,
    ensure_extracted,
    load_market_data,
    predict_model,
    safe_price,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online-style strict TOPK simulation.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--extract-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--start-date", default="2026-01-05", help="First signal date.")
    parser.add_argument("--end-date", default="2026-05-26", help="Last signal date.")
    parser.add_argument("--topks", default="3,5")
    parser.add_argument(
        "--signal-stride",
        type=int,
        default=1,
        help="Use every Nth signal date. 2 can be used to build staggered two-pool tests.",
    )
    parser.add_argument(
        "--signal-offsets",
        default="",
        help="Comma-separated offsets within --signal-stride. Empty means all offsets. Example: 0 or 1 for two pools.",
    )
    parser.add_argument(
        "--entry-lag-days",
        default="1",
        help="Comma-separated entry lag in trading days after signal date. 1 means T signal -> T+1 open entry.",
    )
    parser.add_argument("--hold-days", default="0", help="Comma-separated holding days after entry. 0 means sell entry day close.")
    parser.add_argument("--capitals", default="100000,200000,1000000")
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--reserve-cash-pct", type=float, default=0.02)
    parser.add_argument("--max-position-pct", type=float, default=0.25)
    parser.add_argument("--allocation-mode", choices=["softmax", "rank", "equal"], default="softmax")
    parser.add_argument("--rank-weights", default="", help="Comma-separated weights for rank allocation, e.g. 30,25,20,15,10.")
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--buy-cost-rate", type=float, default=0.0003)
    parser.add_argument("--sell-cost-rate", type=float, default=0.0008)
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument(
        "--use-pred-cache-only",
        action="store_true",
        help="Only read existing per-day prediction caches; fail fast if a cache is missing.",
    )
    return parser.parse_args()


def fuse_one_day(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    denom = sum(DEFAULT_MODELS.values())
    pieces = []
    for model_name, pred in preds.items():
        pieces.append(daily_zscore(pred["score"]) * (DEFAULT_MODELS[model_name] / denom))
    score = sum(pieces)
    out = score.to_frame("score").sort_index()
    return out


def cached_pred_path(cache_dir: Path, model_name: str, signal_date: pd.Timestamp) -> Path:
    day = str(signal_date.date())
    return cache_dir / f"{model_name}_pred_{day}_{day}.pkl"


def load_cached_one_day(cache_dir: Path, model_name: str, signal_date: pd.Timestamp) -> pd.DataFrame:
    path = cached_pred_path(cache_dir, model_name, signal_date)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction cache: {path}. "
            "Run without --use-pred-cache-only to generate missing daily caches first."
        )
    return pd.read_pickle(path)


def rank_signal(signal: pd.DataFrame, signal_date: pd.Timestamp, signal_dir: Path) -> pd.DataFrame:
    day = signal.xs(signal_date, level="datetime").reset_index()
    day = day.sort_values("score", ascending=False).reset_index(drop=True)
    day["rank"] = day.index + 1
    signal_dir.mkdir(parents=True, exist_ok=True)
    day.loc[:, ["rank", "instrument", "score"]].to_csv(signal_dir / f"signal_{signal_date.date()}.csv", index=False)
    return day


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min()) if len(nav) else 0.0


def build_summary(entry_lag_days: int, hold_days: int, topk: int, rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "entry_lag_days": float(entry_lag_days),
            "hold_days": float(hold_days),
            "topk": topk,
            "days": 0.0,
            "cum_return": 0.0,
            "ann_return": 0.0,
            "ir": float("nan"),
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_daily_return": 0.0,
        }
    ret = pd.Series([float(r["net_return"]) for r in rows], index=pd.to_datetime([r["trade_date"] for r in rows]))
    nav = (1.0 + ret).cumprod()
    std = ret.std()
    return {
        "entry_lag_days": float(entry_lag_days),
        "hold_days": float(hold_days),
        "topk": topk,
        "days": float(len(ret)),
        "cum_return": float(nav.iloc[-1] - 1.0),
        "ann_return": float((1.0 + ret.mean()) ** 252 - 1.0),
        "ir": float(ret.mean() / std * np.sqrt(252)) if std and std > 0 else float("nan"),
        "max_drawdown": max_drawdown(nav),
        "win_rate": float((ret > 0).mean()),
        "avg_daily_return": float(ret.mean()),
    }


def softmax_weights(scores: pd.Series, temperature: float) -> pd.Series:
    if scores.empty:
        return scores
    temp = max(float(temperature), 1e-6)
    values = scores.astype(float).to_numpy() / temp
    values = values - np.nanmax(values)
    weights = np.exp(values)
    total = np.nansum(weights)
    if not np.isfinite(total) or total <= 0:
        return pd.Series(np.repeat(1.0 / len(scores), len(scores)), index=scores.index)
    return pd.Series(weights / total, index=scores.index)


def allocation_weights(picks: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    scores = picks.set_index("instrument")["score"]
    if scores.empty:
        return scores
    if args.allocation_mode == "softmax":
        return softmax_weights(scores, args.score_temperature)
    if args.allocation_mode == "equal":
        return pd.Series(np.repeat(1.0 / len(scores), len(scores)), index=scores.index)

    raw_weights = [float(x) for x in args.rank_weights.split(",") if x.strip()]
    if len(raw_weights) < len(scores):
        raise ValueError(f"--rank-weights needs at least {len(scores)} values for topk={len(scores)}")
    weights = np.array(raw_weights[: len(scores)], dtype=float)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid --rank-weights: {args.rank_weights}")
    return pd.Series(weights / total, index=scores.index)


def round_lot_shares(target_value: float, price: float, lot_size: int) -> int:
    if target_value <= 0 or price <= 0:
        return 0
    return int(math.floor(target_value / (price * lot_size)) * lot_size)


def simulate_capital_trade(
    picks: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    start_nav: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    cash = float(start_nav)
    investable = cash * (1.0 - args.reserve_cash_pct)
    weights = allocation_weights(picks, args)
    max_position_value = start_nav * args.max_position_pct
    fill_rows = []

    for inst, weight in weights.items():
        open_price = safe_price(open_px.loc[entry_date], str(inst))
        close_price = safe_price(close_px.loc[exit_date], str(inst))
        if open_price <= 0 or close_price <= 0:
            continue
        target_value = min(investable * float(weight), max_position_value)
        shares = round_lot_shares(target_value, open_price, args.lot_size)
        buy_value = shares * open_price
        buy_cost = buy_value * args.buy_cost_rate
        if shares <= 0 or buy_value + buy_cost > cash:
            continue
        cash -= buy_value + buy_cost
        sell_value = shares * close_price
        sell_cost = sell_value * args.sell_cost_rate
        cash += sell_value - sell_cost
        fill_rows.append(
            {
                "instrument": str(inst),
                "shares": shares,
                "open": open_price,
                "close": close_price,
                "score": float(picks.set_index("instrument").loc[inst, "score"]),
                "buy_value": buy_value,
                "sell_value": sell_value,
                "cost": buy_cost + sell_cost,
            }
        )

    end_nav = cash
    return {
        "start_nav": start_nav,
        "end_nav": end_nav,
        "daily_return": end_nav / start_nav - 1.0 if start_nav > 0 else 0.0,
        "filled_count": len(fill_rows),
        "filled_value": sum(float(r["buy_value"]) for r in fill_rows),
        "trade_cost": sum(float(r["cost"]) for r in fill_rows),
        "fills": fill_rows,
    }


def build_capital_summary(entry_lag_days: int, hold_days: int, capital: float, topk: int, rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "capital": capital,
            "topk": topk,
            "entry_lag_days": float(entry_lag_days),
            "hold_days": float(hold_days),
            "days": 0.0,
            "last_nav": capital,
            "cum_return": 0.0,
            "ann_return": 0.0,
            "ir": float("nan"),
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_daily_return": 0.0,
            "avg_position_ratio": 0.0,
            "avg_filled_count": 0.0,
            "total_cost": 0.0,
        }
    ret = pd.Series([float(r["daily_return"]) for r in rows], index=pd.to_datetime([r["trade_date"] for r in rows]))
    nav = pd.Series([float(r["end_nav"]) for r in rows], index=ret.index)
    std = ret.std()
    return {
        "capital": capital,
        "topk": topk,
        "entry_lag_days": float(entry_lag_days),
        "hold_days": float(hold_days),
        "days": float(len(ret)),
        "last_nav": float(nav.iloc[-1]),
        "cum_return": float(nav.iloc[-1] / capital - 1.0),
        "ann_return": float((1.0 + ret.mean()) ** 252 - 1.0),
        "ir": float(ret.mean() / std * np.sqrt(252)) if std and std > 0 else float("nan"),
        "max_drawdown": max_drawdown(nav),
        "win_rate": float((ret > 0).mean()),
        "avg_daily_return": float(ret.mean()),
        "avg_position_ratio": float(np.mean([float(r["filled_value"]) / float(r["start_nav"]) for r in rows])),
        "avg_filled_count": float(np.mean([float(r["filled_count"]) for r in rows])),
        "total_cost": float(sum(float(r["trade_cost"]) for r in rows)),
    }


def mark_positions_value(positions: list[dict[str, object]], price_date: pd.Timestamp, close_px: pd.DataFrame) -> float:
    if price_date not in close_px.index:
        return 0.0
    total = 0.0
    close_row = close_px.loc[price_date]
    for pos in positions:
        price = safe_price(close_row, str(pos["instrument"]))
        total += int(pos["shares"]) * price
    return total


def sell_due_positions(
    positions: list[dict[str, object]],
    due_date: pd.Timestamp,
    close_px: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], float, float, int]:
    remaining = []
    cash_delta = 0.0
    sell_value = 0.0
    sell_count = 0
    close_row = close_px.loc[due_date]
    for pos in positions:
        if pd.Timestamp(pos["exit_date"]) <= due_date:
            price = safe_price(close_row, str(pos["instrument"]))
            if price <= 0:
                remaining.append(pos)
                continue
            value = int(pos["shares"]) * price
            cost = value * args.sell_cost_rate
            cash_delta += value - cost
            sell_value += value
            sell_count += 1
        else:
            remaining.append(pos)
    return remaining, cash_delta, sell_value, sell_count


def simulate_portfolio_queue(
    signal_days: dict[pd.Timestamp, pd.DataFrame],
    trade_windows: dict[pd.Timestamp, dict[tuple[int, int], tuple[pd.Timestamp, pd.Timestamp]]],
    calendar: pd.DatetimeIndex,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    capital: float,
    topk: int,
    entry_lag_days: int,
    hold_days: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    cash = float(capital)
    positions: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    processed_dates: set[pd.Timestamp] = set()

    def record_day(day: pd.Timestamp, signal_date: pd.Timestamp | None, buy_value: float, sell_value: float, trade_cost: float, n_buy: int, n_sell: int, picks: list[str]) -> None:
        position_value = mark_positions_value(positions, day, close_px)
        nav = cash + position_value
        rows.append(
            {
                "hold_days": hold_days,
                "signal_stride": args.signal_stride,
                "signal_offsets": args.signal_offsets or ",".join(str(x) for x in range(args.signal_stride)),
                "entry_lag_days": entry_lag_days,
                "capital": capital,
                "topk": topk,
                "date": day.date(),
                "signal_date": signal_date.date() if signal_date is not None else "",
                "cash": cash,
                "position_value": position_value,
                "nav": nav,
                "position_ratio": position_value / nav if nav > 0 else 0.0,
                "buy_value": buy_value,
                "sell_value": sell_value,
                "trade_cost": trade_cost,
                "n_buy": n_buy,
                "n_sell": n_sell,
                "n_positions": len(positions),
                "picks": ",".join(picks),
            }
        )
        processed_dates.add(day)

    for signal_date in calendar:
        if signal_date not in signal_days or signal_date not in trade_windows:
            continue
        window_key = (entry_lag_days, hold_days)
        if window_key not in trade_windows[signal_date]:
            continue
        entry_date, exit_date = trade_windows[signal_date][window_key]
        if entry_date not in open_px.index or entry_date not in close_px.index or exit_date not in close_px.index:
            continue

        # If a gap appears, settle positions whose exit date passed before this open.
        stale_exit_dates = sorted({pd.Timestamp(pos["exit_date"]) for pos in positions if pd.Timestamp(pos["exit_date"]) < entry_date})
        for due_date in stale_exit_dates:
            if due_date not in close_px.index:
                continue
            positions, cash_delta, sell_value, n_sell = sell_due_positions(positions, due_date, close_px, args)
            cash += cash_delta
            record_day(due_date, None, 0.0, sell_value, sell_value * args.sell_cost_rate, 0, n_sell, [])

        day = signal_days[signal_date].head(topk).copy()
        picks = day["instrument"].astype(str).tolist()
        equity_before_buy = cash + mark_positions_value(positions, signal_date, close_px)
        reserve_cash = equity_before_buy * args.reserve_cash_pct
        buy_budget = max(cash - reserve_cash, 0.0)
        weights = allocation_weights(day, args)
        max_position_value = equity_before_buy * args.max_position_pct
        open_row = open_px.loc[entry_date]
        buy_value = 0.0
        buy_cost_total = 0.0
        n_buy = 0

        for inst, weight in weights.items():
            price = safe_price(open_row, str(inst))
            if price <= 0:
                continue
            target_value = min(buy_budget * float(weight), max_position_value)
            shares = round_lot_shares(target_value, price, args.lot_size)
            value = shares * price
            cost = value * args.buy_cost_rate
            if shares <= 0 or value + cost > cash:
                continue
            cash -= value + cost
            buy_value += value
            buy_cost_total += cost
            n_buy += 1
            positions.append(
                {
                    "instrument": str(inst),
                    "shares": shares,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": price,
                    "score": float(day.set_index("instrument").loc[inst, "score"]),
                }
            )

        positions, cash_delta, sell_value, n_sell = sell_due_positions(positions, entry_date, close_px, args)
        cash += cash_delta
        sell_cost = sell_value * args.sell_cost_rate
        record_day(entry_date, signal_date, buy_value, sell_value, buy_cost_total + sell_cost, n_buy, n_sell, picks)

    # Settle any remaining positions so the final NAV reflects completed trades.
    for due_date in sorted({pd.Timestamp(pos["exit_date"]) for pos in positions}):
        if due_date in processed_dates or due_date not in close_px.index:
            continue
        positions, cash_delta, sell_value, n_sell = sell_due_positions(positions, due_date, close_px, args)
        cash += cash_delta
        record_day(due_date, None, 0.0, sell_value, sell_value * args.sell_cost_rate, 0, n_sell, [])

    if not rows:
        return {
            "hold_days": float(hold_days),
            "entry_lag_days": float(entry_lag_days),
            "capital": capital,
            "topk": topk,
            "days": 0.0,
            "last_nav": capital,
            "cum_return": 0.0,
            "ann_return": 0.0,
            "ir": float("nan"),
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_daily_return": 0.0,
            "avg_position_ratio": 0.0,
            "avg_positions": 0.0,
            "total_cost": 0.0,
        }, []

    daily = pd.DataFrame(rows).sort_values("date")
    nav = pd.Series(daily["nav"].astype(float).to_numpy(), index=pd.to_datetime(daily["date"]))
    ret = nav.pct_change().fillna(nav.iloc[0] / capital - 1.0)
    std = ret.std()
    summary = {
        "hold_days": float(hold_days),
        "entry_lag_days": float(entry_lag_days),
        "capital": capital,
        "topk": topk,
        "days": float(len(daily)),
        "last_nav": float(nav.iloc[-1]),
        "cum_return": float(nav.iloc[-1] / capital - 1.0),
        "ann_return": float((1.0 + ret.mean()) ** 252 - 1.0),
        "ir": float(ret.mean() / std * np.sqrt(252)) if std and std > 0 else float("nan"),
        "max_drawdown": max_drawdown(nav),
        "win_rate": float((ret > 0).mean()),
        "avg_daily_return": float(ret.mean()),
        "avg_position_ratio": float(daily["position_ratio"].mean()),
        "avg_positions": float(daily["n_positions"].mean()),
        "total_cost": float(daily["trade_cost"].sum()),
    }
    return summary, rows


def main() -> int:
    args = parse_args()
    provider_uri = Path(args.provider_uri).resolve()
    package_path = Path(args.package).resolve()
    extract_dir = Path(args.extract_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Qlib provider: {provider_uri}")
    qlib.init(provider_uri=str(provider_uri), region="cn")
    model_root = None
    if not args.use_pred_cache_only:
        model_root = ensure_extracted(package_path, extract_dir)
        print(f"Model root: {model_root}")
    else:
        print("Model root: skipped (--use-pred-cache-only)")
    print(f"Cache dir: {cache_dir}")

    calendar = pd.DatetimeIndex(D.calendar(start_time=args.start_date, end_time=args.end_date, freq="day"))
    if len(calendar) < 2:
        raise RuntimeError(f"Not enough signal dates in range: {args.start_date} -> {args.end_date}")
    if args.signal_stride < 1:
        raise ValueError("--signal-stride must be positive")
    if args.signal_offsets:
        signal_offsets = {int(x) for x in args.signal_offsets.split(",") if x.strip()}
    else:
        signal_offsets = set(range(args.signal_stride))
    invalid_offsets = sorted(x for x in signal_offsets if x < 0 or x >= args.signal_stride)
    if invalid_offsets:
        raise ValueError(f"--signal-offsets must be in [0, {args.signal_stride - 1}], got {invalid_offsets}")
    signal_calendar = pd.DatetimeIndex(
        [day for idx, day in enumerate(calendar) if idx % args.signal_stride in signal_offsets]
    )
    if len(signal_calendar) < 1:
        raise RuntimeError("No signal dates left after applying --signal-stride/--signal-offsets")
    hold_days_list = [int(x) for x in args.hold_days.split(",") if x.strip()]
    if any(x < 0 for x in hold_days_list):
        raise ValueError("--hold-days must be non-negative")
    entry_lag_days_list = [int(x) for x in args.entry_lag_days.split(",") if x.strip()]
    if any(x < 1 for x in entry_lag_days_list):
        raise ValueError("--entry-lag-days must be positive; 1 means next trading day")
    max_hold_days = max(hold_days_list) if hold_days_list else 0
    max_entry_lag_days = max(entry_lag_days_list) if entry_lag_days_list else 1

    # Need entry day and exit day prices for the last signal date.
    last_future_cal = D.calendar(start_time=str(signal_calendar[-1].date()), freq="day", future=True)
    if len(last_future_cal) <= max_entry_lag_days + max_hold_days:
        raise RuntimeError(
            f"Not enough future calendar dates after {signal_calendar[-1].date()} "
            f"for entry_lag_days={max_entry_lag_days}, hold_days={max_hold_days}"
        )
    last_trade_date = pd.Timestamp(last_future_cal[max_entry_lag_days + max_hold_days])
    market_end = str(last_trade_date.date())
    print(f"Signal dates: {calendar[0].date()} -> {calendar[-1].date()} ({len(calendar)} dates)")
    print(
        f"Selected signal dates: {signal_calendar[0].date()} -> {signal_calendar[-1].date()} "
        f"({len(signal_calendar)} dates), stride={args.signal_stride}, offsets={sorted(signal_offsets)}"
    )
    print(f"Last trade date needed: {market_end}")

    all_instruments = None
    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    capitals = [float(x) for x in args.capitals.split(",") if x.strip()]
    rows_by_case: dict[tuple[int, int, int], list[dict[str, object]]] = {
        (entry_lag_days, hold_days, topk): []
        for entry_lag_days in entry_lag_days_list
        for hold_days in hold_days_list
        for topk in topks
    }
    capital_rows_by_case: dict[tuple[int, int, float, int], list[dict[str, object]]] = {
        (entry_lag_days, hold_days, capital, topk): []
        for entry_lag_days in entry_lag_days_list
        for hold_days in hold_days_list
        for capital in capitals
        for topk in topks
    }
    capital_nav: dict[tuple[int, int, float, int], float] = {
        (entry_lag_days, hold_days, capital, topk): capital
        for entry_lag_days in entry_lag_days_list
        for hold_days in hold_days_list
        for capital in capitals
        for topk in topks
    }
    signal_days: dict[pd.Timestamp, pd.DataFrame] = {}
    trade_windows: dict[pd.Timestamp, dict[tuple[int, int], tuple[pd.Timestamp, pd.Timestamp]]] = {}
    signal_dir = cache_dir / "signals"

    # Load market prices once.  This includes the next trade date used for last signal.
    instruments = D.instruments("csi1000")
    open_px, close_px, amount = load_market_data(str(provider_uri), instruments, args.start_date, market_end)

    for i, signal_date in enumerate(signal_calendar):
        future_cal = D.calendar(start_time=str(signal_date.date()), freq="day", future=True)
        if len(future_cal) <= max_entry_lag_days:
            print(f"skip signal={signal_date.date()}: no enough future trading days for entry_lag_days={max_entry_lag_days}")
            continue
        if len(future_cal) <= max_entry_lag_days + max_hold_days:
            print(
                f"skip signal={signal_date.date()}: not enough future calendar for "
                f"entry_lag_days={max_entry_lag_days}, hold_days={max_hold_days}"
            )
            continue
        if signal_date not in close_px.index:
            print(f"skip signal={signal_date.date()}: missing signal close data")
            continue
        trade_date_by_lag = {entry_lag_days: pd.Timestamp(future_cal[entry_lag_days]) for entry_lag_days in entry_lag_days_list}
        exit_dates = {
            (entry_lag_days, hold_days): pd.Timestamp(future_cal[entry_lag_days + hold_days])
            for entry_lag_days in entry_lag_days_list
            for hold_days in hold_days_list
        }
        missing_entry_dates = [
            entry_date.date() for entry_date in trade_date_by_lag.values() if entry_date not in open_px.index
        ]
        missing_exit_dates = [exit_date.date() for exit_date in exit_dates.values() if exit_date not in close_px.index]
        if missing_entry_dates or missing_exit_dates:
            print(
                f"skip signal={signal_date.date()}: "
                f"missing entry open {missing_entry_dates} or exit close {missing_exit_dates}"
            )
            continue
        print(
            f"strict infer signal={signal_date.date()} "
            f"entry_lags={','.join(str(x) for x in entry_lag_days_list)}",
            flush=True,
        )

        preds = {}
        for model_name in DEFAULT_MODELS:
            if args.use_pred_cache_only:
                preds[model_name] = load_cached_one_day(cache_dir, model_name, signal_date)
            else:
                if model_root is None:
                    raise RuntimeError("model_root is required when generating predictions")
                preds[model_name] = predict_model(
                    model_root,
                    model_name,
                    str(signal_date.date()),
                    str(signal_date.date()),
                    cache_dir,
                    args.force_predict,
                )
        signal = fuse_one_day(preds)
        if all_instruments is None:
            all_instruments = sorted(signal.index.get_level_values("instrument").unique())
        day = rank_signal(signal, signal_date, signal_dir)

        close_row = close_px.loc[signal_date]
        avg_amount = amount.loc[:signal_date].tail(args.lookback).mean(axis=0, skipna=True)
        keep = []
        for inst in day["instrument"].astype(str):
            price = safe_price(close_row, inst)
            amt = float(avg_amount.get(inst, 0.0) or 0.0)
            keep.append(args.min_price <= price <= args.max_price and amt >= args.min_amount)
        day = day.loc[keep].reset_index(drop=True)
        signal_days[signal_date] = day

        for entry_lag_days in entry_lag_days_list:
            trade_date = trade_date_by_lag[entry_lag_days]
            for hold_days in hold_days_list:
                exit_date = exit_dates[(entry_lag_days, hold_days)]
                trade_windows.setdefault(signal_date, {})[(entry_lag_days, hold_days)] = (trade_date, exit_date)
                for topk in topks:
                    picks_df = day.head(topk).copy()
                    picks = picks_df["instrument"].astype(str).tolist()
                    stock_returns = []
                    filled = []
                    for inst in picks:
                        open_price = safe_price(open_px.loc[trade_date], inst)
                        close_price = safe_price(close_px.loc[exit_date], inst)
                        if open_price > 0 and close_price > 0:
                            stock_returns.append(close_price / open_price - 1.0)
                            filled.append(inst)
                    gross_return = float(np.mean(stock_returns)) if stock_returns else 0.0
                    cost = args.buy_cost_rate + args.sell_cost_rate if stock_returns else 0.0
                    rows_by_case[(entry_lag_days, hold_days, topk)].append(
                        {
                    "signal_stride": args.signal_stride,
                    "signal_offsets": ",".join(str(x) for x in sorted(signal_offsets)),
                    "entry_lag_days": entry_lag_days,
                            "hold_days": hold_days,
                            "topk": topk,
                            "signal_date": signal_date.date(),
                            "entry_date": trade_date.date(),
                            "exit_date": exit_date.date(),
                            "trade_date": exit_date.date(),
                            "n_picks": len(filled),
                            "gross_return": gross_return,
                            "net_return": gross_return - cost,
                            "picks": ",".join(picks),
                            "filled_picks": ",".join(filled),
                        }
                    )
                    for capital in capitals:
                        case = (entry_lag_days, hold_days, capital, topk)
                        cap_result = simulate_capital_trade(
                            picks_df, trade_date, exit_date, open_px, close_px, capital_nav[case], args
                        )
                        capital_nav[case] = float(cap_result["end_nav"])
                        capital_rows_by_case[case].append(
                            {
                                "signal_stride": args.signal_stride,
                                "signal_offsets": ",".join(str(x) for x in sorted(signal_offsets)),
                                "entry_lag_days": entry_lag_days,
                                "hold_days": hold_days,
                                "capital": capital,
                                "topk": topk,
                                "signal_date": signal_date.date(),
                                "entry_date": trade_date.date(),
                                "exit_date": exit_date.date(),
                                "trade_date": exit_date.date(),
                                "picks": ",".join(picks),
                                "filled_picks": ",".join(str(fill["instrument"]) for fill in cap_result["fills"]),
                                **{k: v for k, v in cap_result.items() if k != "fills"},
                            }
                        )

    summaries = [
        build_summary(entry_lag_days, hold_days, topk, rows_by_case[(entry_lag_days, hold_days, topk)])
        for entry_lag_days in entry_lag_days_list
        for hold_days in hold_days_list
        for topk in topks
    ]
    daily_rows = [row for rows in rows_by_case.values() for row in rows]
    summary_df = pd.DataFrame(summaries)
    daily_df = pd.DataFrame(daily_rows)
    signal_offset_label = ",".join(str(x) for x in sorted(signal_offsets))
    for df in (summary_df, daily_df):
        if not df.empty:
            if "signal_offsets" not in df.columns:
                df.insert(0, "signal_offsets", signal_offset_label)
            if "signal_stride" not in df.columns:
                df.insert(0, "signal_stride", args.signal_stride)
    capital_summaries = [
        build_capital_summary(
            entry_lag_days,
            hold_days,
            capital,
            topk,
            capital_rows_by_case[(entry_lag_days, hold_days, capital, topk)],
        )
        for entry_lag_days in entry_lag_days_list
        for hold_days in hold_days_list
        for capital in capitals
        for topk in topks
    ]
    capital_summary_df = pd.DataFrame(capital_summaries)
    capital_daily_df = pd.DataFrame([row for rows in capital_rows_by_case.values() for row in rows])
    for df in (capital_summary_df, capital_daily_df):
        if not df.empty:
            if "signal_offsets" not in df.columns:
                df.insert(0, "signal_offsets", signal_offset_label)
            if "signal_stride" not in df.columns:
                df.insert(0, "signal_stride", args.signal_stride)

    queue_summaries = []
    queue_daily_rows = []
    for entry_lag_days in entry_lag_days_list:
        for hold_days in hold_days_list:
            for capital in capitals:
                for topk in topks:
                    summary, rows = simulate_portfolio_queue(
                        signal_days,
                        trade_windows,
                        signal_calendar,
                        open_px,
                        close_px,
                        capital,
                        topk,
                        entry_lag_days,
                        hold_days,
                        args,
                    )
                    queue_summaries.append(summary)
                    queue_daily_rows.extend(rows)
    queue_summary_df = pd.DataFrame(queue_summaries)
    queue_daily_df = pd.DataFrame(queue_daily_rows)
    for df in (queue_summary_df, queue_daily_df):
        if not df.empty:
            if "signal_offsets" not in df.columns:
                df.insert(0, "signal_offsets", signal_offset_label)
            if "signal_stride" not in df.columns:
                df.insert(0, "signal_stride", args.signal_stride)

    summary_path = output_dir / "strict_online_intraday_topk_2026_summary.csv"
    daily_path = output_dir / "strict_online_intraday_topk_2026_daily.csv"
    capital_summary_path = output_dir / "strict_online_intraday_capital_2026_summary.csv"
    capital_daily_path = output_dir / "strict_online_intraday_capital_2026_daily.csv"
    queue_summary_path = output_dir / "strict_online_portfolio_queue_2026_summary.csv"
    queue_daily_path = output_dir / "strict_online_portfolio_queue_2026_daily.csv"
    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    capital_summary_df.to_csv(capital_summary_path, index=False)
    capital_daily_df.to_csv(capital_daily_path, index=False)
    queue_summary_df.to_csv(queue_summary_path, index=False)
    queue_daily_df.to_csv(queue_daily_path, index=False)

    print("===== strict online intraday topk summary =====")
    print(
        summary_df.sort_values(["cum_return", "entry_lag_days", "hold_days", "topk"], ascending=[False, True, True, True]).to_string(index=False)
    )
    print("\n===== strict online intraday capital summary =====")
    print(
        capital_summary_df.sort_values(["cum_return", "entry_lag_days", "hold_days", "capital"], ascending=[False, True, True, True]).to_string(index=False)
    )
    print("\n===== strict online portfolio queue summary =====")
    print(
        queue_summary_df.sort_values(["cum_return", "entry_lag_days", "hold_days", "capital"], ascending=[False, True, True, True]).to_string(index=False)
    )
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {daily_path}")
    print(f"Wrote: {capital_summary_path}")
    print(f"Wrote: {capital_daily_path}")
    print(f"Wrote: {queue_summary_path}")
    print(f"Wrote: {queue_daily_path}")
    print(f"Wrote signals: {signal_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
