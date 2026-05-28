from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from run_daily_fusion_signal import choose_trade_date
from run_prediction_fusion import daily_zscore, load_pred


DEFAULT_PROVIDER_URI = "/root/autodl-tmp/llhh/reference_cn_data/cn_data"
DEFAULT_STATE_DIR = "/root/autodl-tmp/llhh/paper_trading"
DEFAULT_REPORT_DIR = "/root/autodl-tmp/llhh/reports/paper_trading"


@dataclass(frozen=True)
class CostConfig:
    buy_rate: float
    sell_rate: float
    min_cost: float


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def account_path(state_dir: Path) -> Path:
    return state_dir / "paper_account.json"


def positions_path(state_dir: Path) -> Path:
    return state_dir / "paper_positions.csv"


def nav_path(state_dir: Path) -> Path:
    return state_dir / "paper_nav.csv"


def pending_orders_path(state_dir: Path, execution_date: pd.Timestamp) -> Path:
    return state_dir / f"pending_orders_{execution_date.date()}.csv"


def trades_path(state_dir: Path, execution_date: pd.Timestamp) -> Path:
    return state_dir / f"paper_trades_{execution_date.date()}.csv"


def report_path(report_dir: Path, phase: str, date: pd.Timestamp) -> Path:
    return report_dir / f"paper_{phase}_{date.date()}.md"


def load_account(state_dir: Path, capital: float) -> dict:
    path = account_path(state_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "initial_capital": float(capital),
        "cash": float(capital),
        "last_nav": float(capital),
        "total_cost": 0.0,
        "created_at": pd.Timestamp.now().isoformat(),
    }


def save_account(state_dir: Path, account: dict) -> None:
    account_path(state_dir).write_text(json.dumps(account, indent=2, sort_keys=True), encoding="utf-8")


def load_positions(state_dir: Path) -> pd.DataFrame:
    path = positions_path(state_dir)
    if not path.exists():
        return pd.DataFrame(columns=["instrument", "shares", "cost_basis"])
    df = pd.read_csv(path)
    required = {"instrument", "shares"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Position file missing columns: {sorted(missing)}")
    if "cost_basis" not in df.columns:
        df["cost_basis"] = 0.0
    df["instrument"] = df["instrument"].astype(str)
    df["shares"] = df["shares"].astype(int)
    df["cost_basis"] = df["cost_basis"].astype(float)
    return df[df["shares"] > 0].reset_index(drop=True)


def save_positions(state_dir: Path, positions: pd.DataFrame) -> None:
    cols = ["instrument", "shares", "cost_basis"]
    out = positions.loc[:, cols].copy() if not positions.empty else pd.DataFrame(columns=cols)
    out = out[out["shares"] > 0].sort_values("instrument").reset_index(drop=True)
    out.to_csv(positions_path(state_dir), index=False)


def append_nav(state_dir: Path, row: dict) -> None:
    path = nav_path(state_dir)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
        out = out.drop_duplicates(["datetime", "phase"], keep="last")
    else:
        out = df
    out.to_csv(path, index=False)


def calc_cost(value: float, action: str, cost: CostConfig) -> float:
    if value <= 0:
        return 0.0
    rate = cost.buy_rate if action == "BUY" else cost.sell_rate
    return max(value * rate, cost.min_cost)


def round_lot_shares(target_value: float, price: float, lot_size: int) -> int:
    if target_value <= 0 or price <= 0:
        return 0
    return int(math.floor(target_value / (price * lot_size)) * lot_size)


def init_qlib(provider_uri: str) -> None:
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=provider_uri, region=REG_CN)


def load_raw_prices(
    provider_uri: str,
    instruments: Iterable[str],
    date: pd.Timestamp,
    field: str,
) -> pd.DataFrame:
    from qlib.data import D

    instruments = sorted({str(x) for x in instruments if str(x)})
    if not instruments:
        return pd.DataFrame(columns=["instrument", "price", "adjusted_price", "factor"])
    init_qlib(provider_uri)
    qfield = f"${field}"
    data = D.features(instruments, [qfield, "$factor"], start_time=date, end_time=date, freq="day")
    if data is None or data.empty:
        return pd.DataFrame(columns=["instrument", "price", "adjusted_price", "factor"])
    data = data.rename(columns={qfield: "adjusted_price", "$factor": "factor"}).reset_index()
    data["price"] = data["adjusted_price"]
    valid_factor = data["factor"].notna() & (data["factor"] > 0)
    data.loc[valid_factor, "price"] = data.loc[valid_factor, "adjusted_price"] / data.loc[valid_factor, "factor"]
    return data.loc[:, ["instrument", "price", "adjusted_price", "factor"]].dropna(subset=["instrument", "price"])


def next_trade_date(provider_uri: str, signal_date: pd.Timestamp, explicit: str | None) -> pd.Timestamp:
    if explicit:
        return pd.Timestamp(explicit)
    try:
        from qlib.data import D

        init_qlib(provider_uri)
        cal = pd.DatetimeIndex(D.calendar(start_time=signal_date, freq="day"))
        future = cal[cal > signal_date]
        if len(future) > 0:
            return pd.Timestamp(future[0])
    except Exception:
        pass
    # The local data bundle often ends at the latest signal date, so the next
    # exchange date is not always knowable after close. Business-day fallback is
    # good enough for naming the pending order file; after-open can override it.
    return pd.Timestamp(signal_date) + pd.offsets.BDay(1)


def fuse_three(pred_a: pd.DataFrame, pred_b: pd.DataFrame, pred_c: pd.DataFrame, weights: tuple[float, float, float]) -> pd.DataFrame:
    merged = (
        pred_a.rename(columns={"score": "score_a"})
        .join(pred_b.rename(columns={"score": "score_b"}), how="inner")
        .join(pred_c.rename(columns={"score": "score_c"}), how="inner")
    )
    if merged.empty:
        raise ValueError("No overlapping prediction index among the three inputs.")
    wa, wb, wc = weights
    denom = wa + wb + wc
    score = (
        wa / denom * daily_zscore(merged["score_a"])
        + wb / denom * daily_zscore(merged["score_b"])
        + wc / denom * daily_zscore(merged["score_c"])
    )
    return score.to_frame("score").sort_index()


def load_or_fuse_prediction(args: argparse.Namespace) -> pd.DataFrame:
    if args.fused_pred:
        return load_pred(Path(args.fused_pred).expanduser().resolve())
    missing = [name for name in ["pred_a", "pred_b", "pred_c"] if getattr(args, name) is None]
    if missing:
        raise ValueError("Provide either --fused-pred or all of --pred-a/--pred-b/--pred-c.")
    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) != 3:
        raise ValueError("--weights must contain three comma-separated numbers, e.g. 10,1,1")
    return fuse_three(
        load_pred(Path(args.pred_a).expanduser().resolve()),
        load_pred(Path(args.pred_b).expanduser().resolve()),
        load_pred(Path(args.pred_c).expanduser().resolve()),
        weights,  # type: ignore[arg-type]
    )


def ranked_signal(pred: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    day = pred.xs(signal_date, level="datetime").reset_index()
    day = day.sort_values("score", ascending=False).reset_index(drop=True)
    day["rank"] = day.index + 1
    return day.loc[:, ["rank", "instrument", "score"]]


def select_topk_dropout_targets(signal: pd.DataFrame, positions: pd.DataFrame, topk: int, n_drop: int) -> tuple[list[str], list[str], list[str]]:
    ranked = signal["instrument"].astype(str).tolist()
    rank_map = dict(zip(signal["instrument"].astype(str), signal["rank"].astype(int), strict=False))
    current = positions["instrument"].astype(str).tolist() if not positions.empty else []

    if not current:
        targets = ranked[:topk]
        return targets, [], targets

    def rank_or_inf(inst: str) -> float:
        return float(rank_map.get(inst, 10**9))

    outside = [inst for inst in current if rank_or_inf(inst) > topk]
    sell = sorted(outside, key=rank_or_inf, reverse=True)[:n_drop]
    keep = [inst for inst in current if inst not in set(sell)]

    buy_slots = max(len(sell), topk - len(keep))
    buy_slots = min(max(buy_slots, 0), n_drop if sell else topk - len(keep))
    buy = [inst for inst in ranked if inst not in set(keep)][:buy_slots]
    targets = keep + buy
    return targets[:topk], sell, buy


def build_pending_orders(
    *,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    account: dict,
    positions: pd.DataFrame,
    topk: int,
    n_drop: int,
    lot_size: int,
    reserve_cash_pct: float,
    max_position_pct: float,
) -> tuple[pd.DataFrame, dict]:
    price_map = prices.set_index("instrument")["price"].to_dict()
    targets, sell_list, buy_list = select_topk_dropout_targets(signal, positions, topk, n_drop)
    pos_map = positions.set_index("instrument")["shares"].to_dict() if not positions.empty else {}
    current_value = sum(int(shares) * float(price_map.get(inst, 0.0)) for inst, shares in pos_map.items())
    equity = float(account["cash"]) + current_value
    target_value = min(equity * (1.0 - reserve_cash_pct) / topk, equity * max_position_pct)

    order_rows = []
    for inst in sell_list:
        shares = int(pos_map.get(inst, 0))
        if shares > 0:
            order_rows.append(
                {
                    "signal_date": signal_date.date(),
                    "execution_date": execution_date.date(),
                    "instrument": inst,
                    "action": "SELL",
                    "shares": shares,
                    "estimated_price": float(price_map.get(inst, 0.0)),
                    "reason": "dropout",
                }
            )

    for inst in buy_list:
        price = float(price_map.get(inst, 0.0))
        shares = round_lot_shares(target_value, price, lot_size)
        if shares <= 0:
            order_rows.append(
                {
                    "signal_date": signal_date.date(),
                    "execution_date": execution_date.date(),
                    "instrument": inst,
                    "action": "SKIP",
                    "shares": 0,
                    "estimated_price": price,
                    "reason": "cannot_buy_one_lot",
                }
            )
            continue
        order_rows.append(
            {
                "signal_date": signal_date.date(),
                "execution_date": execution_date.date(),
                "instrument": inst,
                "action": "BUY",
                "shares": shares,
                "estimated_price": price,
                "reason": "top_rank",
            }
        )

    orders = pd.DataFrame(order_rows)
    if orders.empty:
        orders = pd.DataFrame(
            columns=["signal_date", "execution_date", "instrument", "action", "shares", "estimated_price", "reason"]
        )
    summary = {
        "signal_date": str(signal_date.date()),
        "execution_date": str(execution_date.date()),
        "topk": int(topk),
        "n_drop": int(n_drop),
        "cash": float(account["cash"]),
        "marked_position_value": float(current_value),
        "marked_equity": float(equity),
        "n_current": int(len(pos_map)),
        "n_targets": int(len(targets)),
        "n_sell": int((orders["action"] == "SELL").sum()) if not orders.empty else 0,
        "n_buy": int((orders["action"] == "BUY").sum()) if not orders.empty else 0,
        "targets": targets,
    }
    return orders, summary


def write_after_close_report(report_dir: Path, summary: dict, orders: pd.DataFrame, signal: pd.DataFrame, topk: int) -> Path:
    date = pd.Timestamp(summary["signal_date"])
    path = report_path(report_dir, "after_close", date)
    report = f"""# Paper Trading After-Close Plan

## Setup

- Signal date: `{summary["signal_date"]}`
- Execution date: `{summary["execution_date"]}`
- Execution price: `T+1 raw open`
- Strategy: `top{summary["topk"]}/drop{summary["n_drop"]}`
- Marked equity: `{summary["marked_equity"]:,.2f}`
- Cash before execution: `{summary["cash"]:,.2f}`
- Current holdings: `{summary["n_current"]}`
- Target holdings after dropout: `{summary["n_targets"]}`

## Orders

{orders.to_markdown(index=False) if not orders.empty else "none"}

## Top Signal Preview

{signal.head(topk).to_markdown(index=False)}
"""
    path.write_text(report, encoding="utf-8")
    return path


def execute_orders(
    orders: pd.DataFrame,
    open_prices: pd.DataFrame,
    account: dict,
    positions: pd.DataFrame,
    cost: CostConfig,
    lot_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    price_map = open_prices.set_index("instrument")["price"].to_dict()
    pos = positions.set_index("instrument").to_dict(orient="index") if not positions.empty else {}
    cash = float(account["cash"])
    trade_rows = []

    executable = orders[orders["action"].isin(["SELL", "BUY"])].copy()
    for _, order in executable[executable["action"] == "SELL"].iterrows():
        inst = str(order["instrument"])
        price = float(price_map.get(inst, float("nan")))
        held = int(pos.get(inst, {}).get("shares", 0))
        shares = min(int(order["shares"]), held)
        if not math.isfinite(price) or price <= 0 or shares <= 0:
            trade_rows.append({**order.to_dict(), "status": "SKIPPED", "fill_price": price, "fill_shares": 0, "cost": 0.0})
            continue
        value = shares * price
        fee = calc_cost(value, "SELL", cost)
        cash += value - fee
        remain = held - shares
        if remain > 0:
            pos[inst]["shares"] = remain
        else:
            pos.pop(inst, None)
        trade_rows.append(
            {**order.to_dict(), "status": "FILLED", "fill_price": price, "fill_shares": shares, "cost": fee}
        )

    for _, order in executable[executable["action"] == "BUY"].iterrows():
        inst = str(order["instrument"])
        price = float(price_map.get(inst, float("nan")))
        shares = int(order["shares"])
        if not math.isfinite(price) or price <= 0 or shares <= 0:
            trade_rows.append({**order.to_dict(), "status": "SKIPPED", "fill_price": price, "fill_shares": 0, "cost": 0.0})
            continue
        while shares > 0:
            value = shares * price
            fee = calc_cost(value, "BUY", cost)
            if value + fee <= cash:
                break
            shares -= lot_size
        if shares <= 0:
            trade_rows.append(
                {**order.to_dict(), "status": "INSUFFICIENT_CASH", "fill_price": price, "fill_shares": 0, "cost": 0.0}
            )
            continue
        value = shares * price
        fee = calc_cost(value, "BUY", cost)
        cash -= value + fee
        old = pos.get(inst, {"shares": 0, "cost_basis": 0.0})
        old_shares = int(old.get("shares", 0))
        old_basis = float(old.get("cost_basis", 0.0))
        new_shares = old_shares + shares
        new_basis = ((old_basis * old_shares) + value) / new_shares if new_shares else 0.0
        pos[inst] = {"shares": new_shares, "cost_basis": new_basis}
        trade_rows.append(
            {**order.to_dict(), "status": "FILLED", "fill_price": price, "fill_shares": shares, "cost": fee}
        )

    new_positions = (
        pd.DataFrame([{"instrument": inst, **vals} for inst, vals in pos.items()])
        if pos
        else pd.DataFrame(columns=["instrument", "shares", "cost_basis"])
    )
    trades = pd.DataFrame(trade_rows)
    position_value = sum(int(vals["shares"]) * float(price_map.get(inst, 0.0)) for inst, vals in pos.items())
    total_cost = float(trades["cost"].sum()) if not trades.empty and "cost" in trades else 0.0
    account["cash"] = float(cash)
    account["last_nav"] = float(cash + position_value)
    account["total_cost"] = float(account.get("total_cost", 0.0) + total_cost)
    summary = {
        "cash": float(cash),
        "position_value_at_open": float(position_value),
        "nav_at_open": float(cash + position_value),
        "trade_cost": total_cost,
        "n_filled": int((trades["status"] == "FILLED").sum()) if not trades.empty else 0,
        "n_skipped": int((trades["status"] != "FILLED").sum()) if not trades.empty else 0,
    }
    return trades, new_positions, summary


def write_after_open_report(report_dir: Path, execution_date: pd.Timestamp, summary: dict, trades: pd.DataFrame, positions: pd.DataFrame) -> Path:
    path = report_path(report_dir, "after_open", execution_date)
    report = f"""# Paper Trading After-Open Execution

## Summary

- Execution date: `{execution_date.date()}`
- Execution price: `raw open`
- NAV at open: `{summary["nav_at_open"]:,.2f}`
- Cash after execution: `{summary["cash"]:,.2f}`
- Position value at open: `{summary["position_value_at_open"]:,.2f}`
- Trade cost: `{summary["trade_cost"]:,.2f}`
- Filled orders: `{summary["n_filled"]}`
- Skipped orders: `{summary["n_skipped"]}`

## Trades

{trades.to_markdown(index=False) if not trades.empty else "none"}

## Positions

{positions.to_markdown(index=False) if not positions.empty else "none"}
"""
    path.write_text(report, encoding="utf-8")
    return path


def after_close(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    ensure_dirs(state_dir, report_dir)
    account = load_account(state_dir, args.capital)
    positions = load_positions(state_dir)

    pred = load_or_fuse_prediction(args)
    signal_date = choose_trade_date(pred, args.signal_date)
    execution_date = next_trade_date(args.provider_uri, signal_date, args.execution_date)
    signal = ranked_signal(pred, signal_date)

    instruments = set(signal.head(max(args.topk * 5, args.topk + args.n_drop + 10))["instrument"].astype(str))
    if not positions.empty:
        instruments |= set(positions["instrument"].astype(str))
    prices = load_raw_prices(args.provider_uri, instruments, signal_date, "close")
    orders, summary = build_pending_orders(
        signal_date=signal_date,
        execution_date=execution_date,
        signal=signal,
        prices=prices,
        account=account,
        positions=positions,
        topk=args.topk,
        n_drop=args.n_drop,
        lot_size=args.lot_size,
        reserve_cash_pct=args.reserve_cash_pct,
        max_position_pct=args.max_position_pct,
    )
    orders.to_csv(pending_orders_path(state_dir, execution_date), index=False)
    save_account(state_dir, account)
    report = write_after_close_report(report_dir, summary, orders, signal, args.topk)
    print(f"Pending orders: {pending_orders_path(state_dir, execution_date)}")
    print(f"Report: {report}")
    print(orders.to_string(index=False) if not orders.empty else "No orders.")


def after_open(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    ensure_dirs(state_dir, report_dir)
    execution_date = pd.Timestamp(args.execution_date)
    orders_file = Path(args.orders).expanduser().resolve() if args.orders else pending_orders_path(state_dir, execution_date)
    if not orders_file.exists():
        raise FileNotFoundError(f"Pending order file not found: {orders_file}")
    orders = pd.read_csv(orders_file)
    account = load_account(state_dir, args.capital)
    positions = load_positions(state_dir)
    instruments = set(orders.loc[orders["action"].isin(["SELL", "BUY"]), "instrument"].astype(str))
    if not positions.empty:
        instruments |= set(positions["instrument"].astype(str))
    prices = load_raw_prices(args.provider_uri, instruments, execution_date, "open")
    cost = CostConfig(args.buy_cost_rate, args.sell_cost_rate, args.min_cost)
    trades, new_positions, summary = execute_orders(orders, prices, account, positions, cost, args.lot_size)
    trades.to_csv(trades_path(state_dir, execution_date), index=False)
    save_positions(state_dir, new_positions)
    save_account(state_dir, account)
    append_nav(
        state_dir,
        {
            "datetime": execution_date.date(),
            "phase": "after_open",
            "nav": summary["nav_at_open"],
            "cash": summary["cash"],
            "position_value": summary["position_value_at_open"],
            "trade_cost": summary["trade_cost"],
        },
    )
    report = write_after_open_report(report_dir, execution_date, summary, trades, new_positions)
    print(f"Trades: {trades_path(state_dir, execution_date)}")
    print(f"Positions: {positions_path(state_dir)}")
    print(f"NAV: {nav_path(state_dir)}")
    print(f"Report: {report}")
    print(trades.to_string(index=False) if not trades.empty else "No trades.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-trading state machine for XGB2 + ADARNN top20/drop2.")
    sub = parser.add_subparsers(dest="action", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
        p.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
        p.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
        p.add_argument("--capital", type=float, default=100000.0)
        p.add_argument("--topk", type=int, default=20)
        p.add_argument("--n-drop", type=int, default=2)
        p.add_argument("--lot-size", type=int, default=100)
        p.add_argument("--reserve-cash-pct", type=float, default=0.02)
        p.add_argument("--max-position-pct", type=float, default=0.12)

    p_close = sub.add_parser("after-close", help="Generate T+1 pending orders from T close signal.")
    add_common(p_close)
    p_close.add_argument("--fused-pred", default=None)
    p_close.add_argument("--pred-a", default=None)
    p_close.add_argument("--pred-b", default=None)
    p_close.add_argument("--pred-c", default=None)
    p_close.add_argument("--weights", default="10,1,1")
    p_close.add_argument("--signal-date", default=None)
    p_close.add_argument("--execution-date", default=None)
    p_close.set_defaults(func=after_close)

    p_open = sub.add_parser("after-open", help="Execute pending orders at T+1 raw open.")
    add_common(p_open)
    p_open.add_argument("--execution-date", required=True)
    p_open.add_argument("--orders", default=None)
    p_open.add_argument("--buy-cost-rate", type=float, default=0.0003)
    p_open.add_argument("--sell-cost-rate", type=float, default=0.0008)
    p_open.add_argument("--min-cost", type=float, default=5.0)
    p_open.set_defaults(func=after_open)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
