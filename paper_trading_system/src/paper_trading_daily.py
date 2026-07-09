from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    buy_rate: float
    sell_rate: float
    min_cost: float


def load_pred(path: Path) -> pd.DataFrame:
    pred = pd.read_pickle(path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if "score" not in pred.columns:
        raise ValueError(f"`score` column not found in {path}")
    pred = pred[["score"]].copy()
    pred.index = pred.index.set_names(["datetime", "instrument"])
    return pred.sort_index()


def daily_zscore(series: pd.Series) -> pd.Series:
    def _zscore(group: pd.Series) -> pd.Series:
        std = group.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return series.groupby(level="datetime", group_keys=False).apply(_zscore)


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


def choose_signal_date(pred: pd.DataFrame, signal_date: str | None) -> pd.Timestamp:
    available = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    if available.empty:
        raise ValueError("Prediction frame is empty.")
    if signal_date is None:
        return pd.Timestamp(available[-1])
    target = pd.Timestamp(signal_date)
    if target not in available:
        raise ValueError(
            f"Signal date {target.date()} not found. Available range: {available[0].date()} -> {available[-1].date()}"
        )
    return target


def account_path(state_dir: Path) -> Path:
    return state_dir / "paper_account.json"


def positions_path(state_dir: Path) -> Path:
    return state_dir / "paper_positions.csv"


def nav_path(state_dir: Path) -> Path:
    return state_dir / "paper_nav.csv"


def pending_orders_path(state_dir: Path, execution_date: pd.Timestamp) -> Path:
    return state_dir / f"pending_orders_{execution_date.date()}.csv"


def approved_orders_path(state_dir: Path, execution_date: pd.Timestamp) -> Path:
    return state_dir / f"approved_orders_{execution_date.date()}.csv"


def approval_status_path(state_dir: Path, execution_date: pd.Timestamp) -> Path:
    return state_dir / f"approval_status_{execution_date.date()}.json"


def trades_path(state_dir: Path, execution_date: pd.Timestamp) -> Path:
    return state_dir / f"paper_trades_{execution_date.date()}.csv"


def load_account(state_dir: Path, capital: float) -> dict:
    path = account_path(state_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"initial_capital": float(capital), "cash": float(capital), "last_nav": float(capital), "total_cost": 0.0}


def save_account(state_dir: Path, account: dict) -> None:
    account_path(state_dir).write_text(json.dumps(account, indent=2, sort_keys=True), encoding="utf-8")


def save_approval_template(state_dir: Path, summary: dict) -> Path:
    path = approval_status_path(state_dir, pd.Timestamp(summary["execution_date"]))
    payload = {
        "signal_date": summary["signal_date"],
        "execution_date": summary["execution_date"],
        "status": "pending_review",
        "reviewer": "",
        "review_time": "",
        "note": "",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_approval_status(state_dir: Path, execution_date: pd.Timestamp) -> dict | None:
    path = approval_status_path(state_dir, execution_date)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_positions(state_dir: Path) -> pd.DataFrame:
    path = positions_path(state_dir)
    if not path.exists():
        return pd.DataFrame(columns=["instrument", "shares", "cost_basis"])
    df = pd.read_csv(path)
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


def init_qlib(provider_uri: str) -> None:
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=provider_uri, region=REG_CN)


def load_raw_prices(provider_uri: str, instruments: set[str], date: pd.Timestamp, field: str) -> pd.DataFrame:
    from qlib.data import D

    instruments = sorted(instruments)
    if not instruments:
        return pd.DataFrame(columns=["instrument", "price"])
    init_qlib(provider_uri)
    qfield = f"${field}"
    data = D.features(instruments, [qfield, "$factor"], start_time=date, end_time=date, freq="day")
    if data is None or data.empty:
        return pd.DataFrame(columns=["instrument", "price"])
    data = data.rename(columns={qfield: "adjusted_price", "$factor": "factor"}).reset_index()
    data["price"] = data["adjusted_price"]
    valid_factor = data["factor"].notna() & (data["factor"] > 0)
    data.loc[valid_factor, "price"] = data.loc[valid_factor, "adjusted_price"] / data.loc[valid_factor, "factor"]
    return data.loc[:, ["instrument", "price", "adjusted_price", "factor"]].dropna(subset=["instrument", "price"])


def load_close_amount_window(
    provider_uri: str,
    instruments: set[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from qlib.data import D

    instruments = sorted(instruments)
    if not instruments:
        empty = pd.DataFrame()
        return empty, empty
    init_qlib(provider_uri)
    data = D.features(instruments, ["$close", "$factor", "$volume"], start_time=start_date, end_time=end_date, freq="day")
    if data is None or data.empty:
        empty = pd.DataFrame()
        return empty, empty
    data = data.rename(columns={"$close": "close_adj", "$factor": "factor", "$volume": "volume"}).reset_index()
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["close"] = data["close_adj"]
    valid_factor = data["factor"].notna() & (data["factor"] > 0)
    data.loc[valid_factor, "close"] = data.loc[valid_factor, "close_adj"] / data.loc[valid_factor, "factor"]
    data["amount"] = data["close"] * data["volume"]
    close_px = data.pivot(index="datetime", columns="instrument", values="close").sort_index()
    amount = data.pivot(index="datetime", columns="instrument", values="amount").sort_index()
    return close_px, amount


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
    return pd.Timestamp(signal_date) + pd.offsets.BDay(1)


def ranked_signal(pred: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    day = pred.xs(signal_date, level="datetime").reset_index()
    day = day.sort_values("score", ascending=False).reset_index(drop=True)
    day["rank"] = day.index + 1
    return day.loc[:, ["rank", "instrument", "score"]]


def safe_price(row: pd.Series, inst: str) -> float:
    try:
        price = float(row.get(inst, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return price if math.isfinite(price) and price > 0 else 0.0


def truthy_market_flag(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "st", "*st"}
    try:
        return bool(float(value) != 0.0)
    except (TypeError, ValueError):
        return bool(value)


def load_security_flags(provider_uri: str, instruments: set[str], date: pd.Timestamp) -> pd.DataFrame:
    """Load optional risk flags from Qlib without making them hard dependencies."""
    from qlib.data import D

    instruments_l = sorted(instruments)
    if not instruments_l:
        return pd.DataFrame(columns=["instrument", "is_st", "paused"])
    init_qlib(provider_uri)
    frames: list[pd.DataFrame] = []
    for field, name in [("$is_st", "is_st"), ("$paused", "paused")]:
        try:
            data = D.features(instruments_l, [field], start_time=date, end_time=date, freq="day")
        except Exception as exc:
            print(f"warning: optional Qlib flag {field} unavailable: {exc}", file=sys.stderr)
            continue
        if data is None or data.empty:
            continue
        part = data.rename(columns={field: name}).reset_index()
        part[name] = part[name].map(truthy_market_flag)
        frames.append(part.loc[:, ["instrument", name]])
    if not frames:
        return pd.DataFrame(columns=["instrument", "is_st", "paused"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="instrument", how="outer")
    for col in ["is_st", "paused"]:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)
    return out.loc[:, ["instrument", "is_st", "paused"]]


def apply_jq_filter(
    signal: pd.DataFrame,
    close_row: pd.Series,
    amount_window: pd.DataFrame,
    min_price: float,
    max_price: float,
    min_amount: float,
    security_flags: pd.DataFrame | None = None,
    filter_st: bool = True,
    filter_paused: bool = True,
) -> pd.DataFrame:
    avg_amount = amount_window.mean(axis=0, skipna=True) if not amount_window.empty else pd.Series(dtype=float)
    flag_map = security_flags.set_index("instrument").to_dict("index") if security_flags is not None and not security_flags.empty else {}
    keep = []
    for inst in signal["instrument"].astype(str):
        price = safe_price(close_row, inst)
        amt = float(avg_amount.get(inst, 0.0) or 0.0)
        flags = flag_map.get(inst, {})
        is_flagged_st = bool(flags.get("is_st", False))
        is_paused = bool(flags.get("paused", False))
        keep.append(
            min_price <= price <= max_price
            and amt >= min_amount
            and not (filter_st and is_flagged_st)
            and not (filter_paused and is_paused)
        )
    filtered = signal.loc[keep].reset_index(drop=True)
    filtered["rank"] = filtered.index + 1
    return filtered


def select_topk_dropout(signal: pd.DataFrame, positions: pd.DataFrame, topk: int, n_drop: int) -> tuple[list[str], list[str]]:
    ranked = signal["instrument"].astype(str).tolist()
    rank_map = dict(zip(signal["instrument"].astype(str), signal["rank"].astype(int), strict=False))
    current = positions["instrument"].astype(str).tolist() if not positions.empty else []
    if not current:
        return [], []

    def rank_or_inf(inst: str) -> float:
        return float(rank_map.get(inst, 10**9))

    sell = sorted([inst for inst in current if rank_or_inf(inst) > topk], key=rank_or_inf, reverse=True)[:n_drop]
    return current, sell


def round_lot_shares(target_value: float, price: float, lot_size: int) -> int:
    if target_value <= 0 or price <= 0:
        return 0
    return int(math.floor(target_value / (price * lot_size)) * lot_size)


def planned_buy_shares(target_value: float, equity: float, price: float, lot_size: int, max_position_pct: float, allow_one_lot_over_target: bool) -> tuple[int, str]:
    shares = round_lot_shares(target_value, price, lot_size)
    if shares > 0 or not allow_one_lot_over_target or price <= 0:
        return shares, "top_rank" if shares > 0 else "cannot_buy_one_lot"
    one_lot_value = price * lot_size
    if one_lot_value <= equity * max_position_pct:
        return lot_size, "one_lot_over_target"
    return 0, "one_lot_exceeds_max_position"


def calc_cost(value: float, action: str, cost: CostConfig) -> float:
    if value <= 0:
        return 0.0
    rate = cost.buy_rate if action == "BUY" else cost.sell_rate
    return max(value * rate, cost.min_cost)


def load_or_fuse(args: argparse.Namespace) -> pd.DataFrame:
    if args.fused_pred:
        return load_pred(Path(args.fused_pred).expanduser().resolve())
    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) != 3:
        raise ValueError("--weights must be like 10,1,1")
    return fuse_three(
        load_pred(Path(args.pred_a).expanduser().resolve()),
        load_pred(Path(args.pred_b).expanduser().resolve()),
        load_pred(Path(args.pred_c).expanduser().resolve()),
        weights,  # type: ignore[arg-type]
    )


def build_orders(args: argparse.Namespace, account: dict, positions: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    signal_date = choose_signal_date(pred, args.signal_date)
    execution_date = next_trade_date(args.provider_uri, signal_date, args.execution_date)
    signal = ranked_signal(pred, signal_date)
    buy_scan_topk = max(args.buy_scan_topk, args.topk)
    instruments = set(signal.head(max(buy_scan_topk, args.topk + args.n_drop + 10))["instrument"].astype(str))
    if not positions.empty:
        instruments |= set(positions["instrument"].astype(str))
    if args.filter_mode == "jq_filter":
        start_date = signal_date - pd.offsets.BDay(max(args.lookback * 2, 10))
        close_px, amount = load_close_amount_window(args.provider_uri, instruments, start_date, signal_date)
        if signal_date in close_px.index:
            close_row = close_px.loc[signal_date]
            amount_window = amount.loc[:signal_date].tail(args.lookback)
            security_flags = load_security_flags(args.provider_uri, instruments, signal_date)
            signal = apply_jq_filter(
                signal,
                close_row,
                amount_window,
                args.min_price,
                args.max_price,
                args.min_amount,
                security_flags,
                args.filter_st,
                args.filter_paused,
            )
    prices = load_raw_prices(args.provider_uri, instruments, signal_date, "close")
    price_map = prices.set_index("instrument")["price"].to_dict()
    rank_map = signal.set_index("instrument")["rank"].to_dict()
    score_map = signal.set_index("instrument")["score"].to_dict()
    current, sell_list = select_topk_dropout(signal, positions, args.topk, args.n_drop)
    pos_map = positions.set_index("instrument")["shares"].to_dict() if not positions.empty else {}
    marked_value = sum(int(shares) * float(price_map.get(inst, 0.0)) for inst, shares in pos_map.items())
    cash = float(account["cash"])
    nav = cash + marked_value
    buy_budget_mode = args.buy_budget_mode.lower()
    if buy_budget_mode in {"capital_pool", "capital", "total_capital"}:
        equity = float(args.capital)
        buy_budget = max(float(args.capital), 0.0)
    elif buy_budget_mode in {"capital_pool_reserved", "capital_reserved"}:
        equity = float(args.capital)
        buy_budget = max(float(args.capital) * (1.0 - args.reserve_cash_pct), 0.0)
    elif buy_budget_mode in {"reserve_nav", "available_cash_reserved"}:
        equity = nav
        buy_budget = max(cash - nav * args.reserve_cash_pct, 0.0)
    elif buy_budget_mode == "available_cash":
        equity = nav
        buy_budget = max(cash, 0.0)
    else:
        raise ValueError(
            "buy_budget_mode must be one of capital_pool, capital_pool_reserved, "
            "available_cash, available_cash_reserved"
        )
    target_value = min(equity * (1.0 - args.reserve_cash_pct) / args.topk, equity * args.max_position_pct)
    kept = [inst for inst in current if inst not in set(sell_list)]
    sell_value = sum(int(pos_map.get(inst, 0)) * float(price_map.get(inst, 0.0)) for inst in sell_list)
    available_cash = buy_budget + sell_value
    buy_slots = max(args.topk - len(kept), 0)
    if current and sell_list:
        buy_slots = min(buy_slots, args.n_drop)
    buy_list: list[str] = []
    skipped_rows = []
    reserved_buy_value = 0.0
    ranked_candidates = signal.head(buy_scan_topk)["instrument"].astype(str).tolist()
    excluded = set(kept) | set(sell_list)
    for inst in ranked_candidates:
        if len(buy_list) >= buy_slots:
            break
        if inst in excluded or inst in set(buy_list):
            continue
        price = float(price_map.get(inst, 0.0))
        shares, reason = planned_buy_shares(
            target_value,
            equity,
            price,
            args.lot_size,
            args.max_position_pct,
            args.allow_one_lot_over_target,
        )
        if shares <= 0:
            skipped_rows.append([signal_date.date(), execution_date.date(), inst, "SKIP", 0, price, reason, "primary", rank_map.get(inst, ""), score_map.get(inst, ""), ""])
            continue
        planned_value = shares * price
        if planned_value + reserved_buy_value > available_cash:
            skipped_rows.append([signal_date.date(), execution_date.date(), inst, "SKIP", 0, price, "insufficient_cash", "primary", rank_map.get(inst, ""), score_map.get(inst, ""), ""])
            continue
        buy_list.append(inst)
        reserved_buy_value += planned_value

    rows = []
    for inst in sell_list:
        shares = int(pos_map.get(inst, 0))
        if shares > 0:
            rows.append([signal_date.date(), execution_date.date(), inst, "SELL", shares, price_map.get(inst, 0.0), "dropout", "primary", rank_map.get(inst, ""), score_map.get(inst, ""), ""])
    for inst in buy_list:
        price = float(price_map.get(inst, 0.0))
        shares, reason = planned_buy_shares(
            target_value,
            equity,
            price,
            args.lot_size,
            args.max_position_pct,
            args.allow_one_lot_over_target,
        )
        action = "BUY" if shares > 0 else "SKIP"
        rows.append([signal_date.date(), execution_date.date(), inst, action, shares, price, reason, "primary", rank_map.get(inst, ""), score_map.get(inst, ""), ""])

    backup_count = max(int(getattr(args, "backup_buy_count", 0)), 0)
    backup_excluded = set(excluded) | set(buy_list)
    backup_rows = []
    for inst in ranked_candidates:
        if len(backup_rows) >= backup_count:
            break
        if inst in backup_excluded:
            continue
        price = float(price_map.get(inst, 0.0))
        shares, reason = planned_buy_shares(
            target_value,
            equity,
            price,
            args.lot_size,
            args.max_position_pct,
            args.allow_one_lot_over_target,
        )
        if shares <= 0:
            continue
        backup_rows.append([signal_date.date(), execution_date.date(), inst, "BUY", shares, price, "backup_rank", "backup", rank_map.get(inst, ""), score_map.get(inst, ""), len(backup_rows) + 1])
        backup_excluded.add(inst)

    rows.extend(backup_rows)
    rows.extend(skipped_rows)

    orders = pd.DataFrame(
        rows,
        columns=[
            "signal_date",
            "execution_date",
            "instrument",
            "action",
            "shares",
            "estimated_price",
            "reason",
            "order_role",
            "model_rank",
            "score",
            "backup_rank",
        ],
    )
    primary = orders["order_role"].eq("primary") if not orders.empty else pd.Series(dtype=bool)
    summary = {
        "signal_date": str(signal_date.date()),
        "execution_date": str(execution_date.date()),
        "filter_mode": args.filter_mode,
        "n_drop": int(args.n_drop),
        "cash": float(account["cash"]),
        "marked_position_value": float(marked_value),
        "marked_equity": float(equity),
        "n_current": int(len(pos_map)),
        "n_targets": int(len(kept) + len(buy_list)),
        "n_buy": int((orders["action"].eq("BUY") & primary).sum()) if not orders.empty else 0,
        "n_sell": int((orders["action"].eq("SELL") & primary).sum()) if not orders.empty else 0,
        "n_backup_buy": int((orders["action"].eq("BUY") & orders["order_role"].eq("backup")).sum()) if not orders.empty else 0,
        "buy_scan_topk": int(buy_scan_topk),
    }
    return orders, signal, summary


def write_plan_report(report_dir: Path, summary: dict, orders: pd.DataFrame, signal: pd.DataFrame, topk: int) -> Path:
    path = report_dir / f"paper_after_close_{summary['signal_date']}.md"
    report = f"""# Paper Trading After-Close Plan

- Signal date: `{summary["signal_date"]}`
- Execution date: `{summary["execution_date"]}`
- Execution price: `T+1 raw open`
- Filter: `{summary["filter_mode"]}`
- Strategy: `top{topk}/drop{summary["n_drop"]}`
- Buy scan topk: `{summary["buy_scan_topk"]}`
- Marked equity: `{summary["marked_equity"]:,.2f}`
- Cash: `{summary["cash"]:,.2f}`
- Current holdings: `{summary["n_current"]}`
- Buy orders: `{summary["n_buy"]}`
- Sell orders: `{summary["n_sell"]}`
- Backup buy candidates: `{summary.get("n_backup_buy", 0)}`

## Orders

{orders.to_markdown(index=False) if not orders.empty else "none"}

## Top Signal Preview

{signal.head(topk).to_markdown(index=False)}
"""
    path.write_text(report, encoding="utf-8")
    return path


def after_close(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    account = load_account(state_dir, args.capital)
    positions = load_positions(state_dir)
    orders, signal, summary = build_orders(args, account, positions, load_or_fuse(args))
    order_path = pending_orders_path(state_dir, pd.Timestamp(summary["execution_date"]))
    approved_path = approved_orders_path(state_dir, pd.Timestamp(summary["execution_date"]))
    signal_path = state_dir / f"signal_rankings_{summary['signal_date']}.csv"
    approval_path = save_approval_template(state_dir, summary)
    orders.to_csv(order_path, index=False)
    signal.to_csv(signal_path, index=False)
    save_account(state_dir, account)
    report = write_plan_report(report_dir, summary, orders, signal, args.topk)
    print(f"Pending orders: {order_path}")
    print(f"Approved orders target: {approved_path}")
    print(f"Approval status template: {approval_path}")
    print(f"Signal rankings: {signal_path}")
    print(f"Report: {report}")
    print(orders.to_string(index=False) if not orders.empty else "No orders.")


def execute_orders(orders: pd.DataFrame, prices: pd.DataFrame, account: dict, positions: pd.DataFrame, cost: CostConfig, lot_size: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    price_map = prices.set_index("instrument")["price"].to_dict()
    pos = positions.set_index("instrument").to_dict(orient="index") if not positions.empty else {}
    cash = float(account["cash"])
    rows = []

    for _, order in orders[orders["action"].isin(["SELL", "BUY"])].iterrows():
        inst = str(order["instrument"])
        action = str(order["action"])
        price = float(price_map.get(inst, float("nan")))
        shares = int(order["shares"])
        if not math.isfinite(price) or price <= 0 or shares <= 0:
            rows.append({**order.to_dict(), "status": "SKIPPED", "fill_price": price, "fill_shares": 0, "cost": 0.0})
            continue
        if action == "SELL":
            shares = min(shares, int(pos.get(inst, {}).get("shares", 0)))
            value = shares * price
            fee = calc_cost(value, action, cost)
            cash += value - fee
            remain = int(pos.get(inst, {}).get("shares", 0)) - shares
            if remain > 0:
                pos[inst]["shares"] = remain
            else:
                pos.pop(inst, None)
        else:
            while shares > 0:
                value = shares * price
                fee = calc_cost(value, action, cost)
                if value + fee <= cash:
                    break
                shares -= lot_size
            if shares <= 0:
                rows.append({**order.to_dict(), "status": "INSUFFICIENT_CASH", "fill_price": price, "fill_shares": 0, "cost": 0.0})
                continue
            value = shares * price
            fee = calc_cost(value, action, cost)
            cash -= value + fee
            old = pos.get(inst, {"shares": 0, "cost_basis": 0.0})
            old_shares = int(old.get("shares", 0))
            new_shares = old_shares + shares
            old_basis = float(old.get("cost_basis", 0.0))
            pos[inst] = {"shares": new_shares, "cost_basis": ((old_basis * old_shares) + value) / new_shares}
        rows.append({**order.to_dict(), "status": "FILLED", "fill_price": price, "fill_shares": shares, "cost": fee})

    new_positions = pd.DataFrame([{"instrument": inst, **vals} for inst, vals in pos.items()]) if pos else pd.DataFrame(columns=["instrument", "shares", "cost_basis"])
    position_value = sum(int(vals["shares"]) * float(price_map.get(inst, 0.0)) for inst, vals in pos.items())
    trades = pd.DataFrame(rows)
    trade_cost = float(trades["cost"].sum()) if not trades.empty else 0.0
    account["cash"] = float(cash)
    account["last_nav"] = float(cash + position_value)
    account["total_cost"] = float(account.get("total_cost", 0.0) + trade_cost)
    return trades, new_positions, {"cash": cash, "position_value": position_value, "nav": cash + position_value, "trade_cost": trade_cost}


def after_open(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    execution_date = pd.Timestamp(args.execution_date)
    approval = load_approval_status(state_dir, execution_date)
    if not args.ignore_approval:
        if approval is None:
            print(f"No approval status found for {execution_date.date()}, skipping execution.")
            return
        status = str(approval.get("status", "")).strip().lower()
        if status not in {"approved", "approved_with_edits"}:
            print(f"Approval status for {execution_date.date()} is `{status or 'missing'}`, skipping execution.")
            return
    order_file = Path(args.orders).expanduser().resolve() if args.orders else approved_orders_path(state_dir, execution_date)
    if not order_file.exists():
        raise FileNotFoundError(f"Approved order file not found: {order_file}")
    orders = pd.read_csv(order_file)
    account = load_account(state_dir, args.capital)
    positions = load_positions(state_dir)
    instruments = set(orders.loc[orders["action"].isin(["SELL", "BUY"]), "instrument"].astype(str))
    if not positions.empty:
        instruments |= set(positions["instrument"].astype(str))
    prices = load_raw_prices(args.provider_uri, instruments, execution_date, "open")
    trades, new_positions, summary = execute_orders(
        orders,
        prices,
        account,
        positions,
        CostConfig(args.buy_cost_rate, args.sell_cost_rate, args.min_cost),
        args.lot_size,
    )
    trades.to_csv(trades_path(state_dir, execution_date), index=False)
    save_positions(state_dir, new_positions)
    save_account(state_dir, account)
    nav_row = pd.DataFrame([{**summary, "datetime": execution_date.date(), "phase": "after_open"}])
    nav_file = nav_path(state_dir)
    if nav_file.exists():
        nav_row = pd.concat([pd.read_csv(nav_file), nav_row], ignore_index=True)
        nav_row = nav_row.drop_duplicates(["datetime", "phase"], keep="last")
    nav_row.to_csv(nav_file, index=False)
    report_file = report_dir / f"paper_after_open_{execution_date.date()}.md"
    report_file.write_text(
        f"""# Paper Trading After-Open Execution

- Execution date: `{execution_date.date()}`
- NAV: `{summary["nav"]:,.2f}`
- Cash: `{summary["cash"]:,.2f}`
- Position value: `{summary["position_value"]:,.2f}`
- Trade cost: `{summary["trade_cost"]:,.2f}`

## Trades

{trades.to_markdown(index=False) if not trades.empty else "none"}

## Positions

{new_positions.to_markdown(index=False) if not new_positions.empty else "none"}
""",
        encoding="utf-8",
    )
    print(f"Trades: {trades_path(state_dir, execution_date)}")
    print(f"Positions: {positions_path(state_dir)}")
    print(f"NAV: {nav_file}")
    print(f"Report: {report_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portable paper-trading state machine.")
    sub = parser.add_subparsers(dest="action", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider-uri", required=True)
        p.add_argument("--state-dir", required=True)
        p.add_argument("--report-dir", required=True)
        p.add_argument("--capital", type=float, default=100000.0)
        p.add_argument("--topk", type=int, default=20)
        p.add_argument("--n-drop", type=int, default=2)
        p.add_argument("--buy-scan-topk", type=int, default=150)
        p.add_argument("--backup-buy-count", type=int, default=0)
        p.add_argument("--lot-size", type=int, default=100)
        p.add_argument("--reserve-cash-pct", type=float, default=0.02)
        p.add_argument(
            "--buy-budget-mode",
            default="capital_pool",
            choices=["capital_pool", "capital_pool_reserved", "available_cash", "available_cash_reserved", "reserve_nav"],
        )
        p.add_argument("--max-position-pct", type=float, default=0.12)
        p.add_argument("--allow-one-lot-over-target", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--filter-mode", choices=["raw", "jq_filter"], default="raw")
        p.add_argument("--min-price", type=float, default=5.0)
        p.add_argument("--max-price", type=float, default=80.0)
        p.add_argument("--min-amount", type=float, default=2e7)
        p.add_argument("--lookback", type=int, default=20)
        p.add_argument("--filter-st", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--filter-paused", action=argparse.BooleanOptionalAction, default=True)

    p = sub.add_parser("after-close")
    add_common(p)
    p.add_argument("--fused-pred", default=None)
    p.add_argument("--pred-a", default=None)
    p.add_argument("--pred-b", default=None)
    p.add_argument("--pred-c", default=None)
    p.add_argument("--weights", default="10,1,1")
    p.add_argument("--signal-date", default=None)
    p.add_argument("--execution-date", default=None)
    p.set_defaults(func=after_close)

    p = sub.add_parser("after-open")
    add_common(p)
    p.add_argument("--execution-date", required=True)
    p.add_argument("--orders", default=None)
    p.add_argument("--ignore-approval", action="store_true")
    p.add_argument("--buy-cost-rate", type=float, default=0.0003)
    p.add_argument("--sell-cost-rate", type=float, default=0.0008)
    p.add_argument("--min-cost", type=float, default=5.0)
    p.set_defaults(func=after_open)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
