#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_env() -> dict[str, str]:
    ops_home = Path(__file__).resolve().parents[1]
    env_path = ops_home / "config" / "env.local"
    if not env_path.exists():
        env_path = ops_home / "config" / "env.local.example"
    env = parse_env(env_path)
    env.setdefault("OPS_HOME", str(ops_home))
    env.setdefault("PAPER_ENGINE_HOME", str(ops_home.parent / "paper_trading_system"))
    env.setdefault("PAPER_STATE_DIR", str(ops_home / "state" / "live_10w_main"))
    env.setdefault("PAPER_REPORT_DIR", str(ops_home / "reports" / "live_10w_main"))
    env.setdefault("LIVE_FILL_DIR", str(ops_home / "live_fills"))
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manually executed fills to the local live account book.")
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--fills-csv", default=None)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--fill-dir", default=None)
    parser.add_argument("--allow-negative-cash", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="Allow applying a fill CSV with no executed shares.")
    return parser.parse_args()


def calc_fee(value: float, action: str, buy_rate: float, sell_rate: float, min_cost: float) -> float:
    rate = buy_rate if action == "BUY" else sell_rate
    return max(float(min_cost), abs(float(value)) * float(rate))


def normalize_fills(fills: pd.DataFrame, execution_date: str, env: dict[str, str]) -> pd.DataFrame:
    required = ["instrument", "action", "shares", "fill_shares", "fill_price"]
    missing = [col for col in required if col not in fills.columns]
    if missing:
        raise ValueError(f"Fill CSV missing columns: {missing}")
    out = fills.copy()
    if "execution_date" in out.columns:
        out["execution_date"] = out["execution_date"].astype(str)
        if not out["execution_date"].eq(execution_date).all():
            bad = sorted(out.loc[~out["execution_date"].eq(execution_date), "execution_date"].unique())
            raise ValueError(f"Fill CSV contains mismatched execution_date values: {bad}")
    else:
        out["execution_date"] = execution_date
    out["instrument"] = out["instrument"].astype(str)
    out["action"] = out["action"].astype(str).str.upper()
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0).astype(int)
    out["fill_shares"] = pd.to_numeric(out["fill_shares"], errors="coerce").fillna(0).astype(int)
    out["fill_price"] = pd.to_numeric(out["fill_price"], errors="coerce").fillna(0.0).astype(float)
    if "fee" not in out.columns:
        out["fee"] = float("nan")
    out["fee"] = pd.to_numeric(out["fee"], errors="coerce")
    if "fill_status" not in out.columns:
        out["fill_status"] = ""
    if "operator_note" not in out.columns:
        out["operator_note"] = ""

    buy_rate = float(env.get("BUY_COST_RATE", "0.0003"))
    sell_rate = float(env.get("SELL_COST_RATE", "0.0008"))
    min_cost = float(env.get("MIN_COST", "5"))
    rows = []
    for _, row in out.iterrows():
        action = str(row["action"])
        fill_shares = int(row["fill_shares"])
        fill_price = float(row["fill_price"])
        if action not in {"BUY", "SELL"}:
            continue
        if fill_shares < 0:
            raise ValueError(f"Negative fill_shares is invalid: {row.to_dict()}")
        if fill_shares > int(row["shares"]):
            raise ValueError(f"fill_shares exceeds requested shares: {row.to_dict()}")
        if fill_shares > 0 and (not math.isfinite(fill_price) or fill_price <= 0):
            raise ValueError(f"Positive fill_shares requires positive fill_price: {row.to_dict()}")
        fee = row["fee"]
        if pd.isna(fee):
            fee = calc_fee(fill_shares * fill_price, action, buy_rate, sell_rate, min_cost) if fill_shares > 0 else 0.0
        status = str(row.get("fill_status", "")).strip().upper()
        if status == "PENDING":
            status = ""
        if not status:
            if fill_shares <= 0:
                status = "UNFILLED"
            elif fill_shares < int(row["shares"]):
                status = "PARTIAL"
            else:
                status = "FILLED"
        row = row.to_dict()
        row.update({"fill_status": status, "fill_shares": fill_shares, "fill_price": fill_price, "fee": float(fee)})
        rows.append(row)
    return pd.DataFrame(rows)


def setup_imports(env: dict[str, str]) -> None:
    paper_src = Path(env["PAPER_ENGINE_HOME"]).expanduser().resolve() / "src"
    qlib_src = env.get("QLIB_SRC", "").strip()
    if qlib_src:
        sys.path.insert(0, str(Path(qlib_src).expanduser().resolve()))
    sys.path.insert(0, str(paper_src))


def main() -> None:
    args = parse_args()
    env = load_env()
    setup_imports(env)
    from paper_trading_daily import (
        load_account,
        load_positions,
        load_raw_prices,
        nav_path,
        positions_path,
        save_account,
        save_positions,
    )

    execution_date = args.execution_date
    state_dir = Path(args.state_dir or env["PAPER_STATE_DIR"]).expanduser().resolve()
    report_dir = Path(args.report_dir or env["PAPER_REPORT_DIR"]).expanduser().resolve()
    fill_dir = Path(args.fill_dir or env["LIVE_FILL_DIR"]).expanduser().resolve()
    fills_csv = Path(args.fills_csv).expanduser().resolve() if args.fills_csv else fill_dir / f"manual_fill_template_{execution_date}.csv"
    if not fills_csv.exists():
        raise FileNotFoundError(f"Manual fill CSV not found: {fills_csv}")

    fills = normalize_fills(pd.read_csv(fills_csv), execution_date, env)
    executed = fills[fills["fill_shares"] > 0].copy()
    if executed.empty and not args.allow_empty:
        raise RuntimeError(
            f"No executed fills found in {fills_csv}. Fill fill_shares and fill_price first, "
            "or use --allow-empty only for an explicit no-fill day."
        )
    account = load_account(state_dir, float(env.get("CAPITAL", "100000")))
    positions = load_positions(state_dir)
    pos = positions.set_index("instrument").to_dict(orient="index") if not positions.empty else {}
    cash = float(account["cash"])

    for _, fill in executed.iterrows():
        inst = str(fill["instrument"])
        action = str(fill["action"])
        shares = int(fill["fill_shares"])
        price = float(fill["fill_price"])
        fee = float(fill["fee"])
        value = shares * price
        if action == "SELL":
            held = int(pos.get(inst, {}).get("shares", 0))
            if shares > held:
                raise ValueError(f"Cannot sell {shares} {inst}; local position only has {held}.")
            cash += value - fee
            remain = held - shares
            if remain > 0:
                pos[inst]["shares"] = remain
            else:
                pos.pop(inst, None)
        else:
            cash -= value + fee
            if cash < -1e-6 and not args.allow_negative_cash:
                raise RuntimeError(f"Cash would become negative ({cash:.2f}) after buying {inst}. Check fills or use --allow-negative-cash.")
            old = pos.get(inst, {"shares": 0, "cost_basis": 0.0})
            old_shares = int(old.get("shares", 0))
            new_shares = old_shares + shares
            old_basis = float(old.get("cost_basis", 0.0))
            pos[inst] = {"shares": new_shares, "cost_basis": ((old_basis * old_shares) + value) / new_shares}

    new_positions = pd.DataFrame([{"instrument": inst, **vals} for inst, vals in pos.items()]) if pos else pd.DataFrame(columns=["instrument", "shares", "cost_basis"])
    instruments = set(new_positions["instrument"].astype(str)) if not new_positions.empty else set()
    mark_prices = load_raw_prices(env["QLIB_PROVIDER_URI"], instruments, pd.Timestamp(execution_date), "close") if instruments else pd.DataFrame(columns=["instrument", "price"])
    price_map = mark_prices.set_index("instrument")["price"].to_dict() if not mark_prices.empty else {}
    position_value = 0.0
    for _, pos_row in new_positions.iterrows():
        price = float(price_map.get(str(pos_row["instrument"]), pos_row["cost_basis"]))
        position_value += int(pos_row["shares"]) * price

    trade_cost = float(executed["fee"].sum()) if not executed.empty else 0.0
    account["cash"] = float(cash)
    account["last_nav"] = float(cash + position_value)
    account["total_cost"] = float(account.get("total_cost", 0.0) + trade_cost)
    save_positions(state_dir, new_positions)
    save_account(state_dir, account)

    live_trades = fill_dir / f"manual_fills_applied_{execution_date}.csv"
    fills.to_csv(live_trades, index=False)
    nav_file = nav_path(state_dir)
    nav_row = pd.DataFrame([{
        "cash": cash,
        "position_value": position_value,
        "nav": cash + position_value,
        "trade_cost": trade_cost,
        "datetime": execution_date,
        "phase": "manual_fill",
    }])
    if nav_file.exists():
        nav_row = pd.concat([pd.read_csv(nav_file), nav_row], ignore_index=True)
        nav_row = nav_row.drop_duplicates(["datetime", "phase"], keep="last")
    nav_row.to_csv(nav_file, index=False)

    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"manual_fill_{execution_date}.md"
    report_file.write_text(
        f"""# Manual Fill Applied

- Execution date: `{execution_date}`
- Fill CSV: `{fills_csv}`
- NAV: `{cash + position_value:,.2f}`
- Cash: `{cash:,.2f}`
- Position value: `{position_value:,.2f}`
- Trade cost: `{trade_cost:,.2f}`

## Fills

{fills.to_markdown(index=False) if not fills.empty else "none"}

## Positions

{new_positions.to_markdown(index=False) if not new_positions.empty else "none"}
""",
        encoding="utf-8",
    )
    print(f"Applied fills: {live_trades}")
    print(f"Positions: {positions_path(state_dir)}")
    print(f"NAV: {nav_file}")
    print(f"Report: {report_file}")
    print(nav_row.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
