#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


def parse_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_env() -> dict[str, str]:
    ops_home = Path(__file__).resolve().parents[1]
    env_path = ops_home / "config" / "env.local"
    if not env_path.exists():
        env_path = ops_home / "config" / "env.local.example"
    env = parse_env(env_path)
    env.setdefault("OPS_HOME", str(ops_home))
    env.setdefault("PAPER_ENGINE_HOME", str(ops_home.parent / "paper_trading_system"))
    env.setdefault("SHORT_HOLD_STATE_DIR", str(ops_home / "state" / "short_hold_single_peak"))
    env.setdefault("SHORT_HOLD_REPORT_DIR", str(ops_home / "reports" / "short_hold_single_peak"))
    env.setdefault("SHORT_HOLD_FILL_DIR", str(ops_home / "short_hold_fills"))
    return env


def setup_imports(env: dict[str, str]) -> None:
    paper_src = Path(env["PAPER_ENGINE_HOME"]).expanduser().resolve() / "src"
    qlib_src = env.get("QLIB_SRC", "").strip()
    if qlib_src:
        sys.path.insert(0, str(Path(qlib_src).expanduser().resolve()))
    sys.path.insert(0, str(paper_src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply short-hold manual fills.")
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--fills-csv", default=None)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def calc_fee(value: float, action: str, env: dict[str, str]) -> float:
    rate = float(env.get("BUY_COST_RATE", "0.0003")) if action == "BUY" else float(env.get("SELL_COST_RATE", "0.0008"))
    return max(float(env.get("MIN_COST", "5")), abs(value) * rate) if value > 0 else 0.0


def account_path(state_dir: Path) -> Path:
    return state_dir / "paper_account.json"


def positions_path(state_dir: Path) -> Path:
    return state_dir / "paper_positions.csv"


def nav_path(state_dir: Path) -> Path:
    return state_dir / "paper_nav.csv"


def load_account(state_dir: Path, capital: float) -> dict:
    path = account_path(state_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"initial_capital": float(capital), "cash": float(capital), "last_nav": float(capital), "total_cost": 0.0}


def load_positions(state_dir: Path) -> pd.DataFrame:
    cols = ["instrument", "shares", "cost_basis", "entry_date", "exit_date", "signal_date", "score"]
    path = positions_path(state_dir)
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for col in cols:
        if col not in df.columns:
            df[col] = "" if col.endswith("_date") or col == "instrument" else 0
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    return df.loc[df["shares"] > 0, cols].reset_index(drop=True)


def save_positions(state_dir: Path, positions: pd.DataFrame) -> None:
    cols = ["instrument", "shares", "cost_basis", "entry_date", "exit_date", "signal_date", "score"]
    out = positions.copy() if not positions.empty else pd.DataFrame(columns=cols)
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    out = out.loc[out["shares"].astype(int) > 0, cols].sort_values(["exit_date", "instrument"]).reset_index(drop=True)
    out.to_csv(positions_path(state_dir), index=False)


def normalize_fills(path: Path, execution_date: str, env: dict[str, str]) -> pd.DataFrame:
    fills = pd.read_csv(path)
    required = ["instrument", "action", "shares", "fill_shares", "fill_price"]
    missing = [col for col in required if col not in fills.columns]
    if missing:
        raise ValueError(f"Fill CSV missing columns: {missing}")
    fills["execution_date"] = fills.get("execution_date", execution_date).astype(str)
    if not fills["execution_date"].eq(execution_date).all():
        raise ValueError("Fill CSV execution_date does not match requested date.")
    fills["instrument"] = fills["instrument"].astype(str)
    fills["action"] = fills["action"].astype(str).str.upper()
    fills["shares"] = pd.to_numeric(fills["shares"], errors="coerce").fillna(0).astype(int)
    fills["fill_shares"] = pd.to_numeric(fills["fill_shares"], errors="coerce").fillna(0).astype(int)
    fills["fill_price"] = pd.to_numeric(fills["fill_price"], errors="coerce").fillna(0.0).astype(float)
    if "fee" not in fills.columns:
        fills["fee"] = float("nan")
    fills["fee"] = pd.to_numeric(fills["fee"], errors="coerce")
    if "fill_status" not in fills.columns:
        fills["fill_status"] = ""
    if "operator_note" not in fills.columns:
        fills["operator_note"] = ""
    rows = []
    for _, row in fills.iterrows():
        action = str(row["action"])
        shares = int(row["fill_shares"])
        price = float(row["fill_price"])
        if action not in {"BUY", "SELL"}:
            continue
        if shares < 0 or shares > int(row["shares"]):
            raise ValueError(f"Invalid fill_shares: {row.to_dict()}")
        if shares > 0 and (not math.isfinite(price) or price <= 0):
            raise ValueError(f"Positive fill_shares requires positive fill_price: {row.to_dict()}")
        fee = row["fee"]
        if pd.isna(fee):
            fee = calc_fee(shares * price, action, env)
        status = str(row.get("fill_status", "")).strip().upper()
        if status == "PENDING":
            status = ""
        if not status:
            status = "UNFILLED" if shares <= 0 else ("PARTIAL" if shares < int(row["shares"]) else "FILLED")
        item = row.to_dict()
        item.update({"fill_status": status, "fill_shares": shares, "fill_price": price, "fee": float(fee)})
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    env = load_env()
    setup_imports(env)
    from paper_trading_daily import load_raw_prices

    state_dir = Path(env["SHORT_HOLD_STATE_DIR"]).expanduser().resolve()
    report_dir = Path(env["SHORT_HOLD_REPORT_DIR"]).expanduser().resolve()
    fill_dir = Path(env["SHORT_HOLD_FILL_DIR"]).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fill_dir.mkdir(parents=True, exist_ok=True)
    fills_csv = Path(args.fills_csv).expanduser().resolve() if args.fills_csv else fill_dir / f"short_hold_fill_template_{args.execution_date}.csv"
    fills = normalize_fills(fills_csv, args.execution_date, env)
    executed = fills.loc[fills["fill_shares"] > 0].copy()
    if executed.empty and not args.allow_empty:
        raise RuntimeError(f"No executed fills found in {fills_csv}.")

    capital = float(env.get("SHORT_HOLD_CAPITAL", env.get("CAPITAL", "100000")))
    account = load_account(state_dir, capital)
    positions = load_positions(state_dir)
    pos = positions.set_index("instrument").to_dict(orient="index") if not positions.empty else {}
    cash = float(account.get("cash", capital))

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
            old = pos.get(inst, {"shares": 0, "cost_basis": 0.0})
            old_shares = int(old.get("shares", 0))
            new_shares = old_shares + shares
            old_basis = float(old.get("cost_basis", 0.0))
            pos[inst] = {
                "shares": new_shares,
                "cost_basis": ((old_basis * old_shares) + value) / new_shares,
                "entry_date": args.execution_date,
                "exit_date": fill.get("exit_date", ""),
                "signal_date": fill.get("signal_date", ""),
                "score": float(fill.get("score", 0.0) or 0.0),
            }

    new_positions = pd.DataFrame([{"instrument": inst, **vals} for inst, vals in pos.items()])
    save_positions(state_dir, new_positions)

    instruments = set(new_positions["instrument"].astype(str)) if not new_positions.empty else set()
    mark_prices = load_raw_prices(env["QLIB_PROVIDER_URI"], instruments, pd.Timestamp(args.execution_date), "close") if instruments else pd.DataFrame()
    price_map = mark_prices.set_index("instrument")["price"].to_dict() if not mark_prices.empty else {}
    position_value = 0.0
    for _, row in new_positions.iterrows():
        price = float(price_map.get(str(row["instrument"]), row["cost_basis"]))
        position_value += int(row["shares"]) * price

    trade_cost = float(executed["fee"].sum()) if not executed.empty else 0.0
    nav = cash + position_value
    account["cash"] = cash
    account["last_nav"] = nav
    account["total_cost"] = float(account.get("total_cost", 0.0) + trade_cost)
    account_path(state_dir).write_text(json.dumps(account, indent=2, sort_keys=True), encoding="utf-8")

    applied = fill_dir / f"short_hold_fills_applied_{args.execution_date}.csv"
    fills.to_csv(applied, index=False)
    nav_row = pd.DataFrame(
        [
            {
                "cash": cash,
                "position_value": position_value,
                "nav": nav,
                "trade_cost": trade_cost,
                "datetime": args.execution_date,
                "phase": "short_hold_manual_fill",
            }
        ]
    )
    nav_file = nav_path(state_dir)
    if nav_file.exists():
        nav_row = pd.concat([pd.read_csv(nav_file), nav_row], ignore_index=True)
        nav_row = nav_row.drop_duplicates(["datetime", "phase"], keep="last")
    nav_row.to_csv(nav_file, index=False)

    report = report_dir / f"short_hold_fill_{args.execution_date}.md"
    report.write_text(
        f"""# Short Hold Fill Applied

- Execution date: `{args.execution_date}`
- NAV: `{nav:,.2f}`
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
    print(f"Applied fills: {applied}")
    print(f"Positions: {positions_path(state_dir)}")
    print(f"NAV: {nav_file}")
    print(f"Report: {report}")
    print(nav_row.tail(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
