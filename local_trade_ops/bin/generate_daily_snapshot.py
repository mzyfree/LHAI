#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def latest_file(directory: Path, prefix: str, suffix: str = "") -> Path | None:
    if not directory.exists():
        return None
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.name.startswith(prefix) and (not suffix or p.name.endswith(suffix))
    ]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100.0, 4)


def pick_nav_rows(nav_rows: list[dict[str, str]], report_date: str) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    if not nav_rows:
        return None, None
    latest = nav_rows[-1]
    previous = None
    for row in reversed(nav_rows[:-1]):
        if row.get("datetime") != latest.get("datetime") or row.get("phase") != latest.get("phase"):
            previous = row
            break
    if previous is None and len(nav_rows) >= 2:
        previous = nav_rows[-2]
    return latest, previous


def order_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    filled = []
    skipped = []
    partial = []
    total_fee = 0.0
    total_buy_cash = 0.0
    for row in rows:
        action = row.get("action", "")
        shares = to_int(row.get("shares"))
        fill_shares = to_int(row.get("fill_shares"))
        fill_price = to_float(row.get("fill_price"))
        fee = to_float(row.get("fee"))
        total_fee += fee
        item = {
            "code": row.get("instrument", ""),
            "action": action,
            "planned_shares": shares,
            "filled_shares": fill_shares,
            "fill_price": fill_price,
            "fee": fee,
            "status": row.get("fill_status", ""),
            "note": row.get("operator_note", ""),
        }
        if fill_shares > 0 and fill_price > 0:
            filled.append(item)
            if action == "BUY":
                total_buy_cash += fill_shares * fill_price + fee
        elif shares > 0:
            skipped.append(item)
        if 0 < fill_shares < shares:
            partial.append(item)
    return {
        "filled_count": len(filled),
        "skipped_count": len(skipped),
        "partial_count": len(partial),
        "total_fee": round(total_fee, 4),
        "total_buy_cash": round(total_buy_cash, 4),
        "filled": filled,
        "skipped": skipped,
        "partial": partial,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a compact daily account snapshot for mini program reporting.")
    parser.add_argument("--date", default="", help="Report date. Defaults to Asia/Shanghai today.")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    ops_home = Path(os.environ.get("OPS_HOME", Path(__file__).resolve().parents[1])).resolve()
    env_file = Path(args.env_file).expanduser().resolve() if args.env_file else ops_home / "config" / "env.local"
    env = {**read_env(env_file), **os.environ}

    report_date = args.date or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    state_dir = Path(env.get("PAPER_STATE_DIR", ops_home / "state" / "live_10w_main")).expanduser()
    fill_dir = Path(env.get("LIVE_FILL_DIR", ops_home / "live_fills")).expanduser()
    submission_dir = Path(env.get("LIVE_SUBMISSION_DIR", ops_home / "live_submissions")).expanduser()
    task_dir = Path(env.get("LIVE_TASK_DIR", ops_home / "live_tasks")).expanduser()
    output_dir = Path(args.output_dir or env.get("DAILY_SNAPSHOT_DIR", ops_home / "daily_snapshots")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    nav_rows = read_csv(state_dir / "paper_nav.csv")
    positions = read_csv(state_dir / "paper_positions.csv")
    latest_nav, previous_nav = pick_nav_rows(nav_rows, report_date)

    initial_capital = to_float(env.get("CAPITAL"), 100000.0)
    latest_nav_value = to_float(latest_nav.get("nav") if latest_nav else None, initial_capital)
    previous_nav_value = to_float(previous_nav.get("nav") if previous_nav else None, initial_capital)
    cash = to_float(latest_nav.get("cash") if latest_nav else None, initial_capital)
    position_value = to_float(latest_nav.get("position_value") if latest_nav else None, 0.0)
    trade_cost = to_float(latest_nav.get("trade_cost") if latest_nav else None, 0.0)
    daily_pnl = latest_nav_value - previous_nav_value if previous_nav else 0.0
    daily_return = latest_nav_value / previous_nav_value - 1.0 if previous_nav_value else None
    cum_return = latest_nav_value / initial_capital - 1.0 if initial_capital else None
    position_ratio = position_value / latest_nav_value if latest_nav_value else 0.0

    applied_fills = read_csv(fill_dir / f"manual_fills_applied_{report_date}.csv")
    if not applied_fills:
        latest_applied = latest_file(fill_dir, "manual_fills_applied_", ".csv")
        applied_fills = read_csv(latest_applied) if latest_applied else []

    broker_ticket = read_csv(submission_dir / f"broker_order_ticket_{report_date}.csv")
    if not broker_ticket:
        latest_ticket = latest_file(submission_dir, "broker_order_ticket_", ".csv")
        broker_ticket = read_csv(latest_ticket) if latest_ticket else []

    live_task = read_csv(task_dir / f"live_order_task_{report_date}.csv")
    if not live_task:
        latest_task = latest_file(task_dir, "live_order_task_", ".csv")
        live_task = read_csv(latest_task) if latest_task else []

    orders = order_summary(applied_fills)
    position_rows = [
        {
            "code": row.get("instrument", ""),
            "shares": to_int(row.get("shares")),
            "cost_basis": to_float(row.get("cost_basis")),
        }
        for row in positions
    ]

    snapshot = {
        "date": report_date,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "account": {
            "initial_capital": initial_capital,
            "nav": round(latest_nav_value, 4),
            "cash": round(cash, 4),
            "position_value": round(position_value, 4),
            "position_ratio_pct": pct(position_ratio),
            "trade_cost": round(trade_cost, 4),
            "daily_pnl": round(daily_pnl, 4),
            "daily_return_pct": pct(daily_return),
            "cum_return_pct": pct(cum_return),
            "latest_nav_date": latest_nav.get("datetime") if latest_nav else None,
            "latest_nav_phase": latest_nav.get("phase") if latest_nav else None,
            "previous_nav": round(previous_nav_value, 4) if previous_nav else None,
        },
        "orders": {
            "planned_count": len(broker_ticket or live_task),
            **orders,
        },
        "positions": position_rows,
        "source_files": {
            "state_dir": str(state_dir),
            "fill_dir": str(fill_dir),
            "submission_dir": str(submission_dir),
            "task_dir": str(task_dir),
        },
    }

    dated_path = output_dir / f"daily_snapshot_{report_date}.json"
    latest_path = output_dir / "daily_snapshot_latest.json"
    text = json.dumps(snapshot, ensure_ascii=False, indent=2)
    dated_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote: {dated_path}")
    print(f"Wrote: {latest_path}")
    print(
        "Summary:",
        f"NAV={snapshot['account']['nav']}",
        f"daily_return_pct={snapshot['account']['daily_return_pct']}",
        f"cum_return_pct={snapshot['account']['cum_return_pct']}",
        f"positions={len(position_rows)}",
        f"filled={snapshot['orders']['filled_count']}",
        f"skipped={snapshot['orders']['skipped_count']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
