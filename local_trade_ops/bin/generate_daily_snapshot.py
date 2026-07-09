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
        order_role = row.get("order_role", "primary") or "primary"
        shares = to_int(row.get("shares"))
        fill_shares = to_int(row.get("fill_shares"))
        fill_price = to_float(row.get("fill_price"))
        fee = to_float(row.get("fee"))
        if order_role == "backup" and fill_shares <= 0:
            continue
        total_fee += fee
        item = {
            "code": row.get("instrument", ""),
            "action": action,
            "order_role": order_role,
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


def first_non_empty(*values: object) -> object:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def format_price(value: float) -> float | None:
    return round(value, 2) if value > 0 else None


def latest_signal_scores(state_dir: Path, report_date: str) -> dict[str, dict[str, object]]:
    candidates = []
    for path in state_dir.glob("signal_rankings_*.csv"):
        date = path.stem.removeprefix("signal_rankings_")
        if date <= report_date:
            candidates.append((date, path))
    if not candidates:
        return {}
    signal_date, path = max(candidates, key=lambda item: item[0])
    scores: dict[str, dict[str, object]] = {}
    for row in read_csv(path):
        inst = row.get("instrument", "")
        if not inst:
            continue
        scores[inst] = {
            "model_rank": first_non_empty(row.get("rank"), ""),
            "score": first_non_empty(row.get("score"), ""),
            "signal_date": signal_date,
        }
    return scores


def build_operations(planned_rows: list[dict[str, str]], fill_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    planned_by_id = {row.get("client_order_id", ""): row for row in planned_rows if row.get("client_order_id")}
    fills_by_id = {row.get("client_order_id", ""): row for row in fill_rows if row.get("client_order_id")}

    ordered_ids: list[str] = []
    for row in planned_rows + fill_rows:
        order_id = row.get("client_order_id", "")
        if order_id and order_id not in ordered_ids:
            ordered_ids.append(order_id)

    operations: list[dict[str, object]] = []
    for order_id in ordered_ids:
        planned = planned_by_id.get(order_id, {})
        fill = fills_by_id.get(order_id, {})
        row = {**planned, **fill}
        action = str(row.get("action", "")).upper()
        planned_shares = to_int(first_non_empty(row.get("shares"), planned.get("shares"), fill.get("shares")))
        filled_shares = to_int(fill.get("fill_shares"))
        fill_price = to_float(fill.get("fill_price"))
        ref_price = to_float(first_non_empty(row.get("estimated_price"), row.get("ref_price")))
        price_3pct = to_float(row.get("price_3pct"))
        price_5pct = to_float(row.get("price_5pct"))
        if action == "BUY" and ref_price > 0:
            if price_3pct <= 0:
                price_3pct = ref_price * 1.03
            if price_5pct <= 0:
                price_5pct = ref_price * 1.05

        if filled_shares >= planned_shares and planned_shares > 0:
            status = "filled"
        elif filled_shares > 0:
            status = "partial"
        elif fill:
            status = "skipped"
        else:
            status = "pending"
        order_role = first_non_empty(row.get("order_role"), "primary")
        note = first_non_empty(row.get("operator_note"))
        if order_role != "backup":
            note = first_non_empty(note, row.get("reason"))
            note = first_non_empty(note, row.get("manual_price_rule"))

        operations.append(
            {
                "client_order_id": order_id,
                "signal_date": first_non_empty(row.get("signal_date"), planned.get("signal_date")),
                "execution_date": first_non_empty(row.get("execution_date"), planned.get("execution_date")),
                "code": first_non_empty(row.get("instrument"), planned.get("instrument")),
                "action": action,
                "order_role": order_role,
                "backup_rank": first_non_empty(row.get("backup_rank"), ""),
                "model_rank": first_non_empty(row.get("model_rank"), ""),
                "score": first_non_empty(row.get("score"), ""),
                "reason": row.get("reason", ""),
                "planned_shares": planned_shares,
                "filled_shares": filled_shares,
                "ref_price": format_price(ref_price),
                "price_3pct": format_price(price_3pct),
                "price_5pct": format_price(price_5pct),
                "fill_price": format_price(fill_price),
                "fee": round(to_float(fill.get("fee")), 4),
                "status": status,
                "note": note,
                "manual_price_rule": row.get("manual_price_rule", ""),
            }
        )
    return operations


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

    planned_orders = broker_ticket or live_task
    orders = order_summary(applied_fills)
    operations = build_operations(planned_orders, applied_fills)
    planned_count = sum(1 for row in planned_orders if (row.get("order_role", "primary") or "primary") == "primary")
    signal_scores = latest_signal_scores(state_dir, report_date)
    position_rows = [
        {
            "code": row.get("instrument", ""),
            "shares": to_int(row.get("shares")),
            "cost_basis": to_float(row.get("cost_basis")),
            "model_rank": signal_scores.get(row.get("instrument", ""), {}).get("model_rank", ""),
            "score": signal_scores.get(row.get("instrument", ""), {}).get("score", ""),
            "signal_date": signal_scores.get(row.get("instrument", ""), {}).get("signal_date", ""),
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
            "planned_count": planned_count,
            "operations": operations,
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
