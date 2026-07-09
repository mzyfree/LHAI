#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


APPROVED_STATUSES = {"approved", "approved_with_edits"}


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
    env.setdefault("PAPER_STATE_DIR", str(ops_home / "state" / "live_10w_main"))
    env.setdefault("LIVE_TASK_DIR", str(ops_home / "live_tasks"))
    env.setdefault("LIVE_TRADING_ENABLED", "0")
    env.setdefault("LIVE_ORDER_HOOK", "")
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and optionally trigger an approved live order task.")
    parser.add_argument("--execution-date", required=True, help="T+1 execution date, e.g. 2026-05-19.")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--task-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing live task files.")
    parser.add_argument("--trigger", action="store_true", help="Run LIVE_ORDER_HOOK if live trading is enabled.")
    return parser.parse_args()


def normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    required = ["signal_date", "execution_date", "instrument", "action", "shares", "estimated_price", "reason"]
    missing = [col for col in required if col not in orders.columns]
    if missing:
        raise ValueError(f"Approved orders missing columns: {missing}")
    optional = ["order_role", "model_rank", "score", "backup_rank"]
    cols = required + [col for col in optional if col in orders.columns]
    out = orders.loc[:, cols].copy()
    if "order_role" not in out.columns:
        out["order_role"] = "primary"
    out["order_role"] = out["order_role"].fillna("primary").astype(str)
    out["instrument"] = out["instrument"].astype(str)
    out["action"] = out["action"].astype(str).str.upper()
    out["shares"] = out["shares"].astype(int)
    out["estimated_price"] = out["estimated_price"].astype(float)
    out = out[out["action"].isin(["BUY", "SELL"]) & (out["shares"] > 0)].reset_index(drop=True)
    out["order_type"] = out["order_role"].map({"backup": "BACKUP_MANUAL"}).fillna("MARKET_ON_OPEN")
    out["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


def main() -> None:
    args = parse_args()
    env = load_env()
    execution_date = args.execution_date
    state_dir = Path(args.state_dir or env["PAPER_STATE_DIR"]).expanduser().resolve()
    task_dir = Path(args.task_dir or env["LIVE_TASK_DIR"]).expanduser().resolve()
    approval_path = state_dir / f"approval_status_{execution_date}.json"
    approved_path = state_dir / f"approved_orders_{execution_date}.csv"

    if not approval_path.exists():
        raise FileNotFoundError(f"Approval status not found: {approval_path}")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    status = str(approval.get("status", "")).strip().lower()
    if status not in APPROVED_STATUSES:
        raise RuntimeError(f"Execution date {execution_date} is not approved. status={status or 'missing'}")
    if not approved_path.exists():
        raise FileNotFoundError(f"Approved orders not found: {approved_path}")

    orders = normalize_orders(pd.read_csv(approved_path))
    task_dir.mkdir(parents=True, exist_ok=True)
    task_csv = task_dir / f"live_order_task_{execution_date}.csv"
    task_json = task_dir / f"live_order_task_{execution_date}.json"
    if (task_csv.exists() or task_json.exists()) and not args.force:
        raise FileExistsError(f"Live task already exists for {execution_date}. Use --force to overwrite.")

    orders.to_csv(task_csv, index=False)
    payload = {
        "execution_date": execution_date,
        "status": "created",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "approval": approval,
        "approved_orders": str(approved_path),
        "task_csv": str(task_csv),
        "n_orders": int(len(orders)),
        "n_primary_orders": int(orders["order_role"].eq("primary").sum()),
        "n_backup_orders": int(orders["order_role"].eq("backup").sum()),
        "buy_notional_est": float((orders.loc[orders["action"].eq("BUY") & orders["order_role"].eq("primary"), "shares"] * orders.loc[orders["action"].eq("BUY") & orders["order_role"].eq("primary"), "estimated_price"]).sum()),
        "sell_notional_est": float((orders.loc[orders["action"].eq("SELL") & orders["order_role"].eq("primary"), "shares"] * orders.loc[orders["action"].eq("SELL") & orders["order_role"].eq("primary"), "estimated_price"]).sum()),
        "live_trading_enabled": str(env.get("LIVE_TRADING_ENABLED", "0")),
        "live_order_hook": str(env.get("LIVE_ORDER_HOOK", "")),
    }
    task_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Live task CSV: {task_csv}")
    print(f"Live task JSON: {task_json}")
    print(orders.to_string(index=False) if not orders.empty else "No live orders.")

    live_enabled = str(env.get("LIVE_TRADING_ENABLED", "0")).lower() in {"1", "true", "yes"}
    hook = str(env.get("LIVE_ORDER_HOOK", "")).strip()
    if args.trigger:
        if not live_enabled:
            print("LIVE_TRADING_ENABLED is not enabled. Task created, broker hook skipped.")
            return
        if not hook:
            raise RuntimeError("LIVE_ORDER_HOOK is empty; cannot trigger broker execution.")
        subprocess.run([hook, str(task_csv), str(task_json)], check=True, env={**os.environ, **env})


if __name__ == "__main__":
    main()
