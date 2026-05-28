#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    env.setdefault("LIVE_TASK_DIR", str(ops_home / "live_tasks"))
    env.setdefault("LIVE_SUBMISSION_DIR", str(ops_home / "live_submissions"))
    env.setdefault("LIVE_FILL_DIR", str(ops_home / "live_fills"))
    env.setdefault("LIVE_BROKER_MODE", "manual")
    env.setdefault("LIVE_TRADING_ENABLED", "0")
    env.setdefault("LIVE_ORDER_HOOK", "")
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit or export an approved live order task.")
    parser.add_argument("--execution-date", required=True, help="Execution date, e.g. 2026-05-19.")
    parser.add_argument("--task-dir", default=None)
    parser.add_argument("--submission-dir", default=None)
    parser.add_argument("--mode", choices=["manual", "hook"], default=None)
    parser.add_argument("--allow-non-today", action="store_true", help="Allow submission for dates other than today's local date.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing submission files.")
    return parser.parse_args()


def today_local() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def normalize_task_orders(orders: pd.DataFrame, execution_date: str) -> pd.DataFrame:
    required = ["signal_date", "execution_date", "instrument", "action", "shares", "estimated_price", "reason", "order_type"]
    missing = [col for col in required if col not in orders.columns]
    if missing:
        raise ValueError(f"Live task missing columns: {missing}")
    out = orders.loc[:, required].copy()
    out["execution_date"] = out["execution_date"].astype(str)
    if not out["execution_date"].eq(execution_date).all():
        bad = sorted(out.loc[~out["execution_date"].eq(execution_date), "execution_date"].unique())
        raise ValueError(f"Task contains mismatched execution_date values: {bad}")
    out["instrument"] = out["instrument"].astype(str)
    out["action"] = out["action"].astype(str).str.upper()
    out["shares"] = out["shares"].astype(int)
    out["estimated_price"] = out["estimated_price"].astype(float)
    out = out[out["action"].isin(["SELL", "BUY"]) & (out["shares"] > 0)].reset_index(drop=True)
    if out.empty:
        raise ValueError("Live task has no valid BUY/SELL orders.")
    out["submit_sequence"] = out["action"].map({"SELL": 0, "BUY": 1}).astype(int)
    out = out.sort_values(["submit_sequence", "instrument"]).reset_index(drop=True)
    out.insert(0, "client_order_id", [f"{execution_date.replace('-', '')}-{i:03d}" for i in range(1, len(out) + 1)])
    out["submit_instruction"] = out["action"] + " " + out["instrument"] + " " + out["shares"].astype(str)
    return out


def run_hook(hook: str, ticket_csv: Path, submission_json: Path, task_csv: Path, task_json: Path, env: dict[str, str]) -> dict:
    result = subprocess.run(
        [hook, str(ticket_csv), str(submission_json), str(task_csv), str(task_json)],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, **env},
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    args = parse_args()
    env = load_env()
    execution_date = args.execution_date
    if not args.allow_non_today and execution_date != today_local():
        raise RuntimeError(
            f"Refusing to submit non-today task. execution_date={execution_date}, today={today_local()}. "
            "Use --allow-non-today only for dry-run/backfill tests."
        )

    task_dir = Path(args.task_dir or env["LIVE_TASK_DIR"]).expanduser().resolve()
    submission_dir = Path(args.submission_dir or env["LIVE_SUBMISSION_DIR"]).expanduser().resolve()
    fill_dir = Path(env["LIVE_FILL_DIR"]).expanduser().resolve()
    task_csv = task_dir / f"live_order_task_{execution_date}.csv"
    task_json = task_dir / f"live_order_task_{execution_date}.json"
    if not task_csv.exists():
        raise FileNotFoundError(f"Live task CSV not found: {task_csv}")
    if not task_json.exists():
        raise FileNotFoundError(f"Live task JSON not found: {task_json}")

    submission_dir.mkdir(parents=True, exist_ok=True)
    fill_dir.mkdir(parents=True, exist_ok=True)
    ticket_csv = submission_dir / f"broker_order_ticket_{execution_date}.csv"
    submission_json = submission_dir / f"live_submission_{execution_date}.json"
    fill_template_csv = fill_dir / f"manual_fill_template_{execution_date}.csv"
    if (ticket_csv.exists() or submission_json.exists() or fill_template_csv.exists()) and not args.force:
        raise FileExistsError(f"Submission already exists for {execution_date}. Use --force to overwrite.")

    orders = normalize_task_orders(pd.read_csv(task_csv), execution_date)
    orders.to_csv(ticket_csv, index=False)
    fill_template = orders.copy()
    fill_template["fill_status"] = "PENDING"
    fill_template["fill_shares"] = ""
    fill_template["fill_price"] = ""
    fill_template["fee"] = ""
    fill_template["operator_note"] = ""
    fill_template.to_csv(fill_template_csv, index=False)

    mode = (args.mode or env.get("LIVE_BROKER_MODE", "manual")).strip().lower()
    if mode not in {"manual", "hook"}:
        raise ValueError(f"Unsupported LIVE_BROKER_MODE={mode}. Use manual or hook.")
    live_enabled = str(env.get("LIVE_TRADING_ENABLED", "0")).lower() in {"1", "true", "yes"}
    hook = str(env.get("LIVE_ORDER_HOOK", "")).strip()

    payload = {
        "execution_date": execution_date,
        "mode": mode,
        "status": "manual_ticket_created",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task_csv": str(task_csv),
        "task_json": str(task_json),
        "ticket_csv": str(ticket_csv),
        "fill_template_csv": str(fill_template_csv),
        "n_orders": int(len(orders)),
        "buy_notional_est": float((orders.loc[orders["action"].eq("BUY"), "shares"] * orders.loc[orders["action"].eq("BUY"), "estimated_price"]).sum()),
        "sell_notional_est": float((orders.loc[orders["action"].eq("SELL"), "shares"] * orders.loc[orders["action"].eq("SELL"), "estimated_price"]).sum()),
        "live_trading_enabled": str(env.get("LIVE_TRADING_ENABLED", "0")),
        "live_order_hook": hook,
        "hook_result": None,
    }
    submission_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if mode == "hook":
        if not live_enabled:
            raise RuntimeError("LIVE_BROKER_MODE=hook requires LIVE_TRADING_ENABLED=1.")
        if not hook:
            raise RuntimeError("LIVE_BROKER_MODE=hook requires LIVE_ORDER_HOOK.")
        hook_result = run_hook(hook, ticket_csv, submission_json, task_csv, task_json, env)
        payload["hook_result"] = hook_result
        payload["status"] = "submitted" if hook_result["returncode"] == 0 else "submit_failed"
        submission_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        if hook_result["returncode"] != 0:
            raise RuntimeError(f"Broker hook failed with code {hook_result['returncode']}. See {submission_json}")

    print(f"Broker ticket CSV: {ticket_csv}")
    print(f"Manual fill template CSV: {fill_template_csv}")
    print(f"Submission JSON: {submission_json}")
    print(f"status={payload['status']}")
    print(orders.to_string(index=False))


if __name__ == "__main__":
    main()
