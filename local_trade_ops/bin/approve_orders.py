#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


VALID_STATUSES = {"approved", "approved_with_edits", "rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve, edit, or reject generated orders.")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    parser.add_argument("--reviewer", default="dylan")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def load_env_state_dir() -> Path:
    here = Path(__file__).resolve()
    ops_home = here.parents[1]
    env_path = ops_home / "config" / "env.local"
    if not env_path.exists():
        env_path = ops_home / "config" / "env.local.example"
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return Path(values["PAPER_STATE_DIR"]).expanduser().resolve()


def main() -> None:
    args = parse_args()
    state_dir = Path(args.state_dir).expanduser().resolve() if args.state_dir else load_env_state_dir()
    execution_date = args.execution_date
    pending = state_dir / f"pending_orders_{execution_date}.csv"
    approved = state_dir / f"approved_orders_{execution_date}.csv"
    status_path = state_dir / f"approval_status_{execution_date}.json"

    if args.status == "approved":
        if not pending.exists():
            raise FileNotFoundError(f"Pending order file not found: {pending}")
        shutil.copyfile(pending, approved)
    elif args.status == "approved_with_edits":
        if not approved.exists():
            raise FileNotFoundError(f"Edited approved order file not found: {approved}")

    payload = {}
    if status_path.exists():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "execution_date": execution_date,
            "status": args.status,
            "reviewer": args.reviewer,
            "review_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": args.note,
        }
    )
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status={args.status}")
    print(f"approval_status={status_path}")
    if approved.exists():
        print(f"approved_orders={approved}")


if __name__ == "__main__":
    main()

