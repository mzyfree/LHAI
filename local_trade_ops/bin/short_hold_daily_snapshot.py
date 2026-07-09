#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
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
    env.setdefault("SHORT_HOLD_STATE_DIR", str(ops_home / "state" / "short_hold_single_peak"))
    env.setdefault("SHORT_HOLD_FILL_DIR", str(ops_home / "short_hold_fills"))
    env.setdefault("SHORT_HOLD_SNAPSHOT_DIR", str(ops_home / "short_hold_daily_snapshots"))
    return env


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def clean_float(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate short-hold daily snapshot JSON.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = load_env()
    state_dir = Path(env["SHORT_HOLD_STATE_DIR"]).expanduser().resolve()
    fill_dir = Path(env["SHORT_HOLD_FILL_DIR"]).expanduser().resolve()
    snapshot_dir = Path(env["SHORT_HOLD_SNAPSHOT_DIR"]).expanduser().resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    nav = read_csv(state_dir / "paper_nav.csv")
    positions = read_csv(state_dir / "paper_positions.csv")
    fills = read_csv(fill_dir / f"short_hold_fills_applied_{args.date}.csv")
    latest_nav = nav.iloc[-1].to_dict() if not nav.empty else {}
    initial = float(nav.iloc[0]["nav"]) if not nav.empty and "nav" in nav.columns else float(env.get("SHORT_HOLD_CAPITAL", env.get("CAPITAL", "100000")))
    current = float(latest_nav.get("nav", initial) or initial)
    prev_nav = float(nav.iloc[-2]["nav"]) if len(nav) >= 2 and "nav" in nav.columns else initial
    operations = []
    if not fills.empty:
        for _, row in fills.iterrows():
            planned = int(pd.to_numeric(row.get("shares", 0), errors="coerce") or 0)
            filled = int(pd.to_numeric(row.get("fill_shares", 0), errors="coerce") or 0)
            status = "filled" if filled >= planned and planned > 0 else ("partial" if filled > 0 else "skipped")
            operations.append(
                {
                    "code": str(row.get("instrument", "")),
                    "action": str(row.get("action", "")),
                    "order_role": str(row.get("order_role", "")),
                    "planned": planned,
                    "filled": filled,
                    "ref_price": clean_float(row.get("estimated_price", 0)),
                    "price_5pct": clean_float(row.get("price_5pct", 0)),
                    "fill_price": clean_float(row.get("fill_price", 0)),
                    "status": status,
                    "note": str(row.get("operator_note", "")),
                    "exit_date": str(row.get("exit_date", "")),
                }
            )
    payload = {
        "strategy": "short_hold_single_peak",
        "date": args.date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "account": {
            "nav": current,
            "cash": float(latest_nav.get("cash", initial) or initial),
            "position_value": float(latest_nav.get("position_value", 0) or 0),
            "daily_pnl": current - prev_nav,
            "daily_return_pct": (current / prev_nav - 1.0) * 100 if prev_nav else 0.0,
            "cum_return_pct": (current / initial - 1.0) * 100 if initial else 0.0,
            "position_ratio_pct": float(latest_nav.get("position_value", 0) or 0) / current * 100 if current else 0.0,
        },
        "orders": {
            "planned_count": len(fills) if not fills.empty else 0,
            "filled_count": sum(1 for item in operations if item["filled"] > 0),
            "skipped_count": sum(1 for item in operations if item["filled"] <= 0),
            "operations": operations,
        },
        "positions": positions.to_dict(orient="records") if not positions.empty else [],
    }
    path = snapshot_dir / f"short_hold_snapshot_{args.date}.json"
    latest = snapshot_dir / "short_hold_snapshot_latest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    latest.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Snapshot: {path}")
    print(f"Latest: {latest}")
    print(json.dumps(payload["account"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
