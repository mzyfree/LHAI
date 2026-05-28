#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def ops_home() -> Path:
    return Path(__file__).resolve().parents[1]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay manual open-buy decisions from historical tick data.")
    parser.add_argument("--execution-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--snapshot-time", default="09:30:30", help="HH:MM:SS, default 09:30:30")
    parser.add_argument("--ticket-csv", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--tick-source", default="sn", choices=["sn", "tt", "nt"])
    return parser.parse_args()


def plain_code(instrument: str) -> str:
    inst = str(instrument).strip()
    if inst.startswith(("SH", "SZ", "BJ")):
        return inst[2:]
    if inst.endswith((".SH", ".SZ", ".BJ")):
        return inst.split(".", 1)[0]
    return inst


def normalize_time(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            parts.append("00")
        return ":".join(part.zfill(2) for part in parts[:3])
    digits = "".join(ch for ch in text if ch.isdigit()).zfill(6)
    return f"{digits[:2]}:{digits[2:4]}:{digits[4:6]}"


def normalize_ticks(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"time", "时间"}:
            rename[col] = "time"
        elif key in {"price", "成交价", "成交价格"}:
            rename[col] = "price"
        elif key in {"volume", "vol", "成交量", "成交手"}:
            rename[col] = "volume"
        elif key in {"amount", "成交金额"}:
            rename[col] = "amount"
        elif key in {"type", "买卖类型"}:
            rename[col] = "type"
    out = df.rename(columns=rename).copy()
    if "time" not in out.columns or "price" not in out.columns:
        raise ValueError(f"Unsupported tick columns: {list(df.columns)}")
    out["time"] = out["time"].map(normalize_time)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"])
    return out.sort_values("time")


def decision(action: str, price: float, price_5pct: float) -> str:
    if action.upper() != "BUY":
        return "卖出/非买入: 不做高开判断"
    if not math.isfinite(price):
        return "历史分笔不可用: 无法复盘"
    if price > price_5pct:
        return "跳过: 09:30:30后首笔价超过5%阈值"
    return "历史复盘可买: 首笔价<=price_5pct"


def main() -> int:
    args = parse_args()
    env = parse_env(ops_home() / "config" / "env.local")
    submission_dir = Path(env.get("LIVE_SUBMISSION_DIR", ops_home() / "live_submissions")).expanduser().resolve()
    ticket_csv = Path(args.ticket_csv).expanduser().resolve() if args.ticket_csv else submission_dir / f"broker_order_ticket_{args.execution_date}.csv"
    if not ticket_csv.exists():
        raise FileNotFoundError(f"Broker ticket not found: {ticket_csv}")

    ticket = pd.read_csv(ticket_csv)
    if ticket.empty:
        raise SystemExit(f"Ticket is empty: {ticket_csv}")
    if "estimated_price" not in ticket.columns:
        raise ValueError("ticket CSV must contain estimated_price")

    import tushare as ts

    rows: list[dict[str, object]] = []
    snapshot = normalize_time(args.snapshot_time)
    for _, raw in ticket.iterrows():
        row = raw.to_dict()
        instrument = str(row.get("instrument", ""))
        action = str(row.get("action", "")).upper()
        estimated = float(row.get("estimated_price"))
        price_3pct = estimated * 1.03
        price_5pct = estimated * 1.05
        snapshot_price = math.nan
        snapshot_tick_time = ""
        tick_error = ""
        if action == "BUY":
            try:
                ticks = ts.get_tick_data(plain_code(instrument), date=args.execution_date, src=args.tick_source)
                ticks = normalize_ticks(ticks if ticks is not None else pd.DataFrame())
                after = ticks[ticks["time"] >= snapshot]
                chosen = after.iloc[0] if not after.empty else ticks.iloc[0]
                snapshot_price = float(chosen["price"])
                snapshot_tick_time = str(chosen["time"])
            except Exception as exc:  # noqa: BLE001 - keep per-instrument failure visible in output.
                tick_error = f"{exc.__class__.__name__}: {exc}"
        rows.append(
            {
                "execution_date": args.execution_date,
                "snapshot_time": snapshot,
                "instrument": instrument,
                "action": action,
                "shares": row.get("shares", ""),
                "estimated_price": round(estimated, 3),
                "price_3pct": round(price_3pct, 3),
                "price_5pct": round(price_5pct, 3),
                "snapshot_tick_time": snapshot_tick_time,
                "snapshot_price": round(snapshot_price, 3) if math.isfinite(snapshot_price) else "",
                "gap_pct": round((snapshot_price / estimated - 1.0) * 100, 2) if math.isfinite(snapshot_price) else "",
                "decision": decision(action, snapshot_price, price_5pct),
                "tick_error": tick_error,
            }
        )

    output = Path(args.output).expanduser().resolve() if args.output else submission_dir / f"historical_rehearsal_{args.execution_date}_{snapshot.replace(':', '')}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False)
    print(json.dumps({"output": str(output), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
