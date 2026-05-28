#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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


def ops_home() -> Path:
    return Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    home = ops_home()
    env = parse_env(home / "config" / "env.local")
    env.setdefault("OPS_HOME", str(home))
    env.setdefault("LIVE_SUBMISSION_DIR", str(home / "live_submissions"))
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch TuShare realtime quotes for manual order decision aid.")
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--ticket-csv", default="")
    parser.add_argument("--env-path", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def to_tushare_code(instrument: str) -> str:
    inst = str(instrument).strip()
    if inst.startswith("SH"):
        return f"{inst[2:]}.SH"
    if inst.startswith("SZ"):
        return f"{inst[2:]}.SZ"
    if inst.endswith((".SH", ".SZ", ".BJ")):
        return inst
    raise ValueError(f"Unsupported instrument code: {instrument}")


def to_plain_code(instrument: str) -> str:
    inst = str(instrument).strip()
    if inst.startswith(("SH", "SZ", "BJ")):
        return inst[2:]
    if inst.endswith((".SH", ".SZ", ".BJ")):
        return inst.split(".", 1)[0]
    return inst


def from_tushare_code(ts_code: str) -> str:
    code, suffix = str(ts_code).split(".", 1)
    if suffix == "SH":
        return f"SH{code}"
    if suffix == "SZ":
        return f"SZ{code}"
    if suffix == "BJ":
        return f"BJ{code}"
    return str(ts_code)


def instrument_from_plain_code(code: str, source_map: dict[str, str]) -> str:
    key = str(code).strip().zfill(6)
    if key in source_map:
        return source_map[key]
    if key.startswith("6"):
        return f"SH{key}"
    if key.startswith(("0", "3")):
        return f"SZ{key}"
    if key.startswith(("4", "8")):
        return f"BJ{key}"
    return key


def to_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def status_for(row: pd.Series) -> str:
    if str(row.get("action", "")).upper() != "BUY":
        return "卖出/非买入: 不做高开判断"
    price_5pct = to_float(row.get("price_5pct"))
    current = to_float(row.get("current_price"))
    ask1 = to_float(row.get("ask_price1"))
    ask_vol1 = to_float(row.get("ask_volume1"))
    if not math.isfinite(price_5pct):
        return "缺少price_5pct"
    if not math.isfinite(current) and not math.isfinite(ask1):
        return "实时行情不可用: 手工看盘"
    if math.isfinite(current) and current > price_5pct:
        return "跳过: 当前价超过5%阈值"
    if math.isfinite(ask1) and ask1 > price_5pct:
        return "跳过: 卖一超过5%阈值"
    if math.isfinite(ask1) and ask1 <= 0:
        return "跳过/观察: 卖一不可用"
    if math.isfinite(ask_vol1) and ask_vol1 <= 0:
        return "观察: 卖一无量"
    return "可人工买入: 限价<=price_5pct"


def recommended_limit(row: pd.Series) -> str:
    if str(row.get("action", "")).upper() != "BUY":
        return ""
    price_5pct = to_float(row.get("price_5pct"))
    ask1 = to_float(row.get("ask_price1"))
    current = to_float(row.get("current_price"))
    base = ask1 if math.isfinite(ask1) and ask1 > 0 else current
    if not (math.isfinite(base) and math.isfinite(price_5pct) and base <= price_5pct):
        return ""
    tick = 0.01
    return f"{min(base + tick, price_5pct):.2f}"


def fetch_legacy_realtime_quotes(ts_module: object, instruments: list[str]) -> pd.DataFrame:
    plain_codes = [to_plain_code(x) for x in instruments]
    source_map = {to_plain_code(x): x for x in instruments}
    quote = ts_module.get_realtime_quotes(plain_codes)
    if quote is None or quote.empty:
        return pd.DataFrame()
    quote = quote.rename(
        columns={
            "price": "current_price",
            "a1_p": "ask_price1",
            "a1_v": "ask_volume1",
            "b1_p": "bid_price1",
            "b1_v": "bid_volume1",
            "time": "quote_time",
        }
    )
    quote["instrument"] = quote["code"].map(lambda code: instrument_from_plain_code(code, source_map))
    return quote


def fetch_pro_rt_k(ts_module: object, token: str, instruments: list[str]) -> pd.DataFrame:
    ts_codes = [to_tushare_code(x) for x in instruments]
    # Pass the token directly so TuShare does not try to persist it under $HOME.
    pro = ts_module.pro_api(token)
    quote = pro.rt_k(ts_code=",".join(ts_codes))
    if quote is None or quote.empty:
        return pd.DataFrame()
    quote = quote.rename(
        columns={
            "close": "current_price",
            "trade_time": "quote_time",
        }
    )
    quote["instrument"] = quote["ts_code"].map(from_tushare_code)
    return quote


def main() -> int:
    args = parse_args()
    env_path = Path(args.env_path).expanduser().resolve() if args.env_path else ops_home() / "config" / "env.local"
    env = parse_env(env_path) or load_env()
    token = env.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("TUSHARE_TOKEN is not configured.")

    submission_dir = Path(env.get("LIVE_SUBMISSION_DIR", ops_home() / "live_submissions")).expanduser().resolve()
    ticket_csv = Path(args.ticket_csv).expanduser().resolve() if args.ticket_csv else submission_dir / f"broker_order_ticket_{args.execution_date}.csv"
    if not ticket_csv.exists():
        raise FileNotFoundError(f"Broker ticket not found: {ticket_csv}")

    ticket = pd.read_csv(ticket_csv)
    if ticket.empty:
        raise SystemExit(f"Ticket is empty: {ticket_csv}")
    if "estimated_price" not in ticket.columns:
        raise ValueError("ticket CSV must contain estimated_price")
    ticket["instrument"] = ticket["instrument"].astype(str)
    ticket["price_3pct"] = pd.to_numeric(ticket["estimated_price"], errors="coerce") * 1.03
    ticket["price_5pct"] = pd.to_numeric(ticket["estimated_price"], errors="coerce") * 1.05
    instruments = ticket["instrument"].tolist()

    import tushare as ts

    quote_source = "get_realtime_quotes"
    try:
        quote = fetch_legacy_realtime_quotes(ts, instruments)
    except Exception as legacy_err:
        quote_source = f"rt_k_fallback_after_{legacy_err.__class__.__name__}"
        quote = fetch_pro_rt_k(ts, token, instruments)
    keep_cols = [
        "instrument",
        "current_price",
        "open",
        "pre_close",
        "high",
        "low",
        "ask_price1",
        "ask_volume1",
        "bid_price1",
        "bid_volume1",
        "quote_time",
    ]
    quote_cols = [c for c in keep_cols if c in quote.columns]
    merged = ticket.merge(quote[quote_cols], on="instrument", how="left") if quote_cols else ticket.copy()
    for col in ["price_3pct", "price_5pct", "current_price", "ask_price1", "bid_price1"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").round(3)
    merged["open_gap_pct"] = ((pd.to_numeric(merged.get("current_price"), errors="coerce") / pd.to_numeric(merged["estimated_price"], errors="coerce") - 1.0) * 100).round(2)
    merged["decision"] = merged.apply(status_for, axis=1)
    merged["recommended_limit_price"] = merged.apply(recommended_limit, axis=1)

    cols = [
        "execution_date",
        "instrument",
        "action",
        "shares",
        "estimated_price",
        "price_3pct",
        "price_5pct",
        "current_price",
        "ask_price1",
        "ask_volume1",
        "bid_price1",
        "bid_volume1",
        "open_gap_pct",
        "recommended_limit_price",
        "decision",
        "quote_time",
    ]
    out = merged[[c for c in cols if c in merged.columns]].copy()
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        output = submission_dir / f"realtime_decision_{args.execution_date}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    payload = {
        "execution_date": args.execution_date,
        "ticket_csv": str(ticket_csv),
        "output": str(output),
        "quote_source": quote_source,
        "rows": json.loads(out.fillna("").to_json(orient="records", force_ascii=False)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
