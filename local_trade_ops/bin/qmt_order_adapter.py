#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
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
    env.setdefault("QMT_ACCOUNT_TYPE", "STOCK")
    env.setdefault("QMT_SESSION_ID", "1001")
    env.setdefault("QMT_ORDER_TYPE", "LATEST_PRICE")
    env.setdefault("QMT_PRICE_TYPE", "MARKET")
    env.setdefault("QMT_STRATEGY_NAME", "llhh_csi1000_main")
    env.setdefault("QMT_ORDER_REMARK", "local_trade_ops")
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QMT/miniQMT broker hook for local_trade_ops.")
    parser.add_argument("ticket_csv", help="broker_order_ticket_YYYY-MM-DD.csv")
    parser.add_argument("submission_json", help="live_submission_YYYY-MM-DD.json")
    parser.add_argument("task_csv", nargs="?", default="")
    parser.add_argument("task_json", nargs="?", default="")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write payload, but do not connect/order.")
    parser.add_argument("--query-only", action="store_true", help="Connect QMT and query account/positions only.")
    return parser.parse_args()


def to_qmt_code(instrument: str) -> str:
    inst = str(instrument).strip().upper()
    if inst.startswith("SH"):
        return f"{inst[2:]}.SH"
    if inst.startswith("SZ"):
        return f"{inst[2:]}.SZ"
    if inst.endswith(".SH") or inst.endswith(".SZ"):
        return inst
    raise ValueError(f"Unsupported instrument format for QMT: {instrument}")


def load_orders(ticket_csv: Path) -> pd.DataFrame:
    if not ticket_csv.exists():
        raise FileNotFoundError(f"Ticket CSV not found: {ticket_csv}")
    orders = pd.read_csv(ticket_csv)
    required = ["client_order_id", "instrument", "action", "shares", "estimated_price"]
    missing = [col for col in required if col not in orders.columns]
    if missing:
        raise ValueError(f"Ticket missing columns: {missing}")
    orders = orders.copy()
    orders["action"] = orders["action"].astype(str).str.upper()
    orders["shares"] = orders["shares"].astype(int)
    orders["estimated_price"] = orders["estimated_price"].astype(float)
    orders = orders[orders["action"].isin(["BUY", "SELL"]) & (orders["shares"] > 0)].reset_index(drop=True)
    orders["qmt_code"] = orders["instrument"].map(to_qmt_code)
    return orders


def import_qmt_modules():
    try:
        from xtquant import xtconstant
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount
    except Exception as exc:  # pragma: no cover - depends on local QMT installation
        raise RuntimeError(
            "xtquant is not available. Install/configure miniQMT/QMT first, then retry. "
            "This adapter is intentionally inert without xtquant."
        ) from exc
    return XtQuantTrader, StockAccount, xtconstant


def qmt_constant(xtconstant, name: str):
    if not hasattr(xtconstant, name):
        raise AttributeError(f"xtconstant.{name} is not available in this xtquant installation.")
    return getattr(xtconstant, name)


def create_trader(env: dict[str, str]):
    client_path = env.get("QMT_CLIENT_PATH", "").strip()
    account_id = env.get("QMT_ACCOUNT_ID", "").strip()
    if not client_path:
        raise RuntimeError("QMT_CLIENT_PATH is empty.")
    if not account_id:
        raise RuntimeError("QMT_ACCOUNT_ID is empty.")
    XtQuantTrader, StockAccount, xtconstant = import_qmt_modules()
    session_id = int(env.get("QMT_SESSION_ID", "1001"))
    trader = XtQuantTrader(client_path, session_id)
    account_type = env.get("QMT_ACCOUNT_TYPE", "STOCK").strip().upper()
    account = StockAccount(account_id, account_type)
    return trader, account, xtconstant


def connect_qmt(env: dict[str, str]):
    trader, account, xtconstant = create_trader(env)
    trader.start()
    connect_result = trader.connect()
    if connect_result not in {0, None}:
        raise RuntimeError(f"QMT connect failed: {connect_result}")
    subscribe_result = trader.subscribe(account)
    if subscribe_result not in {0, None}:
        raise RuntimeError(f"QMT subscribe failed: {subscribe_result}")
    return trader, account, xtconstant


def query_snapshot(trader, account) -> dict:
    asset = trader.query_stock_asset(account)
    positions = trader.query_stock_positions(account)
    return {
        "asset": repr(asset),
        "positions": [repr(p) for p in (positions or [])],
    }


def place_orders(trader, account, xtconstant, orders: pd.DataFrame, env: dict[str, str]) -> list[dict]:
    order_type_name = env.get("QMT_ORDER_TYPE", "LATEST_PRICE").strip().upper()
    price_type_name = env.get("QMT_PRICE_TYPE", "MARKET").strip().upper()
    order_type = qmt_constant(xtconstant, order_type_name)
    price_type = qmt_constant(xtconstant, price_type_name)
    strategy_name = env.get("QMT_STRATEGY_NAME", "llhh_csi1000_main")
    remark_prefix = env.get("QMT_ORDER_REMARK", "local_trade_ops")
    results: list[dict] = []

    for _, order in orders.iterrows():
        volume = int(order["shares"])
        if str(order["action"]) == "SELL":
            volume = -volume
        price = 0.0 if price_type_name == "MARKET" else float(order["estimated_price"])
        remark = f"{remark_prefix}:{order['client_order_id']}"
        try:
            order_id = trader.order_stock(account, str(order["qmt_code"]), order_type, volume, price_type, price, strategy_name, remark)
            status = "submitted"
            error = ""
        except Exception as exc:  # pragma: no cover - depends on live API behavior
            order_id = None
            status = "submit_error"
            error = repr(exc)
        results.append(
            {
                "client_order_id": str(order["client_order_id"]),
                "instrument": str(order["instrument"]),
                "qmt_code": str(order["qmt_code"]),
                "action": str(order["action"]),
                "shares": int(order["shares"]),
                "qmt_volume": volume,
                "price_type": price_type_name,
                "price": price,
                "status": status,
                "qmt_order_id": order_id,
                "error": error,
            }
        )
    return results


def write_result(submission_json: Path, result: dict) -> None:
    payload = {}
    if submission_json.exists():
        payload = json.loads(submission_json.read_text(encoding="utf-8"))
    payload["qmt_adapter_result"] = result
    payload["status"] = result.get("status", payload.get("status", "unknown"))
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    submission_json.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    env = {**load_env(), **os.environ}
    ticket_csv = Path(args.ticket_csv).expanduser().resolve()
    submission_json = Path(args.submission_json).expanduser().resolve()
    orders = load_orders(ticket_csv)

    result: dict = {
        "status": "dry_run" if args.dry_run else "prepared",
        "ticket_csv": str(ticket_csv),
        "n_orders": int(len(orders)),
        "orders": orders.to_dict(orient="records"),
        "account": env.get("QMT_ACCOUNT_ID", ""),
        "client_path": env.get("QMT_CLIENT_PATH", ""),
    }

    if args.dry_run:
        write_result(submission_json, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    trader, account, xtconstant = connect_qmt(env)
    result["snapshot_before"] = query_snapshot(trader, account)
    if args.query_only:
        result["status"] = "query_ok"
    else:
        result["submissions"] = place_orders(trader, account, xtconstant, orders, env)
        result["status"] = "submitted" if all(row["status"] == "submitted" for row in result["submissions"]) else "submit_partial_or_failed"
    write_result(submission_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
