from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from run_daily_fusion_signal import build_day_table, choose_trade_date
from run_prediction_fusion import load_pred


def load_current_positions(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["instrument", "shares"])
    df = pd.read_csv(path)
    required = {"instrument", "shares"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Position file missing columns: {sorted(missing)}")
    df = df.loc[:, ["instrument", "shares"]].copy()
    df["instrument"] = df["instrument"].astype(str)
    df["shares"] = df["shares"].astype(int)
    return df[df["shares"] > 0].reset_index(drop=True)


def load_prices_from_csv(path: Path, trade_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"instrument", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price CSV missing columns: {sorted(missing)}")
    if "datetime" in df.columns and trade_date is not None:
        dates = pd.to_datetime(df["datetime"])
        df = df.loc[dates == trade_date]
    cols = ["instrument", "close"] + (["factor"] if "factor" in df.columns else [])
    out = df.loc[:, cols].copy()
    out["instrument"] = out["instrument"].astype(str)
    out["close"] = out["close"].astype(float)
    if "factor" in out.columns:
        out["factor"] = out["factor"].astype(float)
        out["adjusted_close"] = out["close"]
        out.loc[out["factor"] > 0, "close"] = out.loc[out["factor"] > 0, "adjusted_close"] / out.loc[
            out["factor"] > 0, "factor"
        ]
    return out.dropna().drop_duplicates("instrument", keep="last")


def load_prices_from_qlib(provider_uri: str, instruments: list[str], trade_date: pd.Timestamp) -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    data = D.features(instruments, ["$close", "$factor"], start_time=trade_date, end_time=trade_date, freq="day")
    if data is None or data.empty:
        raise ValueError(f"No Qlib close data for {trade_date.date()}")
    data = data.rename(columns={"$close": "adjusted_close", "$factor": "factor"}).reset_index()
    data["close"] = data["adjusted_close"]
    valid_factor = data["factor"].notna() & (data["factor"] > 0)
    data.loc[valid_factor, "close"] = data.loc[valid_factor, "adjusted_close"] / data.loc[valid_factor, "factor"]
    return data.loc[:, ["instrument", "close", "adjusted_close", "factor"]].dropna(subset=["instrument", "close"])


def round_lot_shares(target_value: float, close: float, lot_size: int) -> int:
    if close <= 0:
        return 0
    lots = math.floor(target_value / (close * lot_size))
    return int(max(lots, 0) * lot_size)


def build_trade_plan(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    capital: float,
    cash: float | None,
    topk: int,
    lot_size: int,
    max_position_pct: float,
    max_lot_value_multiple: float | None,
    reserve_cash_pct: float,
    redistribute_cash: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    price_map = prices.set_index("instrument")["close"].to_dict()
    current = positions.set_index("instrument")["shares"].to_dict()
    current_value = sum(current.get(inst, 0) * price_map.get(inst, 0.0) for inst in current)
    available_capital = capital if cash is None else current_value + cash
    investable = available_capital * (1.0 - reserve_cash_pct)
    equal_target = investable / topk
    max_target = available_capital * max_position_pct
    target_value = min(equal_target, max_target)

    target_rows = []
    skipped = []
    for _, row in signal.head(topk).iterrows():
        inst = str(row["instrument"])
        close = price_map.get(inst)
        if close is None or pd.isna(close) or close <= 0:
            skipped.append({"instrument": inst, "reason": "missing_or_invalid_close"})
            continue
        lot_value = float(close) * lot_size
        if max_lot_value_multiple is not None and lot_value > target_value * max_lot_value_multiple:
            skipped.append(
                {
                    "instrument": inst,
                    "reason": "one_lot_value_too_high",
                    "close": close,
                    "lot_value": lot_value,
                    "limit": target_value * max_lot_value_multiple,
                }
            )
            continue
        shares = round_lot_shares(target_value, float(close), lot_size)
        if shares <= 0:
            skipped.append({"instrument": inst, "reason": "cannot_buy_one_lot", "close": close})
            continue
        value = shares * float(close)
        target_rows.append(
            {
                "rank": int(row["rank"]),
                "instrument": inst,
                "score": float(row["score"]),
                "close": float(close),
                "target_shares": shares,
                "target_value": value,
                "target_weight": value / available_capital if available_capital else 0.0,
            }
        )

    target = pd.DataFrame(target_rows)
    if target.empty:
        trade = pd.DataFrame()
        summary = {
            "available_capital": available_capital,
            "investable": investable,
            "planned_value": 0.0,
            "cash_left_before_cost": available_capital,
            "n_targets": 0,
            "n_skipped": len(skipped),
        }
        return target, trade, summary

    if redistribute_cash:
        max_position_value = available_capital * max_position_pct
        cash_left = available_capital - float(target["target_value"].sum())
        # Use leftover cash to add one lot at a time, prioritizing higher scores.
        # This keeps the plan executable while reducing idle cash from lot rounding.
        changed = True
        while changed:
            changed = False
            target = target.sort_values(["rank", "instrument"]).reset_index(drop=True)
            for idx, row in target.iterrows():
                close = float(row["close"])
                lot_value = close * lot_size
                next_value = float(row["target_value"]) + lot_value
                if lot_value <= cash_left and next_value <= max_position_value:
                    target.loc[idx, "target_shares"] = int(row["target_shares"]) + lot_size
                    target.loc[idx, "target_value"] = next_value
                    target.loc[idx, "target_weight"] = next_value / available_capital if available_capital else 0.0
                    cash_left -= lot_value
                    changed = True

    target_shares = target.set_index("instrument")["target_shares"].to_dict()
    all_inst = sorted(set(current) | set(target_shares))
    trade_rows = []
    for inst in all_inst:
        close = price_map.get(inst)
        if close is None or pd.isna(close) or close <= 0:
            continue
        cur = int(current.get(inst, 0))
        tgt = int(target_shares.get(inst, 0))
        delta = tgt - cur
        if delta == 0:
            action = "HOLD"
        elif delta > 0:
            action = "BUY"
        else:
            action = "SELL"
        trade_rows.append(
            {
                "instrument": inst,
                "action": action,
                "current_shares": cur,
                "target_shares": tgt,
                "delta_shares": delta,
                "close": float(close),
                "trade_value": abs(delta) * float(close),
                "target_value": tgt * float(close),
            }
        )

    trade = pd.DataFrame(trade_rows).sort_values(["action", "instrument"]).reset_index(drop=True)
    planned_value = float(target["target_value"].sum())
    max_target_weight = float(target["target_weight"].max()) if not target.empty else 0.0
    min_target_weight = float(target["target_weight"].min()) if not target.empty else 0.0
    summary = {
        "available_capital": float(available_capital),
        "investable": float(investable),
        "target_value_per_name_before_lot": float(target_value),
        "planned_value": planned_value,
        "cash_left_before_cost": float(available_capital - planned_value),
        "capital_utilization": planned_value / available_capital if available_capital else 0.0,
        "max_target_weight": max_target_weight,
        "min_target_weight": min_target_weight,
        "n_targets": int(len(target)),
        "n_skipped": int(len(skipped)),
        "skipped": skipped,
        "current_value_with_known_prices": float(current_value),
        "buy_value": float(trade.loc[trade["delta_shares"] > 0, "trade_value"].sum()) if not trade.empty else 0.0,
        "sell_value": float(trade.loc[trade["delta_shares"] < 0, "trade_value"].sum()) if not trade.empty else 0.0,
    }
    return target, trade, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an executable A-share lot-size trade plan from a pred.pkl.")
    parser.add_argument("--pred", required=True, help="Fused pred.pkl path.")
    parser.add_argument("--trade-date", default=None, help="Signal date, default latest prediction date.")
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--capital", type=float, required=True, help="Total account capital in RMB.")
    parser.add_argument("--cash", type=float, default=None, help="Optional current cash. If omitted, assume all cash.")
    parser.add_argument("--positions", default=None, help="Optional CSV with instrument,shares.")
    parser.add_argument("--price-csv", default=None, help="Optional CSV with instrument,close[,datetime].")
    parser.add_argument("--provider-uri", default="/root/autodl-tmp/llhh/reference_cn_data/cn_data")
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--max-position-pct", type=float, default=0.12)
    parser.add_argument(
        "--max-lot-value-multiple",
        type=float,
        default=None,
        help="Skip stocks whose one-lot value exceeds target value per name times this multiple.",
    )
    parser.add_argument("--reserve-cash-pct", type=float, default=0.02)
    parser.add_argument(
        "--no-redistribute-cash",
        action="store_true",
        help="Disable second-pass leftover cash allocation.",
    )
    parser.add_argument("--output-dir", default="/root/autodl-tmp/llhh/reports/trade_plans")
    args = parser.parse_args()

    pred = load_pred(Path(args.pred).expanduser().resolve())
    trade_date = choose_trade_date(pred, args.trade_date)
    signal = build_day_table(pred, trade_date, args.topk)
    positions = load_current_positions(Path(args.positions).expanduser().resolve() if args.positions else None)

    if args.price_csv:
        prices = load_prices_from_csv(Path(args.price_csv).expanduser().resolve(), trade_date=trade_date)
    else:
        instruments = sorted(set(signal["instrument"]) | set(positions["instrument"]))
        prices = load_prices_from_qlib(args.provider_uri, instruments, trade_date)

    target, trade, summary = build_trade_plan(
        signal,
        prices,
        positions,
        capital=args.capital,
        cash=args.cash,
        topk=args.topk,
        lot_size=args.lot_size,
        max_position_pct=args.max_position_pct,
        max_lot_value_multiple=args.max_lot_value_multiple,
        reserve_cash_pct=args.reserve_cash_pct,
        redistribute_cash=not args.no_redistribute_cash,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"trade_plan_{trade_date.date()}_top{args.topk}_cap{int(args.capital)}"
    target_path = output_dir / f"{prefix}_targets.csv"
    trade_path = output_dir / f"{prefix}_orders.csv"
    summary_path = output_dir / f"{prefix}_summary.md"

    target.to_csv(target_path, index=False)
    trade.to_csv(trade_path, index=False)

    skipped_text = "\n".join(f"- {item}" for item in summary.get("skipped", [])) or "none"
    report = f"""# Trade Plan

## Setup

- Prediction: `{Path(args.pred).expanduser().resolve()}`
- Trade date: `{trade_date.date()}`
- TopK: `{args.topk}`
- Capital: `{args.capital:,.2f}`
- Lot size: `{args.lot_size}`
- Max position pct: `{args.max_position_pct:.2%}`
- Max lot value multiple: `{args.max_lot_value_multiple}`
- Reserve cash pct: `{args.reserve_cash_pct:.2%}`
- Redistribute cash: `{not args.no_redistribute_cash}`

## Summary

| Metric | Value |
| --- | ---: |
| Available capital | {summary["available_capital"]:,.2f} |
| Investable | {summary["investable"]:,.2f} |
| Target value per name before lot | {summary["target_value_per_name_before_lot"]:,.2f} |
| Planned value | {summary["planned_value"]:,.2f} |
| Cash left before cost | {summary["cash_left_before_cost"]:,.2f} |
| Capital utilization | {summary["capital_utilization"]:.2%} |
| Max target weight | {summary["max_target_weight"]:.2%} |
| Min target weight | {summary["min_target_weight"]:.2%} |
| Buy value | {summary["buy_value"]:,.2f} |
| Sell value | {summary["sell_value"]:,.2f} |
| Number of targets | {summary["n_targets"]} |
| Number skipped | {summary["n_skipped"]} |

## Skipped

{skipped_text}

## Targets

{target.to_markdown(index=False) if not target.empty else "none"}

## Orders

{trade.to_markdown(index=False) if not trade.empty else "none"}
"""
    summary_path.write_text(report, encoding="utf-8")

    print("Trade plan complete.")
    print(f"- target: {target_path}")
    print(f"- orders: {trade_path}")
    print(f"- summary: {summary_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
