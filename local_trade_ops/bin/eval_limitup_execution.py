from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_HOME = Path(__file__).resolve().parents[2]
PAPER_BIN = PROJECT_HOME / "paper_trading_system" / "bin"
if str(PAPER_BIN) not in sys.path:
    sys.path.insert(0, str(PAPER_BIN))

from eval_smallcap_ensemble_grid import daily_zscore  # noqa: E402
from eval_smallcap_models import filter_signal, load_market_data, load_pred, max_drawdown, safe_price  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate live execution when T+1 open limit-up buy orders cannot fill.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--benchmark", default="SH000852")
    parser.add_argument("--pred-a", required=True)
    parser.add_argument("--pred-b", required=True)
    parser.add_argument("--pred-c", required=True)
    parser.add_argument("--weights", default="3,1,1")
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--topk", type=int, default=7)
    parser.add_argument("--n-drop", type=int, default=1)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--reserve-cash-pct", type=float, default=0.02)
    parser.add_argument("--max-position-pct", type=float, default=0.12)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--buy-cost-rate", type=float, default=0.0003)
    parser.add_argument("--sell-cost-rate", type=float, default=0.0008)
    parser.add_argument("--min-cost", type=float, default=5.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--scan-topk", default="7,20,30,50,150")
    parser.add_argument(
        "--max-buy-gap",
        default="0.05",
        help="Comma-separated thresholds for gap_skip/gap_replenish modes. Example: 0.03,0.04,0.05 means skip buys opening above +3%%/+4%%/+5%%.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--daily-output", required=True)
    return parser.parse_args()


def fuse_three(pred_a: pd.DataFrame, pred_b: pd.DataFrame, pred_c: pd.DataFrame, weights: tuple[float, float, float]) -> pd.DataFrame:
    merged = (
        pred_a.rename(columns={"score": "score_a"})
        .join(pred_b.rename(columns={"score": "score_b"}), how="inner")
        .join(pred_c.rename(columns={"score": "score_c"}), how="inner")
    )
    wa, wb, wc = weights
    denom = wa + wb + wc
    score = (
        wa / denom * daily_zscore(merged["score_a"])
        + wb / denom * daily_zscore(merged["score_b"])
        + wc / denom * daily_zscore(merged["score_c"])
    )
    return score.to_frame("score").sort_index()


def limit_rate(inst: str) -> float:
    code = inst[2:] if inst[:2] in {"SH", "SZ"} else inst
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def open_gap(close_row: pd.Series, open_row: pd.Series, inst: str) -> float:
    close_price = safe_price(close_row, inst)
    open_price = safe_price(open_row, inst)
    if close_price <= 0 or open_price <= 0:
        return math.inf
    return open_price / close_price - 1.0


def is_open_limit_up(close_row: pd.Series, open_row: pd.Series, inst: str) -> bool:
    gap = open_gap(close_row, open_row, inst)
    if not math.isfinite(gap):
        return False
    return gap >= limit_rate(inst) * 0.999


def calc_cost(value: float, rate: float, min_cost: float) -> float:
    if value <= 0:
        return 0.0
    return max(value * rate, min_cost)


def round_lot_shares(target_value: float, price: float, lot_size: int) -> int:
    if target_value <= 0 or price <= 0:
        return 0
    return int(math.floor(target_value / (price * lot_size)) * lot_size)


def can_buy(mode: str, close_row: pd.Series, open_row: pd.Series, inst: str, max_buy_gap: float) -> tuple[bool, str]:
    gap = open_gap(close_row, open_row, inst)
    if mode == "ideal":
        return True, "ideal"
    if is_open_limit_up(close_row, open_row, inst):
        return False, "open_limit_up"
    if mode in {"gap_skip", "gap_replenish"} and (not math.isfinite(gap) or gap > max_buy_gap):
        return False, "gap_too_high"
    return True, "tradable"


def run_mode(
    *,
    mode: str,
    scan_topk: int,
    max_buy_gap: float,
    pred: pd.DataFrame,
    dates: pd.DatetimeIndex,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    amount: pd.DataFrame,
    bench_open: pd.Series,
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    cash = float(args.capital)
    positions: dict[str, int] = {}
    total_cost = 0.0
    daily_rows = []
    rejected_rows = []

    for i, signal_date in enumerate(dates[:-1]):
        execution_date = dates[i + 1]
        day = pred.xs(signal_date, level="datetime").reset_index().sort_values("score", ascending=False).reset_index(drop=True)
        day["rank"] = day.index + 1
        start_i = max(0, i - args.lookback + 1)
        day = filter_signal(day, close_px.loc[signal_date], amount.iloc[start_i : i + 1], args)
        day["rank"] = day.index + 1
        ranked = day["instrument"].astype(str).tolist()
        rank_map = dict(zip(ranked, day["rank"], strict=False))

        close_row = close_px.loc[signal_date]
        open_row = open_px.loc[execution_date]
        equity = cash + sum(shares * safe_price(close_row, inst) for inst, shares in positions.items())
        target_value = min(equity * (1.0 - args.reserve_cash_pct) / args.topk, equity * args.max_position_pct)

        sell = sorted(
            [inst for inst in positions if rank_map.get(inst, 10**9) > args.topk],
            key=lambda inst: rank_map.get(inst, 10**9),
            reverse=True,
        )[: args.n_drop]
        kept = [inst for inst in positions if inst not in set(sell)]
        buy_slots = max(args.topk - len(kept), 0)
        if positions and sell:
            buy_slots = min(buy_slots, args.n_drop)

        buy_candidates = ranked if mode in {"replenish", "gap_replenish"} else ranked[: args.topk]
        if mode in {"replenish", "gap_replenish"}:
            buy_candidates = ranked[:scan_topk]

        buy: list[tuple[str, int]] = []
        rejected = []
        reserved = 0.0
        available_cash = cash + sum(positions.get(inst, 0) * safe_price(close_row, inst) for inst in sell)
        excluded = set(kept) | set(sell)
        for inst in buy_candidates:
            if len(buy) >= buy_slots:
                break
            if inst in excluded or any(item[0] == inst for item in buy):
                continue
            tradable, reason = can_buy(mode, close_row, open_row, inst, max_buy_gap)
            if not tradable:
                rejected.append((inst, reason))
                continue
            estimate_price = safe_price(close_row, inst)
            shares = round_lot_shares(target_value, estimate_price, args.lot_size)
            if shares <= 0 and estimate_price * args.lot_size <= equity * args.max_position_pct:
                shares = args.lot_size
            if shares <= 0:
                rejected.append((inst, "cannot_buy_one_lot"))
                continue
            estimate_value = shares * estimate_price
            if estimate_value + reserved > available_cash:
                rejected.append((inst, "insufficient_cash"))
                continue
            buy.append((inst, shares))
            reserved += estimate_value

        for inst, reason in rejected:
            rejected_rows.append(
                {
                    "mode": mode,
                    "scan_topk": scan_topk,
                    "max_buy_gap": max_buy_gap,
                    "signal_date": signal_date.date(),
                    "execution_date": execution_date.date(),
                    "instrument": inst,
                    "rank": int(rank_map.get(inst, 0)),
                    "reason": reason,
                    "open_gap_pct": open_gap(close_row, open_row, inst) * 100,
                }
            )

        for inst in sell:
            shares = positions.get(inst, 0)
            price = safe_price(open_row, inst)
            if shares <= 0 or price <= 0:
                continue
            value = shares * price
            fee = calc_cost(value, args.sell_cost_rate, args.min_cost)
            cash += value - fee
            total_cost += fee
            positions.pop(inst, None)

        filled_buys = 0
        for inst, shares in buy:
            price = safe_price(open_row, inst)
            while shares > 0 and price > 0:
                value = shares * price
                fee = calc_cost(value, args.buy_cost_rate, args.min_cost)
                if value + fee <= cash:
                    break
                shares -= args.lot_size
            if shares <= 0 or price <= 0:
                continue
            value = shares * price
            fee = calc_cost(value, args.buy_cost_rate, args.min_cost)
            cash -= value + fee
            total_cost += fee
            positions[inst] = positions.get(inst, 0) + shares
            filled_buys += 1

        position_value = sum(shares * safe_price(open_row, inst) for inst, shares in positions.items())
        nav = cash + position_value
        daily_rows.append(
            {
                "mode": mode,
                "scan_topk": scan_topk,
                "max_buy_gap": max_buy_gap,
                "datetime": execution_date,
                "nav": nav,
                "cash": cash,
                "position_value": position_value,
                "position_ratio": position_value / nav if nav > 0 else np.nan,
                "holdings": len(positions),
                "n_sell": len(sell),
                "n_buy_signal": buy_slots,
                "n_buy_filled": filled_buys,
                "n_buy_rejected": len(rejected),
                "total_cost": total_cost,
            }
        )

    daily = pd.DataFrame(daily_rows)
    nav = daily["nav"]
    bench = bench_open.reindex(dates[1:]).dropna()
    benchmark_return = float(bench.iloc[-1] / bench.iloc[0] - 1.0) if len(bench) else np.nan
    result = {
        "mode": mode,
        "scan_topk": scan_topk,
        "max_buy_gap": max_buy_gap,
        "capital": args.capital,
        "last_nav": float(nav.iloc[-1]),
        "cum_return": float(nav.iloc[-1] / args.capital - 1.0),
        "benchmark_return": benchmark_return,
        "excess_return": float(nav.iloc[-1] / args.capital - 1.0 - benchmark_return),
        "max_drawdown": max_drawdown(nav),
        "avg_position_ratio": float(daily["position_ratio"].mean()),
        "min_position_ratio": float(daily["position_ratio"].min()),
        "final_holdings": int(daily.iloc[-1]["holdings"]),
        "total_cost": float(total_cost),
        "total_rejected": int(daily["n_buy_rejected"].sum()),
        "total_filled_buys": int(daily["n_buy_filled"].sum()),
    }
    rejected_df = pd.DataFrame(rejected_rows)
    if not rejected_df.empty:
        result["limit_rejected"] = int((rejected_df["reason"] == "open_limit_up").sum())
        result["gap_rejected"] = int((rejected_df["reason"] == "gap_too_high").sum())
    else:
        result["limit_rejected"] = 0
        result["gap_rejected"] = 0
    return result, daily


def main() -> None:
    args = parse_args()
    weights = tuple(float(x.strip()) for x in args.weights.split(",") if x.strip())
    if len(weights) != 3:
        raise SystemExit("--weights must contain exactly 3 values")
    pred = fuse_three(load_pred(Path(args.pred_a)), load_pred(Path(args.pred_b)), load_pred(Path(args.pred_c)), weights)  # type: ignore[arg-type]
    dates = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    end_date = args.end_date or str(dates[-1].date())
    dates = dates[(dates >= pd.Timestamp(args.start_date)) & (dates <= pd.Timestamp(end_date))]
    instruments = sorted(pred.index.get_level_values("instrument").unique())
    open_px, close_px, amount, bench_open = load_market_data(args.provider_uri, instruments, str(dates[0].date()), str(dates[-1].date()), args.benchmark)

    rows = []
    daily_frames = []
    scan_values = [int(x) for x in args.scan_topk.split(",") if x.strip()]
    gap_values = [float(x) for x in str(args.max_buy_gap).split(",") if x.strip()]
    scenarios: list[tuple[str, int, float]] = [("ideal", args.topk, gap_values[0]), ("limit_skip", args.topk, gap_values[0])]
    scenarios += [("gap_skip", args.topk, gap) for gap in gap_values]
    scenarios += [("replenish", scan, gap_values[0]) for scan in scan_values if scan >= args.topk]
    scenarios += [("gap_replenish", scan, gap) for scan in scan_values if scan >= args.topk for gap in gap_values]

    seen = set()
    for mode, scan_topk, max_buy_gap in scenarios:
        key = (mode, scan_topk, max_buy_gap)
        if key in seen:
            continue
        seen.add(key)
        result, daily = run_mode(
            mode=mode,
            scan_topk=scan_topk,
            max_buy_gap=max_buy_gap,
            pred=pred,
            dates=dates,
            open_px=open_px,
            close_px=close_px,
            amount=amount,
            bench_open=bench_open,
            args=args,
        )
        rows.append(result)
        daily_frames.append(daily)

    summary = pd.DataFrame(rows).sort_values(["cum_return", "avg_position_ratio"], ascending=False)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    for path, frame in [(args.output, summary), (args.daily_output, daily_all)]:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)

    print("===== limit-up execution summary =====")
    pct_cols = ["cum_return", "benchmark_return", "excess_return", "max_drawdown", "avg_position_ratio", "min_position_ratio"]
    show = summary.copy()
    for col in pct_cols:
        show[col] = show[col] * 100
    print(show.to_string(index=False))
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.daily_output}")


if __name__ == "__main__":
    main()
