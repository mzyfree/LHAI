#!/usr/bin/env python3
"""Point-in-time auction hot-sector and naked-K leader backtest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
@dataclass(frozen=True)
class AuctionConfig:
    hot_sector_count: int = 3
    min_sector_size: int = 5
    min_sector_breadth: float = 0.55
    min_sector_median_gap: float = 0.005
    min_sector_top20_gap: float = 0.02
    min_amount_ma20: float = 30_000_000.0
    min_price: float = 3.0
    max_entry_gap: float = 0.05
    max_return20: float = 0.30
    max_holding_days: int = 5
    fixed_stop_loss: float = -1.0
    signal_low_buffer: float = 0.0
    daily_drop_threshold: float = -0.05
    min_leader_score: float = 0.0
    min_stock_sector_open_edge: float = float("-inf")
    min_pattern_count: int = 0
    buy_cost_rate: float = 0.0003
    sell_cost_rate: float = 0.0008
    min_cost: float = 5.0
    lot_size: int = 100
    buy_slippage_bps: float = 0.0
    sell_slippage_bps: float = 0.0
    min_market_open_breadth: float = 0.0
    min_market_prior_up_breadth: float = 0.0
    min_market_prior_above_ma20: float = 0.0
    min_market_prior_5d_median: float = -1.0
    allow_preemption: bool = False
    preemption_score_margin: float = 0.0
    preemption_min_new_score: float = 0.0
    preemption_reference: str = "current_candidate"
    preemption_max_incumbent_return: float = float("inf")
    soft_market_weak_threshold: float = -1.0
    soft_market_strong_threshold: float = -1.0
    soft_market_neutral_exposure: float = 1.0
    exit_old_position_on_weak_open: bool = False
    cooldown_after_loss_days: int = 0


@dataclass(frozen=True)
class LeaderWeights:
    open_strength: float = 0.40
    prior_strength: float = 0.25
    naked_structure: float = 0.20
    limit_memory: float = 0.15


def fee(value: float, rate: float, minimum: float) -> float:
    return max(value * rate, minimum)


def load_daily(start: str, end: str) -> pd.DataFrame:
    from qlib.data import D

    fields = ["$open", "$high", "$low", "$close", "$factor", "$volume"]
    rows = D.features(D.instruments("all"), fields, start_time=start, end_time=end, freq="day")
    rows = rows.rename(columns=dict(zip(fields, ["open", "high", "low", "close", "factor", "volume"]))).reset_index()
    rows["datetime"] = pd.to_datetime(rows["datetime"]).dt.normalize()
    for column in ["open", "high", "low", "close", "factor", "volume"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        rows[f"raw_{column}"] = rows[column] / rows["factor"]
    rows["amount"] = rows["raw_close"] * rows["volume"]
    return rows.sort_values(["instrument", "datetime"]).reset_index(drop=True)


def add_point_in_time_features(rows: pd.DataFrame, stock_meta: pd.DataFrame) -> pd.DataFrame:
    """Build 09:25 features; every non-opening feature is shifted by one session."""
    out = rows.copy().sort_values(["instrument", "datetime"]).reset_index(drop=True)
    meta = stock_meta[["instrument", "name", "industry", "list_date"]].drop_duplicates("instrument")
    out = out.merge(meta, on="instrument", how="left", validate="many_to_one")
    by_stock = out.groupby("instrument", group_keys=False)
    out["history_days"] = by_stock.cumcount()
    out["prior_close"] = by_stock["close"].shift(1)
    out["prior_raw_close"] = by_stock["raw_close"].shift(1)
    out["prior_low"] = by_stock["low"].shift(1)
    returns = {days: by_stock["close"].pct_change(days, fill_method=None) for days in (1, 5, 20)}
    for days, values in returns.items():
        out[f"prior_return_{days}d"] = values.groupby(out["instrument"]).shift(1)
    out["prior_ma5"] = by_stock["close"].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    out["prior_ma20"] = by_stock["close"].transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())
    # The prior day's breakout compares its close with highs ending two sessions ago.
    out["prior_high20"] = by_stock["high"].transform(lambda x: x.shift(2).rolling(20, min_periods=20).max())
    out["prior_low20"] = by_stock["low"].transform(lambda x: x.shift(2).rolling(20, min_periods=20).min())
    out["prior_amount_ma20"] = by_stock["amount"].transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())
    limit_like = returns[1].ge(out["instrument"].map(limit_threshold))
    out["prior_limit_count60"] = limit_like.groupby(out["instrument"]).transform(
        lambda x: x.shift(1).rolling(60, min_periods=20).sum()
    )
    prior_limit_1 = limit_like.groupby(out["instrument"]).shift(1).eq(True)
    prior_limit_2 = limit_like.groupby(out["instrument"]).shift(2).eq(True)
    out["prior_two_consecutive_limits"] = prior_limit_1 & prior_limit_2
    out["open_gap"] = out["raw_open"] / out["prior_raw_close"] - 1.0
    out["prior_ma20_distance"] = out["prior_close"] / out["prior_ma20"] - 1.0
    out["prior_range20"] = out["prior_high20"] / out["prior_low20"] - 1.0
    out["prior_breakout"] = (
        (out["prior_close"] >= out["prior_high20"] * 1.002)
        & (out["prior_range20"] <= 0.25)
    )
    out["prior_support"] = (
        out["prior_ma20_distance"].between(-0.02, 0.04)
        & out["prior_return_1d"].between(0.0, 0.06)
    )
    out["prior_weak_to_strong"] = (
        out["prior_return_1d"].between(-0.08, -0.005)
        & (out["prior_close"] >= out["prior_ma20"] * 0.98)
    )
    out["pattern_count"] = out[["prior_breakout", "prior_support", "prior_weak_to_strong"]].sum(axis=1)
    return out


def limit_threshold(instrument: str) -> float:
    code = str(instrument).upper()
    if code.startswith("BJ"):
        return 0.295
    if code.startswith(("SZ300", "SZ301", "SH688")):
        return 0.195
    return 0.095


def rank_auction_candidates(
    day: pd.DataFrame,
    config: AuctionConfig,
    weights: LeaderWeights = LeaderWeights(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = day[
        day["industry"].notna()
        & day["open_gap"].notna()
        & np.isfinite(day["raw_open"])
        & day["raw_open"].gt(0)
        & np.isfinite(day["prior_raw_close"])
        & day["prior_raw_close"].gt(0)
        & day["prior_amount_ma20"].ge(config.min_amount_ma20)
        & day["prior_raw_close"].ge(config.min_price)
        & day["history_days"].ge(120)
    ].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    market_open_breadth = float(valid["open_gap"].gt(0).mean())
    prior_return_1d = valid.get("prior_return_1d", pd.Series(np.nan, index=valid.index))
    market_prior_up_breadth = float(prior_return_1d.gt(0).mean())
    market_prior_above_ma20 = float(valid["prior_close"].ge(valid["prior_ma20"]).mean())
    market_prior_5d_median = float(valid["prior_return_5d"].median())
    if (
        market_open_breadth < config.min_market_open_breadth
        or market_prior_up_breadth < config.min_market_prior_up_breadth
        or market_prior_above_ma20 < config.min_market_prior_above_ma20
        or market_prior_5d_median < config.min_market_prior_5d_median
    ):
        return pd.DataFrame(), pd.DataFrame()
    sector = valid.groupby("industry", observed=True)["open_gap"].agg(
        sector_size="size",
        sector_breadth=lambda x: float((x > 0).mean()),
        sector_median_gap="median",
        sector_top20_gap=lambda x: float(x.nlargest(max(1, int(np.ceil(len(x) * 0.2)))).mean()),
    ).reset_index()
    sector = sector[
        sector["sector_size"].ge(config.min_sector_size)
        & sector["sector_breadth"].ge(config.min_sector_breadth)
        & sector["sector_median_gap"].ge(config.min_sector_median_gap)
        & sector["sector_top20_gap"].ge(config.min_sector_top20_gap)
    ].copy()
    for column in ["sector_breadth", "sector_median_gap", "sector_top20_gap"]:
        sector[f"{column}_rank"] = sector[column].rank(pct=True)
    sector["sector_score"] = (
        0.35 * sector["sector_breadth_rank"]
        + 0.45 * sector["sector_median_gap_rank"]
        + 0.20 * sector["sector_top20_gap_rank"]
    )
    hot = sector.nlargest(config.hot_sector_count, "sector_score")
    candidates = valid.merge(hot, on="industry", how="inner", validate="many_to_one")
    candidates = candidates[
        candidates["open_gap"].le(config.max_entry_gap)
        & candidates["open_gap"].lt(candidates["instrument"].map(limit_threshold))
        & candidates["prior_close"].ge(candidates["prior_ma20"])
        & candidates["prior_return_20d"].le(config.max_return20)
        & candidates["prior_limit_count60"].ge(1)
    ].copy()
    if candidates.empty:
        return hot, candidates
    candidates["stock_vs_sector_open"] = candidates["open_gap"] - candidates["sector_median_gap"]
    candidates["open_strength_rank"] = candidates.groupby("industry", observed=True)["stock_vs_sector_open"].rank(pct=True)
    candidates["prior_strength_rank"] = candidates.groupby("datetime")["prior_return_5d"].rank(pct=True)
    candidates["memory_rank"] = candidates.groupby("datetime")["prior_limit_count60"].rank(pct=True)
    candidates["leader_score"] = (
        weights.open_strength * candidates["open_strength_rank"]
        + weights.prior_strength * candidates["prior_strength_rank"]
        + weights.naked_structure * candidates["pattern_count"].clip(upper=2) / 2
        + weights.limit_memory * candidates["memory_rank"]
    )
    candidates = candidates[
        candidates["leader_score"].ge(config.min_leader_score)
        & candidates["stock_vs_sector_open"].ge(config.min_stock_sector_open_edge)
        & candidates["pattern_count"].ge(config.min_pattern_count)
    ].copy()
    candidates["market_open_breadth"] = market_open_breadth
    candidates["market_prior_up_breadth"] = market_prior_up_breadth
    candidates["market_prior_above_ma20"] = market_prior_above_ma20
    candidates["market_prior_5d_median"] = market_prior_5d_median
    candidates = candidates.sort_values(["leader_score", "open_gap", "instrument"], ascending=[False, False, True])
    return hot, candidates


def market_exposure(candidates: pd.DataFrame, config: AuctionConfig) -> float:
    """Return the opening target exposure using only 09:25/prior-session fields."""
    if candidates.empty or config.soft_market_weak_threshold < 0:
        return 1.0
    fields = ["market_open_breadth", "market_prior_up_breadth", "market_prior_above_ma20"]
    values = [float(candidates.iloc[0][field]) for field in fields]
    if min(values) < config.soft_market_weak_threshold:
        return 0.0
    if min(values) < config.soft_market_strong_threshold:
        return config.soft_market_neutral_exposure
    return 1.0


def replay(
    rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    topk: int,
    config: AuctionConfig,
    stock_snapshots: pd.DataFrame | None = None,
    index_snapshots: pd.DataFrame | None = None,
    weights: LeaderWeights = LeaderWeights(),
):
    calendar = sorted(rows.loc[rows["datetime"].between(start, end), "datetime"].unique())
    by_date = {date: frame.copy() for date, frame in rows.groupby("datetime", sort=False)}
    indexed = rows.set_index(["datetime", "instrument"]).sort_index()
    cash = 100_000.0
    positions: list[dict] = []
    trades: list[dict] = []
    selections: list[pd.DataFrame] = []
    nav_rows: list[dict] = []
    blocked_entry_dates: set[pd.Timestamp] = set()
    snap_index = None if stock_snapshots is None else stock_snapshots.set_index(["trade_date", "instrument"])
    index_context = {}
    if index_snapshots is not None:
        context = index_snapshots.copy()
        context["index_return_0935"] = context["price_0935"] / context["open"] - 1.0
        index_context = context.groupby("trade_date")["index_return_0935"].agg(
            index_median_return="median", index_positive_count=lambda x: int((x >= 0).sum())
        ).to_dict("index")
    for date in calendar:
        day = by_date[date]
        hot, candidates = rank_auction_candidates(day, config, weights)
        target_exposure = market_exposure(candidates, config)
        exited_at_open = False
        if config.exit_old_position_on_weak_open and target_exposure == 0.0:
            for position in list(positions):
                key = (date, position["instrument"])
                if pd.Timestamp(position["entry_date"]) >= pd.Timestamp(date) or key not in indexed.index:
                    continue
                row = indexed.loc[key]
                exit_price = float(row["raw_open"]) * (1.0 - config.sell_slippage_bps / 10_000.0)
                if not np.isfinite(exit_price) or exit_price <= 0:
                    continue
                gross = position["shares"] * exit_price
                sell_fee = fee(gross, config.sell_cost_rate, config.min_cost)
                cash += gross - sell_fee
                invested = position["entry_gross"] + position["buy_fee"]
                trades.append({**position, "exit_date": date, "exit_price": exit_price,
                               "exit_reason": "weak_market_open", "sell_fee": sell_fee,
                               "trade_return": (gross - sell_fee) / invested - 1})
                positions.remove(position)
                exited_at_open = True
        if not candidates.empty:
            # Confirmation may reject a planned leader, but must not silently replace it
            # with a lower-ranked name because that changes the strategy being tested.
            selected = candidates.head(topk).copy()
            selected["selected_rank"] = np.arange(1, len(selected) + 1)
            selections.append(selected)
            if config.allow_preemption and topk == 1 and positions:
                challenger = selected.iloc[0]
                incumbent = positions[0]
                incumbent_rows = candidates.loc[candidates["instrument"].eq(incumbent["instrument"])]
                if config.preemption_reference == "entry_score":
                    incumbent_score = float(incumbent["leader_score"])
                else:
                    incumbent_score = float(incumbent_rows.iloc[0]["leader_score"]) if not incumbent_rows.empty else 0.0
                score_edge = float(challenger["leader_score"]) - incumbent_score
                incumbent_key = (date, incumbent["instrument"])
                can_sell = pd.Timestamp(incumbent["entry_date"]) < pd.Timestamp(date) and incumbent_key in indexed.index
                incumbent_open_return = float("inf")
                if can_sell:
                    incumbent_open_return = float(indexed.loc[incumbent_key]["raw_open"]) / float(incumbent["entry_price"]) - 1.0
                if (
                    challenger["instrument"] != incumbent["instrument"]
                    and float(challenger["leader_score"]) >= config.preemption_min_new_score
                    and score_edge >= config.preemption_score_margin
                    and incumbent_open_return <= config.preemption_max_incumbent_return
                    and can_sell
                ):
                    incumbent_row = indexed.loc[incumbent_key]
                    exit_price = float(incumbent_row["raw_open"]) * (1.0 - config.sell_slippage_bps / 10_000.0)
                    if np.isfinite(exit_price) and exit_price > 0:
                        gross = incumbent["shares"] * exit_price
                        sell_fee = fee(gross, config.sell_cost_rate, config.min_cost)
                        cash += gross - sell_fee
                        invested = incumbent["entry_gross"] + incumbent["buy_fee"]
                        trades.append({
                            **incumbent,
                            "exit_date": date,
                            "exit_price": exit_price,
                            "exit_reason": "preempted_at_open",
                            "sell_fee": sell_fee,
                            "trade_return": (gross - sell_fee) / invested - 1,
                            "challenger": challenger["instrument"],
                            "challenger_score": float(challenger["leader_score"]),
                            "incumbent_score_today": incumbent_score,
                            "preemption_score_edge": score_edge,
                            "incumbent_open_return": incumbent_open_return,
                        })
                        positions.remove(incumbent)
            for candidate in selected.itertuples(index=False):
                if exited_at_open or target_exposure <= 0.0 or pd.Timestamp(date) in blocked_entry_dates:
                    continue
                if len(positions) >= topk or any(p["instrument"] == candidate.instrument for p in positions):
                    continue
                entry_price = candidate.raw_open
                if snap_index is not None:
                    snap_key = (pd.Timestamp(date).strftime("%Y-%m-%d"), candidate.instrument)
                    market = index_context.get(pd.Timestamp(date).strftime("%Y-%m-%d"))
                    if snap_key not in snap_index.index or market is None:
                        continue
                    snap = snap_index.loc[snap_key]
                    if isinstance(snap, pd.DataFrame):
                        snap = snap.iloc[0]
                    if market["index_median_return"] < -0.003 or market["index_positive_count"] < 1:
                        continue
                    if snap["price_0935"] < snap["open"] or snap["price_0935"] < snap["vwap_0935"]:
                        continue
                    if snap["price_0935"] / candidate.prior_raw_close - 1 > config.max_entry_gap:
                        continue
                    entry_price = float(snap["price_0935"])
                entry_price *= 1.0 + config.buy_slippage_bps / 10_000.0
                if not np.isfinite(entry_price) or entry_price <= 0:
                    continue
                slots = topk - len(positions)
                current_market_value = sum(p["shares"] * p["last_price"] for p in positions)
                current_nav = cash + current_market_value
                exposure_budget = max(0.0, current_nav * target_exposure - current_market_value)
                budget = min(cash / slots, exposure_budget / slots)
                shares = int(budget // (entry_price * config.lot_size)) * config.lot_size
                gross = shares * entry_price
                buy_fee = fee(gross, config.buy_cost_rate, config.min_cost) if shares else 0.0
                while shares and gross + buy_fee > cash:
                    shares -= config.lot_size
                    gross = shares * entry_price
                    buy_fee = fee(gross, config.buy_cost_rate, config.min_cost) if shares else 0.0
                if not shares:
                    continue
                cash -= gross + buy_fee
                positions.append({
                    "signal_date": date, "entry_date": date, "instrument": candidate.instrument,
                    "industry": candidate.industry, "leader_score": candidate.leader_score,
                    "shares": shares, "entry_price": entry_price, "entry_gross": gross,
                    "buy_fee": buy_fee, "signal_low": candidate.prior_low, "holding_days": 0,
                    "last_price": entry_price,
                })
        # Close decisions happen after the opening-entry phase. Proceeds from a
        # close exit are therefore available only from the next session onward.
        for position in list(positions):
            key = (date, position["instrument"])
            if key not in indexed.index:
                continue
            row = indexed.loc[key]
            if not np.isfinite(row["raw_close"]) or row["raw_close"] <= 0:
                continue
            position["last_price"] = float(row["raw_close"])
            if pd.Timestamp(position["entry_date"]) == pd.Timestamp(date):
                continue
            position["holding_days"] += 1
            reason = None
            trade_return_at_close = float(row["raw_close"]) / float(position["entry_price"]) - 1.0
            if config.fixed_stop_loss > -1.0 and trade_return_at_close <= config.fixed_stop_loss:
                reason = "fixed_stop_loss"
            elif row["close"] < position["signal_low"] * (1.0 - config.signal_low_buffer):
                reason = "signal_low_broken"
            elif row["close"] / row["prior_close"] - 1 <= config.daily_drop_threshold:
                reason = "daily_drop_threshold"
            elif row["close"] < row["prior_ma5"]:
                reason = "close_below_prior_ma5"
            elif position["holding_days"] >= config.max_holding_days:
                reason = "max_holding_days"
            if reason:
                exit_price = float(row["raw_close"]) * (1.0 - config.sell_slippage_bps / 10_000.0)
                gross = position["shares"] * exit_price
                sell_fee = fee(gross, config.sell_cost_rate, config.min_cost)
                cash += gross - sell_fee
                invested = position["entry_gross"] + position["buy_fee"]
                trades.append({**position, "exit_date": date, "exit_price": exit_price, "exit_reason": reason,
                               "sell_fee": sell_fee, "trade_return": (gross - sell_fee) / invested - 1})
                trade_return = (gross - sell_fee) / invested - 1
                if trade_return < 0 and config.cooldown_after_loss_days > 0:
                    date_index = calendar.index(date)
                    blocked_entry_dates.update(
                        pd.Timestamp(x)
                        for x in calendar[date_index + 1:date_index + 1 + config.cooldown_after_loss_days]
                    )
                positions.remove(position)
        market_value = 0.0
        for position in positions:
            key = (date, position["instrument"])
            price = indexed.loc[key, "raw_close"] if key in indexed.index else position["last_price"]
            if not np.isfinite(price) or price <= 0:
                price = position["last_price"]
            else:
                position["last_price"] = float(price)
            market_value += position["shares"] * float(price)
        nav_rows.append({"datetime": date, "nav": cash + market_value, "cash": cash, "positions": len(positions)})
    nav = pd.DataFrame(nav_rows)
    nav["daily_return"] = nav["nav"].pct_change().fillna(0.0)
    nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0
    summary = {
        "topk": topk, "days": len(nav), "completed_trades": len(trades),
        "cum_return": float(nav.iloc[-1]["nav"] / 100_000.0 - 1),
        "max_drawdown": float(nav["drawdown"].min()),
        "trade_win_rate": float(pd.Series([x["trade_return"] for x in trades]).gt(0).mean()) if trades else np.nan,
    }
    return summary, nav, pd.DataFrame(trades), pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def main() -> int:
    import qlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--stock-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-start", default="2025-01-01")
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--topks", default="1,3")
    parser.add_argument("--stock-snapshots", type=Path)
    parser.add_argument("--index-snapshots", type=Path)
    parser.add_argument(
        "--leader-weights", default="0.40,0.25,0.20,0.15",
        help="Comma-separated weights: opening, prior-5d, naked structure, limit memory.",
    )
    parser.add_argument("--min-market-open-breadth", type=float, default=0.0)
    parser.add_argument("--min-market-prior-up-breadth", type=float, default=0.0)
    parser.add_argument("--min-market-prior-above-ma20", type=float, default=0.0)
    parser.add_argument("--min-market-prior-5d-median", type=float, default=-1.0)
    parser.add_argument("--buy-slippage-bps", type=float, default=0.0)
    parser.add_argument("--sell-slippage-bps", type=float, default=0.0)
    parser.add_argument("--allow-preemption", action="store_true")
    parser.add_argument("--preemption-score-margin", type=float, default=0.0)
    parser.add_argument("--preemption-min-new-score", type=float, default=0.0)
    parser.add_argument("--preemption-reference", choices=["current_candidate", "entry_score"], default="current_candidate")
    parser.add_argument("--preemption-max-incumbent-return", type=float, default=float("inf"))
    parser.add_argument("--max-holding-days", type=int, default=5)
    parser.add_argument("--fixed-stop-loss", type=float, default=-1.0)
    parser.add_argument("--signal-low-buffer", type=float, default=0.0)
    parser.add_argument("--daily-drop-threshold", type=float, default=-0.05)
    parser.add_argument("--min-leader-score", type=float, default=0.0)
    parser.add_argument("--min-stock-sector-open-edge", type=float, default=float("-inf"))
    parser.add_argument("--min-pattern-count", type=int, default=0)
    parser.add_argument("--cooldown-after-loss-days", type=int, default=0)
    parser.add_argument("--soft-market-weak-threshold", type=float, default=-1.0)
    parser.add_argument("--soft-market-strong-threshold", type=float, default=-1.0)
    parser.add_argument("--soft-market-neutral-exposure", type=float, default=1.0)
    parser.add_argument("--exit-old-position-on-weak-open", action="store_true")
    args = parser.parse_args()
    qlib.init(provider_uri=args.provider_uri, region="cn")
    rows = add_point_in_time_features(load_daily(args.feature_start, args.end_date), pd.read_csv(args.stock_meta))
    stock_snapshots = pd.read_csv(args.stock_snapshots, dtype={"trade_date": str}) if args.stock_snapshots else None
    index_snapshots = pd.read_csv(args.index_snapshots, dtype={"trade_date": str}) if args.index_snapshots else None
    mode = "0935" if stock_snapshots is not None else "0930"
    weight_values = [float(value) for value in args.leader_weights.split(",")]
    if len(weight_values) != 4 or not np.isclose(sum(weight_values), 1.0):
        raise ValueError("leader weights must contain four values summing to 1")
    weights = LeaderWeights(*weight_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    config = AuctionConfig(
        min_market_open_breadth=args.min_market_open_breadth,
        min_market_prior_up_breadth=args.min_market_prior_up_breadth,
        min_market_prior_above_ma20=args.min_market_prior_above_ma20,
        min_market_prior_5d_median=args.min_market_prior_5d_median,
        buy_slippage_bps=args.buy_slippage_bps,
        sell_slippage_bps=args.sell_slippage_bps,
        allow_preemption=args.allow_preemption,
        preemption_score_margin=args.preemption_score_margin,
        preemption_min_new_score=args.preemption_min_new_score,
        preemption_reference=args.preemption_reference,
        preemption_max_incumbent_return=args.preemption_max_incumbent_return,
        max_holding_days=args.max_holding_days,
        fixed_stop_loss=args.fixed_stop_loss,
        signal_low_buffer=args.signal_low_buffer,
        daily_drop_threshold=args.daily_drop_threshold,
        min_leader_score=args.min_leader_score,
        min_stock_sector_open_edge=args.min_stock_sector_open_edge,
        min_pattern_count=args.min_pattern_count,
        cooldown_after_loss_days=args.cooldown_after_loss_days,
        soft_market_weak_threshold=args.soft_market_weak_threshold,
        soft_market_strong_threshold=args.soft_market_strong_threshold,
        soft_market_neutral_exposure=args.soft_market_neutral_exposure,
        exit_old_position_on_weak_open=args.exit_old_position_on_weak_open,
    )
    for topk in [int(x) for x in args.topks.split(",")]:
        summary, nav, trades, selections = replay(
            rows, pd.Timestamp(args.start_date), pd.Timestamp(args.end_date), topk, config,
            stock_snapshots, index_snapshots, weights,
        )
        summaries.append(summary)
        nav.to_csv(args.output_dir / f"auction_{mode}_top{topk}_nav.csv", index=False)
        trades.to_csv(args.output_dir / f"auction_{mode}_top{topk}_trades.csv", index=False)
        selections.to_parquet(args.output_dir / f"auction_{mode}_top{topk}_selections.parquet", index=False)
    result = pd.DataFrame(summaries)
    result.to_csv(args.output_dir / f"auction_{mode}_summary.csv", index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
