#!/usr/bin/env python3
"""Hierarchical DeepSeek replay with independent positions and a strict no-add rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_naked_k_qwen as llm
from backtest_naked_k_qwen import prompt, qwen
from backtest_auction_hot_sector import add_point_in_time_features, limit_threshold, load_daily
from backtest_naked_k_deepseek_intraday import (
    compact_bars, download_5m, fee, hierarchical_shortlist, hierarchy_universe,
    intraday_facts,
)


UPPER_WICK_RANGE_THRESHOLD = 0.04
UPPER_WICK_RATIO_THRESHOLD = 0.35


def dynamic_intraday_top3(
    candidates: list[dict],
    sector_limit: int = 5,
    stock_limit: int = 3,
) -> tuple[list[str], list[dict], list[dict]]:
    """Refresh intraday hotspots and Top3 using only facts visible at the decision time."""
    if not candidates:
        return [], [], []
    frame = pd.DataFrame([
        {
            "code": item["code"],
            "industry": item.get("industry") or "未知",
            "intradayReturn": float(item.get("intradayReturn") or 0),
            "aboveVwap": item.get("facts", {}).get("aboveVwap") is True,
            "dailySupport": item.get("facts", {}).get("dailySupport") is True,
            "threeLows": item.get("facts", {}).get("threeLowsNonDecreasing") is True,
            "volumeBreakout": item.get("facts", {}).get("volumeBreakout") is True,
        }
        for item in candidates
    ])
    sectors = frame.groupby("industry", as_index=False).agg(
        size=("code", "size"),
        breadth=("intradayReturn", lambda values: float((values > 0).mean())),
        medianReturn=("intradayReturn", "median"),
        aboveVwapRate=("aboveVwap", "mean"),
    )
    for field in ["breadth", "medianReturn", "aboveVwapRate"]:
        sectors[f"{field}Rank"] = sectors[field].rank(pct=True, method="average")
    sectors["score"] = (
        .40 * sectors.breadthRank
        + .35 * sectors.medianReturnRank
        + .25 * sectors.aboveVwapRateRank
    )
    sectors = sectors.sort_values(["score", "medianReturn", "size"], ascending=False)
    hot_sectors = sectors.head(sector_limit).industry.tolist()
    sector_medians = sectors.set_index("industry").medianReturn.to_dict()
    sector_scores = sectors.set_index("industry").score.to_dict()
    frame["strongerThanSector"] = frame.apply(
        lambda row: row.intradayReturn > float(sector_medians.get(row.industry, 0)), axis=1,
    )
    frame["evidenceScore"] = (
        frame.industry.isin(hot_sectors).astype(int)
        + frame.strongerThanSector.astype(int)
        + frame.aboveVwap.astype(int)
        + frame.threeLows.astype(int)
        + frame.volumeBreakout.astype(int)
        + frame.dailySupport.astype(int)
    )
    frame["sectorScore"] = frame.industry.map(sector_scores).fillna(0)
    selected = frame.sort_values(
        ["evidenceScore", "sectorScore", "intradayReturn", "code"],
        ascending=[False, False, False, True],
    ).head(stock_limit)
    return hot_sectors, selected.code.tolist(), sectors.to_dict("records")


def cached_qwen(prompt_text: str, key: str, cache_path: Path) -> tuple[dict, dict]:
    """Cache deterministic temperature-zero decisions by their complete prompt."""
    digest = hashlib.sha256(prompt_text.encode()).hexdigest()
    with sqlite3.connect(cache_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS responses (hash TEXT PRIMARY KEY, decision TEXT NOT NULL, usage TEXT NOT NULL)")
        row = conn.execute("SELECT decision, usage FROM responses WHERE hash = ?", (digest,)).fetchone()
        if row:
            return json.loads(row[0]), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cache_hit": True}
    decision, usage = qwen(prompt_text, key, "deepseek-chat")
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO responses(hash, decision, usage) VALUES (?, ?, ?)",
            (digest, json.dumps(decision, ensure_ascii=False), json.dumps(usage, ensure_ascii=False)),
        )
    return decision, usage


def is_large_upper_wick(frame: pd.DataFrame, execution_time: pd.Timestamp) -> tuple[bool, dict]:
    """Evaluate the completed signal bar immediately before execution."""
    completed = frame[frame.trade_time < execution_time]
    if completed.empty:
        return False, {}
    bar = completed.iloc[-1]
    span = float(bar.high - bar.low)
    amplitude = span / float(bar.open) if float(bar.open) > 0 else 0.0
    upper_ratio = (float(bar.high) - max(float(bar.open), float(bar.close))) / span if span > 0 else 0.0
    facts = {
        "signalBarTime": bar.trade_time.isoformat(),
        "amplitude": amplitude,
        "upperWickRatio": upper_ratio,
    }
    return amplitude >= UPPER_WICK_RANGE_THRESHOLD and upper_ratio >= UPPER_WICK_RATIO_THRESHOLD, facts


def mark_position(position: dict, bars: dict[str, pd.DataFrame], now: pd.Timestamp) -> float:
    frame = bars.get(position["stock"])
    seen = frame[frame.trade_time <= now] if frame is not None else pd.DataFrame()
    price = float(seen.iloc[-1].close) if not seen.empty else position["entry_price"]
    return position["shares"] * price


def position_context(position: dict, bars: dict[str, pd.DataFrame], now: pd.Timestamp, hot_sectors: list[str]) -> dict:
    frame = bars.get(position["stock"])
    seen = frame[frame.trade_time <= now] if frame is not None else pd.DataFrame()
    mark = float(seen.iloc[-1].close) if not seen.empty else position["entry_price"]
    volume = float(seen.vol.sum()) if not seen.empty else 0
    vwap = float(seen.amount.sum() / volume) if volume > 0 else None
    tail = seen.tail(3)
    return {
        "stock": position["stock"], "industry": position["industry"],
        "entryDate": str(position["entry_date"].date()), "entryPrice": position["entry_price"],
        "return": mark / position["entry_price"] - 1,
        "facts": {
            "vwap": vwap, "aboveVwap": None if vwap is None else mark > vwap,
            "support": position.get("signal_support"),
            "supportBroken": None if position.get("signal_support") is None else mark < position["signal_support"],
            "consecutiveWeakBarsWithRisingVolume": bool(len(tail) == 3 and (tail.close < tail.open).all() and tail.vol.is_monotonic_increasing),
            "hotspotRetreated": position["industry"] not in hot_sectors,
        },
        "bars": compact_bars(frame, now) if frame is not None else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--stock-meta", required=True, type=Path)
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hierarchy-audit", type=Path)
    parser.add_argument("--single-position", action="store_true")
    parser.add_argument("--full-position", action="store_true")
    parser.add_argument("--min-buy-signal-time", default="09:35")
    parser.add_argument("--min-buy-confidence", type=float, default=0.0)
    parser.add_argument("--min-buy-open-breadth", type=float, default=0.30)
    parser.add_argument("--intraday-breadth", type=Path, help="Optional anchored intraday breadth CSV for recovery-gate studies")
    parser.add_argument(
        "--intraday-recovery-threshold", type=float,
        help="Breadth required to reopen entries after the opening gate failed; defaults to min-buy-open-breadth",
    )
    parser.add_argument("--recovery-confirm-bars", type=int, default=1)
    parser.add_argument("--execute-at-signal-price", action="store_true")
    parser.add_argument("--allow-same-signal-rotation", action="store_true")
    parser.add_argument(
        "--recovery-market-vwap", type=Path,
        help="Optional 5-minute market proxy CSV; recovery requires close above cumulative VWAP",
    )
    parser.add_argument("--decision-cache", type=Path)
    parser.add_argument(
        "--baseline-audit",
        type=Path,
        help="Reuse baseline decisions while sellable and locked position states still match.",
    )
    parser.add_argument("--block-same-day-rebuy", action="store_true")
    parser.add_argument("--block-after-two-limit-ups", action="store_true")
    parser.add_argument("--require-structure-or-volume", action="store_true")
    parser.add_argument(
        "--dynamic-top3",
        action="store_true",
        help="Refresh hotspots and Top3 every five minutes from the frozen 09:25 Top20 pool.",
    )
    parser.add_argument("--dynamic-pool-size", type=int, default=20)
    parser.add_argument("--min-prior-limit-count60", type=int)
    parser.add_argument("--max-prior-return20", type=float)
    parser.add_argument(
        "--direct-open-breadth", type=float,
        help="When flat, buy the frozen Top1 at 09:30 if 09:25 breadth reaches this threshold.",
    )
    parser.add_argument(
        "--block-first-bar-vwap-only-sell",
        action="store_true",
        help="Reject a T+1 opening-bar sell when below-VWAP is the only negative position fact.",
    )
    args = parser.parse_args()
    if args.full_position and not args.single_position:
        parser.error("--full-position requires --single-position")
    recovery_threshold = (
        args.intraday_recovery_threshold
        if args.intraday_recovery_threshold is not None
        else args.min_buy_open_breadth
    )
    if not 0 <= recovery_threshold <= 1:
        parser.error("--intraday-recovery-threshold must be between 0 and 1")
    if args.recovery_confirm_bars < 1:
        parser.error("--recovery-confirm-bars must be positive")
    if args.direct_open_breadth is not None and not 0 <= args.direct_open_breadth <= 1:
        parser.error("--direct-open-breadth must be between 0 and 1")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    baseline_audit = {}
    if args.baseline_audit:
        baseline_audit = {
            row["dateTime"]: row for row in json.loads(args.baseline_audit.read_text())
        }
    intraday_breadth = {}
    if args.intraday_breadth:
        breadth_frame = pd.read_csv(args.intraday_breadth)
        value_column = "adjusted" if "adjusted" in breadth_frame else "breadth"
        intraday_breadth = {
            (str(row.date), str(row.time)): float(getattr(row, value_column))
            for row in breadth_frame.itertuples() if pd.notna(getattr(row, value_column))
        }

    def breadth_at(date: pd.Timestamp, now: pd.Timestamp, opening: float) -> float:
        if not intraday_breadth or opening >= args.min_buy_open_breadth:
            return opening
        return intraday_breadth.get((str(date.date()), now.strftime("%H:%M")), opening)

    market_above_vwap = {}
    if args.recovery_market_vwap:
        market = pd.read_csv(args.recovery_market_vwap)
        market["trade_time"] = pd.to_datetime(market["trade_time"])
        market = market.sort_values("trade_time")
        market["cum_vwap"] = market.groupby(market.trade_time.dt.date, sort=False).apply(
            lambda frame: frame.amount.cumsum() / frame.vol.cumsum()
        ).reset_index(level=0, drop=True)
        market_above_vwap = {
            (str(row.trade_time.date()), row.trade_time.strftime("%H:%M")): bool(row.close > row.cum_vwap)
            for row in market.itertuples()
        }

    import baostock as bs
    import qlib
    qlib.init(provider_uri=args.provider_uri, region="cn")
    llm.ENDPOINT = "https://api.deepseek.com/chat/completions"
    meta = pd.read_parquet(args.stock_meta) if args.stock_meta.suffix == ".parquet" else pd.read_csv(args.stock_meta)
    if "instrument" not in meta and "ts_code" in meta:
        meta["instrument"] = meta.ts_code.str[-2:] + meta.ts_code.str[:6]
    load_end = (pd.Timestamp(args.start_date) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    rows = add_point_in_time_features(load_daily("2025-01-01", load_end), meta)
    prior_two_limit_lookup = rows.set_index(["datetime", "instrument"])["prior_two_consecutive_limits"].to_dict()
    dates = sorted(pd.Timestamp(x) for x in rows.loc[rows.datetime.ge(args.start_date), "datetime"].unique())[:args.sessions]
    by_date = {pd.Timestamp(k): v for k, v in rows.groupby("datetime")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "minutes"; cache.mkdir(exist_ok=True)

    daily, hierarchy_logs = {}, []
    frozen_hierarchy = None
    if args.hierarchy_audit:
        frozen_hierarchy = json.loads(args.hierarchy_audit.read_text())
    for date in dates:
        if frozen_hierarchy is None:
            market, sectors, stocks, logs = hierarchical_shortlist(by_date[date], date, key)
        else:
            date_text = str(date.date())
            logs = [
                {key: value for key, value in row.items() if key != "date"}
                for row in frozen_hierarchy if row.get("date") == date_text
            ]
            market, sectors, tradable = hierarchy_universe(by_date[date])
            if args.min_prior_limit_count60 is not None:
                tradable = tradable[tradable.prior_limit_count60.ge(args.min_prior_limit_count60)]
            if args.max_prior_return20 is not None:
                tradable = tradable[tradable.prior_return_20d.le(args.max_prior_return20)]
            shortlist_phase = "top20" if args.dynamic_top3 else "top3"
            final_result = next(x["result"] for x in logs if x["phase"] == shortlist_phase) if any(x["phase"] == shortlist_phase for x in logs) else {"codes": []}
            final_codes = final_result.get("codes", [])[:args.dynamic_pool_size if args.dynamic_top3 else 3]
            columns = [
                "instrument", "name", "industry", "raw_open", "open_gap", "sectorEdge",
                "prior_return_1d", "prior_return_5d", "prior_return_20d",
                "prior_amount_ma20", "prior_limit_count60", "pattern_count",
                "prior_ma20_distance", "prior_range20", "prior_raw_close",
                "prior_low_raw", "prior_ma5_raw",
            ]
            stocks = tradable[tradable.instrument.isin(final_codes)][columns].replace({np.nan: None}).to_dict("records")
            rank = {code: index for index, code in enumerate(final_codes)}
            stocks.sort(key=lambda item: rank.get(item["instrument"], len(rank)))
        market_result = next(x["result"] for x in logs if x["phase"] == "market")
        sector_result = next(x["result"] for x in logs if x["phase"] == "sectors")
        daily[date] = {
            "market": market, "sectors": sectors, "stocks": stocks, "bars": {},
            "maxExposure": float(market_result.get("maxExposure", 0)),
            "hotSectors": sector_result.get("sectors", []),
        }
        hierarchy_logs.extend({"date": str(date.date()), **x} for x in logs)

    all_codes = sorted({x["instrument"] for d in daily.values() for x in d["stocks"]})
    login = None
    requested_codes = {
        date: ([x["instrument"] for x in daily[date]["stocks"]] if args.dynamic_top3 else all_codes)
        for date in dates
    }
    missing_cache = [
        (date, code) for date in dates for code in requested_codes[date]
        if not (cache / f"{code}_{date:%Y%m%d}.csv").exists()
    ]
    if missing_cache and os.environ.get("NAKED_K_SKIP_BAOSTOCK") != "1":
        for attempt in range(3):
            login = bs.login()
            if login.error_code == "0":
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
        if login is None or login.error_code != "0":
            # download_5m falls back to TuShare when BaoStock is unavailable.
            login = None
    failures = []
    try:
        for date, info in daily.items():
            for code in requested_codes[date]:
                try: info["bars"][code] = download_5m(bs if login is not None else None, code, date, cache)
                except Exception as exc: failures.append({"date": str(date.date()), "stock": code, "error": str(exc)})
            info["stocks"] = [x for x in info["stocks"] if x["instrument"] in info["bars"]]
            print(date.date(), "candidates", len(info["stocks"]), "tracked", len(info["bars"]), flush=True)
    finally:
        if login is not None and login.error_code == "0":
            bs.logout()

    cash, positions, pending = 100_000.0, [], {"sell": [], "buy": None}
    audit, trades, nav, execution_rejections = [], [], [], []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in hierarchy_logs:
        for field in usage_total:
            usage_total[field] += int(row["usage"].get(field, 0))

    def execute_orders(orders, date, now, info, recovery_ready, sold_today, price_field):
        nonlocal cash, positions
        for code in orders["sell"]:
            pos = next((x for x in positions if x["stock"] == code and date > x["entry_date"]), None)
            frame = info["bars"].get(code)
            current = frame[frame.trade_time.eq(now)] if frame is not None else pd.DataFrame()
            if pos and not current.empty:
                px = float(current.iloc[0][price_field])
                gross = pos["shares"] * px
                cost = fee(gross, sell=True)
                cash += gross - cost
                trades.append({**pos, "exit_date": date, "exit_time": now, "exit_price": px, "sell_fee": cost, "return": (gross-cost)/(pos["gross"]+pos["buy_fee"])-1})
                positions.remove(pos)
                sold_today.add(code)

        buy = orders["buy"]
        if not buy:
            return
        code = buy.get("stock")
        held = {x["stock"] for x in positions}
        item = next((x for x in info["stocks"] if x["instrument"] == code), None)
        frame = info["bars"].get(code)
        current = frame[frame.trade_time.eq(now)] if frame is not None else pd.DataFrame()
        rebuy_allowed = not args.block_same_day_rebuy or code not in sold_today
        if not (item and code not in held and rebuy_allowed and not current.empty and recovery_ready):
            return
        px = float(current.iloc[0][price_field])
        prior_close = float(item["prior_raw_close"])
        wick_cutoff = now + pd.Timedelta(nanoseconds=1) if price_field == "close" else now
        upper_wick_blocked, upper_wick_facts = is_large_upper_wick(frame, wick_cutoff)
        if px / prior_close - 1 >= limit_threshold(code):
            execution_rejections.append({"dateTime": now.isoformat(), "stock": code, "reason": "limit_up_at_execution"})
            return
        if upper_wick_blocked:
            execution_rejections.append({"dateTime": now.isoformat(), "stock": code, "reason": "large_upper_wick", **upper_wick_facts})
            return
        equity = cash + sum(mark_position(x, info["bars"], now) for x in positions)
        if args.full_position:
            budget = cash
        else:
            invested = equity - cash
            total_capacity = max(0., equity * min(1., max(0., info["maxExposure"])) - invested)
            tier = float(buy.get("targetExposure", 0))
            tier = tier if tier in (.3, .6, 1.0) else 0
            budget = min(cash, total_capacity, equity * tier)
        shares = int(budget/(px*100))*100
        gross = shares*px
        cost = fee(gross) if shares else 0
        while shares and gross+cost > cash:
            shares -= 100
            gross = shares*px
            cost = fee(gross) if shares else 0
        if shares:
            cash -= gross+cost
            positions.append({"stock":code,"industry":item["industry"],"entry_date":date,"entry_time":now,"entry_price":px,"signal_support":item.get("prior_low_raw"),"shares":shares,"gross":gross,"buy_fee":cost})

    for date in dates:
        info = daily[date]
        missing_held = [x["stock"] for x in positions if x["stock"] not in info["bars"]]
        held_login = None
        if missing_held:
            held_login = bs.login()
            if held_login.error_code != "0":
                held_login = None
            for code in missing_held:
                try:
                    info["bars"][code] = download_5m(bs if held_login is not None else None, code, date, cache)
                except Exception as exc:
                    failures.append({"date": str(date.date()), "stock": code, "error": f"held_position:{exc}"})
            if held_login is not None:
                bs.logout()
        timeline = sorted(set().union(*(set(x.trade_time) for x in info["bars"].values())))
        if not timeline:
            value = cash + sum(x["shares"] * x["entry_price"] for x in positions)
            nav.append({"date":str(date.date()),"nav":value,"cash":cash,"positions":"|".join(x["stock"] for x in positions)})
            continue
        pending = {"sell": [], "buy": None}
        sold_today: set[str] = set()
        recovery_streak = 0
        if (
            args.direct_open_breadth is not None
            and not positions
            and float(info["market"].get("openBreadth", 0.0)) >= args.direct_open_breadth
            and info["stocks"]
        ):
            item = info["stocks"][0]
            code = item["instrument"]
            frame = info["bars"].get(code)
            first_bar = frame.iloc[0] if frame is not None and not frame.empty else None
            if first_bar is not None:
                px = float(first_bar.open)
                prior_close = float(item["prior_raw_close"])
                if px / prior_close - 1 < limit_threshold(code):
                    shares = int(cash / (px * 100)) * 100
                    gross = shares * px
                    cost = fee(gross) if shares else 0
                    while shares and gross + cost > cash:
                        shares -= 100
                        gross = shares * px
                        cost = fee(gross) if shares else 0
                    if shares:
                        cash -= gross + cost
                        entry_time = pd.Timestamp(f"{date.date()} 09:30:00")
                        positions.append({
                            "stock": code, "industry": item["industry"], "entry_date": date,
                            "entry_time": entry_time, "entry_price": px,
                            "signal_support": item.get("prior_low_raw"), "shares": shares,
                            "gross": gross, "buy_fee": cost,
                        })
                        audit.append({
                            "dateTime": entry_time.isoformat(), "positions": [], "locked": [],
                            "decision": {
                                "sellCodes": [],
                                "buy": {"stock": code, "targetExposure": 1.0},
                                "confidence": None,
                                "reasons": [
                                    f"09:25开盘广度{float(info['market']['openBreadth']):.4f}达到"
                                    f"{args.direct_open_breadth:.4f}，按消融规则09:30直接买入固定Top1"
                                ],
                                "risks": ["直接开盘买入尚无盘中量价确认"],
                            },
                            "executionBlock": None,
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            "strategyOverride": "direct_open_top1",
                        })
        for bar_index, now in enumerate(timeline):
            current_breadth = breadth_at(date, now, float(info["market"]["openBreadth"]))
            recovery_day = bool(intraday_breadth) and float(info["market"]["openBreadth"]) < args.min_buy_open_breadth
            active_breadth_threshold = recovery_threshold if recovery_day else args.min_buy_open_breadth
            if recovery_day:
                recovery_streak = recovery_streak + 1 if current_breadth >= active_breadth_threshold else 0
            market_vwap_ok = (
                not args.recovery_market_vwap
                or market_above_vwap.get((str(date.date()), now.strftime("%H:%M")), False)
            )
            recovery_ready = (
                current_breadth >= active_breadth_threshold
                if not recovery_day
                else (recovery_streak >= args.recovery_confirm_bars and market_vwap_ok)
            )
            # Execute previous bar's orders at this bar open.
            if not args.execute_at_signal_price:
                execute_orders(pending, date, now, info, recovery_ready, sold_today, "open")
            pending = {"sell": [], "buy": None}

            held_codes = {x["stock"] for x in positions}
            locked = [{"stock":x["stock"],"entryDate":str(x["entry_date"].date()),"sellLocked":True} for x in positions if date == x["entry_date"]]
            candidates = []
            can_rotate = args.allow_same_signal_rotation and any(date > x["entry_date"] for x in positions)
            suppress_new_candidates = args.single_position and positions and not can_rotate
            candidate_source = info["stocks"] if args.dynamic_top3 else ([] if suppress_new_candidates else info["stocks"])
            for item in candidate_source:
                code = item["instrument"]
                if code in held_codes or (args.block_same_day_rebuy and code in sold_today): continue  # Strict no-add/cooldown rule.
                visible = info["bars"][code][info["bars"][code].trade_time <= now]
                if visible.empty: continue
                last = visible.iloc[-1]
                base = {"code":code,"name":item["name"],"industry":item["industry"],"openGap":item["open_gap"],"prior5d":item["prior_return_5d"],"prior20d":item["prior_return_20d"],"patternCount":item["pattern_count"],"limitMemory60":item["prior_limit_count60"],"intradayReturn":float(last.close/visible.iloc[0].open-1)}
                base["facts"] = intraday_facts(info["bars"][code], now, float(item["sectorEdge"]), base)
                base["evidenceChecklist"] = {
                    "hotSector": base.get("industry") in info["hotSectors"],
                    **{
                        field: base["facts"].get(field) is True
                        for field in ["strongerThanSector", "aboveVwap", "threeLowsNonDecreasing", "volumeBreakout", "dailySupport"]
                    },
                }
                base["evidenceCount"] = sum(base["evidenceChecklist"].values())
                base["bars"] = compact_bars(info["bars"][code], now)
                candidates.append(base)
            active_hot_sectors = info["hotSectors"]
            dynamic_sector_facts = []
            if args.dynamic_top3 and candidates:
                active_hot_sectors, active_codes, dynamic_sector_facts = dynamic_intraday_top3(candidates)
                active_code_set = set(active_codes)
                sector_medians = {
                    row["industry"]: float(row["medianReturn"])
                    for row in dynamic_sector_facts
                }
                candidates = [item for item in candidates if item["code"] in active_code_set]
                for item in candidates:
                    item["facts"]["strongerThanSector"] = bool(
                        item["intradayReturn"] > sector_medians.get(item["industry"], 0)
                    )
                if suppress_new_candidates:
                    candidates = []
            eligible = [position_context(x, info["bars"], now, active_hot_sectors) for x in positions if date > x["entry_date"]]
            equity = cash + sum(mark_position(x, info["bars"], now) for x in positions)
            exposure = 0 if equity <= 0 else (equity-cash)/equity
            if args.full_position:
                # Share-lot rounding leaves immaterial cash dust. Do not let that
                # alter otherwise identical LLM prompts in paired experiments.
                exposure = 1.0 if positions else 0.0
                cash_ratio = 0.0 if positions else 1.0
            else:
                cash_ratio = cash/equity if equity else 0
            # A decision on the final bar has no same-day execution bar. Never carry it
            # into tomorrow, where both the candidate set and market regime have changed.
            if bar_index == len(timeline) - 1 and not args.execute_at_signal_price:
                continue
            # On an opening-gate failure day, a flat account cannot buy until
            # the recovery threshold is met. Skip an LLM call that the hard
            # execution gate would reject regardless of the model response.
            if recovery_day and not positions and not recovery_ready:
                continue
            # Locked T+1 holdings need no sell analysis. If no fresh candidate remains,
            # there is no executable choice and therefore no reason to spend an LLM call.
            if not eligible and not candidates:
                continue
            max_total_exposure = 1.0 if args.full_position else info["maxExposure"]
            payload = {"dateTime":now.isoformat(),"marketAtOpen":info["market"],"maxTotalExposure":max_total_exposure,"currentExposure":exposure,"cashRatio":cash_ratio,"hotSectors":active_hot_sectors,"sellablePositions":eligible,"lockedTodayPositions":locked,"candidates":candidates}
            if args.dynamic_top3:
                payload["dynamicSectorFacts"] = dynamic_sector_facts
            entry_recovery_context = recovery_day and bool(candidates)
            if entry_recovery_context:
                payload["intradayBreadth"] = current_breadth
            schema = '{"sellCodes":["可卖持仓代码"],"buy":{"stock":"代码或null","targetExposure":0.3|0.6|1.0}|null,"confidence":0到1,"reasons":["..."],"risks":["..."]}'
            if args.single_position and args.allow_same_signal_rotation:
                mode_rule = "账户最多持有一只股票；T+1持仓可与新候选比较，只有同时卖出全部旧仓才允许换入一只新股票。新仓使用全部可用现金。"
            else:
                mode_rule = "账户最多持有一只股票；有持仓时禁止寻找新股票。新仓使用全部可用现金。" if args.single_position else "允许管理多只独立持仓。"
            breadth_rule = (
                "intradayBreadth低于配置门槛时禁止买入；marketAtOpen仅是09:25快照，盘中恢复后以intradayBreadth为准。"
                if entry_recovery_context else "市场开盘广度低于30%禁止买入。"
            )
            task = f"""管理A股组合。{mode_rule}只对sellablePositions判断卖出，lockedTodayPositions买入当日不可卖且无需评价。持仓卖出只引用position.facts。空余总仓位可从candidates买入最多一只；已持有代码不会出现在候选，坚决禁止加仓。代码已在candidates[].evidenceChecklist和evidenceCount中完成六项证据计数，必须直接引用，禁止自行重算；evidenceCount少于4禁止买入。达到4项仅代表具备买入资格，不等于必须买；若达到4项却wait，必须写明额外拒绝理由，禁止声称未达到门槛。targetExposure只能是30%、60%、100%。{breadth_rule}决策在下一根5分钟K线开盘执行。"""
            prompt_text = prompt(task, schema, payload)
            baseline_row = baseline_audit.get(now.isoformat())
            current_sellable = sorted(x["stock"] for x in eligible)
            current_locked = sorted(x["stock"] for x in locked)
            baseline_sellable = sorted(x["stock"] for x in baseline_row.get("positions", [])) if baseline_row else []
            baseline_locked = sorted(x["stock"] for x in baseline_row.get("locked", [])) if baseline_row else []
            if baseline_row and current_sellable == baseline_sellable and current_locked == baseline_locked:
                decision = baseline_row["decision"]
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "baseline_reused": True}
            else:
                decision, usage = (
                    cached_qwen(prompt_text, key, args.decision_cache)
                    if args.decision_cache else qwen(prompt_text, key, "deepseek-chat")
                )
            for field in usage_total: usage_total[field] += int(usage.get(field,0))
            sellable_codes = {x["stock"] for x in positions if date > x["entry_date"]}
            pending["sell"] = [x for x in decision.get("sellCodes",[]) if x in sellable_codes]
            if args.block_first_bar_vwap_only_sell and pending["sell"]:
                guarded = []
                for code in pending["sell"]:
                    context = next((x for x in eligible if x["stock"] == code), None)
                    facts = context.get("facts", {}) if context else {}
                    vwap_only = bool(
                        context
                        and len(context.get("bars", [])) == 1
                        and facts.get("aboveVwap") is False
                        and facts.get("supportBroken") is not True
                        and facts.get("consecutiveWeakBarsWithRisingVolume") is not True
                        and facts.get("hotspotRetreated") is not True
                    )
                    if vwap_only:
                        execution_rejections.append({
                            "dateTime": now.isoformat(),
                            "stock": code,
                            "reason": "first_bar_vwap_only_sell",
                        })
                    else:
                        guarded.append(code)
                pending["sell"] = guarded
            buy = decision.get("buy")
            allowed = {x["instrument"] for x in info["stocks"]} - held_codes
            if args.block_same_day_rebuy:
                allowed -= sold_today
            block = None
            if buy and buy.get("stock") in allowed:
                selected = next((x for x in candidates if x["code"] == buy.get("stock")), None)
                facts = selected.get("facts", {}) if selected else {}
                evidence_count = int(bool(selected and selected["industry"] in active_hot_sectors)) + sum(
                    int(facts.get(field) is True) for field in (
                        "strongerThanSector", "aboveVwap", "threeLowsNonDecreasing",
                        "volumeBreakout", "dailySupport",
                    )
                )
                if not recovery_ready:
                    block = "market_recovery_not_confirmed"
                elif now.strftime("%H:%M") < args.min_buy_signal_time:
                    block = "buy_signal_before_min_time"
                elif float(decision.get("confidence", 0) or 0) < args.min_buy_confidence:
                    block = "buy_confidence_below_threshold"
                elif evidence_count < 4:
                    block = "buy_evidence_below_four"
                elif args.require_structure_or_volume and not (
                    facts.get("threeLowsNonDecreasing") is True
                    or facts.get("volumeBreakout") is True
                ):
                    block = "buy_structure_or_volume_not_confirmed"
                elif facts.get("threeLowsNonDecreasing") is False and facts.get("volumeBreakout") is False:
                    # A nominal four-factor score can still be a failed intraday
                    # breakout when neither price structure nor volume confirms it.
                    block = "buy_structure_and_volume_both_false"
                elif args.block_after_two_limit_ups and prior_two_limit_lookup.get((date, buy.get("stock")), False):
                    block = "buy_after_two_consecutive_limit_ups"
                elif args.single_position and positions and not {x["stock"] for x in positions}.issubset(set(pending["sell"])):
                    block = "rotation_requires_all_positions_sold"
                else: pending["buy"] = buy
            audit.append({"dateTime":now.isoformat(),"positions":eligible,"locked":locked,"decision":decision,"executionBlock":block,"usage":usage})
            if args.execute_at_signal_price:
                execute_orders(pending, date, now, info, recovery_ready, sold_today, "close")
                pending = {"sell": [], "buy": None}

        close_time = timeline[-1]
        value = cash + sum(mark_position(x, info["bars"], close_time) for x in positions)
        nav.append({"date":str(date.date()),"nav":value,"cash":cash,"positions":"|".join(x["stock"] for x in positions)})
        (args.output_dir/"audit.partial.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str))
        pd.DataFrame(nav).to_csv(args.output_dir/"nav.partial.csv",index=False)
        print(date.date(),"nav",round(value,2),"positions",nav[-1]["positions"],flush=True)

    nf=pd.DataFrame(nav); curve=pd.concat([pd.Series([100000.]),nf.nav],ignore_index=True)
    summary={"start":nav[0]["date"],"end":nav[-1]["date"],"sessions":len(dates),"calls":len(audit),"cumReturn":float(nf.nav.iloc[-1]/100000-1),"maxDrawdown":float((curve/curve.cummax()-1).min()),"completedTrades":len(trades),"usage":usage_total,"openPositions":positions,"minuteFailures":failures,"singlePosition":args.single_position,"fullPosition":args.full_position,"executeAtSignalPrice":args.execute_at_signal_price,"allowSameSignalRotation":args.allow_same_signal_rotation,"candidateRefresh":{"dynamicTop3":args.dynamic_top3,"openingPoolSize":args.dynamic_pool_size if args.dynamic_top3 else 3},"openingCandidateFilters":{"minPriorLimitCount60":args.min_prior_limit_count60,"maxPriorReturn20":args.max_prior_return20},"directOpen":{"enabled":args.direct_open_breadth is not None,"breadthThreshold":args.direct_open_breadth,"selection":"frozen_top1","execution":"09:30_first_bar_open"},"executionRejections":len(execution_rejections),"sellFilters":{"blockFirstBarVwapOnly":args.block_first_bar_vwap_only_sell},"intradayBreadth":str(args.intraday_breadth) if args.intraday_breadth else None,"intradayRecoveryThreshold":recovery_threshold if args.intraday_breadth else None,"recoveryConfirmBars":args.recovery_confirm_bars,"recoveryMarketVwap":str(args.recovery_market_vwap) if args.recovery_market_vwap else None,"upperWickFilter":{"amplitude":UPPER_WICK_RANGE_THRESHOLD,"upperWickRatio":UPPER_WICK_RATIO_THRESHOLD},"buyFilters":{"minSignalTime":args.min_buy_signal_time,"minConfidence":args.min_buy_confidence,"minOpenBreadth":args.min_buy_open_breadth,"minEvidenceCount":4,"blockStructureAndVolumeBothFalse":True,"requireStructureOrVolume":args.require_structure_or_volume,"blockAfterTwoLimitUps":args.block_after_two_limit_ups,"blockSameDayRebuy":args.block_same_day_rebuy}}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)); (args.output_dir/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str)); (args.output_dir/"hierarchy_audit.json").write_text(json.dumps(hierarchy_logs,ensure_ascii=False,indent=2,default=str)); (args.output_dir/"execution_rejections.json").write_text(json.dumps(execution_rejections,ensure_ascii=False,indent=2,default=str)); nf.to_csv(args.output_dir/"nav.csv",index=False); pd.DataFrame(trades).to_csv(args.output_dir/"trades.csv",index=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str)); return 0


if __name__ == "__main__": raise SystemExit(main())
