#!/usr/bin/env python3
"""Strict point-in-time DeepSeek replay using frozen daily facts and 5-minute bars."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_naked_k_qwen as llm
from backtest_naked_k_qwen import prompt, qwen, valid_day
from backtest_auction_hot_sector import add_point_in_time_features, limit_threshold, load_daily


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT.parent / "reports" / "naked_k_deepseek_intraday_3d"


def ts_code(instrument: str) -> str:
    value = str(instrument).upper()
    return f"{value[2:]}.{value[:2]}"


def baostock_code(instrument: str) -> str:
    value = str(instrument).upper()
    return f"{value[:2].lower()}.{value[2:]}"


def baostock_frame(result) -> pd.DataFrame:
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(result.error_msg)
    return pd.DataFrame(rows, columns=result.fields)


def download_5m(bs, instrument: str, date: pd.Timestamp, cache: Path) -> pd.DataFrame:
    target = cache / f"{instrument}_{date:%Y%m%d}.csv"
    if target.exists():
        frame = pd.read_csv(target)
    else:
        result = bs.query_history_k_data_plus(
            baostock_code(instrument), "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=f"{date:%Y-%m-%d}", end_date=f"{date:%Y-%m-%d}",
            frequency="5", adjustflag="3",
        )
        frame = baostock_frame(result)
        if frame.empty:
            raise RuntimeError(f"no 5m data: {instrument} {date.date()}")
        frame.to_csv(target, index=False)
    if "trade_time" not in frame:
        frame["trade_time"] = pd.to_datetime(frame["time"].astype(str), format="%Y%m%d%H%M%S%f", errors="coerce")
    else:
        frame["trade_time"] = pd.to_datetime(frame["trade_time"])
    if "vol" not in frame and "volume" in frame:
        frame["vol"] = frame["volume"]
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("trade_time").reset_index(drop=True)


def compact_bars(frame: pd.DataFrame, now: pd.Timestamp, limit: int = 12) -> list[dict]:
    seen = frame[frame.trade_time <= now].tail(limit)
    return [
        {"t": x.trade_time.strftime("%H:%M"), "o": round(x.open, 3), "h": round(x.high, 3),
         "l": round(x.low, 3), "c": round(x.close, 3), "v": round(x.vol, 1)}
        for x in seen.itertuples()
    ]


def intraday_facts(frame: pd.DataFrame, now: pd.Timestamp, sector_edge: float, item: dict) -> dict:
    seen = frame[frame.trade_time <= now]
    last = seen.iloc[-1]
    amount = float(seen.amount.sum())
    volume = float(seen.vol.sum())
    vwap = amount / volume if amount > 0 and volume > 0 else None
    lows_rising = None if len(seen) < 3 else bool(seen.low.tail(3).is_monotonic_increasing)
    breakout = None
    volume_ratio = None
    if len(seen) >= 4:
        prior = seen.iloc[:-1]
        prior_volume = float(prior.vol.tail(3).mean())
        volume_ratio = float(last.vol / prior_volume) if prior_volume > 0 else None
        breakout = bool(last.close > prior.high.max() and volume_ratio is not None and volume_ratio >= 1.2)
    daily_support = sum([
        float(item.get("prior5d") or 0) > 0,
        int(item.get("patternCount") or 0) > 0,
        float(item.get("limitMemory60") or 0) >= 1,
    ]) >= 2
    return {
        "strongerThanSector": bool(sector_edge > 0),
        "aboveVwap": None if vwap is None else bool(last.close > vwap),
        "vwap": None if vwap is None else round(vwap, 4),
        "lastClose": float(last.close),
        "threeLowsNonDecreasing": lows_rising,
        "volumeRatioVsPrior3Bars": None if volume_ratio is None else round(volume_ratio, 3),
        "volumeBreakout": breakout,
        "dailySupport": bool(daily_support),
    }


def fee(value: float, sell: bool = False) -> float:
    return max(5.0, value * (0.0008 if sell else 0.0003))


def broad_contexts(day: pd.DataFrame, count: int) -> tuple[dict, list[dict], list[dict]]:
    valid = valid_day(day)
    market = {
        "stockCount": len(valid), "openBreadth": float(valid.open_gap.gt(0).mean()),
        "medianOpenGap": float(valid.open_gap.median()),
        "priorUpBreadth": float(valid.prior_return_1d.gt(0).mean()),
        "priorAboveMa20": float(valid.prior_close.ge(valid.prior_ma20).mean()),
    }
    sectors = valid.groupby("industry", observed=True).open_gap.agg(
        size="size", breadth=lambda x: float((x > 0).mean()), medianGap="median",
        top20Gap=lambda x: float(x.nlargest(max(1, int(np.ceil(len(x) * .2)))).mean()),
    ).reset_index()
    sectors = sectors[sectors["size"] >= 5].copy()
    for col in ["breadth", "medianGap", "top20Gap"]:
        sectors[col + "Rank"] = sectors[col].rank(pct=True)
    sectors["factScore"] = .35 * sectors.breadthRank + .45 * sectors.medianGapRank + .2 * sectors.top20GapRank
    sector_rows = sectors.nlargest(15, "factScore")[["industry", "size", "breadth", "medianGap", "top20Gap"]].to_dict("records")
    pool = valid.merge(sectors[["industry", "medianGap", "factScore"]], on="industry", how="inner")
    pool = pool[
        pool.open_gap.le(.05) & pool.open_gap.lt(pool.instrument.map(limit_threshold))
        & pool.prior_return_20d.le(.50)
    ].copy()
    pool["sectorEdge"] = pool.open_gap - pool.medianGap
    pool["broadScore"] = .50 * pool.sectorEdge.rank(pct=True) + .20 * pool.factScore + .15 * pool.prior_return_5d.rank(pct=True) + .15 * pool.pattern_count.clip(upper=2) / 2
    pool = pool.nlargest(count, "broadScore")
    pool["prior_low_raw"] = pool.prior_low / pool.factor
    pool["prior_ma5_raw"] = pool.prior_ma5 / pool.factor
    cols = ["instrument", "name", "industry", "raw_open", "open_gap", "sectorEdge", "prior_return_1d", "prior_return_5d", "prior_return_20d", "prior_amount_ma20", "prior_limit_count60", "pattern_count", "prior_raw_close", "prior_low_raw", "prior_ma5_raw"]
    return market, sector_rows, pool[cols].replace({np.nan: None}).to_dict("records")


def hierarchy_universe(day: pd.DataFrame) -> tuple[dict, list[dict], pd.DataFrame]:
    valid = valid_day(day)
    market = {
        "stockCount": len(valid), "openBreadth": float(valid.open_gap.gt(0).mean()),
        "medianOpenGap": float(valid.open_gap.median()),
        "priorUpBreadth": float(valid.prior_return_1d.gt(0).mean()),
        "priorAboveMa20": float(valid.prior_close.ge(valid.prior_ma20).mean()),
        "strongOpenCount": int(valid.open_gap.ge(.03).sum()),
        "weakOpenCount": int(valid.open_gap.le(-.03).sum()),
    }
    sectors = valid.groupby("industry", observed=True).open_gap.agg(
        size="size", breadth=lambda x: float((x > 0).mean()), medianGap="median",
        top20Gap=lambda x: float(x.nlargest(max(1, int(np.ceil(len(x) * .2)))).mean()),
    ).reset_index()
    sectors = sectors[sectors["size"] >= 5].sort_values("medianGap", ascending=False)
    sector_rows = sectors.replace({np.nan: None}).to_dict("records")
    tradable = valid[
        valid.open_gap.le(.05)
        & valid.open_gap.lt(valid.instrument.map(limit_threshold))
    ].copy().merge(sectors[["industry", "medianGap"]], on="industry", how="inner")
    tradable["sectorEdge"] = tradable.open_gap - tradable.medianGap
    tradable["prior_low_raw"] = tradable.prior_low / tradable.factor
    tradable["prior_ma5_raw"] = tradable.prior_ma5 / tradable.factor
    return market, sector_rows, tradable


def hierarchical_shortlist(day: pd.DataFrame, date: pd.Timestamp, key: str) -> tuple[dict, list[dict], list[dict], list[dict]]:
    market, sectors, tradable = hierarchy_universe(day)
    logs = []
    market_decision, usage = qwen(prompt(
        "判断今日09:25市场状态和可接受的新仓风险。",
        '{"decision":"risk_on|cautious|risk_off","maxExposure":0到1,"confidence":0到1,"reasons":["..."],"risks":["..."]}',
        {"date": str(date.date()), "market": market}), key, "deepseek-chat")
    logs.append({"phase": "market", "result": market_decision, "usage": usage})
    sector_decision, usage = qwen(prompt(
        "比较全部行业聚合事实，选择0到5个具备真实联动的热点板块。",
        '{"decision":"accept|none","sectors":["最多5个"],"confidence":0到1,"reasons":["..."],"risks":["..."]}',
        {"date": str(date.date()), "marketDecision": market_decision, "allSectors": sectors}), key, "deepseek-chat")
    logs.append({"phase": "sectors", "result": sector_decision, "usage": usage})
    selected_sectors = sector_decision.get("sectors", [])[:5] if sector_decision.get("decision") == "accept" else []
    pool = tradable[tradable.industry.isin(selected_sectors)].copy()
    compact_cols = ["instrument", "name", "industry", "raw_open", "open_gap", "sectorEdge", "prior_return_1d", "prior_return_5d", "prior_return_20d", "prior_amount_ma20", "prior_limit_count60", "pattern_count", "prior_ma20_distance", "prior_range20"]
    compact = pool[compact_cols].replace({np.nan: None}).to_dict("records")
    if not compact:
        return market, sectors, [], logs
    screen_payload = {"date": str(date.date()), "marketDecision": market_decision, "sectorDecision": sector_decision, "stocks": compact}
    try:
        screen, usage = qwen(prompt(
            "从热点板块全部可交易股票中筛选0到20只观察股。不得选择输入以外代码；本层仅返回代码，不逐股解释。",
            '{"decision":"select|skip","codes":["最多20个代码"]}',
            screen_payload), key, "deepseek-chat")
    except RuntimeError as exc:
        if "Content Exists Risk" not in str(exc):
            raise
        # Some company abbreviations trip provider moderation. Names are redundant
        # here because this phase selects codes from numeric/industry facts only.
        safe_stocks = [{k: v for k, v in row.items() if k != "name"} for row in compact]
        screen, usage = qwen(prompt(
            "从热点板块全部可交易股票中筛选0到20只观察股。不得选择输入以外代码；本层仅返回代码，不逐股解释。",
            '{"decision":"select|skip","codes":["最多20个代码"]}',
            {**screen_payload, "stocks": safe_stocks}), key, "deepseek-chat")
        usage = {**usage, "content_filter_fallback": True}
    logs.append({"phase": "top20", "result": screen, "usage": usage})
    allowed = set(pool.instrument)
    codes = [x for x in screen.get("codes", []) if x in allowed][:20]
    details = pool[pool.instrument.isin(codes)][compact_cols + ["prior_raw_close", "prior_low_raw", "prior_ma5_raw"]].replace({np.nan: None}).to_dict("records")
    if not details:
        return market, sectors, [], logs
    final, usage = qwen(prompt(
        "综合市场、热点和裸K事实，从观察股精排0到3只盘中跟踪股。允许全部放弃；理由和风险各最多3条，每条不超过40字。",
        '{"decision":"select|skip","codes":["最多3个代码"],"confidence":0到1,"reasons":["最多3条"],"risks":["最多3条"]}',
        {"date": str(date.date()), "marketDecision": market_decision, "sectorDecision": sector_decision, "top20": details}), key, "deepseek-chat")
    logs.append({"phase": "top3", "result": final, "usage": usage})
    detail_by_code = {x["instrument"]: x for x in details}
    top3 = [detail_by_code[x] for x in final.get("codes", []) if x in detail_by_code][:3]
    return market, sectors, top3, logs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--stock-meta", required=True, type=Path)
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    import qlib
    import baostock as bs
    qlib.init(provider_uri=args.provider_uri, region="cn")
    llm.ENDPOINT = "https://api.deepseek.com/chat/completions"
    meta = pd.read_parquet(args.stock_meta) if args.stock_meta.suffix == ".parquet" else pd.read_csv(args.stock_meta)
    if "instrument" not in meta and "ts_code" in meta:
        meta["instrument"] = meta["ts_code"].str[-2:] + meta["ts_code"].str[:6]
    load_end = (pd.Timestamp(args.start_date) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    rows = add_point_in_time_features(load_daily("2025-01-01", load_end), meta)
    dates = sorted(pd.Timestamp(x) for x in rows.loc[rows.datetime.ge(args.start_date), "datetime"].unique())[:args.sessions]
    by_date = {pd.Timestamp(k): v for k, v in rows.groupby("datetime")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "minutes"
    cache.mkdir(exist_ok=True)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)

    daily: dict[pd.Timestamp, dict] = {}
    hierarchy_logs: list[dict] = []
    failures: list[dict] = []
    for date in dates:
        if args.hierarchical:
            market, sectors, shortlist, stage_logs = hierarchical_shortlist(by_date[date], date, deepseek_key)
            hierarchy_logs.extend({"date": str(date.date()), **x} for x in stage_logs)
            market_result = next(x["result"] for x in stage_logs if x["phase"] == "market")
            sector_result = next(x["result"] for x in stage_logs if x["phase"] == "sectors")
        else:
            market, sectors, shortlist = broad_contexts(by_date[date], args.candidate_count)
            market_result, sector_result = {"maxExposure": 1.0}, {"sectors": [x["industry"] for x in sectors[:5]]}
        daily[date] = {"market": market, "sectors": sectors, "stocks": shortlist, "bars": {}, "maxExposure": float(market_result.get("maxExposure", 1.0)), "hotSectors": sector_result.get("sectors", [])}
    all_codes = sorted({item["instrument"] for info in daily.values() for item in info["stocks"]})
    for date in dates:
        info = daily[date]
        bars = {}
        for code in all_codes:
            try:
                bars[code] = download_5m(bs, code, date, cache)
            except Exception as exc:
                failures.append({"date": str(date.date()), "instrument": code, "error": str(exc)})
        info["bars"] = bars
        info["stocks"] = [x for x in info["stocks"] if x["instrument"] in bars]
        print(date.date(), "candidates", len(info["stocks"]), "tracked", len(bars), flush=True)
    bs.logout()

    cash, position, pending = 100_000.0, None, None
    audit, trades, nav = [], [], []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in hierarchy_logs:
        for key in usage_total:
            usage_total[key] += int(row["usage"].get(key, 0))
    for date in dates:
        info = daily[date]
        timeline = sorted(set().union(*(set(x.trade_time) for x in info["bars"].values())))
        for now in timeline:
            # A decision made at the previous completed bar executes at this bar's open.
            if pending:
                code = pending.get("stock")
                bar = info["bars"].get(code)
                current = bar[bar.trade_time.eq(now)] if bar is not None else pd.DataFrame()
                if not current.empty:
                    px = float(current.iloc[0].open)
                    action = pending.get("action")
                    if action == "buy" and position is None:
                        budget = cash * min(1.0, max(0.0, info["maxExposure"]))
                        shares = int(budget / (px * 100)) * 100
                        gross = shares * px
                        cost = fee(gross) if shares else 0
                        while shares and gross + cost > cash:
                            shares -= 100; gross = shares * px; cost = fee(gross) if shares else 0
                        if shares:
                            cash -= gross + cost
                            selected_item = next(x for x in info["stocks"] if x["instrument"] == code)
                            position = {"stock": code, "industry": selected_item["industry"], "entry_date": date, "entry_time": now, "entry_price": px, "signal_support": selected_item.get("prior_low_raw"), "shares": shares, "gross": gross, "buy_fee": cost}
                    elif action == "sell" and position and position["stock"] == code and date > position["entry_date"]:
                        gross = position["shares"] * px; cost = fee(gross, sell=True); cash += gross - cost
                        trades.append({**position, "exit_date": date, "exit_time": now, "exit_price": px, "sell_fee": cost, "return": (gross - cost) / (position["gross"] + position["buy_fee"]) - 1})
                        position = None
                pending = None

            stock_context = []
            for item in info["stocks"]:
                code = item["instrument"]
                visible = info["bars"][code][info["bars"][code].trade_time <= now]
                if visible.empty:
                    continue
                last = visible.iloc[-1]
                base_item = {
                    "code": code, "name": item["name"], "industry": item["industry"],
                    "openGap": item["open_gap"], "prior5d": item["prior_return_5d"],
                    "prior20d": item["prior_return_20d"], "patternCount": item["pattern_count"],
                    "limitMemory60": item["prior_limit_count60"], "intradayReturn": float(last.close / visible.iloc[0].open - 1),
                }
                base_item["facts"] = intraday_facts(info["bars"][code], now, float(item["sectorEdge"]), base_item)
                base_item["bars"] = compact_bars(info["bars"][code], now)
                stock_context.append(base_item)
            public_position = None
            if position:
                held_bars = info["bars"].get(position["stock"])
                held_seen = held_bars[held_bars.trade_time <= now] if held_bars is not None else pd.DataFrame()
                mark = float(held_seen.iloc[-1].close) if not held_seen.empty else position["entry_price"]
                held_vwap = None
                weak_bars = None
                if not held_seen.empty:
                    held_volume = float(held_seen.vol.sum())
                    held_vwap = float(held_seen.amount.sum() / held_volume) if held_volume > 0 else None
                    tail3 = held_seen.tail(3)
                    weak_bars = bool(len(tail3) == 3 and (tail3.close < tail3.open).all() and tail3.vol.is_monotonic_increasing)
                public_position = {
                    "stock": position["stock"], "industry": position["industry"],
                    "entryDate": str(position["entry_date"].date()), "entryPrice": position["entry_price"],
                    "return": mark / position["entry_price"] - 1, "canSell": date > position["entry_date"],
                    "facts": {"vwap": held_vwap, "aboveVwap": None if held_vwap is None else mark > held_vwap,
                              "support": position.get("signal_support"), "supportBroken": None if position.get("signal_support") is None else mark < position["signal_support"],
                              "consecutiveWeakBarsWithRisingVolume": weak_bars,
                              "hotspotRetreated": position["industry"] not in info["hotSectors"]},
                    "bars": compact_bars(held_bars, now) if held_bars is not None else []}
            payload = {"dateTime": now.isoformat(), "marketAtOpen": info["market"], "sectorAtOpen": info["sectors"][:10], "position": public_position, "candidates": stock_context}
            schema = '{"action":"buy|sell|hold|wait","stock":"代码或null","confidence":0到1,"hotSectors":["..."],"reasons":["..."],"risks":["..."]}'
            task = """完全根据截至当前的5分钟K线与裸K事实决定操作，决策在下一根5分钟K线开盘执行。
空仓时主动比较候选，不能仅以T+1、可能回落或数据有限作为永久wait理由。只允许按candidates[].facts计数；值为null或false一律不算满足，禁止自行估算或从单根K线推断。以下证据至少4项为true时应优先buy最强一只：所属板块位于hotSectors；strongerThanSector；aboveVwap；threeLowsNonDecreasing；volumeBreakout；dailySupport。若不足4项则wait。
持仓时遵守A股T+1：买入当日只能hold；次日起只允许引用position.facts。若supportBroken、aboveVwap=false且连续无法收复、consecutiveWeakBarsWithRisingVolume或hotspotRetreated形成明确风险则sell，否则hold。禁止自行计算未提供指标，普通波动不能单独触发卖出。
必须在reasons中逐项写明满足和不满足的证据数量，并逐字引用facts字段；不得引用INPUT之外的信息。"""
            decision, usage = qwen(prompt(task, schema, payload), deepseek_key, "deepseek-chat")
            for key in usage_total:
                usage_total[key] += int(usage.get(key, 0))
            action, code = decision.get("action"), decision.get("stock")
            allowed = {x["instrument"] for x in info["stocks"]}
            execution_block = None
            if action == "buy" and position is None and code in allowed:
                if float(info["market"].get("openBreadth", 0.0)) < 0.30:
                    execution_block = "market_open_breadth_below_30pct"
                else:
                    pending = decision
            elif action == "sell" and position and code == position["stock"] and date > position["entry_date"]:
                pending = decision
            audit.append({"dateTime": now.isoformat(), "position": public_position, "decision": decision, "executionBlock": execution_block, "usage": usage})

        mark = 0.0
        if position:
            held = info["bars"].get(position["stock"])
            px = float(held.iloc[-1].close) if held is not None and not held.empty else position["entry_price"]
            mark = position["shares"] * px
        nav.append({"date": str(date.date()), "nav": cash + mark, "cash": cash, "position": position["stock"] if position else None})
        (args.output_dir / "audit.partial.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(nav).to_csv(args.output_dir / "nav.partial.csv", index=False)
        print(date.date(), "nav", round(nav[-1]["nav"], 2), "position", nav[-1]["position"], flush=True)

    frame = pd.DataFrame(nav)
    nav_with_initial = pd.concat([pd.Series([100_000.0]), frame.nav], ignore_index=True)
    summary = {"start": nav[0]["date"], "end": nav[-1]["date"], "sessions": len(dates), "calls": len(audit), "cumReturn": frame.iloc[-1].nav / 100_000 - 1, "maxDrawdown": float((nav_with_initial / nav_with_initial.cummax() - 1).min()), "completedTrades": len(trades), "usage": usage_total, "openPosition": position, "minuteFailures": failures}
    (args.output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "hierarchy_audit.json").write_text(json.dumps(hierarchy_logs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(nav).to_csv(args.output_dir / "nav.csv", index=False)
    pd.DataFrame(trades).to_csv(args.output_dir / "trades.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
