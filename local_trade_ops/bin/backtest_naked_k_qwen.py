#!/usr/bin/env python3
"""Small point-in-time Qwen replay for the full naked-K decision pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from backtest_auction_hot_sector import add_point_in_time_features, limit_threshold, load_daily


OPS_HOME = Path(__file__).resolve().parents[1]
OUT_DEFAULT = OPS_HOME.parent / "reports" / "naked_k_qwen_pilot_2026"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def qwen(prompt: str, key: str, model: str, attempts: int = 3) -> tuple[dict, dict]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1000, "response_format": {"type": "json_object"}}
    last = None
    for attempt in range(attempts):
        try:
            response = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=90)
        except requests.RequestException as exc:
            last = f"request failed: {exc}"
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
                continue
            break
        if response.status_code == 200:
            raw = response.json()
            text = raw["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            try:
                return json.loads(text), raw.get("usage", {})
            except json.JSONDecodeError as exc:
                last = f"invalid JSON response: {exc}"
                time.sleep(2 ** attempt)
                continue
        last = f"HTTP {response.status_code}: {response.text[:500]}"
        retryable_content_filter = response.status_code == 400 and "Content Exists Risk" in response.text
        if response.status_code not in (429, 500, 502, 503, 504) and not retryable_content_filter:
            break
        time.sleep(2 ** attempt)
    raise RuntimeError(last or "Qwen request failed")


def prompt(task: str, schema: str, payload: dict) -> str:
    return f"""你是A股裸K短线系统的独立决策节点。只允许使用INPUT内截至当前时点的数据，禁止新闻、传闻和未来数据。
任务：{task}
规则：数据不足时保守；confidence是证据完整度而非上涨概率；reasons必须引用字段；只返回JSON对象。
OUTPUT_SCHEMA：{schema}
INPUT：{json.dumps(payload, ensure_ascii=False, default=str)}"""


def valid_day(day: pd.DataFrame) -> pd.DataFrame:
    return day[day.industry.notna() & day.raw_open.gt(0) & day.prior_raw_close.gt(0) & day.prior_amount_ma20.ge(30_000_000) & day.prior_raw_close.ge(3) & day.history_days.ge(120)].copy()


def contexts(day: pd.DataFrame) -> tuple[dict, list[dict], list[dict]]:
    valid = valid_day(day)
    market = {
        "stockCount": len(valid), "openBreadth": float(valid.open_gap.gt(0).mean()),
        "medianOpenGap": float(valid.open_gap.median()), "priorUpBreadth": float(valid.prior_return_1d.gt(0).mean()),
        "priorAboveMa20": float(valid.prior_close.ge(valid.prior_ma20).mean()), "prior5dMedian": float(valid.prior_return_5d.median()),
        "strongOpenCount": int(valid.open_gap.ge(.03).sum()), "weakOpenCount": int(valid.open_gap.le(-.03).sum()),
    }
    sectors = valid.groupby("industry", observed=True).open_gap.agg(size="size", breadth=lambda x: float((x > 0).mean()), medianGap="median", top20Gap=lambda x: float(x.nlargest(max(1, int(np.ceil(len(x) * .2)))).mean())).reset_index()
    sectors = sectors[sectors["size"] >= 5].copy()
    for col in ["breadth", "medianGap", "top20Gap"]:
        sectors[col + "Rank"] = sectors[col].rank(pct=True)
    sectors["factScore"] = .35 * sectors.breadthRank + .45 * sectors.medianGapRank + .2 * sectors.top20GapRank
    sector_rows = sectors.nlargest(15, "factScore")[["industry", "size", "breadth", "medianGap", "top20Gap"]].to_dict("records")
    names = [x["industry"] for x in sector_rows[:10]]
    # Execution constraints stay deterministic; the LLM must not waive them.
    pool = valid[
        valid.industry.isin(names)
        & valid.open_gap.le(.05)
        & valid.open_gap.lt(valid.instrument.map(limit_threshold))
        & valid.prior_return_20d.le(.25)
        & valid.prior_limit_count60.ge(1)
    ].copy().merge(sectors[["industry", "medianGap"]], on="industry", how="left")
    pool["sectorEdge"] = pool.open_gap - pool.medianGap
    pool["sectorRank"] = pool.groupby("industry").sectorEdge.rank(method="first", ascending=False)
    pool["prior_low_raw"] = pool["prior_low"] / pool["factor"]
    pool["prior_ma5_raw"] = pool["prior_ma5"] / pool["factor"]
    pool = pool[pool.sectorRank <= 5]
    cols = ["instrument", "name", "industry", "raw_open", "open_gap", "sectorEdge", "prior_return_1d", "prior_return_5d", "prior_return_20d", "prior_amount_ma20", "prior_limit_count60", "pattern_count", "prior_raw_close", "prior_low_raw", "prior_ma5_raw"]
    return market, sector_rows, pool.sort_values(["industry", "sectorRank"])[cols].replace({np.nan: None}).to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--stock-meta", required=True, type=Path)
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")
    import qlib
    qlib.init(provider_uri=args.provider_uri, region="cn")
    end_load = (pd.Timestamp(args.start_date) + pd.Timedelta(days=50)).strftime("%Y-%m-%d")
    rows = add_point_in_time_features(load_daily("2025-01-01", end_load), pd.read_csv(args.stock_meta))
    calendar = sorted(pd.Timestamp(x) for x in rows.loc[rows.datetime.ge(pd.Timestamp(args.start_date)), "datetime"].unique())[:args.sessions]
    by_date = {pd.Timestamp(k): v for k, v in rows.groupby("datetime")}
    cash, position, trades, logs, nav = 100_000.0, None, [], [], []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for date in calendar:
        day = by_date[date]
        date_text = date.strftime("%Y-%m-%d")
        occupied_at_open = position is not None
        if position is not None:
            held = day[day.instrument.eq(position["instrument"])]
            if not held.empty and np.isfinite(held.iloc[0].raw_close) and date > position["entry_date"]:
                row = held.iloc[0]
                position["holding_days"] += 1
                public_position = {key: value for key, value in position.items() if key not in {"last_price", "gross", "buy_fee"}}
                exit_input = {"date": date_text, "position": public_position, "closeFacts": {"close": row.raw_close, "dailyReturn": row.raw_close / row.prior_raw_close - 1, "returnSinceEntry": row.raw_close / position["entry_price"] - 1, "priorMa5Raw": row.prior_ma5 / row.factor, "signalLowRaw": position["signal_low_raw"], "holdingDays": position["holding_days"]}}
                result, usage = qwen(prompt("判断当前持仓收盘应hold或sell。", '{"decision":"hold|sell","confidence":0到1,"reasons":["..."],"risks":["..."]}', exit_input), key, args.model)
                logs.append({"date": date_text, "phase": "exit", "input": exit_input, "result": result, "usage": usage})
                for k in total_usage: total_usage[k] += int(usage.get(k, 0))
                if result.get("decision") == "sell" or position["holding_days"] >= 5:
                    gross = position["shares"] * float(row.raw_close); sell_fee = max(5.0, gross * .0008); cash += gross - sell_fee
                    trades.append({**position, "exit_date": date_text, "exit_price": float(row.raw_close), "sell_fee": sell_fee, "return": (gross - sell_fee) / (position["gross"] + position["buy_fee"]) - 1, "exit_model": result})
                    position = None
        # A close sale cannot fund a purchase at the same day's open.
        if position is None and not occupied_at_open:
            market, sectors, pool = contexts(day)
            stages = []
            m, usage = qwen(prompt("判断今日开盘风险状态。", '{"decision":"risk_on|cautious|risk_off","confidence":0到1,"reasons":["..."],"risks":["..."]}', {"date": date_text, "market": market}), key, args.model); stages.append(("market", m, usage))
            if m.get("decision") != "risk_off":
                h, usage = qwen(prompt("从事实表识别最多3个有真实联动的热点板块。", '{"decision":"accept|none","sectors":["..."],"confidence":0到1,"reasons":["..."],"risks":["..."]}', {"date": date_text, "marketDecision": m, "sectors": sectors}), key, args.model); stages.append(("hotspot", h, usage))
                allowed_pool = [x for x in pool if x["industry"] in h.get("sectors", [])]
                if h.get("decision") == "accept" and allowed_pool:
                    s, usage = qwen(prompt("从股票池选择最多1只股票，也可以skip。", '{"decision":"select|skip","selectedCode":"代码或null","confidence":0到1,"reasons":["..."],"risks":["..."]}', {"date": date_text, "marketDecision": m, "hotspotDecision": h, "stockPool": allowed_pool}), key, args.model); stages.append(("selection", s, usage))
                    selected = next((x for x in allowed_pool if x["instrument"] == s.get("selectedCode")), None)
                    if s.get("decision") == "select" and selected:
                        e, usage = qwen(prompt("判断目标股票开盘应buy、wait或skip。历史回放无法在盘中执行wait，因此wait视为不买。", '{"decision":"buy|wait|skip","confidence":0到1,"reasons":["..."],"risks":["..."]}', {"date": date_text, "marketDecision": m, "hotspotDecision": h, "selectionDecision": s, "stock": selected}), key, args.model); stages.append(("entry", e, usage))
                        if e.get("decision") == "buy":
                            price = float(selected["raw_open"]); shares = int(cash // (price * 100)) * 100; gross = shares * price; buy_fee = max(5.0, gross * .0003) if shares else 0
                            while shares and gross + buy_fee > cash: shares -= 100; gross = shares * price; buy_fee = max(5.0, gross * .0003) if shares else 0
                            if shares:
                                cash -= gross + buy_fee
                                factor = float(day.loc[day.instrument.eq(selected["instrument"]), "factor"].iloc[0])
                                position = {"instrument": selected["instrument"], "name": selected["name"], "industry": selected["industry"], "entry_date": date, "entry_price": price, "last_price": price, "shares": shares, "gross": gross, "buy_fee": buy_fee, "signal_low_raw": float(selected["prior_low_raw"]), "holding_days": 0}
            for phase, result, usage in stages:
                logs.append({"date": date_text, "phase": phase, "result": result, "usage": usage})
                for k in total_usage: total_usage[k] += int(usage.get(k, 0))
        close_value = 0.0
        if position:
            held = day[day.instrument.eq(position["instrument"])]
            if not held.empty and np.isfinite(held.iloc[0].raw_close):
                position["last_price"] = float(held.iloc[0].raw_close)
            close_value = position["shares"] * position.get("last_price", position["entry_price"])
        nav.append({"date": date_text, "nav": cash + close_value, "cash": cash, "position": position["instrument"] if position else None})
        print(date_text, "nav", round(nav[-1]["nav"], 2), "position", nav[-1]["position"], flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(nav).to_csv(args.output_dir / "nav.csv", index=False)
    pd.DataFrame([{k: v for k, v in x.items() if k != "exit_model"} for x in trades]).to_csv(args.output_dir / "trades.csv", index=False)
    frame = pd.DataFrame(nav); drawdown = frame.nav / frame.nav.cummax() - 1
    summary = {"model": args.model, "sessions": len(calendar), "start": nav[0]["date"], "end": nav[-1]["date"], "cumReturn": nav[-1]["nav"] / 100_000 - 1, "maxDrawdown": float(drawdown.min()), "completedTrades": len(trades), "usage": total_usage, "openPosition": position}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
