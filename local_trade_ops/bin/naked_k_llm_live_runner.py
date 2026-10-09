#!/usr/bin/env python3
"""Run one auditable phase of the frozen naked-K LLM live assistant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests


OPS_HOME = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.environ.get("NAKED_K_LIVE_STATE_DIR", OPS_HOME / "state" / "naked_k_llm_live"))
os.environ.setdefault("NAKED_K_STATE_DIR", str(STATE_DIR / "market_runtime"))
CONFIG_PATH = Path(os.environ.get("NAKED_K_LIVE_CONFIG", OPS_HOME / "config" / "naked_k_llm_live.json"))
API_BASE = os.environ.get("NAKED_K_LIVE_API", "http://127.0.0.1:8787").rstrip("/")

sys.path.insert(0, str(OPS_HOME / "bin"))
import backtest_naked_k_qwen as llm  # noqa: E402
from backtest_naked_k_qwen import prompt, qwen  # noqa: E402
from backtest_naked_k_deepseek_intraday import board_allowed, compact_bars, intraday_facts  # noqa: E402
import naked_k_daily_runner as daily  # noqa: E402


def load_local_env() -> None:
    path = OPS_HOME / "config" / "env.local"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def update_system(**patch) -> None:
    path = STATE_DIR / "system.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(patch)
    current["updatedAt"] = datetime.now().astimezone().isoformat()
    atomic_json(path, current)


def post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"live API {response.status_code}: {response.text[:500]}")
    return response.json()


def get_status() -> dict:
    response = requests.get(f"{API_BASE}/api/naked-k-live/status", timeout=15)
    response.raise_for_status()
    return response.json()


def deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY")
    llm.ENDPOINT = "https://api.deepseek.com/chat/completions"
    return key


def phase_preflight(date: str) -> dict:
    if not daily.is_trade_day(date):
        result = {"date": date, "status": "closed", "reason": "not_a_trade_day"}
        update_system(
            scheduler={"status": "market_closed", "lastPhase": "preflight", "lastSuccessAt": datetime.now().astimezone().isoformat()},
            marketData={"status": "market_closed"},
            llm={"status": "not_required", "model": "deepseek-chat"},
        )
        return result
    release = daily.latest_release()
    result = daily.preflight({"strategy_id": config()["strategyId"]}, release, date)
    update_system(
        scheduler={"status": "ready", "lastPhase": "preflight", "lastSuccessAt": datetime.now().astimezone().isoformat()},
        marketData={"status": "ready", "release": release.name, "historyEndDate": result["historyEndDate"]},
        llm={"status": "configured" if os.environ.get("DEEPSEEK_API_KEY") else "missing_key", "model": "deepseek-chat"},
    )
    return result


def safe_qwen(text: str, key: str, strip_names_payload: dict | None = None) -> tuple[dict, dict]:
    try:
        return qwen(text, key, "deepseek-chat")
    except RuntimeError as exc:
        if "Content Exists Risk" not in str(exc) or strip_names_payload is None:
            raise
        safe = json.loads(json.dumps(strip_names_payload, ensure_ascii=False))
        for stock in safe.get("stocks", []):
            stock.pop("name", None)
        return qwen(prompt(
            "从热点板块全部可交易股票中筛选0到20只观察股。不得选择输入以外代码；本层仅返回代码。",
            '{"decision":"select|skip","codes":["最多20个代码"]}', safe,
        ), key, "deepseek-chat")


def limit_threshold(code: str) -> float:
    if code.endswith(".BJ"):
        return 0.295
    if code.startswith(("300", "301", "688")):
        return 0.195
    return 0.095


def frozen_open_pool(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        if not board_allowed(str(row.get("ts_code", ""))):
            continue
        gap = row.get("open_gap")
        if gap is None:
            continue
        if float(gap) > 0.05 or float(gap) >= limit_threshold(str(row.get("ts_code", ""))):
            continue
        result.append(row)
    return result


def phase_open(date: str) -> dict:
    cfg = config()
    if not daily.is_trade_day(date):
        result = post("/api/naked-k-live/open", {
            "date": date, "status": "failed", "openBreadth": None, "top3": [], "hotSectors": [],
            "reasons": ["今日不是交易日"], "risks": ["休市，不运行盘中决策"],
            "dataCutoff": f"{date}T09:25:00+08:00",
        })
        update_system(scheduler={"status": "market_closed", "lastPhase": "open"}, marketData={"status": "market_closed"})
        return result
    legacy = json.loads((OPS_HOME / "config" / "auction_top1_forward_20260908.json").read_text(encoding="utf-8"))
    legacy["entry"]["min_market_open_breadth"] = cfg["openBreadthThreshold"]
    release = daily.latest_release()
    raw = daily.open_decision(legacy, release, date)
    breadth = float(raw["marketOpenBreadth"])
    if breadth < cfg["openBreadthThreshold"]:
        result = post("/api/naked-k-live/open", {
            "date": date, "openBreadth": breadth, "top3": [], "hotSectors": [],
            "reasons": [f"09:25开盘广度{breadth:.2%}低于37%门槛"], "risks": ["今日禁止新开仓"],
            "dataCutoff": f"{date}T09:25:00+08:00",
        })
        update_system(scheduler={"status": "running", "lastPhase": "open", "lastSuccessAt": datetime.now().astimezone().isoformat()})
        return result

    key = deepseek_key()
    context = raw["llmContext"]
    context["sectorLeaderboard"] = [row for row in context["sectorLeaderboard"] if int(row.get("sector_size") or 0) >= 5]
    context["stockPool"] = frozen_open_pool(context["stockPool"])
    market_decision, market_usage = qwen(prompt(
        "解释今日09:25市场状态。开仓许可只由代码中的37%广度门槛决定。",
        '{"decision":"risk_on|cautious|risk_off","confidence":0到1,"reasons":["..."],"risks":["..."]}',
        {"date": date, "market": context["market"]},
    ), key, "deepseek-chat")
    sector_decision, sector_usage = qwen(prompt(
        "比较行业聚合事实，选择0到5个具备真实联动的热点板块。",
        '{"decision":"accept|none","sectors":["最多5个"],"confidence":0到1,"reasons":["..."],"risks":["..."]}',
        {"date": date, "marketDecision": market_decision, "allSectors": context["sectorLeaderboard"]},
    ), key, "deepseek-chat")
    sectors = sector_decision.get("sectors", [])[:5] if sector_decision.get("decision") == "accept" else []
    stocks = [row for row in context["stockPool"] if row.get("industry") in sectors]
    top20_payload = {"date": date, "marketDecision": market_decision, "sectorDecision": sector_decision, "stocks": stocks}
    screen, screen_usage = safe_qwen(prompt(
        "从热点板块全部可交易股票中筛选0到20只观察股。不得选择输入以外代码；本层仅返回代码。",
        '{"decision":"select|skip","codes":["最多20个代码"]}', top20_payload,
    ), key, top20_payload)
    allowed = {row["ts_code"] for row in stocks}
    codes = [code for code in screen.get("codes", []) if code in allowed][:20]
    details = [row for row in stocks if row["ts_code"] in codes]
    final, final_usage = qwen(prompt(
        "综合市场、热点和裸K事实，从观察股精排0到3只盘中跟踪股。允许全部放弃。",
        '{"decision":"select|skip","codes":["最多3个代码"],"confidence":0到1,"reasons":["..."],"risks":["..."]}',
        {"date": date, "marketDecision": market_decision, "sectorDecision": sector_decision, "top20": details},
    ), key, "deepseek-chat")
    top_codes = final.get("codes", [])[:3] if final.get("decision") == "select" else []
    by_code = {row["ts_code"]: row for row in details}
    top3 = [{
        "code": code, "name": by_code[code].get("name", code), "industry": by_code[code].get("industry"),
        "price": by_code[code].get("open"), "context": by_code[code],
    } for code in top_codes if code in by_code]
    result = post("/api/naked-k-live/open", {
        "date": date, "openBreadth": breadth, "top3": top3, "hotSectors": sectors,
        "reasons": final.get("reasons", []) or market_decision.get("reasons", []),
        "risks": final.get("risks", []) or market_decision.get("risks", []),
        "dataCutoff": f"{date}T09:25:00+08:00",
    })
    usage = [market_usage, sector_usage, screen_usage, final_usage]
    update_system(
        scheduler={"status": "running", "lastPhase": "open", "lastSuccessAt": datetime.now().astimezone().isoformat()},
        llm={"status": "ok", "model": "deepseek-chat", "lastCallAt": datetime.now().astimezone().isoformat(), "usage": usage},
    )
    return result


def normalize_5m(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = ["trade_time", "open", "close", "high", "low", "vol", "amount"]
    if frame.empty or not set(required).issubset(frame.columns):
        raise RuntimeError(f"{source}没有返回有效5分钟行情")
    result = frame[required].copy()
    result["trade_time"] = pd.to_datetime(result["trade_time"])
    for column in required[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["trade_time", "open", "close", "high", "low"]).sort_values("trade_time").reset_index(drop=True)
    result.attrs["source"] = source
    return result


def tencent_5m(code: str, date: str) -> pd.DataFrame:
    if date != datetime.now().astimezone().strftime("%Y-%m-%d"):
        raise RuntimeError("腾讯分钟行情只提供当前交易日")
    symbol, exchange = code.split(".")
    prefix = "sh" if exchange == "SH" else "sz" if exchange == "SZ" else "bj"
    key = prefix + symbol
    response = requests.get(
        "https://web.ifzq.gtimg.cn/appstock/app/minute/query",
        params={"code": key}, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}, timeout=15,
    )
    response.raise_for_status()
    lines = (((response.json().get("data") or {}).get(key) or {}).get("data") or {}).get("data") or []
    rows = []
    for line in lines:
        cells = line.split()
        if len(cells) < 4:
            continue
        rows.append({"trade_time": pd.Timestamp(f"{date} {cells[0][:2]}:{cells[0][2:]}") , "price": float(cells[1]),
                     "cum_vol": float(cells[2]), "cum_amount": float(cells[3])})
    minute = pd.DataFrame(rows).sort_values("trade_time").reset_index(drop=True)
    if minute.empty:
        raise RuntimeError(f"腾讯没有取得{code}在{date}的分钟行情")
    clock = minute.trade_time.dt.hour * 60 + minute.trade_time.dt.minute
    continuous_session = clock.between(9 * 60 + 30, 11 * 60 + 30) | clock.between(13 * 60, 15 * 60)
    minute = minute[continuous_session].copy()
    # Tencent reports cumulative volume in hands while amount is in yuan.
    minute["vol"] = minute.cum_vol.diff().fillna(minute.cum_vol).clip(lower=0) * 100
    minute["amount"] = minute.cum_amount.diff().fillna(minute.cum_amount).clip(lower=0)
    clock = minute.trade_time.dt.hour * 60 + minute.trade_time.dt.minute
    session_minute = np.where(clock <= 11 * 60 + 30, clock - 9 * 60 - 30, 120 + clock - 13 * 60)
    minute["bucket"] = np.maximum(1, np.ceil(session_minute / 5)).astype(int)
    bars = minute.groupby("bucket", sort=True).agg(
        trade_time=("trade_time", "max"), open=("price", "first"), close=("price", "last"),
        high=("price", "max"), low=("price", "min"), vol=("vol", "sum"), amount=("amount", "sum"),
    ).reset_index(drop=True)
    return normalize_5m(bars, "tencent_minute_aggregated")


def tushare_5m(code: str, date: str) -> pd.DataFrame:
    frame = daily.pro_client().stk_mins(
        ts_code=code, start_date=f"{date} 09:30:00", end_date=f"{date} 15:00:00", freq="5min",
    )
    return normalize_5m(frame, "tushare_stk_mins")


def eastmoney_5m(code: str, date: str) -> pd.DataFrame:
    symbol, exchange = code.split(".")
    secid = f"1.{symbol}" if exchange == "SH" else f"0.{symbol}"
    compact = date.replace("-", "")
    errors = []
    for attempt in range(2):
        try:
            response = requests.get(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                params={"secid": secid, "klt": 5, "fqt": 1, "beg": compact, "end": compact,
                        "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"},
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}, timeout=15,
            )
            response.raise_for_status()
            klines = (response.json().get("data") or {}).get("klines") or []
            rows = []
            for line in klines:
                cells = line.split(",")
                if len(cells) >= 7:
                    rows.append({"trade_time": cells[0], "open": cells[1], "close": cells[2], "high": cells[3],
                                 "low": cells[4], "vol": float(cells[5]) * 100, "amount": cells[6]})
            return normalize_5m(pd.DataFrame(rows), "eastmoney_5m")
        except Exception as exc:
            errors.append(f"eastmoney:{exc}")
            time.sleep(attempt + 1)
    for source in [tencent_5m, tushare_5m]:
        try:
            return source(code, date)
        except Exception as exc:
            errors.append(f"{source.__name__}:{exc}")
    raise RuntimeError(f"没有取得{code}在{date}的5分钟行情: {'; '.join(errors)}")


def phase_intraday(date: str, at: str | None) -> dict:
    status = get_status()
    state = status["state"]
    if state in {"ENTRY_BLOCKED", "BUY_PENDING", "SELL_PENDING", "T0_LOCKED", "PREMARKET"}:
        return {"status": "skipped", "state": state, "reason": "当前状态没有可执行的LLM决策"}
    key = deepseek_key()
    cutoff = pd.Timestamp(f"{date} {at}") if at else pd.Timestamp.now().floor("5min")
    open_state = status["day"]["open"]
    hot_sectors = [row if isinstance(row, str) else row.get("industry", "") for row in open_state.get("hotSectors", [])]
    if state == "WATCHING":
        candidates = []
        frames = {}
        for item in open_state.get("top3", []):
            frame = eastmoney_5m(item["code"], date)
            frames[item["code"]] = frame
            visible = frame[frame.trade_time <= cutoff]
            if visible.empty:
                continue
            context = item.get("context") or {}
            base = {
                "code": item["code"], "name": item["name"], "industry": item.get("industry"),
                "openGap": context.get("open_gap"), "prior5d": context.get("prior_return_5d"),
                "prior20d": context.get("prior_return_20d"), "patternCount": context.get("pattern_count"),
                "limitMemory60": context.get("prior_limit_count60"),
                "priorClose": context.get("prior_close"),
                "intradayReturn": float(visible.iloc[-1].close / visible.iloc[0].open - 1),
            }
            base["facts"] = intraday_facts(frame, cutoff, float(context.get("stock_vs_sector_open") or 0), base)
            base["evidenceChecklist"] = {
                "hotSector": base.get("industry") in hot_sectors,
                **{
                    field: base["facts"].get(field) is True
                    for field in ["strongerThanSector", "aboveVwap", "threeLowsNonDecreasing", "volumeBreakout", "dailySupport"]
                },
            }
            base["evidenceCount"] = sum(base["evidenceChecklist"].values())
            base["bars"] = compact_bars(frame, cutoff)
            candidates.append(base)
        schema = '{"action":"buy|wait","stock":"代码或null","confidence":0到1,"reasons":["..."],"risks":["..."]}'
        task = """空仓时比较固定Top3。代码已经在每只股票的evidenceChecklist与evidenceCount中完成六项证据计数，必须直接引用evidenceCount，禁止自行重算。evidenceCount少于4必须wait；达到4仅代表具备买入资格，不等于必须buy，仍可因动能、量价质量或追高风险wait。每只股票必须准确写明“满足evidenceCount/6项”；若达到4却wait，必须明确说明额外拒绝理由，禁止写成“未达到4项”。不得选择输入外股票。"""
        decision, usage = qwen(prompt(task, schema, {"dateTime": cutoff.isoformat(), "hotSectors": hot_sectors, "candidates": candidates}), key, "deepseek-chat")
        action = decision.get("action", "wait")
        code = decision.get("stock") if action == "buy" else None
        selected = next((item for item in candidates if item["code"] == code), None)
        hard_block = None
        evidence = None
        if action == "buy":
            if cutoff.strftime("%H:%M") > config()["lastBuyDecision"]:
                action, hard_block = "blocked", "after_last_buy_decision"
            elif not selected:
                action, hard_block = "blocked", "stock_not_in_fixed_top3"
            else:
                facts = selected["facts"]
                evidence = int(selected.get("industry") in hot_sectors) + sum(int(facts.get(field) is True) for field in ["strongerThanSector", "aboveVwap", "threeLowsNonDecreasing", "volumeBreakout", "dailySupport"])
                if evidence < 4:
                    action, hard_block = "blocked", "buy_evidence_below_four"
                elif facts.get("threeLowsNonDecreasing") is False and facts.get("volumeBreakout") is False:
                    action, hard_block = "blocked", "buy_structure_and_volume_both_false"
                else:
                    bar = frames[code][frames[code].trade_time <= cutoff].iloc[-1]
                    span = float(bar.high - bar.low)
                    amplitude = span / float(bar.open) if bar.open else 0
                    upper = (float(bar.high) - max(float(bar.open), float(bar.close))) / span if span else 0
                    if amplitude >= .04 and upper >= .35:
                        action, hard_block = "blocked", "large_upper_wick"
                    prior_close = selected.get("priorClose")
                    if prior_close and float(bar.close) >= float(prior_close) * (1 + limit_threshold(code)):
                        action, hard_block = "blocked", "limit_up_not_buyable"
        name = selected["name"] if selected else ""
        price = selected["facts"]["lastClose"] if selected else None
    else:
        position = status["account"]["position"]
        frame = eastmoney_5m(position["code"], date)
        visible = frame[frame.trade_time <= cutoff]
        volume = float(visible.vol.sum())
        vwap = float(visible.amount.sum() / volume) if volume else None
        tail = visible.tail(3)
        entry_context = position.get("entryContext") or {}
        raw_support = entry_context.get("prior_low")
        support = float(raw_support) if raw_support is not None and np.isfinite(float(raw_support)) and float(raw_support) > 0 else None
        entry_industry = entry_context.get("industry")
        facts = {"aboveVwap": bool(visible.iloc[-1].close > vwap) if vwap else None,
                 "vwap": vwap, "support": support,
                 "supportBroken": bool(visible.iloc[-1].close < support) if support else None,
                 "consecutiveWeakBarsWithRisingVolume": bool(len(tail) == 3 and (tail.close < tail.open).all() and tail.vol.is_monotonic_increasing),
                 "hotspotRetreated": bool(entry_industry and entry_industry not in hot_sectors)}
        payload = {"dateTime": cutoff.isoformat(), "position": {**position, "facts": facts, "bars": compact_bars(frame, cutoff)}}
        decision, usage = qwen(prompt("只判断当前T+1持仓继续持有或卖出。卖出必须引用position.facts中的明确走弱证据。首根5分钟K线只有aboveVwap=false而其他风险事实均不为true时必须hold，等待下一根K线确认。", '{"action":"hold|sell","stock":"持仓代码","confidence":0到1,"reasons":["..."],"risks":["..."]}', payload), key, "deepseek-chat")
        action = decision.get("action", "hold")
        code, name, price = position["code"], position["name"], float(visible.iloc[-1].close)
        hard_block, evidence = None, None
        if action == "sell" and decision.get("stock") != code:
            action, hard_block = "hold", "sell_stock_mismatch"
        position_facts, visible_bar_count = facts, len(visible)
    if not status["account"].get("position"):
        position_facts, visible_bar_count = None, None
    result = post("/api/naked-k-live/decisions", {
        "id": f"intraday-{date}-{cutoff.strftime('%H%M')}", "date": date, "generatedAt": datetime.now().astimezone().isoformat(),
        "dataCutoff": cutoff.tz_localize("Asia/Shanghai").isoformat() if cutoff.tzinfo is None else cutoff.isoformat(),
        "action": action, "code": code, "name": name, "referencePrice": price,
        "confidence": decision.get("confidence"), "evidenceCount": evidence,
        "reasons": decision.get("reasons", []), "risks": decision.get("risks", []), "hardBlock": hard_block,
        "positionFacts": position_facts, "visibleBarCount": visible_bar_count,
    })
    update_system(
        scheduler={"status": "running", "lastPhase": "intraday", "lastSuccessAt": datetime.now().astimezone().isoformat()},
        marketData={"status": "ready", "last5mCutoff": cutoff.isoformat()},
        llm={"status": "ok", "model": "deepseek-chat", "lastCallAt": datetime.now().astimezone().isoformat(), "usage": usage},
    )
    return result


def main() -> int:
    load_local_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "open", "intraday"], required=True)
    parser.add_argument("--date", default=datetime.now().astimezone().strftime("%Y-%m-%d"))
    parser.add_argument("--at", help="Intraday cutoff HH:MM, mainly for deterministic tests")
    args = parser.parse_args()
    try:
        if args.phase == "preflight":
            result = phase_preflight(args.date)
        elif args.phase == "open":
            result = phase_open(args.date)
        else:
            result = phase_intraday(args.date, args.at)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        update_system(
            scheduler={"status": "failed", "lastPhase": args.phase, "lastFailureAt": datetime.now().astimezone().isoformat(), "error": str(exc)},
            **({"marketData": {"status": "failed", "error": str(exc)}} if args.phase != "open" else {}),
        )
        print(json.dumps({"status": "failed", "phase": args.phase, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
