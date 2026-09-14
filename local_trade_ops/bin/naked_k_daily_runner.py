#!/usr/bin/env python3
"""Run the frozen auction naked-K strategy as auditable daily phases."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
import requests


OPS_HOME = Path(__file__).resolve().parents[1]
PROJECT_HOME = OPS_HOME.parent
CONFIG_PATH = OPS_HOME / "config" / "auction_top1_forward_20260908.json"
STATE_DIR = Path(os.environ.get("NAKED_K_STATE_DIR", OPS_HOME / "state" / "naked_k_forward"))
DATA_HOME = OPS_HOME / "data" / "naked_k"


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


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def latest_release() -> Path:
    releases = sorted(path for path in (DATA_HOME / "releases").glob("*") if (path / "manifest.json").exists())
    if not releases:
        raise RuntimeError("没有可用的裸K历史数据快照")
    return releases[-1]


def pro_client():
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("缺少 TUSHARE_TOKEN，无法更新昨日正式行情")
    return ts.pro_api(token)


def is_trade_day(date: str) -> bool:
    compact = date.replace("-", "")
    calendar = pro_client().trade_cal(exchange="SSE", start_date=compact, end_date=compact)
    return bool(len(calendar) and int(calendar.iloc[0]["is_open"]) == 1)


def update_missing_history(release: Path, target_date: str) -> list[str]:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    start = pd.Timestamp(str(manifest["end_date"])) + pd.Timedelta(days=1)
    end = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    if start > end:
        return []
    pro = pro_client()
    calendar = pro.trade_cal(exchange="SSE", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    dates = sorted(calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"].astype(str))
    updated = []
    for trade_date in dates:
        for folder, method in [("daily", pro.daily), ("adj_factor", pro.adj_factor), ("stk_limit", pro.stk_limit)]:
            path = STATE_DIR / "market_data" / folder / trade_date[:4] / f"{trade_date}.parquet"
            if path.exists():
                continue
            frame = method(trade_date=trade_date)
            if frame is None or len(frame) < 3000:
                raise RuntimeError(f"{folder}:{trade_date} 数据不完整 rows={0 if frame is None else len(frame)}")
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        updated.append(trade_date)
    return updated


def fetch_tencent_realtime(codes: list[str]) -> pd.DataFrame:
    symbols = []
    for code in codes:
        symbol, exchange = code.split(".")
        prefix = "sh" if exchange == "SH" else "sz" if exchange == "SZ" else "bj"
        symbols.append(prefix + symbol)
    batches = [symbols[index:index + 80] for index in range(0, len(symbols), 80)]

    def fetch(batch: list[str]) -> list[dict]:
        response = requests.get(
            "https://qt.gtimg.cn/q=" + ",".join(batch),
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}, timeout=10,
        )
        response.raise_for_status()
        rows = []
        for line in response.content.decode("gbk", errors="ignore").split(";"):
            if '="' not in line:
                continue
            values = line.split('="', 1)[1].rstrip('"').split("~")
            if len(values) < 6 or not values[2]:
                continue
            raw = line.split('="', 1)[0].rsplit("_", 1)[-1]
            exchange = "SH" if raw.startswith("sh") else "SZ" if raw.startswith("sz") else "BJ"
            rows.append({"TS_CODE": f"{values[2]}.{exchange}", "NAME": values[1], "PRICE": values[3], "CLOSE": values[4], "OPEN": values[5]})
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(fetch, batch) for batch in batches]
        for future in as_completed(futures):
            rows.extend(future.result())
    frame = pd.DataFrame(rows)
    if len(frame) < 3000:
        raise RuntimeError(f"腾讯实时行情不完整: rows={len(frame)}")
    return frame


def fetch_realtime(codes: list[str], attempts: int = 3) -> pd.DataFrame:
    last_error = None
    for attempt in range(attempts):
        try:
            frame = ts.realtime_list(src="dc")
            frame.columns = [str(column).upper() for column in frame.columns]
            required = {"TS_CODE", "NAME", "PRICE", "OPEN", "CLOSE"}
            if not required.issubset(frame.columns) or len(frame) < 3000:
                raise RuntimeError(f"实时行情不完整: rows={len(frame)}")
            return frame
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    try:
        return fetch_tencent_realtime(codes)
    except Exception as backup_error:
        raise RuntimeError(f"实时行情连续失败: dc={last_error}; tencent={backup_error}") from backup_error


def load_history(release: Path, sessions: int = 140) -> pd.DataFrame:
    files_by_date = {path.stem: path for path in (release / "daily").glob("*/*.parquet")}
    files_by_date.update({path.stem: path for path in (STATE_DIR / "market_data" / "daily").glob("*/*.parquet")})
    files = [files_by_date[key] for key in sorted(files_by_date)[-sessions:]]
    factors = {path.stem: path for path in (release / "adj_factor").glob("*/*.parquet")}
    factors.update({path.stem: path for path in (STATE_DIR / "market_data" / "adj_factor").glob("*/*.parquet")})
    frames = []
    for path in files:
        daily = pd.read_parquet(path)
        factor_path = factors.get(path.stem)
        if factor_path:
            factor = pd.read_parquet(factor_path)[["ts_code", "adj_factor"]]
            daily = daily.merge(factor, on="ts_code", how="left")
        else:
            daily["adj_factor"] = 1.0
        frames.append(daily)
    history = pd.concat(frames, ignore_index=True)
    history["trade_date"] = history["trade_date"].astype(str)
    history["adj_close"] = pd.to_numeric(history["close"], errors="coerce") * pd.to_numeric(history["adj_factor"], errors="coerce")
    history["adj_high"] = pd.to_numeric(history["high"], errors="coerce") * pd.to_numeric(history["adj_factor"], errors="coerce")
    history["adj_low"] = pd.to_numeric(history["low"], errors="coerce") * pd.to_numeric(history["adj_factor"], errors="coerce")
    history["amount_cny"] = pd.to_numeric(history["amount"], errors="coerce") * 1000.0
    return history.sort_values(["ts_code", "trade_date"])


def build_prior_features(history: pd.DataFrame) -> pd.DataFrame:
    records = []
    for code, frame in history.groupby("ts_code", sort=False):
        frame = frame.dropna(subset=["adj_close"]).tail(121)
        if len(frame) < 121:
            continue
        close = frame["adj_close"].to_numpy(float)
        high = frame["adj_high"].to_numpy(float)
        low = frame["adj_low"].to_numpy(float)
        amount = frame["amount_cny"].to_numpy(float)
        one_day = close[-1] / close[-2] - 1
        high20 = np.nanmax(high[-21:-1])
        low20 = np.nanmin(low[-21:-1])
        ma20 = np.nanmean(close[-20:])
        range20 = high20 / low20 - 1 if low20 > 0 else np.nan
        breakout = close[-1] >= high20 * 1.002 and range20 <= 0.25
        support = -0.02 <= close[-1] / ma20 - 1 <= 0.04 and 0 <= one_day <= 0.06
        weak_to_strong = -0.08 <= one_day <= -0.005 and close[-1] >= ma20 * 0.98
        returns = frame["adj_close"].pct_change(fill_method=None)
        threshold = 0.295 if code.endswith(".BJ") else 0.195 if code.startswith(("300", "301", "688")) else 0.095
        records.append({
            "ts_code": code, "prior_close": float(frame.iloc[-1]["close"]), "prior_low": float(frame.iloc[-1]["low"]),
            "prior_return_1d": one_day, "prior_return_5d": close[-1] / close[-6] - 1,
            "prior_return_20d": close[-1] / close[-21] - 1, "prior_ma5_raw": float(frame["close"].tail(5).mean()),
            "prior_ma20_ok": close[-1] >= ma20, "prior_amount_ma20": float(np.nanmean(amount[-20:])),
            "prior_limit_count60": int((returns.tail(60) >= threshold).sum()),
            "pattern_count": int(breakout) + int(support) + int(weak_to_strong),
        })
    return pd.DataFrame(records)


def open_decision(config: dict, release: Path, date: str) -> dict:
    meta = pd.read_parquet(release / "reference" / "stock_basic.parquet")
    listed = meta.loc[meta["requested_list_status"].eq("L"), "ts_code"].astype(str).tolist()
    realtime = fetch_realtime(listed)
    snapshot_dir = STATE_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    realtime.to_parquet(snapshot_dir / f"open_{date.replace('-', '')}.parquet", index=False)
    history = load_history(release)
    features = build_prior_features(history)
    live = realtime.rename(columns={"TS_CODE": "ts_code", "NAME": "live_name", "OPEN": "open", "CLOSE": "live_prior_close"})
    for column in ["open", "live_prior_close"]:
        live[column] = pd.to_numeric(live[column], errors="coerce")
    day = features.merge(meta[["ts_code", "name", "industry", "list_date"]], on="ts_code", how="left").merge(
        live[["ts_code", "live_name", "open", "live_prior_close"]], on="ts_code", how="inner"
    )
    day["open_gap"] = day["open"] / day["live_prior_close"] - 1
    valid = day[day["industry"].notna() & day["open"].gt(0) & day["prior_amount_ma20"].ge(config["entry"]["min_amount_ma20"]) & day["prior_close"].ge(config["entry"]["min_price"])].copy()
    breadth = float(valid["open_gap"].gt(0).mean()) if len(valid) else 0.0
    result = {"date": date, "generatedAt": datetime.now().astimezone().isoformat(), "marketOpenBreadth": breadth, "decision": "empty", "reason": "", "top1": None, "hotSectors": [], "candidates": [], "llmContext": {"market": {}, "sectorLeaderboard": [], "stockPool": []}}
    result["llmContext"]["market"] = {
        "validStockCount": int(len(valid)),
        "upCount": int(valid["open_gap"].gt(0).sum()),
        "downCount": int(valid["open_gap"].lt(0).sum()),
        "flatCount": int(valid["open_gap"].eq(0).sum()),
        "openBreadth": breadth,
        "medianOpenGap": float(valid["open_gap"].median()) if len(valid) else None,
        "strongOpenCount": int(valid["open_gap"].ge(.03).sum()),
        "weakOpenCount": int(valid["open_gap"].le(-.03).sum()),
    }
    if breadth < config["entry"]["min_market_open_breadth"]:
        result["reason"] = "market_open_breadth_below_threshold"
        return result
    sector_all = valid.groupby("industry")["open_gap"].agg(sector_size="size", sector_breadth=lambda x: float((x > 0).mean()), sector_median_gap="median", sector_top20_gap=lambda x: float(x.nlargest(max(1, int(np.ceil(len(x) * .2)))).mean())).reset_index()
    for column in ["sector_breadth", "sector_median_gap", "sector_top20_gap"]:
        sector_all[column + "_rank"] = sector_all[column].rank(pct=True)
    sector_all["sector_score"] = .35 * sector_all.sector_breadth_rank + .45 * sector_all.sector_median_gap_rank + .2 * sector_all.sector_top20_gap_rank
    sector_columns = ["industry", "sector_size", "sector_breadth", "sector_median_gap", "sector_top20_gap", "sector_score"]
    result["llmContext"]["sectorLeaderboard"] = sector_all.nlargest(15, "sector_score")[sector_columns].replace({np.nan: None}).to_dict("records")
    llm_sectors = set(result["llmContext"]["sectorLeaderboard"][index]["industry"] for index in range(min(10, len(result["llmContext"]["sectorLeaderboard"]))))
    llm_pool = valid[valid["industry"].isin(llm_sectors)].copy()
    sector_medians = sector_all.set_index("industry")["sector_median_gap"]
    llm_pool["stock_vs_sector_open"] = llm_pool["open_gap"] - llm_pool["industry"].map(sector_medians)
    llm_pool["within_sector_rank"] = llm_pool.groupby("industry")["stock_vs_sector_open"].rank(method="first", ascending=False)
    llm_pool = llm_pool[llm_pool["within_sector_rank"] <= 5]
    llm_columns = ["ts_code", "name", "industry", "open", "open_gap", "stock_vs_sector_open", "prior_return_1d", "prior_return_5d", "prior_return_20d", "prior_amount_ma20", "prior_limit_count60", "pattern_count", "prior_low", "prior_ma5_raw"]
    result["llmContext"]["stockPool"] = llm_pool.sort_values(["industry", "within_sector_rank"])[llm_columns].replace({np.nan: None}).to_dict("records")
    sector = sector_all[(sector_all.sector_size >= 5) & (sector_all.sector_breadth >= .55) & (sector_all.sector_median_gap >= .005) & (sector_all.sector_top20_gap >= .02)].copy()
    hot = sector.nlargest(config["entry"]["hot_sector_count"], "sector_score")
    result["hotSectors"] = hot[["industry", "sector_score", "sector_breadth", "sector_median_gap"]].to_dict("records")
    candidates = valid.merge(hot, on="industry", how="inner")
    candidates["stock_vs_sector_open"] = candidates.open_gap - candidates.sector_median_gap
    candidates = candidates[(candidates.open_gap <= config["entry"]["max_entry_gap"]) & candidates.prior_ma20_ok & (candidates.prior_return_20d <= config["entry"]["max_prior_return_20d"]) & (candidates.prior_limit_count60 >= config["entry"]["min_prior_limit_count_60d"]) & (candidates.stock_vs_sector_open >= config["entry"]["min_stock_sector_open_edge"])].copy()
    if candidates.empty:
        result["reason"] = "no_candidate_after_filters"
        return result
    candidates["open_strength_rank"] = candidates.groupby("industry")["stock_vs_sector_open"].rank(pct=True)
    candidates["prior_strength_rank"] = candidates["prior_return_5d"].rank(pct=True)
    candidates["memory_rank"] = candidates["prior_limit_count60"].rank(pct=True)
    w = config["leader_weights"]
    candidates["leader_score"] = w["open_strength"] * candidates.open_strength_rank + w["prior_5d_strength"] * candidates.prior_strength_rank + w["naked_k_structure"] * candidates.pattern_count.clip(upper=2) / 2 + w["limit_memory_60d"] * candidates.memory_rank
    columns = ["ts_code", "name", "industry", "open", "open_gap", "leader_score", "open_strength_rank", "prior_strength_rank", "pattern_count", "prior_limit_count60", "prior_low", "prior_ma5_raw"]
    ranked = candidates.sort_values(["leader_score", "open_gap", "ts_code"], ascending=[False, False, True]).head(10)
    result["candidates"] = ranked[columns].replace({np.nan: None}).to_dict("records")
    result["top1"] = result["candidates"][0]
    result["decision"] = "buy"
    result["reason"] = "top1_selected"
    return result


def preflight(config: dict, release: Path, date: str) -> dict:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    updated = update_missing_history(release, date)
    runtime = sorted((STATE_DIR / "market_data" / "daily").glob("*/*.parquet"))
    latest = runtime[-1].stem if runtime else str(manifest["end_date"])
    return {"date": date, "generatedAt": datetime.now().astimezone().isoformat(), "status": "ready", "release": release.name, "historyEndDate": f"{latest[:4]}-{latest[4:6]}-{latest[6:]}", "updatedDates": updated, "strategyId": config["strategy_id"]}


def account_from_ledger() -> tuple[dict[str, dict], float, float]:
    ledger_path = STATE_DIR / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"initialCapital": 100000, "fills": []}
    positions: dict[str, dict] = {}
    realized_pnl = 0.0
    net_cash_flow = 0.0
    for fill in ledger.get("fills", []):
        code, count = fill["code"], int(fill["shares"])
        position = positions.setdefault(code, {"name": fill["name"], "shares": 0, "cost": 0.0})
        if fill["side"] == "buy":
            old_cost = position["shares"] * position["cost"]
            position["shares"] += count
            position["cost"] = (old_cost + float(fill["gross"]) + float(fill["fee"])) / position["shares"]
            net_cash_flow -= float(fill["gross"]) + float(fill["fee"])
        else:
            realized_pnl += float(fill["gross"]) - float(fill["fee"]) - position["cost"] * count
            position["shares"] -= count
            net_cash_flow += float(fill["gross"]) - float(fill["fee"])
    return {code: item for code, item in positions.items() if item["shares"] > 0}, realized_pnl, net_cash_flow


def settle_account(date: str) -> dict:
    holdings, realized_pnl, net_cash_flow = account_from_ledger()
    quotes = fetch_realtime(list(holdings)).set_index("TS_CODE") if holdings else pd.DataFrame()
    positions, market_value, open_cost, unrealized_pnl = [], 0.0, 0.0, 0.0
    for code, holding in holdings.items():
        if code not in quotes.index:
            raise RuntimeError(f"收盘行情没有持仓股票 {code}")
        price = float(quotes.loc[code, "PRICE"])
        count, cost = holding["shares"], holding["cost"]
        value = count * price
        market_value += value
        open_cost += count * cost
        unrealized_pnl += value - count * cost
        positions.append({"code": code, "name": holding["name"], "shares": count, "cost": cost, "close": price, "marketValue": value, "unrealizedPnl": value - count * cost})
    path = STATE_DIR / "forward_performance.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows = [row for row in rows if row.get("date") != date]
    row = {"date": date, "generatedAt": datetime.now().astimezone().isoformat(), "openCost": open_cost, "marketValue": market_value, "realizedPnl": realized_pnl, "unrealizedPnl": unrealized_pnl, "totalPnl": realized_pnl + unrealized_pnl, "netCashFlow": net_cash_flow, "positions": positions}
    rows.append(row)
    atomic_json(path, rows)
    return {"status": "settled", **row}


def tail_decision(config: dict, release: Path, date: str) -> dict:
    ledger_path = STATE_DIR / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"fills": []}
    shares = {}
    for fill in ledger.get("fills", []):
        shares[fill["code"]] = shares.get(fill["code"], 0) + (fill["shares"] if fill["side"] == "buy" else -fill["shares"])
    held = [code for code, count in shares.items() if count > 0]
    result = {"date": date, "generatedAt": datetime.now().astimezone().isoformat(), "decision": "hold", "reason": "no_position", "position": None, "checks": {}}
    if not held:
        return result
    realtime = fetch_realtime(held).set_index("TS_CODE")
    code = held[0]
    if code not in realtime.index:
        raise RuntimeError(f"实时行情没有持仓股票 {code}")
    row = realtime.loc[code]
    price, prior = float(row["PRICE"]), float(row["CLOSE"])
    daily_return = price / prior - 1
    history = load_history(release)
    features = build_prior_features(history).set_index("ts_code")
    prior_ma5 = float(features.loc[code, "prior_ma5_raw"]) if code in features.index else np.nan
    buys = [fill for fill in ledger.get("fills", []) if fill["side"] == "buy" and fill["code"] == code]
    entry_date = max((fill["date"] for fill in buys), default=date)
    sessions = sorted(history.loc[(history.ts_code == code) & (history.trade_date > entry_date.replace("-", "")), "trade_date"].unique())
    holding_days = len(sessions) + (1 if date.replace("-", "") not in sessions and date > entry_date else 0)
    open_phase = STATE_DIR / "phases" / f"{entry_date}_open.json"
    signal = json.loads(open_phase.read_text(encoding="utf-8")) if open_phase.exists() else {}
    signal_low = next((float(item["prior_low"]) for item in signal.get("candidates", []) if item["ts_code"] == code), np.nan)
    checks = {"dailyReturn": daily_return, "priorMa5": prior_ma5 if np.isfinite(prior_ma5) else None, "signalLow": signal_low if np.isfinite(signal_low) else None, "holdingDays": holding_days}
    result["checks"] = checks
    result["position"] = {"code": code, "name": row["NAME"], "price": price, "dailyReturn": daily_return, "shares": shares[code], "entryDate": entry_date, "holdingDays": holding_days}
    if np.isfinite(signal_low) and price < signal_low * (1 - float(config["exit"]["signal_low_buffer"])):
        result.update(decision="sell", reason="signal_low_broken")
    elif daily_return <= config["exit"]["daily_drop_threshold"]:
        result.update(decision="sell", reason="daily_drop_threshold")
    elif config["exit"]["close_below_prior_ma5"] and np.isfinite(prior_ma5) and price < prior_ma5:
        result.update(decision="sell", reason="close_below_prior_ma5")
    elif holding_days >= int(config["exit"]["max_holding_days"]):
        result.update(decision="sell", reason="max_holding_days")
    else:
        result.update(decision="hold", reason="no_exit_rule_triggered")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "open", "tail", "settle"], required=True)
    parser.add_argument("--date", default=datetime.now().astimezone().date().isoformat())
    args = parser.parse_args()
    load_local_env()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    release = latest_release()
    try:
        if not is_trade_day(args.date):
            payload = {"date": args.date, "generatedAt": datetime.now().astimezone().isoformat(), "status": "skipped", "reason": "not_trade_day"}
        elif args.phase == "preflight":
            payload = preflight(config, release, args.date)
        elif args.phase == "open":
            payload = open_decision(config, release, args.date)
        elif args.phase == "tail":
            payload = tail_decision(config, release, args.date)
        else:
            payload = settle_account(args.date)
        atomic_json(STATE_DIR / "phases" / f"{args.date}_{args.phase}.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        atomic_json(STATE_DIR / "phases" / f"{args.date}_{args.phase}.json", {"date": args.date, "generatedAt": datetime.now().astimezone().isoformat(), "status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
