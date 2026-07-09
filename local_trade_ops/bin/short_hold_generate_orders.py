#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_env() -> dict[str, str]:
    ops_home = Path(__file__).resolve().parents[1]
    env_path = ops_home / "config" / "env.local"
    if not env_path.exists():
        env_path = ops_home / "config" / "env.local.example"
    env = parse_env(env_path)
    env.setdefault("OPS_HOME", str(ops_home))
    env.setdefault("PAPER_ENGINE_HOME", str(ops_home.parent / "paper_trading_system"))
    env.setdefault("SHORT_HOLD_STATE_DIR", str(ops_home / "state" / "short_hold_single_peak"))
    env.setdefault("SHORT_HOLD_REPORT_DIR", str(ops_home / "reports" / "short_hold_single_peak"))
    env.setdefault("SHORT_HOLD_TASK_DIR", str(ops_home / "short_hold_tasks"))
    env.setdefault("SHORT_HOLD_SUBMISSION_DIR", str(ops_home / "short_hold_submissions"))
    env.setdefault("SHORT_HOLD_FILL_DIR", str(ops_home / "short_hold_fills"))
    return env


def setup_imports(env: dict[str, str]) -> None:
    ops_home = Path(env["OPS_HOME"]).expanduser().resolve()
    paper_src = Path(env["PAPER_ENGINE_HOME"]).expanduser().resolve() / "src"
    qlib_src = env.get("QLIB_SRC", "").strip()
    if qlib_src:
        sys.path.insert(0, str(Path(qlib_src).expanduser().resolve()))
    sys.path.insert(0, str(ops_home))
    sys.path.insert(0, str(paper_src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate short-hold single-peak manual order ticket.")
    parser.add_argument("--signal-date", default=None)
    parser.add_argument("--execution-date", default=None)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--task-dir", default=None)
    parser.add_argument("--submission-dir", default=None)
    parser.add_argument("--fill-dir", default=None)
    return parser.parse_args()


def account_path(state_dir: Path) -> Path:
    return state_dir / "paper_account.json"


def positions_path(state_dir: Path) -> Path:
    return state_dir / "paper_positions.csv"


def nav_path(state_dir: Path) -> Path:
    return state_dir / "paper_nav.csv"


def load_account(state_dir: Path, capital: float) -> dict:
    path = account_path(state_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"initial_capital": float(capital), "cash": float(capital), "last_nav": float(capital), "total_cost": 0.0}


def load_positions(state_dir: Path) -> pd.DataFrame:
    path = positions_path(state_dir)
    cols = ["instrument", "shares", "cost_basis", "entry_date", "exit_date", "signal_date", "score"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for col in cols:
        if col not in df.columns:
            df[col] = "" if col.endswith("_date") or col == "instrument" else 0
    df["instrument"] = df["instrument"].astype(str)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce").fillna(0.0).astype(float)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).astype(float)
    return df.loc[df["shares"] > 0, cols].reset_index(drop=True)


def save_initial_nav(state_dir: Path, account: dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    account_path(state_dir).write_text(json.dumps(account, indent=2, sort_keys=True), encoding="utf-8")
    path = nav_path(state_dir)
    if not path.exists():
        pd.DataFrame(
            [
                {
                    "cash": account["cash"],
                    "position_value": 0.0,
                    "nav": account["last_nav"],
                    "trade_cost": 0.0,
                    "datetime": "",
                    "phase": "init",
                }
            ]
        ).to_csv(path, index=False)


def softmax_weights(scores: pd.Series, temperature: float) -> pd.Series:
    if scores.empty:
        return scores
    temp = max(float(temperature), 1e-6)
    values = scores.astype(float).to_numpy() / temp
    values = values - np.nanmax(values)
    weights = np.exp(values)
    total = np.nansum(weights)
    if not np.isfinite(total) or total <= 0:
        return pd.Series(np.repeat(1.0 / len(scores), len(scores)), index=scores.index)
    return pd.Series(weights / total, index=scores.index)


def round_lot_shares(target_value: float, price: float, lot_size: int) -> int:
    if target_value <= 0 or price <= 0:
        return 0
    return int(math.floor(target_value / (price * lot_size)) * lot_size)


def model_filter_reason(
    inst: str,
    close_row: pd.Series | None,
    amount_window: pd.DataFrame,
    security_flags: pd.DataFrame,
    min_price: float,
    max_price: float,
    min_amount: float,
    exclude_restricted: bool,
    filter_st: bool,
    filter_paused: bool,
) -> tuple[bool, str, float, float]:
    reasons: list[str] = []
    price = float("nan")
    amount_avg = float("nan")

    if close_row is None or inst not in close_row.index or pd.isna(close_row.get(inst)):
        reasons.append("缺少信号日收盘价")
    else:
        price = float(close_row.get(inst))
        if price < min_price:
            reasons.append(f"收盘价 {price:.2f} < 最低价 {min_price:g}")
        if price > max_price:
            reasons.append(f"收盘价 {price:.2f} > 最高价 {max_price:g}")

    if inst not in amount_window.columns:
        reasons.append("缺少成交额窗口")
    else:
        amount_avg = float(pd.to_numeric(amount_window[inst], errors="coerce").mean())
        if not np.isfinite(amount_avg):
            reasons.append("20日均成交额无效")
        elif amount_avg < min_amount:
            reasons.append(f"20日均成交额 {amount_avg / 10000:.0f}万 < {min_amount / 10000:.0f}万")

    if not permission_allowed(inst, exclude_restricted):
        reasons.append("账户权限过滤：创业板/科创板")

    if not security_flags.empty:
        matched = security_flags.loc[security_flags["instrument"].astype(str) == inst]
        if not matched.empty:
            row = matched.iloc[0]
            if filter_st and bool(row.get("is_st", False)):
                reasons.append("风险过滤：ST/*ST")
            if filter_paused and bool(row.get("paused", False)):
                reasons.append("风险过滤：停牌")

    return not reasons, "通过" if not reasons else "；".join(reasons), price, amount_avg


def permission_allowed(inst: str, exclude_restricted: bool) -> bool:
    if not exclude_restricted:
        return True
    # Default to main-board names only for manual accounts without ChiNext/STAR permissions.
    return not (inst.startswith("SZ300") or inst.startswith("SZ301") or inst.startswith("SH688"))


def _order_score_column(candidates: pd.DataFrame) -> str:
    return "final_score" if "final_score" in candidates.columns else "score"


def _v3_extra_values(pick: pd.Series) -> dict[str, object]:
    extras: dict[str, object] = {}
    for column in [
        "return_score",
        "buyability_risk",
        "strong_prob",
        "liquidity_risk",
        "final_score",
        "score_source",
    ]:
        if column in pick.index:
            extras[column] = pick.get(column)
    return extras


def _load_side_model(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _apply_v3_scoring(
    candidates: pd.DataFrame,
    provider_uri: str,
    signal_date: pd.Timestamp,
    side_model_dir: Path,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    from bin.short_hold_v3_features import build_inference_features, load_qlib_ohlcv
    from bin.short_hold_v3_scoring import score_candidates_v3, sort_candidates_for_orders
    from paper_trading_daily import init_qlib

    if candidates.empty:
        return candidates.copy()

    buyability_path = side_model_dir / "buyability_model.pkl"
    strong_path = side_model_dir / "strong_model.pkl"
    missing = [str(path) for path in [buyability_path, strong_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"SHORT_HOLD_V3_SIDE_MODEL_DIR missing artifacts: {missing}")

    instruments = candidates["instrument"].astype(str).tolist()
    start_date = pd.Timestamp(signal_date) - pd.offsets.BDay(60)
    init_qlib(provider_uri)
    ohlcv = load_qlib_ohlcv(instruments, start_date, signal_date)
    features = build_inference_features(candidates, ohlcv, signal_date)
    if features.empty:
        return candidates.copy()

    buyability_model = _load_side_model(buyability_path)
    strong_model = _load_side_model(strong_path)
    side_scores = pd.DataFrame(
        {
            "instrument": features["instrument"].astype(str),
            "buyability_risk": buyability_model.predict_proba(features)[:, 1],
            "strong_prob": strong_model.predict_proba(features)[:, 1],
        }
    )
    scored = score_candidates_v3(features, side_scores, alpha=alpha, beta=beta, gamma=gamma)
    return sort_candidates_for_orders(scored)


def main() -> int:
    args = parse_args()
    env = load_env()
    setup_imports(env)

    from paper_trading_daily import (
        apply_jq_filter,
        choose_signal_date,
        fuse_three,
        load_close_amount_window,
        load_pred,
        load_raw_prices,
        load_security_flags,
        next_trade_date,
        ranked_signal,
    )

    state_dir = Path(args.state_dir or env["SHORT_HOLD_STATE_DIR"]).expanduser().resolve()
    task_dir = Path(args.task_dir or env["SHORT_HOLD_TASK_DIR"]).expanduser().resolve()
    submission_dir = Path(args.submission_dir or env["SHORT_HOLD_SUBMISSION_DIR"]).expanduser().resolve()
    fill_dir = Path(args.fill_dir or env["SHORT_HOLD_FILL_DIR"]).expanduser().resolve()
    for directory in [state_dir, task_dir, submission_dir, fill_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    capital = float(env.get("SHORT_HOLD_CAPITAL", env.get("CAPITAL", "100000")))
    topk = int(env.get("SHORT_HOLD_TOPK", "4"))
    min_amount = float(env.get("SHORT_HOLD_MIN_AMOUNT", "30000000"))
    max_position_pct = float(env.get("SHORT_HOLD_MAX_POSITION_PCT", "0.70"))
    reserve_cash_pct = float(env.get("SHORT_HOLD_RESERVE_CASH_PCT", "0.02"))
    buy_budget_mode = env.get("SHORT_HOLD_BUY_BUDGET_MODE", env.get("BUY_BUDGET_MODE", "capital_pool")).strip().lower()
    score_temperature = float(env.get("SHORT_HOLD_SCORE_TEMPERATURE", "1.0"))
    backup_count = int(env.get("SHORT_HOLD_BACKUP_COUNT", "3"))
    scoring_mode = env.get("SHORT_HOLD_SCORING_MODE", "v2").strip().lower()
    v3_side_model_dir = Path(
        env.get(
            "SHORT_HOLD_V3_SIDE_MODEL_DIR",
            str(Path(env["OPS_HOME"]) / "model_packages" / "short_hold_v3_side_models"),
        )
    ).expanduser().resolve()
    v3_alpha = float(env.get("SHORT_HOLD_V3_ALPHA", "1.0"))
    v3_beta = float(env.get("SHORT_HOLD_V3_BETA", "2.0"))
    v3_gamma = float(env.get("SHORT_HOLD_V3_GAMMA", "0.0"))
    exclude_restricted_markets = env.get("SHORT_HOLD_EXCLUDE_RESTRICTED_MARKETS", "1").strip() not in {"0", "false", "False", "no", "NO"}
    filter_st = env.get("SHORT_HOLD_FILTER_ST", env.get("FILTER_ST", "1")).strip() not in {"0", "false", "False", "no", "NO"}
    filter_paused = env.get("SHORT_HOLD_FILTER_PAUSED", env.get("FILTER_PAUSED", "1")).strip() not in {"0", "false", "False", "no", "NO"}
    lot_size = int(env.get("LOT_SIZE", "100"))
    scan_topk = int(env.get("SHORT_HOLD_SCAN_TOPK", "160"))
    provider_uri = env["QLIB_PROVIDER_URI"]
    weights = tuple(float(x) for x in env.get("WEIGHTS", "3,1,1").split(","))
    if len(weights) != 3:
        raise ValueError("WEIGHTS must be like 3,1,1")

    pred = fuse_three(
        load_pred(Path(env["PRED_A"]).expanduser().resolve()),
        load_pred(Path(env["PRED_B"]).expanduser().resolve()),
        load_pred(Path(env["PRED_C"]).expanduser().resolve()),
        weights,  # type: ignore[arg-type]
    )
    signal_date = choose_signal_date(pred, args.signal_date)
    execution_date = next_trade_date(provider_uri, signal_date, args.execution_date)
    exit_date = next_trade_date(provider_uri, execution_date, None)

    account = load_account(state_dir, capital)
    save_initial_nav(state_dir, account)
    positions = load_positions(state_dir)
    signal = ranked_signal(pred, signal_date)

    candidates = signal.head(max(scan_topk, topk * 20)).copy()
    candidates["model_rank"] = candidates["rank"].astype(int)
    instruments = set(candidates["instrument"].astype(str))
    if not positions.empty:
        instruments |= set(positions["instrument"].astype(str))

    start_date = signal_date - pd.offsets.BDay(max(int(env.get("LOOKBACK", "20")) * 2, 10))
    close_px, amount = load_close_amount_window(provider_uri, instruments, start_date, signal_date)
    close_row = close_px.loc[signal_date] if signal_date in close_px.index else None
    amount_window = amount.loc[:signal_date].tail(int(env.get("LOOKBACK", "20")))
    security_flags = load_security_flags(provider_uri, instruments, signal_date)
    min_price = float(env.get("MIN_PRICE", "5"))
    max_price = float(env.get("MAX_PRICE", "80"))

    top_model_reviews: list[dict[str, object]] = []
    for _, pick in signal.head(5).iterrows():
        inst = str(pick["instrument"])
        passed, reason, price, amount_avg = model_filter_reason(
            inst,
            close_row,
            amount_window,
            security_flags,
            min_price,
            max_price,
            min_amount,
            exclude_restricted_markets,
            filter_st,
            filter_paused,
        )
        top_model_reviews.append(
            {
                "signal_date": signal_date.date(),
                "execution_date": execution_date.date(),
                "instrument": inst,
                "model_rank": int(pick["rank"]),
                "score": float(pick["score"]),
                "close": price if np.isfinite(price) else "",
                "amount20": amount_avg if np.isfinite(amount_avg) else "",
                "filter_status": "PASS" if passed else "FILTERED",
                "filter_reason": reason,
            }
        )
    if signal_date in close_px.index:
        candidates = apply_jq_filter(
            candidates,
            close_px.loc[signal_date],
            amount_window,
            min_price,
            max_price,
            min_amount,
            security_flags,
            filter_st,
            filter_paused,
        )
    candidates = candidates.loc[
        candidates["instrument"].astype(str).map(lambda inst: permission_allowed(inst, exclude_restricted_markets))
    ].copy()
    if scoring_mode not in {"v2", "v3"}:
        raise ValueError("SHORT_HOLD_SCORING_MODE must be v2 or v3")
    if scoring_mode == "v3":
        candidates = _apply_v3_scoring(
            candidates,
            provider_uri,
            signal_date,
            v3_side_model_dir,
            v3_alpha,
            v3_beta,
            v3_gamma,
        )

    prices = load_raw_prices(provider_uri, instruments, signal_date, "close")
    price_map = prices.set_index("instrument")["price"].to_dict()
    held = set(positions["instrument"].astype(str)) if not positions.empty else set()

    rows: list[dict[str, object]] = []
    timeline_note = (
        f"{signal_date.date()} 信号；"
        f"{execution_date.date()} 开盘人工买；"
        f"{exit_date.date()} 尾盘/收盘卖"
    )
    for _, pos in positions.iterrows():
        if str(pos.get("exit_date", "")) and pd.Timestamp(pos["exit_date"]) <= execution_date:
            inst = str(pos["instrument"])
            rows.append(
                {
                    "signal_date": pos.get("signal_date") or signal_date.date(),
                    "execution_date": execution_date.date(),
                    "exit_date": pos.get("exit_date") or execution_date.date(),
                    "instrument": inst,
                    "action": "SELL",
                    "shares": int(pos["shares"]),
                    "estimated_price": float(price_map.get(inst, pos["cost_basis"])),
                    "reason": "short_hold_exit_due",
                    "order_role": "primary",
                    "model_rank": "",
                    "score": float(pos.get("score", 0.0) or 0.0),
                    "operator_note": f"{execution_date.date()} 到期卖出，优先完成",
                }
            )

    cash = float(account.get("cash", capital))
    nav = float(account.get("last_nav", cash))
    if buy_budget_mode in {"capital_pool", "capital", "total_capital"}:
        buy_budget = max(capital, 0.0)
        position_cap_base = max(capital, 0.0)
    elif buy_budget_mode in {"capital_pool_reserved", "capital_reserved"}:
        buy_budget = max(capital * (1.0 - reserve_cash_pct), 0.0)
        position_cap_base = max(capital, 0.0)
    elif buy_budget_mode in {"reserve_nav", "available_cash_reserved"}:
        buy_budget = max(cash - nav * reserve_cash_pct, 0.0)
        position_cap_base = max(nav, 0.0)
    elif buy_budget_mode == "available_cash":
        buy_budget = max(cash, 0.0)
        position_cap_base = max(nav, 0.0)
    else:
        raise ValueError(
            "SHORT_HOLD_BUY_BUDGET_MODE must be one of "
            "capital_pool, capital_pool_reserved, available_cash, available_cash_reserved"
        )
    available_candidates = candidates.loc[~candidates["instrument"].astype(str).isin(held)].copy()
    if scoring_mode == "v3":
        from bin.short_hold_v3_scoring import sort_candidates_for_orders

        available_candidates = sort_candidates_for_orders(available_candidates)
    else:
        available_candidates = available_candidates.sort_values("score", ascending=False).reset_index(drop=True)
    buy_candidates = available_candidates.head(topk).copy()
    extra_backup_candidates = available_candidates.iloc[topk:].copy()
    order_score_column = _order_score_column(buy_candidates)
    weights_by_inst = softmax_weights(buy_candidates.set_index("instrument")[order_score_column], score_temperature)
    candidate_reviews: list[dict[str, object]] = []
    promoted_backup_instruments: set[str] = set()
    promoted_backup_picks: list[pd.Series] = []
    primary_order_values: list[float] = []
    for _, pick in buy_candidates.iterrows():
        inst = str(pick["instrument"])
        price = float(price_map.get(inst, 0.0))
        weight = float(weights_by_inst.get(inst, 0.0))
        target_value = min(buy_budget * weight, position_cap_base * max_position_pct)
        shares = round_lot_shares(target_value, price, lot_size)
        skip_reason = ""
        if price <= 0:
            skip_reason = "无有效参考价"
        elif shares <= 0:
            min_lot_value = price * lot_size
            skip_reason = f"softmax 分配金额 {target_value:.2f} 不足一手 {min_lot_value:.2f}"
        candidate_reviews.append(
            {
                "signal_date": signal_date.date(),
                "execution_date": execution_date.date(),
                "exit_date": exit_date.date(),
                "instrument": inst,
                "model_rank": int(pick.get("model_rank", pick["rank"])),
                "score": float(pick["score"]),
                **_v3_extra_values(pick),
                "estimated_price": price,
                "softmax_weight": weight,
                "target_value": target_value,
                "planned_shares": max(shares, 0),
                "planned_value": max(shares, 0) * price,
                "status": "ORDERED" if shares > 0 else "BACKUP",
                "order_role": "primary" if shares > 0 else "backup",
                "reason": skip_reason or "进入短持有下单清单",
            }
        )
        if shares <= 0:
            promoted_backup_instruments.add(inst)
            promoted_backup_picks.append(pick)
            continue
        primary_order_values.append(shares * price)
        rows.append(
            {
                "signal_date": signal_date.date(),
                "execution_date": execution_date.date(),
                "exit_date": exit_date.date(),
                "instrument": inst,
                "action": "BUY",
                "shares": shares,
                "estimated_price": price,
                "planned_value": shares * price,
                "reason": "short_hold_top_rank",
                "order_role": "primary",
                "model_rank": int(pick.get("model_rank", pick["rank"])),
                "score": float(pick["score"]),
                **_v3_extra_values(pick),
                "operator_note": timeline_note,
            }
        )

    fallback_budget = min(buy_budget, position_cap_base * max_position_pct)
    released_primary_budget = min(sum(primary_order_values) if primary_order_values else fallback_budget, fallback_budget)
    backup_pick_rows: list[pd.Series] = promoted_backup_picks[: max(backup_count, 0)]
    if len(backup_pick_rows) < max(backup_count, 0):
        for _, pick in extra_backup_candidates.iterrows():
            inst = str(pick["instrument"])
            if inst in promoted_backup_instruments:
                continue
            backup_pick_rows.append(pick)
            if len(backup_pick_rows) >= max(backup_count, 0):
                break

    backup_df = pd.DataFrame(backup_pick_rows)
    backup_weights_by_inst = (
        softmax_weights(backup_df.set_index("instrument")[_order_score_column(backup_df)], score_temperature)
        if not backup_df.empty
        else pd.Series(dtype=float)
    )
    for _, pick in backup_df.iterrows():
        inst = str(pick["instrument"])
        price = float(price_map.get(inst, 0.0))
        backup_weight = float(backup_weights_by_inst.get(inst, 0.0))
        backup_target_value = min(released_primary_budget * backup_weight, position_cap_base * max_position_pct)
        shares = round_lot_shares(backup_target_value, price, lot_size)
        skip_reason = ""
        if price <= 0:
            skip_reason = "无有效参考价"
        elif shares <= 0:
            min_lot_value = price * lot_size
            skip_reason = f"backup softmax 分配金额 {backup_target_value:.2f} 不足一手 {min_lot_value:.2f}"
        updated_existing_review = False
        for review in candidate_reviews:
            if review.get("instrument") == inst:
                review["softmax_weight"] = backup_weight
                review["target_value"] = backup_target_value
                review["backup_released_budget"] = released_primary_budget
                review["backup_planned_shares"] = max(shares, 0)
                review["backup_planned_value"] = max(shares, 0) * price
                review["planned_shares"] = max(shares, 0)
                review["planned_value"] = max(shares, 0) * price
                review["status"] = "BACKUP" if shares > 0 else "SKIPPED"
                review["reason"] = skip_reason or "TOP 内分配不足一手，转为 backup 组合；按释放主单资金重新 softmax"
                updated_existing_review = True
                break
        if not updated_existing_review:
            candidate_reviews.append(
                {
                    "signal_date": signal_date.date(),
                    "execution_date": execution_date.date(),
                    "exit_date": exit_date.date(),
                    "instrument": inst,
                    "model_rank": int(pick.get("model_rank", pick["rank"])),
                    "score": float(pick["score"]),
                    **_v3_extra_values(pick),
                    "estimated_price": price,
                    "softmax_weight": backup_weight,
                    "target_value": backup_target_value,
                    "backup_released_budget": released_primary_budget,
                    "planned_shares": max(shares, 0),
                    "planned_value": max(shares, 0) * price,
                    "backup_planned_shares": max(shares, 0),
                    "backup_planned_value": max(shares, 0) * price,
                    "status": "BACKUP" if shares > 0 else "SKIPPED",
                    "order_role": "backup",
                    "reason": skip_reason or "backup 组合：仅 primary 买不进时使用；按释放主单资金重新 softmax",
                }
            )
        if shares <= 0:
            continue
        rows.append(
            {
                "signal_date": signal_date.date(),
                "execution_date": execution_date.date(),
                "exit_date": exit_date.date(),
                "instrument": inst,
                "action": "BUY",
                "shares": shares,
                "estimated_price": price,
                "planned_value": shares * price,
                "replacement_budget": released_primary_budget,
                "softmax_weight": backup_weight,
                "target_value": backup_target_value,
                "reason": "short_hold_backup_rank",
                "order_role": "backup",
                "model_rank": int(pick.get("model_rank", pick["rank"])),
                "score": float(pick["score"]),
                **_v3_extra_values(pick),
                "operator_note": f"backup 组合：仅 primary 买不进时使用；按释放主单资金重新 softmax；{timeline_note}",
            }
        )

    orders = pd.DataFrame(rows)
    if orders.empty:
        raise RuntimeError("No short-hold orders generated. Check cash, positions, filters, and predictions.")

    order_path = state_dir / f"pending_orders_{execution_date.date()}.csv"
    approved_path = state_dir / f"approved_orders_{execution_date.date()}.csv"
    task_path = task_dir / f"short_hold_order_task_{execution_date.date()}.csv"
    ticket_path = submission_dir / f"short_hold_broker_ticket_{execution_date.date()}.csv"
    fill_path = fill_dir / f"short_hold_fill_template_{execution_date.date()}.csv"
    candidate_review_path = state_dir / f"short_hold_candidate_review_{execution_date.date()}.csv"
    top_review_path = state_dir / f"short_hold_model_top_review_{execution_date.date()}.csv"
    status_path = state_dir / f"approval_status_{execution_date.date()}.json"
    signal_path = state_dir / f"signal_rankings_{signal_date.date()}.csv"

    orders.to_csv(order_path, index=False)
    orders.to_csv(approved_path, index=False)
    orders.to_csv(task_path, index=False)
    orders.to_csv(ticket_path, index=False)
    fill_template = orders.copy()
    fill_template["fill_status"] = "PENDING"
    fill_template["fill_shares"] = ""
    fill_template["fill_price"] = ""
    fill_template["fee"] = ""
    fill_template.to_csv(fill_path, index=False)
    pd.DataFrame(top_model_reviews).to_csv(top_review_path, index=False)
    pd.DataFrame(candidate_reviews).to_csv(candidate_review_path, index=False)
    signal.to_csv(signal_path, index=False)
    status_path.write_text(
        json.dumps(
            {
                "signal_date": str(signal_date.date()),
                "execution_date": str(execution_date.date()),
                "exit_date": str(exit_date.date()),
                "timeline_note": timeline_note,
                "status": "auto_approved",
                "strategy": "short_hold_single_peak",
                "scoring_mode": scoring_mode,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Signal date: {signal_date.date()}")
    print(f"Execution date: {execution_date.date()}")
    print(f"Exit date: {exit_date.date()}")
    print(f"Orders: {order_path}")
    print(f"Ticket: {ticket_path}")
    print(f"Fill template: {fill_path}")
    print(f"Model top review: {top_review_path}")
    print(f"Candidate review: {candidate_review_path}")
    print(orders.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
