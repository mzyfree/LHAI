#!/usr/bin/env python3
"""Cached strict intraday TOPK simulation from packaged Qlib models.

This runner avoids re-initializing Qlib DatasetH for every signal date.  It
loads each packaged model once, predicts the full point-in-time test segment
once, writes per-day signal caches, then simulates:

    signal_date close -> next trading day open buy -> same day close sell

The current packaged Alpha158 configs use no infer processors, so feature rows
are point-in-time after the Qlib expression loader has produced them.  We still
remove learn processors during inference to avoid label-driven filtering.
"""

from __future__ import annotations

import argparse
import copy
import math
import pickle
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.utils import init_instance_by_config


DEFAULT_MODELS = {
    "xgb_csi1000_long_prod2026": 3.0,
    "doubleensemble_csi1000_short_prod2026": 1.0,
    "catboost_csi1000_long_prod2026": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached strict intraday TOPK simulation from Qlib model package.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--extract-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="2026-05-26")
    parser.add_argument("--topks", default="3,5")
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--buy-cost-rate", type=float, default=0.0003)
    parser.add_argument("--sell-cost-rate", type=float, default=0.0008)
    parser.add_argument("--force-predict", action="store_true", help="Ignore cached model predictions and regenerate them.")
    return parser.parse_args()


def ensure_extracted(package_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in extract_dir.iterdir() if p.is_dir() and p.name.startswith("csi1000_main_")]
    if candidates:
        return sorted(candidates)[-1]

    if not package_path.exists():
        raise FileNotFoundError(package_path)

    with tarfile.open(package_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    candidates = [p for p in extract_dir.iterdir() if p.is_dir() and p.name.startswith("csi1000_main_")]
    if not candidates:
        raise RuntimeError(f"No csi1000_main_* directory found after extracting: {package_path}")
    return sorted(candidates)[-1]


def load_task(model_dir: Path) -> dict:
    task_path = model_dir / "artifacts" / "task"
    with task_path.open("rb") as f:
        task = pickle.load(f)
    if not isinstance(task, dict) or "dataset" not in task:
        raise RuntimeError(f"Unexpected task artifact: {task_path}")
    return task


def validate_infer_config(task: dict, model_name: str) -> None:
    handler_kwargs = task["dataset"]["kwargs"]["handler"]["kwargs"]
    infer_processors = handler_kwargs.get("infer_processors", [])
    process_type = handler_kwargs.get("process_type", "append")
    if infer_processors:
        raise RuntimeError(
            f"{model_name} has infer_processors={infer_processors}; cached strict runner needs review before use."
        )
    print(f"{model_name}: infer_processors=[], process_type={process_type}")


def build_dataset(task: dict, start_date: str, end_date: str) -> Dataset:
    dataset_config = copy.deepcopy(task["dataset"])
    handler_kwargs = dataset_config["kwargs"]["handler"]["kwargs"]
    handler_kwargs["end_time"] = end_date
    # Live inference cannot know next-day labels.  The packaged Alpha158 configs
    # use DropnaLabel/label normalization only on learn data; remove them here.
    handler_kwargs["learn_processors"] = []
    dataset_config["kwargs"]["segments"]["test"] = [start_date, end_date]
    return init_instance_by_config(dataset_config, accept_types=Dataset)


def predict_model(model_root: Path, model_name: str, start_date: str, end_date: str, cache_dir: Path, force: bool) -> pd.DataFrame:
    pred_path = cache_dir / f"{model_name}_pred_{start_date}_{end_date}.pkl"
    if pred_path.exists() and not force:
        return pd.read_pickle(pred_path)

    model_dir = model_root / model_name
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    task = load_task(model_dir)
    validate_infer_config(task, model_name)
    dataset = build_dataset(task, start_date, end_date)

    with (model_dir / "artifacts" / "params.pkl").open("rb") as f:
        model = pickle.load(f)

    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    elif "score" not in pred.columns:
        if len(pred.columns) != 1:
            raise RuntimeError(f"Unexpected prediction columns for {model_name}: {list(pred.columns)}")
        pred = pred.rename(columns={pred.columns[0]: "score"})

    pred.index = pred.index.set_names(["datetime", "instrument"])
    pred = pred[["score"]].sort_index()
    pred.to_pickle(pred_path)
    print(f"Wrote model prediction cache: {pred_path}")
    return pred


def daily_zscore(score: pd.Series) -> pd.Series:
    def _z(group: pd.Series) -> pd.Series:
        std = group.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return group * 0.0
        return (group - group.mean()) / std

    return score.groupby(level="datetime", group_keys=False).apply(_z)


def fuse_predictions(preds: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    denom = sum(weights.values())
    pieces = []
    for model_name, pred in preds.items():
        pieces.append(daily_zscore(pred["score"]) * (weights[model_name] / denom))
    score = sum(pieces)
    return score.to_frame("score").sort_index()


def write_signal_cache(pred: pd.DataFrame, dates: pd.DatetimeIndex, cache_dir: Path) -> None:
    signal_dir = cache_dir / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    for signal_date in dates:
        path = signal_dir / f"signal_{signal_date.date()}.csv"
        if path.exists():
            continue
        day = pred.xs(signal_date, level="datetime").reset_index()
        day = day.sort_values("score", ascending=False).reset_index(drop=True)
        day["rank"] = day.index + 1
        day.loc[:, ["rank", "instrument", "score"]].to_csv(path, index=False)


def load_market_data(provider_uri: str, instruments: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fields = ["$open", "$close", "$factor", "$volume"]
    data = D.features(instruments, fields, start_time=start, end_time=end, freq="day")
    data = data.rename(columns={"$open": "open_adj", "$close": "close_adj", "$factor": "factor", "$volume": "volume"}).reset_index()
    data["datetime"] = pd.to_datetime(data["datetime"])
    valid = data["factor"].notna() & (data["factor"] > 0)
    data["open"] = data["open_adj"]
    data["close"] = data["close_adj"]
    data.loc[valid, "open"] = data.loc[valid, "open_adj"] / data.loc[valid, "factor"]
    data.loc[valid, "close"] = data.loc[valid, "close_adj"] / data.loc[valid, "factor"]
    data["amount"] = data["close"] * data["volume"]
    open_px = data.pivot(index="datetime", columns="instrument", values="open").sort_index()
    close_px = data.pivot(index="datetime", columns="instrument", values="close").sort_index()
    amount = data.pivot(index="datetime", columns="instrument", values="amount").sort_index()
    return open_px, close_px, amount


def safe_price(row: pd.Series, inst: str) -> float:
    try:
        value = float(row.get(inst, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())


def simulate_intraday(
    pred: pd.DataFrame,
    dates: pd.DatetimeIndex,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    amount: pd.DataFrame,
    topk: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    daily_returns: list[float] = []
    trade_days: list[pd.Timestamp] = []
    detail_rows: list[dict[str, object]] = []

    for i, signal_date in enumerate(dates[:-1]):
        trade_date = dates[i + 1]
        day = pred.xs(signal_date, level="datetime").reset_index()
        day = day.sort_values("score", ascending=False).reset_index(drop=True)
        day["rank"] = day.index + 1

        avg_amount = amount.iloc[max(0, i - args.lookback + 1) : i + 1].mean(axis=0, skipna=True)
        close_row = close_px.loc[signal_date]
        keep = []
        for inst in day["instrument"].astype(str):
            price = safe_price(close_row, inst)
            amt = float(avg_amount.get(inst, 0.0) or 0.0)
            keep.append(args.min_price <= price <= args.max_price and amt >= args.min_amount)
        day = day.loc[keep].reset_index(drop=True)
        picks = day.head(topk)["instrument"].astype(str).tolist()

        stock_returns = []
        for inst in picks:
            open_price = safe_price(open_px.loc[trade_date], inst)
            close_price = safe_price(close_px.loc[trade_date], inst)
            if open_price > 0 and close_price > 0:
                stock_returns.append(close_price / open_price - 1.0)

        gross_return = float(np.mean(stock_returns)) if stock_returns else 0.0
        cost = args.buy_cost_rate + args.sell_cost_rate if stock_returns else 0.0
        net_return = gross_return - cost
        daily_returns.append(net_return)
        trade_days.append(trade_date)

        detail_rows.append(
            {
                "topk": topk,
                "signal_date": signal_date.date(),
                "trade_date": trade_date.date(),
                "n_picks": len(stock_returns),
                "gross_return": gross_return,
                "net_return": net_return,
                "picks": ",".join(picks),
            }
        )

    ret = pd.Series(daily_returns, index=trade_days)
    nav = (1.0 + ret).cumprod()
    summary = {
        "topk": topk,
        "days": float(len(ret)),
        "cum_return": float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
        "ann_return": float((1.0 + ret.mean()) ** 252 - 1.0) if len(ret) else 0.0,
        "ir": float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) and ret.std() > 0 else float("nan"),
        "max_drawdown": max_drawdown(nav) if len(nav) else 0.0,
        "win_rate": float((ret > 0).mean()) if len(ret) else 0.0,
        "avg_daily_return": float(ret.mean()) if len(ret) else 0.0,
    }
    return summary, detail_rows


def main() -> int:
    args = parse_args()
    provider_uri = Path(args.provider_uri).resolve()
    package_path = Path(args.package).resolve()
    extract_dir = Path(args.extract_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_root = ensure_extracted(package_path, extract_dir)
    print(f"Qlib provider: {provider_uri}")
    print(f"Model root: {model_root}")
    print(f"Cache dir: {cache_dir}")

    qlib.init(provider_uri=str(provider_uri), region="cn")

    preds = {
        model_name: predict_model(model_root, model_name, args.start_date, args.end_date, cache_dir, args.force_predict)
        for model_name in DEFAULT_MODELS
    }
    pred = fuse_predictions(preds, DEFAULT_MODELS)

    dates = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    dates = dates[(dates >= pd.Timestamp(args.start_date)) & (dates <= pd.Timestamp(args.end_date))]
    instruments = sorted(pred.index.get_level_values("instrument").unique())
    print(f"Prediction dates: {dates[0].date()} -> {dates[-1].date()} ({len(dates)} dates)")
    print(f"Instruments: {len(instruments)}")

    write_signal_cache(pred, dates, cache_dir)
    open_px, close_px, amount = load_market_data(str(provider_uri), instruments, str(dates[0].date()), str(dates[-1].date()))

    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    summaries = []
    daily_rows = []
    for topk in topks:
        summary, rows = simulate_intraday(pred, dates, open_px, close_px, amount, topk, args)
        summaries.append(summary)
        daily_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    daily_df = pd.DataFrame(daily_rows)
    summary_path = output_dir / "strict_cached_intraday_topk_2026_summary.csv"
    daily_path = output_dir / "strict_cached_intraday_topk_2026_daily.csv"
    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)

    print("===== strict cached intraday topk summary =====")
    print(summary_df.to_string(index=False))
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {daily_path}")
    print(f"Wrote signals: {cache_dir / 'signals'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
