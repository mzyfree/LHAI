from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_smallcap_models import (  # noqa: E402
    capital_sim,
    load_market_data,
    load_pred,
    max_drawdown,
    model_name,
    simulate_topdrop,
)


DEFAULT_ENSEMBLES = (
    "lgb_xgb_long=lgb_long:1,xgb_long:1;"
    "lgb_xgb_long_2_1=lgb_long:2,xgb_long:1;"
    "lgb_cat_long=lgb_long:1,catboost_long:1;"
    "lgb_xgb_cat_long=lgb_long:1,xgb_long:1,catboost_long:1;"
    "tree_long=lgb_long:1,xgb_long:1,catboost_long:1,doubleensemble_long:1;"
    "lgb_long_mid=lgb_long:1,lgb_mid:1;"
    "xgb_long_mid=xgb_long:1,xgb_mid:1;"
    "best_four_mixed=lgb_long:2,xgb_long:1,catboost_long:1,doubleensemble_short:1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CSI small-cap prediction ensembles.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--benchmark", default="SH000852")
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="2026-05-18")
    parser.add_argument("--topk-pairs", default="7:1,10:1,15:1,20:2")
    parser.add_argument("--ensembles", default=DEFAULT_ENSEMBLES)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capital-output", required=True)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=3e7)
    parser.add_argument("--lookback", type=int, default=20)
    return parser.parse_args()


def daily_zscore(score: pd.Series) -> pd.Series:
    def _z(group: pd.Series) -> pd.Series:
        std = group.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return group * 0.0
        return (group - group.mean()) / std

    return score.groupby(level="datetime", group_keys=False).apply(_z)


def parse_ensembles(spec: str) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        name, body = item.split("=", 1)
        parts: list[tuple[str, float]] = []
        for token in body.split(","):
            model, weight = token.split(":", 1)
            parts.append((model.strip(), float(weight)))
        out[name.strip()] = parts
    return out


def fuse_predictions(preds: dict[str, pd.DataFrame], parts: list[tuple[str, float]]) -> pd.DataFrame:
    total_weight = sum(weight for _, weight in parts)
    if total_weight <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")

    merged: pd.DataFrame | None = None
    for model, weight in parts:
        if model not in preds:
            raise KeyError(f"Missing model prediction: {model}")
        score = daily_zscore(preds[model]["score"]).rename(model)
        weighted = score * (weight / total_weight)
        merged = weighted.to_frame("score") if merged is None else merged.join(weighted.to_frame("score"), how="inner", rsuffix="_r")
        if merged.shape[1] > 1:
            merged["score"] = merged.sum(axis=1)
            merged = merged[["score"]]
    if merged is None:
        raise ValueError("Empty ensemble.")
    return merged.sort_index()


def main() -> None:
    args = parse_args()
    pred_paths = sorted(Path(args.pred_dir).glob("*.pkl"))
    if not pred_paths:
        raise FileNotFoundError(f"No predictions in {args.pred_dir}")

    preds = {model_name(p, args.market): load_pred(p) for p in pred_paths}
    ensembles = parse_ensembles(args.ensembles)
    fused = {}
    for name, parts in ensembles.items():
        try:
            fused[name] = fuse_predictions(preds, parts)
        except KeyError as exc:
            print(f"skip {name}: {exc}", file=sys.stderr)

    if not fused:
        raise RuntimeError("No valid ensembles were built.")

    first = next(iter(fused.values()))
    dates = pd.DatetimeIndex(sorted(first.index.get_level_values("datetime").unique()))
    dates = dates[(dates >= pd.Timestamp(args.start_date)) & (dates <= pd.Timestamp(args.end_date))]
    instruments = sorted(first.index.get_level_values("instrument").unique())
    open_px, close_px, amount, bench_open = load_market_data(args.provider_uri, instruments, str(dates[0].date()), str(dates[-1].date()), args.benchmark)
    bench = bench_open.reindex(dates[1:]).dropna()
    bench_return = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
    pairs = [tuple(int(x) for x in item.split(":")) for item in args.topk_pairs.split(",") if item.strip()]

    rows = []
    for name, pred in fused.items():
        for topk, n_drop in pairs:
            for filter_name, use_filter in [("raw", False), ("jq_filter", True)]:
                result = simulate_topdrop(pred, dates, open_px, close_px, amount, topk, n_drop, use_filter, args)
                rows.append({"ensemble": name, "filter": filter_name, "topk": topk, "n_drop": n_drop, **result, "benchmark_return": bench_return})
    out = pd.DataFrame(rows)
    out["excess_return"] = out["cum_return"] - out["benchmark_return"]
    out = out.sort_values(["excess_return", "ir"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("===== top ensemble results =====")
    print(out.head(40).to_string(index=False))

    cap_rows = []
    for _, row in out.head(20).iterrows():
        pred = fused[row["ensemble"]]
        for capital in [100000.0, 200000.0, 1000000.0]:
            result = capital_sim(pred, dates, open_px, close_px, amount, int(row["topk"]), int(row["n_drop"]), capital, row["filter"] == "jq_filter", args)
            cap_rows.append({"ensemble": row["ensemble"], "filter": row["filter"], "topk": row["topk"], "n_drop": row["n_drop"], **result, "benchmark_return": bench_return})
    cap = pd.DataFrame(cap_rows)
    cap["excess_return"] = cap["cum_return"] - cap["benchmark_return"]
    cap = cap.sort_values(["excess_return", "capital"], ascending=[False, True])
    cap.to_csv(args.capital_output, index=False)
    print("===== top capital results =====")
    print(cap.head(60).to_string(index=False))
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.capital_output}")


if __name__ == "__main__":
    main()
