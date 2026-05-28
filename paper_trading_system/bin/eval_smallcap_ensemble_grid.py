from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_smallcap_models import (  # noqa: E402
    capital_sim,
    first_seen_dates,
    load_list_dates,
    load_market_data,
    load_pred,
    model_name,
    simulate_topdrop,
)


DEFAULT_MODELS = "lgb_long,xgb_long,catboost_long,doubleensemble_short"
DEFAULT_COMBO_SIZES = "1,2,3,4"
DEFAULT_TOPK_PAIRS = "5:1,7:1,10:1,15:1,20:2"
DEFAULT_CAPITALS = "100000,200000,500000,1000000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search small-cap model ensembles across combos, weights, top/drop, and capital.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--benchmark", default="SH000852")
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="2026-05-18")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--combo-sizes", default=DEFAULT_COMBO_SIZES)
    parser.add_argument("--topk-pairs", default=DEFAULT_TOPK_PAIRS)
    parser.add_argument("--capitals", default=DEFAULT_CAPITALS)
    parser.add_argument(
        "--weight-patterns",
        default="",
        help="Optional semicolon-separated weight tuples, e.g. '1;1,1;2,1;1,2;1,1,1;2,1,1;1,2,1'. "
        "Patterns are matched by combo size; when omitted, built-in defaults are used.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--capital-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=3e7)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-list-days", type=int, default=120)
    parser.add_argument("--max-open-gap", type=float, default=0.05)
    parser.add_argument("--max-volatility", type=float, default=0.08)
    parser.add_argument("--max-order-amount-pct", type=float, default=0.005)
    parser.add_argument("--list-date-csv", default="")
    return parser.parse_args()


def daily_zscore(score: pd.Series) -> pd.Series:
    def _z(group: pd.Series) -> pd.Series:
        std = group.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return group * 0.0
        return (group - group.mean()) / std

    return score.groupby(level="datetime", group_keys=False).apply(_z)


def default_weight_patterns(size: int) -> list[tuple[float, ...]]:
    if size == 1:
        return [(1.0,)]
    base = [(1.0,) * size]
    # Put more weight on the first model in each ordered combo. We sort combos with
    # the strongest prior model first via --models, so this creates intuitive grids.
    for lead in (2.0, 3.0):
        base.append((lead,) + (1.0,) * (size - 1))
    if size == 2:
        base.append((1.0, 2.0))
    return base


def parse_weight_patterns(spec: str) -> dict[int, list[tuple[float, ...]]]:
    patterns: dict[int, list[tuple[float, ...]]] = {}
    if not spec.strip():
        return patterns

    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = tuple(float(x.strip()) for x in chunk.split(",") if x.strip())
        if not parts:
            continue
        patterns.setdefault(len(parts), [])
        if parts not in patterns[len(parts)]:
            patterns[len(parts)].append(parts)
    return patterns


def weight_patterns(size: int, custom_patterns: dict[int, list[tuple[float, ...]]]) -> list[tuple[float, ...]]:
    return custom_patterns.get(size, default_weight_patterns(size))


def fuse_predictions(preds: dict[str, pd.DataFrame], combo: tuple[str, ...], weights: tuple[float, ...]) -> pd.DataFrame:
    denom = sum(weights)
    pieces = []
    for model, weight in zip(combo, weights, strict=True):
        pieces.append(daily_zscore(preds[model]["score"]) * (weight / denom))
    score = sum(pieces)
    return score.to_frame("score").sort_index()


def fmt_weights(combo: tuple[str, ...], weights: tuple[float, ...]) -> str:
    return ",".join(f"{m}:{int(w) if float(w).is_integer() else w:g}" for m, w in zip(combo, weights, strict=True))


def main() -> None:
    args = parse_args()
    pred_paths = sorted(Path(args.pred_dir).glob("*.pkl"))
    if not pred_paths:
        raise FileNotFoundError(f"No predictions in {args.pred_dir}")

    all_preds = {model_name(p, args.market): load_pred(p) for p in pred_paths}
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    missing = [m for m in models if m not in all_preds]
    if missing:
        raise KeyError(f"Missing models in pred-dir: {missing}")

    combo_sizes = [int(x) for x in args.combo_sizes.split(",") if x.strip()]
    pairs = [tuple(int(x) for x in item.split(":")) for item in args.topk_pairs.split(",") if item.strip()]
    capitals = [float(x) for x in args.capitals.split(",") if x.strip()]
    custom_weight_patterns = parse_weight_patterns(args.weight_patterns)

    first = all_preds[models[0]]
    first_seen = load_list_dates(args.list_date_csv) or first_seen_dates(first)
    dates = pd.DatetimeIndex(sorted(first.index.get_level_values("datetime").unique()))
    dates = dates[(dates >= pd.Timestamp(args.start_date)) & (dates <= pd.Timestamp(args.end_date))]
    instruments = sorted(first.index.get_level_values("instrument").unique())
    open_px, close_px, amount, bench_open = load_market_data(args.provider_uri, instruments, str(dates[0].date()), str(dates[-1].date()), args.benchmark)
    bench = bench_open.reindex(dates[1:]).dropna()
    benchmark_return = float(bench.iloc[-1] / bench.iloc[0] - 1.0)

    topdrop_rows = []
    capital_rows = []
    for size in combo_sizes:
        for combo in itertools.combinations(models, size):
            for weights in weight_patterns(size, custom_weight_patterns):
                pred = fuse_predictions(all_preds, combo, weights)
                combo_name = "+".join(combo)
                weights_text = fmt_weights(combo, weights)
                for topk, n_drop in pairs:
                    for filter_name, use_filter in [("raw", False), ("jq_filter", True), ("jq_filter_v1", True)]:
                        result = simulate_topdrop(pred, dates, open_px, close_px, amount, topk, n_drop, use_filter, args, filter_name, first_seen)
                        row = {
                            "combo": combo_name,
                            "weights": weights_text,
                            "filter": filter_name,
                            "topk": topk,
                            "n_drop": n_drop,
                            **result,
                            "benchmark_return": benchmark_return,
                        }
                        row["excess_return"] = row["cum_return"] - benchmark_return
                        topdrop_rows.append(row)
                        for capital in capitals:
                            cap = capital_sim(pred, dates, open_px, close_px, amount, topk, n_drop, capital, use_filter, args, filter_name, first_seen)
                            cap_row = {
                                "combo": combo_name,
                                "weights": weights_text,
                                "filter": filter_name,
                                "topk": topk,
                                "n_drop": n_drop,
                                **cap,
                                "benchmark_return": benchmark_return,
                            }
                            cap_row["excess_return"] = cap_row["cum_return"] - benchmark_return
                            capital_rows.append(cap_row)

    topdrop = pd.DataFrame(topdrop_rows).sort_values(["excess_return", "ir"], ascending=False)
    capital = pd.DataFrame(capital_rows).sort_values(["excess_return", "capital"], ascending=[False, True])

    summary = (
        capital.groupby(["combo", "weights", "filter", "topk", "n_drop"], as_index=False)
        .agg(
            avg_return=("cum_return", "mean"),
            min_return=("cum_return", "min"),
            avg_excess=("excess_return", "mean"),
            min_excess=("excess_return", "min"),
            worst_drawdown=("max_drawdown", "min"),
            avg_position_ratio=("avg_position_ratio", "mean"),
            min_position_ratio=("avg_position_ratio", "min"),
            avg_holdings=("final_holdings", "mean"),
            avg_cost=("total_cost", "mean"),
        )
        .sort_values(["min_excess", "avg_excess", "avg_position_ratio"], ascending=False)
    )

    for path, frame in [(args.output, topdrop), (args.capital_output, capital), (args.summary_output, summary)]:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)

    print("===== top infinite/equal-weight screen =====")
    print(topdrop.head(30).to_string(index=False))
    print("\n===== top capital rows =====")
    print(capital.head(40).to_string(index=False))
    print("\n===== stable summary across capitals =====")
    print(summary.head(40).to_string(index=False))
    print(f"\nWrote: {args.output}")
    print(f"Wrote: {args.capital_output}")
    print(f"Wrote: {args.summary_output}")


if __name__ == "__main__":
    main()
