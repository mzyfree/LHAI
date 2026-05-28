from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow.record_temp import SignalRecord

from run_prediction_fusion import (
    backtest_metrics,
    daily_rank,
    daily_zscore,
    load_config,
    load_pred,
    signal_metrics,
)


DEFAULT_MODELS = [
    "lgb_alpha158",
    "linear",
    "tabnet",
    "lstm",
    "catboost",
    "adarnn",
    "xgboost",
]

DEFAULT_WEIGHT_SETS = [
    "1,1,1,1,1,1,1",
    "3,2,2,2,1,1,1",
    "3,2,2,1,1,1,1",
    "3,2,2,1,0,0,1",
    "3,2,2,1,0,1,0",
    "4,2,2,1,1,1,1",
    "4,3,2,1,1,1,1",
    "5,3,2,1,1,1,1",
]


def parse_csv(spec: str) -> list[str]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty comma-separated spec: {spec!r}")
    return values


def parse_weight_sets(spec: str, n_models: int) -> list[list[float]]:
    weight_sets = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        weights = [float(x.strip()) for x in item.split(",") if x.strip()]
        if len(weights) != n_models:
            raise ValueError(f"Weight set {weights} has {len(weights)} weights, expected {n_models}.")
        if sum(weights) <= 0:
            raise ValueError(f"Weight set must have positive sum: {weights}")
        weight_sets.append(weights)
    if not weight_sets:
        raise ValueError("No valid weight sets parsed.")
    return weight_sets


def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [w / total for w in weights]


def load_named_predictions(models: list[str], pred_dir: Path) -> dict[str, pd.DataFrame]:
    preds = {}
    missing = []
    for model in models:
        path = pred_dir / f"{model}_reference_long.pkl"
        if not path.exists():
            missing.append(str(path))
            continue
        preds[model] = load_pred(path)
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))
    return preds


def fuse_many(preds: dict[str, pd.DataFrame], method: str, weights: list[float]) -> pd.DataFrame:
    if len(preds) != len(weights):
        raise ValueError("Number of predictions and weights must match.")

    series = []
    for name, pred in preds.items():
        score = pred["score"].rename(name)
        if method == "mean":
            series.append(score)
        elif method == "rank_mean":
            series.append(daily_rank(score).rename(name))
        elif method == "zscore_mean":
            series.append(daily_zscore(score).rename(name))
        else:
            raise ValueError(f"Unsupported method: {method}")

    merged = pd.concat(series, axis=1, join="inner").dropna()
    if merged.empty:
        raise ValueError("No overlapping prediction rows after alignment.")

    norm_weights = normalize_weights(weights)
    fused_score = sum(merged[col] * weight for col, weight in zip(merged.columns, norm_weights))
    return fused_score.to_frame("score").sort_index()


def metric_value(analysis: dict[str, pd.DataFrame], bucket: str, metric: str) -> float:
    return float(analysis[bucket].loc[metric, "risk"])


def format_weights(models: list[str], weights: list[float]) -> str:
    return ",".join(f"{model}:{weight:g}" for model, weight in zip(models, weights))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search multi-model ensembles for reference_long predictions.")
    parser.add_argument(
        "--config",
        default="configs/reference_long/workflow_config_lgb_alpha158_reference_long.yaml",
        help="Workflow config used for labels and backtest.",
    )
    parser.add_argument(
        "--pred-dir",
        default="/root/autodl-tmp/llhh/preds",
        help="Directory containing *_reference_long.pkl predictions.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names. Order must match weight sets.",
    )
    parser.add_argument(
        "--methods",
        default="rank_mean,zscore_mean,mean",
        help="Comma-separated fusion methods.",
    )
    parser.add_argument(
        "--weight-sets",
        default=";".join(DEFAULT_WEIGHT_SETS),
        help="Semicolon-separated weight sets. Each set is comma-separated and must match --models length.",
    )
    parser.add_argument(
        "--sort-by",
        default="annualized_return",
        choices=["annualized_return", "information_ratio", "max_drawdown", "Rank IC", "IC"],
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of rows to include in markdown top table.")
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/llhh/reports/reference_long_ensemble_search.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--csv-output",
        default="/root/autodl-tmp/llhh/reports/reference_long_ensemble_search.csv",
        help="CSV result path.",
    )
    parser.add_argument(
        "--save-best-pred",
        default="/root/autodl-tmp/llhh/preds/ensemble_reference_long_best.pkl",
        help="Where to save the best fused prediction.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    pred_dir = Path(args.pred_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    csv_output_path = Path(args.csv_output).expanduser().resolve()
    save_best_path = Path(args.save_best_pred).expanduser().resolve()

    models = parse_csv(args.models)
    methods = parse_csv(args.methods)
    weight_sets = parse_weight_sets(args.weight_sets, len(models))

    config = load_config(config_path)
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"], region=REG_CN)
    dataset = init_instance_by_config(config["task"]["dataset"])
    label = SignalRecord.generate_label(dataset)
    if label is None:
        raise ValueError("Failed to generate test label from dataset.")

    preds = load_named_predictions(models, pred_dir)

    rows = []
    best_row = None
    best_pred = None
    best_score = None

    for method, weights in itertools.product(methods, weight_sets):
        fused = fuse_many(preds, method=method, weights=weights)
        aligned_label = label.loc[fused.index]
        sig = signal_metrics(fused, aligned_label)
        # backtest_metrics/fill_placeholder mutates the strategy config by
        # replacing <PRED>; isolate each candidate from previous evaluations.
        analysis, _ = backtest_metrics(fused, copy.deepcopy(config["port_analysis_config"]))

        row = {
            "method": method,
            "weights": format_weights(models, weights),
            "IC": sig["IC"],
            "ICIR": sig["ICIR"],
            "Rank IC": sig["Rank IC"],
            "Rank ICIR": sig["Rank ICIR"],
            "annualized_return": metric_value(analysis, "excess_return_with_cost", "annualized_return"),
            "information_ratio": metric_value(analysis, "excess_return_with_cost", "information_ratio"),
            "max_drawdown": metric_value(analysis, "excess_return_with_cost", "max_drawdown"),
        }
        rows.append(row)

        current_score = row[args.sort_by]
        if best_score is None or current_score > best_score:
            best_score = current_score
            best_row = row
            best_pred = fused

    result_df = pd.DataFrame(rows).sort_values(args.sort_by, ascending=False).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_best_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(csv_output_path, index=False)
    if best_pred is not None:
        best_pred.to_pickle(save_best_path)

    report = f"""# Reference Long Ensemble Search

## Setup

- Config: `{config_path}`
- Pred dir: `{pred_dir}`
- Models: `{", ".join(models)}`
- Methods: `{", ".join(methods)}`
- Sort by: `{args.sort_by}`
- Best pred: `{save_best_path}`
- CSV: `{csv_output_path}`

## Best Candidate

{pd.DataFrame([best_row]).to_markdown(index=False) if best_row else "none"}

## Top {args.top_k}

{result_df.head(args.top_k).to_markdown(index=False)}
"""
    output_path.write_text(report, encoding="utf-8")

    print("Reference long ensemble search complete.")
    print(f"- output: {output_path}")
    print(f"- csv: {csv_output_path}")
    print(f"- best pred: {save_best_path}")
    if best_row:
        print("- best:")
        for key, value in best_row.items():
            print(f"  - {key}: {value}")
    print()
    print(result_df.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
