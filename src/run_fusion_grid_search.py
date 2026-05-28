from __future__ import annotations

import argparse
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
    fuse_scores,
    load_config,
    load_pred,
    signal_metrics,
)


def parse_weights(spec: str) -> list[tuple[float, float]]:
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":")
        pairs.append((float(left), float(right)))
    if not pairs:
        raise ValueError("No valid weight pairs parsed.")
    return pairs


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    if not methods:
        raise ValueError("No valid methods parsed.")
    return methods


def metric_value(analysis: dict[str, pd.DataFrame], bucket: str, metric: str) -> float:
    return float(analysis[bucket].loc[metric, "risk"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small fusion grid search over methods and weights.")
    parser.add_argument("--config", required=True, help="Qlib workflow config path.")
    parser.add_argument("--pred-a", required=True, help="First pred.pkl path.")
    parser.add_argument("--pred-b", required=True, help="Second pred.pkl path.")
    parser.add_argument(
        "--methods",
        default="rank_mean,zscore_mean",
        help="Comma-separated methods. e.g. rank_mean,zscore_mean",
    )
    parser.add_argument(
        "--weights",
        default="1:1,2:1,1:2,3:1,1:3",
        help="Comma-separated weight pairs. e.g. 1:1,2:1,1:2",
    )
    parser.add_argument(
        "--sort-by",
        default="annualized_return",
        choices=["annualized_return", "information_ratio", "Rank IC", "IC"],
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/fusion_grid_search.md"),
        help="Markdown summary output path.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    methods = parse_methods(args.methods)
    weight_pairs = parse_weights(args.weights)

    config = load_config(config_path)
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"], region=REG_CN)
    dataset = init_instance_by_config(config["task"]["dataset"])
    label = SignalRecord.generate_label(dataset)
    if label is None:
        raise ValueError("Failed to generate test label from dataset.")

    pred_a = load_pred(Path(args.pred_a).expanduser().resolve())
    pred_b = load_pred(Path(args.pred_b).expanduser().resolve())

    rows = []
    best_payload = None
    best_score = None

    for method, (wa, wb) in itertools.product(methods, weight_pairs):
        fused = fuse_scores(pred_a, pred_b, method, wa, wb)
        aligned_label = label.loc[fused.index]
        sig = signal_metrics(fused, aligned_label)
        analysis, _ = backtest_metrics(fused, config["port_analysis_config"])

        row = {
            "method": method,
            "weights": f"{wa}:{wb}",
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
            best_payload = row

    result_df = pd.DataFrame(rows).sort_values(args.sort_by, ascending=False).reset_index(drop=True)

    report = f"""# Fusion Grid Search

## Setup

- Config: `{config_path}`
- Pred A: `{Path(args.pred_a).name}`
- Pred B: `{Path(args.pred_b).name}`
- Methods: `{", ".join(methods)}`
- Weight pairs: `{", ".join(f"{wa}:{wb}" for wa, wb in weight_pairs)}`
- Sorted by: `{args.sort_by}`

## Best Candidate

{pd.DataFrame([best_payload]).to_markdown(index=False) if best_payload else "none"}

## Full Ranking

{result_df.to_markdown(index=False)}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("Fusion grid search complete.")
    print(f"- output: {output_path}")
    if best_payload:
        print("- best:")
        for k, v in best_payload.items():
            print(f"  - {k}: {v}")
    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
