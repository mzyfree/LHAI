from __future__ import annotations

import argparse
import copy
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

from run_prediction_fusion import backtest_metrics, load_config, load_pred, signal_metrics


def parse_pairs(spec: str) -> list[tuple[int, int]]:
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        topk, n_drop = item.split(":")
        pairs.append((int(topk), int(n_drop)))
    if not pairs:
        raise ValueError("No valid topk:n_drop pairs parsed.")
    return pairs


def metric_value(analysis: dict[str, pd.DataFrame], bucket: str, metric: str) -> float:
    return float(analysis[bucket].loc[metric, "risk"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Search TopkDropoutStrategy topk/n_drop parameters.")
    parser.add_argument(
        "--config",
        default="configs/reference_long/workflow_config_lgb_alpha158_reference_long.yaml",
        help="Workflow config used for labels and backtest.",
    )
    parser.add_argument("--pred", required=True, help="Prediction pkl path.")
    parser.add_argument(
        "--pairs",
        default="10:1,20:2,30:3,50:5,80:8,100:10,150:15",
        help="Comma-separated topk:n_drop pairs.",
    )
    parser.add_argument(
        "--sort-by",
        default="annualized_return",
        choices=["annualized_return", "information_ratio", "max_drawdown", "Rank IC", "IC"],
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/llhh/reports/reference_long_topk_search.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--csv-output",
        default="/root/autodl-tmp/llhh/reports/reference_long_topk_search.csv",
        help="CSV result path.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    pred_path = Path(args.pred).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    csv_output_path = Path(args.csv_output).expanduser().resolve()
    pairs = parse_pairs(args.pairs)

    config = load_config(config_path)
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"], region=REG_CN)
    dataset = init_instance_by_config(config["task"]["dataset"])
    label = SignalRecord.generate_label(dataset)
    if label is None:
        raise ValueError("Failed to generate test label from dataset.")

    pred = load_pred(pred_path)
    aligned_label = label.loc[pred.index]
    sig = signal_metrics(pred, aligned_label)

    rows = []
    for topk, n_drop in pairs:
        port_cfg = copy.deepcopy(config["port_analysis_config"])
        strategy_kwargs = port_cfg["strategy"].setdefault("kwargs", {})
        strategy_kwargs["topk"] = topk
        strategy_kwargs["n_drop"] = n_drop

        analysis, _ = backtest_metrics(pred, port_cfg)
        rows.append(
            {
                "topk": topk,
                "n_drop": n_drop,
                "IC": sig["IC"],
                "ICIR": sig["ICIR"],
                "Rank IC": sig["Rank IC"],
                "Rank ICIR": sig["Rank ICIR"],
                "annualized_return": metric_value(analysis, "excess_return_with_cost", "annualized_return"),
                "information_ratio": metric_value(analysis, "excess_return_with_cost", "information_ratio"),
                "max_drawdown": metric_value(analysis, "excess_return_with_cost", "max_drawdown"),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(args.sort_by, ascending=False).reset_index(drop=True)
    best = result_df.iloc[0].to_dict() if not result_df.empty else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_output_path, index=False)

    report = f"""# Reference Long TopK Search

## Setup

- Config: `{config_path}`
- Pred: `{pred_path}`
- Pairs: `{", ".join(f"{topk}:{n_drop}" for topk, n_drop in pairs)}`
- Sort by: `{args.sort_by}`
- CSV: `{csv_output_path}`

## Best Candidate

{pd.DataFrame([best]).to_markdown(index=False) if best else "none"}

## Full Ranking

{result_df.to_markdown(index=False)}
"""
    output_path.write_text(report, encoding="utf-8")

    print("Reference long topk search complete.")
    print(f"- output: {output_path}")
    print(f"- csv: {csv_output_path}")
    if best:
        print("- best:")
        for key, value in best.items():
            print(f"  - {key}: {value}")
    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
