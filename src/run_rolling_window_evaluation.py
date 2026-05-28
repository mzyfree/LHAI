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


def parse_windows(spec: str) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = [part.strip() for part in item.split(":")]
        windows.append((start, end))
    if not windows:
        raise ValueError("No valid windows parsed.")
    return windows


def slice_frame(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dt_index = df.index.get_level_values("datetime")
    mask = (dt_index >= pd.Timestamp(start)) & (dt_index <= pd.Timestamp(end))
    return df.loc[mask]


def metric_value(analysis: dict[str, pd.DataFrame], bucket: str, metric: str) -> float:
    return float(analysis[bucket].loc[metric, "risk"])


def window_port_config(base_config: dict, start: str, end: str) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["backtest"]["start_time"] = start
    cfg["backtest"]["end_time"] = end
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pred.pkl on rolling test windows.")
    parser.add_argument("--config", required=True, help="Qlib workflow config path.")
    parser.add_argument("--pred", required=True, help="Prediction file path.")
    parser.add_argument(
        "--provider-uri",
        default=None,
        help="Optional provider_uri override for configs copied from another machine.",
    )
    parser.add_argument(
        "--windows",
        default="2017-01-01:2018-12-31,2018-01-01:2019-12-31,2019-01-01:2020-08-01",
        help="Comma-separated start:end windows.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/rolling_window_evaluation.md"),
        help="Markdown report output path.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    pred_path = Path(args.pred).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    windows = parse_windows(args.windows)

    config = load_config(config_path)
    provider_uri = args.provider_uri or config["qlib_init"]["provider_uri"]
    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=REG_CN)

    dataset = init_instance_by_config(config["task"]["dataset"])
    label = SignalRecord.generate_label(dataset)
    if label is None or label.empty:
        raise ValueError("Failed to generate labels from dataset.")

    pred = load_pred(pred_path)
    label = label.loc[pred.index]

    rows: list[dict[str, float | str]] = []
    for start, end in windows:
        pred_window = slice_frame(pred, start, end)
        label_window = label.loc[pred_window.index]
        if pred_window.empty or label_window.empty:
            raise ValueError(f"No prediction/label overlap for window {start} to {end}.")

        sig = signal_metrics(pred_window, label_window)
        analysis, _ = backtest_metrics(pred_window, window_port_config(config["port_analysis_config"], start, end))

        rows.append(
            {
                "window": f"{start} -> {end}",
                "IC": sig["IC"],
                "ICIR": sig["ICIR"],
                "Rank IC": sig["Rank IC"],
                "Rank ICIR": sig["Rank ICIR"],
                "annualized_return": metric_value(analysis, "excess_return_with_cost", "annualized_return"),
                "information_ratio": metric_value(analysis, "excess_return_with_cost", "information_ratio"),
                "max_drawdown": metric_value(analysis, "excess_return_with_cost", "max_drawdown"),
            }
        )

    result_df = pd.DataFrame(rows)

    report = f"""# Rolling Window Evaluation

## Setup

- Config: `{config_path}`
- Pred: `{pred_path}`
- Provider URI: `{provider_uri}`

## Windows

{result_df.to_markdown(index=False)}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("Rolling window evaluation complete.")
    print(f"- output: {output_path}")
    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
