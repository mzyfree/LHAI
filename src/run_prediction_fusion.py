from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

import pandas as pd
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import qlib
from qlib.backtest import backtest as normal_backtest
from qlib.constant import REG_CN
from qlib.contrib.evaluate import indicator_analysis, risk_analysis
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import fill_placeholder, init_instance_by_config
from qlib.workflow.record_temp import SignalRecord


def load_config(config_path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    return yaml.load(config_path.read_text(encoding="utf-8"))


def load_pred(path: Path) -> pd.DataFrame:
    pred = pd.read_pickle(path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if "score" not in pred.columns:
        raise ValueError(f"`score` column not found in {path}")
    pred = pred[["score"]].copy()
    pred.index = pred.index.set_names(["datetime", "instrument"])
    return pred.sort_index()


def daily_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level="datetime").rank(pct=True)


def daily_zscore(series: pd.Series) -> pd.Series:
    def _zscore(group: pd.Series) -> pd.Series:
        std = group.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return series.groupby(level="datetime", group_keys=False).apply(_zscore)


def fuse_scores(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    method: str,
    weight_a: float,
    weight_b: float,
) -> pd.DataFrame:
    merged = pred_a.rename(columns={"score": "score_a"}).join(
        pred_b.rename(columns={"score": "score_b"}), how="inner"
    )
    if merged.empty:
        raise ValueError("No overlapping prediction index between the two inputs.")

    wa = weight_a / (weight_a + weight_b)
    wb = weight_b / (weight_a + weight_b)

    if method == "mean":
        score = wa * merged["score_a"] + wb * merged["score_b"]
    elif method == "rank_mean":
        score = wa * daily_rank(merged["score_a"]) + wb * daily_rank(merged["score_b"])
    elif method == "zscore_mean":
        score = wa * daily_zscore(merged["score_a"]) + wb * daily_zscore(merged["score_b"])
    else:
        raise ValueError(f"Unsupported method: {method}")

    return score.to_frame("score").sort_index()


def load_test_label(dataset) -> pd.DataFrame:
    with DatasetH.class_casting(dataset, DatasetH):
        params = dict(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
        try:
            label = dataset.prepare(**params)
        except TypeError:
            del params["data_key"]
            label = dataset.prepare(**params)
    if label is None or label.empty:
        raise ValueError("Failed to load test labels from dataset.")
    return label


def signal_metrics(pred: pd.DataFrame, label: pd.DataFrame) -> dict[str, float]:
    from qlib.contrib.eva.alpha import calc_ic

    ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
    return {
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean() / ic.std()),
        "Rank IC": float(ric.mean()),
        "Rank ICIR": float(ric.mean() / ric.std()),
    }


def backtest_metrics(pred: pd.DataFrame, port_cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    strategy_config = port_cfg["strategy"]
    executor_config = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    backtest_config = port_cfg["backtest"].copy()

    placeholder_value = {"<PRED>": pred}
    strategy_config = fill_placeholder(strategy_config, placeholder_value)
    executor_config = fill_placeholder(executor_config, placeholder_value)

    portfolio_metric_dict, indicator_dict = normal_backtest(
        executor=executor_config,
        strategy=strategy_config,
        **backtest_config,
    )

    freq_key = "day" if "day" in portfolio_metric_dict else "1day"
    if freq_key not in portfolio_metric_dict:
        raise KeyError(f"Expected 'day' or '1day' in portfolio metrics, got {list(portfolio_metric_dict.keys())}")
    report_normal, _ = portfolio_metric_dict[freq_key]
    indicators_normal = indicator_dict[freq_key][0]

    analysis = {
        "benchmark": risk_analysis(report_normal["bench"], freq="day"),
        "excess_return_without_cost": risk_analysis(report_normal["return"] - report_normal["bench"], freq="day"),
        "excess_return_with_cost": risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"],
            freq="day",
        ),
    }
    indicator_df = indicator_analysis(indicators_normal)
    return analysis, indicator_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse two Qlib prediction files and evaluate them.")
    parser.add_argument("--config", required=True, help="Qlib workflow yaml used for dataset/backtest.")
    parser.add_argument("--pred-a", required=True, help="First pred.pkl path.")
    parser.add_argument("--pred-b", required=True, help="Second pred.pkl path.")
    parser.add_argument(
        "--method",
        default="rank_mean",
        choices=["mean", "rank_mean", "zscore_mean"],
        help="Fusion method.",
    )
    parser.add_argument("--weight-a", type=float, default=1.0)
    parser.add_argument("--weight-b", type=float, default=1.0)
    parser.add_argument("--save-pred", default=None, help="Optional path to save fused pred.pkl")
    parser.add_argument("--output", default=None, help="Optional markdown report path.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    qlib_init = config["qlib_init"]
    provider_uri = str(Path(qlib_init["provider_uri"]).expanduser())
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    dataset = init_instance_by_config(config["task"]["dataset"])
    label = SignalRecord.generate_label(dataset)
    if label is None:
        raise ValueError("Failed to generate test label from dataset.")

    pred_a = load_pred(Path(args.pred_a).expanduser().resolve())
    pred_b = load_pred(Path(args.pred_b).expanduser().resolve())
    fused = fuse_scores(pred_a, pred_b, args.method, args.weight_a, args.weight_b)

    if args.save_pred:
        save_pred_path = Path(args.save_pred).expanduser().resolve()
        save_pred_path.parent.mkdir(parents=True, exist_ok=True)
        fused.to_pickle(save_pred_path)
        print(f"Saved fused pred to {save_pred_path}")

    label = label.loc[fused.index]
    sig = signal_metrics(fused, label)
    analysis, indicator_df = backtest_metrics(fused, config["port_analysis_config"])

    print("Fusion setup:")
    print(f"- config: {config_path}")
    print(f"- pred_a: {args.pred_a}")
    print(f"- pred_b: {args.pred_b}")
    print(f"- method: {args.method}")
    print(f"- weights: {args.weight_a}:{args.weight_b}")
    print()
    pprint(sig)
    print()
    print("The following are analysis results of benchmark return(1day).")
    pprint(analysis["benchmark"])
    print("The following are analysis results of the excess return without cost(1day).")
    pprint(analysis["excess_return_without_cost"])
    print("The following are analysis results of the excess return with cost(1day).")
    pprint(analysis["excess_return_with_cost"])
    print("The following are analysis results of indicators(1day).")
    pprint(indicator_df)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        report = f"""# Prediction Fusion Report

## Setup

- Config: `{config_path}`
- Pred A: `{Path(args.pred_a).name}`
- Pred B: `{Path(args.pred_b).name}`
- Method: `{args.method}`
- Weights: `{args.weight_a}:{args.weight_b}`

## Signal Metrics

| Metric | Value |
| --- | ---: |
| IC | {sig["IC"]:.6f} |
| ICIR | {sig["ICIR"]:.6f} |
| Rank IC | {sig["Rank IC"]:.6f} |
| Rank ICIR | {sig["Rank ICIR"]:.6f} |

## Excess Return With Cost (1day)

{analysis["excess_return_with_cost"].to_markdown()}

## Indicators (1day)

{indicator_df.to_markdown()}
"""
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
