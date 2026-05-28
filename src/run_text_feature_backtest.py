from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pformat

import lightgbm as lgb
import pandas as pd
from ruamel.yaml import YAML
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

from run_prediction_fusion import backtest_metrics
from run_text_feature_pilot import (
    add_cross_sectional_text_features,
    add_interaction_features,
    ensure_text_features,
    load_text_features,
    prepare_split,
)


def load_config(config_path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    return yaml.load(config_path.read_text(encoding="utf-8"))


def attach_text_features(features: pd.DataFrame, text_features: pd.DataFrame) -> pd.DataFrame:
    base = features.copy()
    merged = base.join(text_features, how="left")
    text_cols = [c for c in merged.columns if c not in base.columns]
    if text_cols:
        merged[text_cols] = merged[text_cols].fillna(0.0)
    add_cross_sectional_text_features(merged, text_cols)
    add_interaction_features(merged, text_cols)
    return merged


def train_lgbm_from_config(
    model_config: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[lgb.Booster, pd.DataFrame]:
    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)

    kwargs = model_config.get("kwargs", {}).copy()
    params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
    }
    params.update(kwargs)

    num_boost_round = int(params.pop("num_boost_round", 500))
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))
    loss = params.pop("loss", None)
    if loss and "objective" not in params:
        params["objective"] = loss

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(20)],
    )
    pred = pd.DataFrame(
        {"score": model.predict(x_test, num_iteration=model.best_iteration)},
        index=x_test.index,
    )
    pred.index = pred.index.set_names(["datetime", "instrument"])
    return model, pred.sort_index()


def signal_metrics(pred: pd.DataFrame, label: pd.Series) -> dict[str, float]:
    df = pred.join(label.rename("label"), how="inner").dropna()
    ic_vals = []
    ric_vals = []
    for _, group in df.groupby(level="datetime"):
        if group["score"].nunique() < 2 or group["label"].nunique() < 2:
            continue
        ic_vals.append(group["score"].corr(group["label"]))
        ric_vals.append(spearmanr(group["score"], group["label"]).statistic)
    ic = pd.Series(ic_vals)
    ric = pd.Series(ric_vals)
    return {
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean() / ic.std()),
        "Rank IC": float(ric.mean()),
        "Rank ICIR": float(ric.mean() / ric.std()),
    }


def coverage_stats(x_test_text: pd.DataFrame, text_columns: list[str], text_features: pd.DataFrame) -> dict[str, object]:
    if not text_columns:
        return {
            "matched_rows": 0,
            "coverage": 0.0,
            "raw_instruments": [],
            "matched_instruments": [],
        }
    matched_rows = int(x_test_text[text_columns].abs().sum(axis=1).gt(0).sum())
    coverage = matched_rows / len(x_test_text) if len(x_test_text) else 0.0
    raw_instruments = sorted(text_features.index.get_level_values("instrument").unique())
    test_instruments = set(x_test_text.index.get_level_values("instrument"))
    matched_instruments = [inst for inst in raw_instruments if inst in test_instruments]
    return {
        "matched_rows": matched_rows,
        "coverage": coverage,
        "raw_instruments": raw_instruments,
        "matched_instruments": matched_instruments,
    }


def metric_value(analysis: dict[str, pd.DataFrame], bucket: str, metric: str) -> float:
    return float(analysis[bucket].loc[metric, "risk"])


def prediction_similarity(pred_base: pd.DataFrame, pred_text: pd.DataFrame, topk: int) -> dict[str, float]:
    merged = pred_base.rename(columns={"score": "base"}).join(
        pred_text.rename(columns={"score": "text"}),
        how="inner",
    )
    overlaps = []
    for _, group in merged.groupby(level="datetime"):
        base_inst = set(group["base"].nlargest(topk).index.get_level_values("instrument"))
        text_inst = set(group["text"].nlargest(topk).index.get_level_values("instrument"))
        overlaps.append(len(base_inst & text_inst) / topk)
    return {
        "pearson": float(merged["base"].corr(merged["text"])),
        "spearman": float(spearmanr(merged["base"], merged["text"]).statistic),
        "avg_topk_overlap": float(pd.Series(overlaps).mean()),
        "min_topk_overlap": float(pd.Series(overlaps).min()),
        "max_topk_overlap": float(pd.Series(overlaps).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs text-feature LightGBM backtest under one config.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "src/configs/workflow_config_lightgbm_csi500_h10_text_window.yaml"),
    )
    parser.add_argument(
        "--raw-events",
        default=str(PROJECT_ROOT / "data/raw/text_events.csv"),
    )
    parser.add_argument(
        "--text-features",
        default=str(PROJECT_ROOT / "data/processed/text_event_features_v2.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/stage2_text_feature_backtest_v2.md"),
    )
    parser.add_argument(
        "--pred-dir",
        default=str(PROJECT_ROOT / "preds"),
        help="Directory to save baseline/text pred.pkl outputs.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    raw_events_path = Path(args.raw_events).expanduser().resolve()
    text_features_path = Path(args.text_features).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    pred_dir = Path(args.pred_dir).expanduser().resolve()

    config = load_config(config_path)
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"], region=REG_CN)

    dataset = init_instance_by_config(config["task"]["dataset"])
    x_train, y_train = prepare_split(dataset, "train")
    x_valid, y_valid = prepare_split(dataset, "valid")
    x_test, y_test = prepare_split(dataset, "test")

    text_feature_path = ensure_text_features(raw_events_path, text_features_path)
    text_features = load_text_features(text_feature_path)

    x_train_text = attach_text_features(x_train, text_features)
    x_valid_text = attach_text_features(x_valid, text_features)
    x_test_text = attach_text_features(x_test, text_features)
    text_columns = [c for c in x_test_text.columns if c not in x_test.columns]
    coverage = coverage_stats(x_test_text, text_columns, text_features)

    model_cfg = config["task"]["model"]
    _, pred_base = train_lgbm_from_config(model_cfg, x_train, y_train, x_valid, y_valid, x_test)
    _, pred_text = train_lgbm_from_config(model_cfg, x_train_text, y_train, x_valid_text, y_valid, x_test_text)

    sig_base = signal_metrics(pred_base, y_test)
    sig_text = signal_metrics(pred_text, y_test)
    analysis_base, indicator_base = backtest_metrics(pred_base, config["port_analysis_config"])
    analysis_text, indicator_text = backtest_metrics(pred_text, config["port_analysis_config"])
    topk = int(config["port_analysis_config"]["strategy"]["kwargs"]["topk"])
    similarity = prediction_similarity(pred_base, pred_text, topk)

    pred_dir.mkdir(parents=True, exist_ok=True)
    baseline_pred_path = pred_dir / "text_feature_backtest_baseline.pkl"
    text_pred_path = pred_dir / "text_feature_backtest_text_v2.pkl"
    pred_base.to_pickle(baseline_pred_path)
    pred_text.to_pickle(text_pred_path)

    report = f"""# Stage 2 Text Feature Backtest v2

## Setup

- Config: `{config_path}`
- Raw event file: `{raw_events_path.name}`
- Aggregated text feature file: `{text_feature_path.name}`
- Saved baseline pred: `{baseline_pred_path}`
- Saved text pred: `{text_pred_path}`

## Test Coverage

- Test rows with any text feature present: `{coverage["coverage"]:.4%}`
- Matched test rows: `{coverage["matched_rows"]}`
- Raw event instruments: `{", ".join(coverage["raw_instruments"]) if coverage["raw_instruments"] else "none"}`
- Raw event instruments present in test pool: `{", ".join(coverage["matched_instruments"]) if coverage["matched_instruments"] else "none"}`

## Signal Comparison

| Metric | Baseline | Baseline + Text Features v2 |
| --- | ---: | ---: |
| IC | {sig_base["IC"]:.4f} | {sig_text["IC"]:.4f} |
| ICIR | {sig_base["ICIR"]:.4f} | {sig_text["ICIR"]:.4f} |
| Rank IC | {sig_base["Rank IC"]:.4f} | {sig_text["Rank IC"]:.4f} |
| Rank ICIR | {sig_base["Rank ICIR"]:.4f} | {sig_text["Rank ICIR"]:.4f} |

## Prediction Similarity

- Pearson correlation: `{similarity["pearson"]:.4f}`
- Spearman correlation: `{similarity["spearman"]:.4f}`
- Avg top-{topk} overlap by day: `{similarity["avg_topk_overlap"]:.4%}`
- Min top-{topk} overlap by day: `{similarity["min_topk_overlap"]:.4%}`
- Max top-{topk} overlap by day: `{similarity["max_topk_overlap"]:.4%}`

## Backtest Comparison (Excess Return With Cost, 1day)

| Metric | Baseline | Baseline + Text Features v2 |
| --- | ---: | ---: |
| annualized_return | {metric_value(analysis_base, "excess_return_with_cost", "annualized_return"):.6f} | {metric_value(analysis_text, "excess_return_with_cost", "annualized_return"):.6f} |
| information_ratio | {metric_value(analysis_base, "excess_return_with_cost", "information_ratio"):.6f} | {metric_value(analysis_text, "excess_return_with_cost", "information_ratio"):.6f} |
| max_drawdown | {metric_value(analysis_base, "excess_return_with_cost", "max_drawdown"):.6f} | {metric_value(analysis_text, "excess_return_with_cost", "max_drawdown"):.6f} |

## Baseline Backtest Detail

{analysis_base["excess_return_with_cost"].to_markdown()}

## Text v2 Backtest Detail

{analysis_text["excess_return_with_cost"].to_markdown()}

## Baseline Indicators

{indicator_base.to_markdown()}

## Text v2 Indicators

{indicator_text.to_markdown()}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("Text feature backtest setup:")
    print(f"- config: {config_path}")
    print(f"- raw_events: {raw_events_path}")
    print(f"- text_features: {text_feature_path}")
    print(f"- output: {output_path}")
    print(f"- baseline_pred: {baseline_pred_path}")
    print(f"- text_pred: {text_pred_path}")
    print()
    print("Signal metrics:")
    print(pformat({"baseline": sig_base, "text_v2": sig_text}))
    print()
    print("Prediction similarity:")
    print(pformat(similarity))
    print()
    print("Baseline excess return with cost (1day):")
    print(analysis_base["excess_return_with_cost"])
    print("Text v2 excess return with cost (1day):")
    print(analysis_text["excess_return_with_cost"])
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
