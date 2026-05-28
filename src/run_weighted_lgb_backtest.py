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
from run_text_feature_pilot import prepare_split


def load_config(config_path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    return yaml.load(config_path.read_text(encoding="utf-8"))


def train_lgbm_from_config(
    model_config: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    train_weight: pd.Series | None = None,
) -> tuple[lgb.Booster, pd.DataFrame]:
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        weight=None if train_weight is None else train_weight.to_numpy(),
    )
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


def build_time_decay_weights(index: pd.MultiIndex, half_life_days: int) -> pd.Series:
    dt = pd.to_datetime(index.get_level_values("datetime"))
    age_days = (dt.max() - dt).days.astype(float)
    weights = 0.5 ** (age_days / float(half_life_days))
    return pd.Series(weights, index=index, name="time_decay_weight")


def metric_value(analysis: dict[str, pd.DataFrame], bucket: str, metric: str) -> float:
    return float(analysis[bucket].loc[metric, "risk"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare plain LightGBM vs time-decay weighted LightGBM.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "src/configs/workflow_config_lightgbm_csi500_h10.yaml"),
    )
    parser.add_argument(
        "--half-life-days",
        type=int,
        default=252,
        help="Half-life in calendar days for time-decay weights.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/stage2_weighted_lgb_backtest.md"),
    )
    parser.add_argument(
        "--pred-dir",
        default=str(PROJECT_ROOT / "preds"),
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    pred_dir = Path(args.pred_dir).expanduser().resolve()

    config = load_config(config_path)
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"], region=REG_CN)
    dataset = init_instance_by_config(config["task"]["dataset"])

    x_train, y_train = prepare_split(dataset, "train")
    x_valid, y_valid = prepare_split(dataset, "valid")
    x_test, y_test = prepare_split(dataset, "test")

    train_weight = build_time_decay_weights(x_train.index, args.half_life_days)

    model_cfg = config["task"]["model"]
    _, pred_plain = train_lgbm_from_config(model_cfg, x_train, y_train, x_valid, y_valid, x_test)
    _, pred_weighted = train_lgbm_from_config(
        model_cfg,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        train_weight=train_weight,
    )

    sig_plain = signal_metrics(pred_plain, y_test)
    sig_weighted = signal_metrics(pred_weighted, y_test)
    analysis_plain, indicator_plain = backtest_metrics(pred_plain, config["port_analysis_config"])
    analysis_weighted, indicator_weighted = backtest_metrics(pred_weighted, config["port_analysis_config"])

    pred_dir.mkdir(parents=True, exist_ok=True)
    plain_pred_path = pred_dir / "weighted_lgb_plain.pkl"
    weighted_pred_path = pred_dir / f"weighted_lgb_time_decay_hl_{args.half_life_days}.pkl"
    pred_plain.to_pickle(plain_pred_path)
    pred_weighted.to_pickle(weighted_pred_path)

    report = f"""# Stage 2 Weighted LightGBM Backtest

## Setup

- Config: `{config_path}`
- Weighting scheme: `time_decay`
- Half-life days: `{args.half_life_days}`
- Saved plain pred: `{plain_pred_path}`
- Saved weighted pred: `{weighted_pred_path}`

## Weight Example

- Newest train sample weight: `{train_weight.max():.6f}`
- Oldest train sample weight: `{train_weight.min():.6f}`
- Median train sample weight: `{train_weight.median():.6f}`

## Signal Comparison

| Metric | Plain | Time-decay weighted |
| --- | ---: | ---: |
| IC | {sig_plain["IC"]:.4f} | {sig_weighted["IC"]:.4f} |
| ICIR | {sig_plain["ICIR"]:.4f} | {sig_weighted["ICIR"]:.4f} |
| Rank IC | {sig_plain["Rank IC"]:.4f} | {sig_weighted["Rank IC"]:.4f} |
| Rank ICIR | {sig_plain["Rank ICIR"]:.4f} | {sig_weighted["Rank ICIR"]:.4f} |

## Backtest Comparison (Excess Return With Cost, 1day)

| Metric | Plain | Time-decay weighted |
| --- | ---: | ---: |
| annualized_return | {metric_value(analysis_plain, "excess_return_with_cost", "annualized_return"):.6f} | {metric_value(analysis_weighted, "excess_return_with_cost", "annualized_return"):.6f} |
| information_ratio | {metric_value(analysis_plain, "excess_return_with_cost", "information_ratio"):.6f} | {metric_value(analysis_weighted, "excess_return_with_cost", "information_ratio"):.6f} |
| max_drawdown | {metric_value(analysis_plain, "excess_return_with_cost", "max_drawdown"):.6f} | {metric_value(analysis_weighted, "excess_return_with_cost", "max_drawdown"):.6f} |

## Plain Backtest Detail

{analysis_plain["excess_return_with_cost"].to_markdown()}

## Weighted Backtest Detail

{analysis_weighted["excess_return_with_cost"].to_markdown()}

## Plain Indicators

{indicator_plain.to_markdown()}

## Weighted Indicators

{indicator_weighted.to_markdown()}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("Weighted LightGBM setup:")
    print(f"- config: {config_path}")
    print(f"- half_life_days: {args.half_life_days}")
    print(f"- output: {output_path}")
    print(f"- plain_pred: {plain_pred_path}")
    print(f"- weighted_pred: {weighted_pred_path}")
    print()
    print("Signal metrics:")
    print(pformat({"plain": sig_plain, "weighted": sig_weighted}))
    print()
    print("Weight summary:")
    print(
        pformat(
            {
                "newest_weight": float(train_weight.max()),
                "oldest_weight": float(train_weight.min()),
                "median_weight": float(train_weight.median()),
            }
        )
    )
    print()
    print("Plain excess return with cost (1day):")
    print(analysis_plain["excess_return_with_cost"])
    print("Weighted excess return with cost (1day):")
    print(analysis_weighted["excess_return_with_cost"])
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
