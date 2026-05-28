from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from src.features.build_text_event_features import build_features, load_rows, write_rows


def load_config(config_path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    return yaml.load(config_path.read_text(encoding="utf-8"))


def ensure_text_features(input_path: Path, output_path: Path) -> Path:
    if output_path.exists():
        return output_path
    rows = load_rows(input_path)
    features = build_features(rows)
    write_rows(output_path, features)
    return output_path


def prepare_split(dataset, segment: str) -> tuple[pd.DataFrame, pd.Series]:
    df = dataset.prepare(segment)
    if isinstance(df.columns, pd.MultiIndex):
        features = df["feature"].copy()
        label = df["label"].iloc[:, 0].copy()
    elif "LABEL0" in df.columns:
        features = df.drop(columns=["LABEL0"]).copy()
        label = df["LABEL0"].copy()
    else:
        raise ValueError(f"Unexpected dataset format from Qlib: {list(df.columns[:10])}")
    features.index = features.index.set_names(["datetime", "instrument"])
    label.index = label.index.set_names(["datetime", "instrument"])
    return features, label


def load_text_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index(["datetime", "instrument"]).sort_index()


def attach_text_features(features: pd.DataFrame, text_features: pd.DataFrame) -> pd.DataFrame:
    base = features.copy()
    merged = base.join(text_features, how="left")
    text_cols = [c for c in merged.columns if c not in base.columns]
    if text_cols:
        merged[text_cols] = merged[text_cols].fillna(0.0)
    add_cross_sectional_text_features(merged, text_cols)
    add_interaction_features(merged, text_cols)
    return merged


def add_cross_sectional_text_features(df: pd.DataFrame, text_cols: list[str]) -> None:
    if not text_cols:
        return

    for col in text_cols:
        grouped = df[col].groupby(level="datetime", group_keys=False)
        df[f"{col}_cs_rank"] = grouped.rank(pct=True)

        def _zscore(group: pd.Series) -> pd.Series:
            std = group.std()
            if pd.isna(std) or std == 0:
                return pd.Series(0.0, index=group.index)
            return (group - group.mean()) / std

        df[f"{col}_cs_zscore"] = grouped.apply(_zscore)


def add_interaction_features(df: pd.DataFrame, text_cols: list[str]) -> None:
    if not text_cols:
        return

    interaction_specs = [
        ("event_count", ["STD20", "STD10", "STD5", "VSTD20", "VSTD10", "VSTD5"]),
        ("event_count_roll_5", ["STD20", "VSTD20", "VSTD10"]),
        ("positive_keyword_hits", ["ROC10", "ROC20", "ROC60"]),
        ("positive_keyword_hits_roll_5", ["ROC10", "ROC20"]),
        ("sentiment_balance", ["ROC10", "ROC20", "KLEN"]),
        ("sentiment_balance_roll_5", ["ROC10", "STD20"]),
    ]

    for left_col, right_candidates in interaction_specs:
        if left_col not in df.columns:
            continue
        for right_col in right_candidates:
            if right_col not in df.columns:
                continue
            df[f"{left_col}_x_{right_col}"] = df[left_col] * df[right_col]


def train_and_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
) -> pd.Series:
    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.2,
        "colsample_bytree": 0.8879,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 8,
        "verbosity": -1,
    }
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    pred = model.predict(x_test, num_iteration=model.best_iteration)
    return pd.Series(pred, index=x_test.index, name="score")


def daily_ic(pred: pd.Series, label: pd.Series) -> float:
    df = pd.concat([pred, label.rename("label")], axis=1).dropna()
    vals = []
    for _, g in df.groupby(level="datetime"):
        if g["score"].nunique() < 2 or g["label"].nunique() < 2:
            continue
        vals.append(g["score"].corr(g["label"]))
    return float(pd.Series(vals).mean()) if vals else float("nan")


def daily_rank_ic(pred: pd.Series, label: pd.Series) -> float:
    df = pd.concat([pred, label.rename("label")], axis=1).dropna()
    vals = []
    for _, g in df.groupby(level="datetime"):
        if g["score"].nunique() < 2 or g["label"].nunique() < 2:
            continue
        vals.append(spearmanr(g["score"], g["label"]).statistic)
    return float(pd.Series(vals).mean()) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot comparison: baseline vs baseline + text event features.")
    parser.add_argument(
        "--config",
        default="/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500_h10.yaml",
    )
    parser.add_argument(
        "--raw-events",
        default="/Users/Dylan.Min/Documents/Code/learn/LHAI/data/raw/text_events_template.csv",
    )
    parser.add_argument(
        "--text-features",
        default="/Users/Dylan.Min/Documents/Code/learn/LHAI/data/processed/text_event_features.csv",
    )
    parser.add_argument(
        "--output",
        default="/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage2_text_feature_pilot.md",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"], region=REG_CN)
    dataset = init_instance_by_config(config["task"]["dataset"])

    x_train, y_train = prepare_split(dataset, "train")
    x_valid, y_valid = prepare_split(dataset, "valid")
    x_test, y_test = prepare_split(dataset, "test")

    text_feature_path = ensure_text_features(Path(args.raw_events), Path(args.text_features))
    text_features = load_text_features(text_feature_path)
    raw_instruments = sorted(text_features.index.get_level_values("instrument").unique())
    test_instruments = set(x_test.index.get_level_values("instrument"))

    pred_base = train_and_predict(x_train, y_train, x_valid, y_valid, x_test)
    x_train_text = attach_text_features(x_train, text_features)
    x_valid_text = attach_text_features(x_valid, text_features)
    x_test_text = attach_text_features(x_test, text_features)
    pred_text = train_and_predict(
        x_train_text,
        y_train,
        x_valid_text,
        y_valid,
        x_test_text,
    )

    matched_rows = int(x_test_text[text_features.columns].abs().sum(axis=1).gt(0).sum())
    coverage = matched_rows / len(x_test_text) if len(x_test_text) else 0.0
    matched_instruments = [inst for inst in raw_instruments if inst in test_instruments]

    report = f"""# Stage 2 Text Feature Pilot

## Setup

- Base config: `src/configs/workflow_config_lightgbm_csi500_h10.yaml`
- Raw event file: `{Path(args.raw_events).name}`
- Aggregated text feature file: `{Path(args.text_features).name}`

## Test Coverage

- Test rows with any text feature present: `{coverage:.4%}`
- Matched test rows: `{matched_rows}`
- Raw event instruments: `{", ".join(raw_instruments) if raw_instruments else "none"}`
- Raw event instruments present in test pool: `{", ".join(matched_instruments) if matched_instruments else "none"}`

## Prediction Comparison

| Metric | Baseline | Baseline + Text Features |
| --- | --- | --- |
| IC | {daily_ic(pred_base, y_test):.4f} | {daily_ic(pred_text, y_test):.4f} |
| Rank IC | {daily_rank_ic(pred_base, y_test):.4f} | {daily_rank_ic(pred_text, y_test):.4f} |

## Interpretation

这次只是 pilot：

- 它验证了文本事件特征已经可以按 `datetime + instrument` 并入现有训练样本。
- 如果原始文本事件样本非常少，结果不代表真实策略效果，只能说明流程已经打通。
- 要判断文本特征是否真的有效，需要更大规模、真实覆盖度更高的公告或新闻数据。
"""

    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
