#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OPS_HOME = Path(__file__).resolve().parents[1]
if str(OPS_HOME) not in sys.path:
    sys.path.insert(0, str(OPS_HOME))

from bin.short_hold_v3_features import V3FeatureColumns


class QuantileSideModel:
    def __init__(self, column: str, direction: str):
        if direction not in {"high", "low"}:
            raise ValueError("direction must be 'high' or 'low'")
        self.column = column
        self.direction = direction
        self.quantiles: list[float] = []

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "QuantileSideModel":
        series = pd.to_numeric(features[self.column], errors="coerce").fillna(0.0)
        label_values = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
        positive = series[label_values == 1]
        if positive.empty:
            positive = series
        self.quantiles = [float(positive.quantile(q)) for q in (0.25, 0.50, 0.75)]
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if not self.quantiles:
            raise ValueError("model must be fit before predict_proba")

        series = pd.to_numeric(features[self.column], errors="coerce").fillna(0.0)
        q1, q2, q3 = self.quantiles
        if self.direction == "high":
            score = np.select(
                [series >= q3, series >= q2, series >= q1],
                [0.85, 0.65, 0.45],
                default=0.20,
            )
        else:
            score = np.select(
                [series <= q1, series <= q2, series <= q3],
                [0.85, 0.65, 0.45],
                default=0.20,
            )
        positive_score = np.asarray(score, dtype=float)
        return np.column_stack([1.0 - positive_score, positive_score])


if __name__ == "__main__":
    sys.modules.setdefault("bin.train_short_hold_v3_side_models", sys.modules[__name__])
    QuantileSideModel.__module__ = "bin.train_short_hold_v3_side_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train short-hold v3 side scoring models.")
    parser.add_argument("--samples", required=True, help="CSV with v3 features and labels.")
    parser.add_argument("--output-dir", required=True, help="Directory for side model artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples)
    samples = pd.read_csv(samples_path)
    feature_columns = list(V3FeatureColumns().numeric)
    required = set(feature_columns) | {"buyability_bad", "strong_label"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"sample file {samples_path} is missing required columns: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buyability_model = QuantileSideModel("distance_to_limit_up", "low").fit(samples, samples["buyability_bad"])
    strong_model = QuantileSideModel("return_score", "high").fit(samples, samples["strong_label"])

    with (output_dir / "buyability_model.pkl").open("wb") as handle:
        pickle.dump(buyability_model, handle)
    with (output_dir / "strong_model.pkl").open("wb") as handle:
        pickle.dump(strong_model, handle)

    metadata = {
        "feature_columns": feature_columns,
        "buyability_label": "buyability_bad",
        "strong_label": "strong_label",
        "model_type": "quantile_side_model",
    }
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    (output_dir / "metadata.json").write_text(metadata_text, encoding="utf-8")

    print(f"Wrote side models: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
