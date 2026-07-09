from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QuantileSideModel:
    column: str
    direction: str
    invalid_probability: float = 0.20
    quantiles: list[float] = field(default_factory=list)
    training_summary: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"high", "low"}:
            raise ValueError("direction must be 'high' or 'low'")
        if not 0.0 <= float(self.invalid_probability) <= 1.0:
            raise ValueError("invalid_probability must be between 0 and 1")
        self.invalid_probability = float(self.invalid_probability)

    def fit(self, samples: pd.DataFrame, label_col: str) -> "QuantileSideModel":
        missing = [column for column in (self.column, label_col) if column not in samples.columns]
        if missing:
            raise ValueError(f"samples are missing required columns: {missing}")

        feature_values = pd.to_numeric(samples[self.column], errors="coerce")
        label_values = pd.to_numeric(samples[label_col], errors="coerce")
        finite_feature_mask = np.isfinite(feature_values.to_numpy(dtype=float))
        finite_features = feature_values[finite_feature_mask]
        if finite_features.empty:
            raise ValueError(f"{self.column} has no finite training samples")

        positive_mask = finite_feature_mask & np.isfinite(label_values.to_numpy(dtype=float)) & (label_values == 1)
        positive_features = feature_values[positive_mask]
        selected = positive_features if not positive_features.empty else finite_features

        self.quantiles = [float(selected.quantile(q)) for q in (0.25, 0.50, 0.75)]
        self.training_summary = {
            "column": self.column,
            "label_column": label_col,
            "direction": self.direction,
            "invalid_probability": self.invalid_probability,
            "row_count": int(len(samples)),
            "finite_feature_count": int(finite_feature_mask.sum()),
            "invalid_feature_count": int((~finite_feature_mask).sum()),
            "positive_label_finite_feature_count": int(len(positive_features)),
            "selected_sample_count": int(len(selected)),
            "fallback_to_all_finite": bool(positive_features.empty),
        }
        return self

    def predict_proba(self, samples: pd.DataFrame) -> np.ndarray:
        if not self.quantiles:
            raise ValueError("model must be fit before predict_proba")
        if self.column not in samples.columns:
            raise ValueError(f"samples are missing required column: {self.column}")

        feature_values = pd.to_numeric(samples[self.column], errors="coerce")
        values = feature_values.to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        positive_score = np.full(len(samples), self.invalid_probability, dtype=float)

        q1, q2, q3 = self.quantiles
        finite_values = values[finite_mask]
        if self.direction == "high":
            finite_score = np.select(
                [finite_values >= q3, finite_values >= q2, finite_values >= q1],
                [0.85, 0.65, 0.45],
                default=0.20,
            )
        else:
            finite_score = np.select(
                [finite_values <= q1, finite_values <= q2, finite_values <= q3],
                [0.85, 0.65, 0.45],
                default=0.20,
            )
        positive_score[finite_mask] = np.asarray(finite_score, dtype=float)
        return np.column_stack([1.0 - positive_score, positive_score])
