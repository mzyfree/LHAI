#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import pandas as pd

OPS_HOME = Path(__file__).resolve().parents[1]
if str(OPS_HOME) not in sys.path:
    sys.path.insert(0, str(OPS_HOME))

from bin.short_hold_v3_features import V3FeatureColumns
from bin.short_hold_v3_side_model import QuantileSideModel


BUYABILITY_LABEL = "buyability_bad"
STRONG_LABEL = "strong_label"
BUYABILITY_LABEL_ALIASES = ("buyability_bad_label",)
STRONG_LABEL_ALIASES = ("strong_next_label",)
BUYABILITY_ARTIFACT = "buyability_model.pkl"
STRONG_ARTIFACT = "strong_model.pkl"
METADATA_ARTIFACT = "metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train short-hold v3 side scoring models.")
    parser.add_argument("--samples", required=True, help="CSV with v3 features and labels.")
    parser.add_argument("--output-dir", required=True, help="Directory for side model artifacts.")
    return parser.parse_args()


def _resolve_label_column(samples: pd.DataFrame, primary: str, aliases: tuple[str, ...]) -> str:
    if primary in samples.columns:
        return primary
    for alias in aliases:
        if alias in samples.columns:
            return alias
    choices = ", ".join((primary, *aliases))
    raise ValueError(f"sample file is missing required label column; expected one of: {choices}")


def _pickle_bytes(model: QuantileSideModel) -> bytes:
    handle = io.BytesIO()
    pickle.dump(model, handle)
    return handle.getvalue()


def _verify_pickle(path: Path) -> QuantileSideModel:
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, QuantileSideModel):
        raise TypeError(f"{path.name} did not reload as QuantileSideModel")
    return model


def _publish_artifacts(output_dir: Path, artifacts: dict[str, bytes]) -> None:
    with tempfile.TemporaryDirectory(prefix=".tmp_short_hold_v3_side_models_", dir=output_dir) as tmpdir:
        tmp = Path(tmpdir)
        for name, payload in artifacts.items():
            (tmp / name).write_bytes(payload)

        _verify_pickle(tmp / BUYABILITY_ARTIFACT)
        _verify_pickle(tmp / STRONG_ARTIFACT)
        json.loads((tmp / METADATA_ARTIFACT).read_text(encoding="utf-8"))

        for name in (BUYABILITY_ARTIFACT, STRONG_ARTIFACT, METADATA_ARTIFACT):
            os.replace(tmp / name, output_dir / name)


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples)
    samples = pd.read_csv(samples_path)
    feature_columns = list(V3FeatureColumns().numeric)
    missing = sorted(set(feature_columns) - set(samples.columns))
    if missing:
        raise ValueError(f"sample file {samples_path} is missing required columns: {missing}")
    buyability_label = _resolve_label_column(samples, BUYABILITY_LABEL, BUYABILITY_LABEL_ALIASES)
    strong_label = _resolve_label_column(samples, STRONG_LABEL, STRONG_LABEL_ALIASES)
    if samples.empty:
        raise ValueError(f"sample file {samples_path} contains no rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buyability_model = QuantileSideModel(
        "distance_to_limit_up",
        "low",
        invalid_probability=0.95,
    ).fit(samples, buyability_label)
    strong_model = QuantileSideModel(
        "turnover_change_5",
        "high",
        invalid_probability=0.05,
    ).fit(samples, strong_label)

    metadata = {
        "feature_columns": feature_columns,
        "rows": int(len(samples)),
        "buyability_label": buyability_label,
        "strong_label": strong_label,
        "model_type": "quantile_side_model",
        "models": {
            BUYABILITY_ARTIFACT: dict(buyability_model.training_summary),
            STRONG_ARTIFACT: dict(strong_model.training_summary),
        },
    }
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"

    _publish_artifacts(
        output_dir,
        {
            BUYABILITY_ARTIFACT: _pickle_bytes(buyability_model),
            STRONG_ARTIFACT: _pickle_bytes(strong_model),
            METADATA_ARTIFACT: metadata_text.encode("utf-8"),
        },
    )

    print(f"Wrote side models: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
