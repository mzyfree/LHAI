#!/usr/bin/env python3
"""Run local inference from packaged Qlib models and refresh prediction pkls."""

from __future__ import annotations

import argparse
import copy
import pickle
import shutil
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import qlib
from qlib.data.dataset import Dataset
from qlib.utils import init_instance_by_config


DEFAULT_MODELS = {
    "xgb_csi1000_long_prod2026": "xgb_csi1000_long_prod2026.pkl",
    "doubleensemble_csi1000_short_prod2026": "doubleensemble_csi1000_short_prod2026.pkl",
    "catboost_csi1000_long_prod2026": "catboost_csi1000_long_prod2026.pkl",
}

PACKAGE_ROOT_PREFIXES = ("csi1000_main_", "csi1000_short_hold_v2_")


def latest_calendar_date(provider_uri: Path) -> str:
    calendar_path = provider_uri / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise SystemExit(f"Missing Qlib calendar: {calendar_path}")
    dates = [line.strip() for line in calendar_path.read_text().splitlines() if line.strip()]
    if not dates:
        raise SystemExit(f"Empty Qlib calendar: {calendar_path}")
    return dates[-1]


def ensure_extracted(package_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not package_path.exists():
        raise SystemExit(f"Model package not found: {package_path}")

    with tarfile.open(package_path, "r:gz") as tar:
        package_roots = sorted({
            Path(member.name).parts[0]
            for member in tar.getmembers()
            if Path(member.name).parts and Path(member.name).parts[0].startswith(PACKAGE_ROOT_PREFIXES)
        })
        for root in reversed(package_roots):
            candidate = extract_dir / root
            if candidate.is_dir():
                return candidate
        tar.extractall(extract_dir)

    candidates = [extract_dir / root for root in package_roots if (extract_dir / root).is_dir()]
    if not candidates:
        prefixes = ", ".join(f"{prefix}*" for prefix in PACKAGE_ROOT_PREFIXES)
        raise SystemExit(f"No supported package root ({prefixes}) found after extracting: {package_path}")
    return sorted(candidates)[-1]


def load_task(model_dir: Path) -> dict:
    task_path = model_dir / "artifacts" / "task"
    with task_path.open("rb") as f:
        task = pickle.load(f)
    if not isinstance(task, dict) or "dataset" not in task:
        raise SystemExit(f"Unexpected task artifact: {task_path}")
    return task


def build_dataset(task: dict, end_date: str) -> Dataset:
    dataset_config = copy.deepcopy(task["dataset"])
    handler_kwargs = dataset_config["kwargs"]["handler"]["kwargs"]
    handler_kwargs["end_time"] = end_date
    # Daily live inference cannot know the next-day label for the newest bar.
    # Training configs use DropnaLabel in learn_processors, which would drop
    # that newest bar and make T close -> T+1 open trading impossible.
    handler_kwargs["learn_processors"] = []
    dataset_config["kwargs"]["segments"]["test"] = ["2026-01-01", end_date]
    return init_instance_by_config(dataset_config, accept_types=Dataset)


def predict_one(model_dir: Path, provider_latest: str) -> pd.DataFrame:
    task = load_task(model_dir)
    dataset = build_dataset(task, provider_latest)

    with (model_dir / "artifacts" / "params.pkl").open("rb") as f:
        model = pickle.load(f)

    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    elif "score" not in pred.columns:
        if len(pred.columns) != 1:
            raise SystemExit(f"Prediction output has unexpected columns for {model_dir.name}: {list(pred.columns)}")
        pred = pred.rename(columns={pred.columns[0]: "score"})

    dates = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    if str(dates[-1].date()) != provider_latest:
        raise SystemExit(
            f"{model_dir.name} prediction ended at {dates[-1].date()}, "
            f"expected latest Qlib date {provider_latest}"
        )
    return pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--extract-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    provider_uri = Path(args.provider_uri).resolve()
    package_path = Path(args.package).resolve()
    extract_dir = Path(args.extract_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    provider_latest = latest_calendar_date(provider_uri)
    package_root = ensure_extracted(package_path, extract_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Qlib provider: {provider_uri}")
    print(f"Latest Qlib calendar date: {provider_latest}")
    print(f"Model package root: {package_root}")

    qlib.init(provider_uri=str(provider_uri), region="cn")

    staged: list[tuple[Path, Path]] = []
    # Keep the staging directory beside the final output so atomic replace works
    # even when /tmp and the output directory live on different filesystems.
    with tempfile.TemporaryDirectory(prefix=".pred_stage_", dir=output_dir) as tmp:
        tmp_dir = Path(tmp)
        for model_name, output_name in DEFAULT_MODELS.items():
            model_dir = package_root / model_name
            if not model_dir.exists():
                raise SystemExit(f"Missing model directory in package: {model_dir}")

            print(f"Predicting {model_name}...")
            pred = predict_one(model_dir, provider_latest)
            tmp_path = tmp_dir / output_name
            pred.to_pickle(tmp_path)

            dates = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
            print(f"- {output_name}: {dates[0].date()} -> {dates[-1].date()} ({len(dates)} dates)")
            staged.append((tmp_path, output_dir / output_name))

        for src, dst in staged:
            try:
                src.replace(dst)
            except OSError:
                shutil.copy2(src, dst)
                src.unlink()
            print(f"Wrote: {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
