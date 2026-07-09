#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


MAIN_CONFIGS = [
    "workflow_config_xgb_csi1000_long_prod2026.yaml",
    "workflow_config_doubleensemble_csi1000_short_prod2026.yaml",
    "workflow_config_catboost_csi1000_long_prod2026.yaml",
]

LABEL_MODES = {
    "default": None,
    # T close data -> buy at T+1 open -> sell at T+2 close.
    # This aligns the training target with the short-hold execution path.
    "short_hold_v2": ["Ref($close, -2) / Ref($open, -1) - 1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weekly production training configs.")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--valid-start", required=True)
    parser.add_argument("--valid-end", required=True)
    parser.add_argument("--infer-date", required=True)
    parser.add_argument("--backtest-start", default=None)
    parser.add_argument("--label-mode", choices=sorted(LABEL_MODES), default="default")
    return parser.parse_args()


def patch_config(config: dict, args: argparse.Namespace) -> dict:
    config["qlib_init"]["provider_uri"] = args.provider_uri

    handler_kwargs = config["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    original_start = handler_kwargs["start_time"]
    handler_kwargs["end_time"] = args.infer_date
    handler_kwargs["fit_start_time"] = original_start
    handler_kwargs["fit_end_time"] = args.train_end
    label = LABEL_MODES[args.label_mode]
    if label is not None:
        handler_kwargs["label"] = label

    segments = config["task"]["dataset"]["kwargs"]["segments"]
    segments["train"] = [original_start, args.train_end]
    segments["valid"] = [args.valid_start, args.valid_end]
    segments["test"] = [args.infer_date, args.infer_date]

    if "port_analysis_config" in config:
        backtest = config["port_analysis_config"]["backtest"]
        backtest["start_time"] = args.backtest_start or args.valid_start
        backtest["end_time"] = args.valid_end

    return config


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in MAIN_CONFIGS:
        src = base_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        with src.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = patch_config(config, args)
        dst = output_dir / name
        with dst.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        written.append(dst)

    print("Generated weekly production configs:")
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
