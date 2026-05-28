from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


PAPER_HOME = Path(os.environ.get("PAPER_HOME", Path(__file__).resolve().parents[1])).expanduser().resolve()
QLIB_SRC = Path(os.environ.get("QLIB_SRC", "")).expanduser().resolve()
PROVIDER_URI = os.environ["QLIB_PROVIDER_URI"]
OUT_DIR = PAPER_HOME / "config" / "smallcap"


MODELS = {
    "lgb": ("LightGBM/workflow_config_lightgbm_Alpha158.yaml", "Alpha158"),
    "xgb": ("XGBoost/workflow_config_xgboost_Alpha158.yaml", "Alpha158"),
    "linear": ("Linear/workflow_config_linear_Alpha158.yaml", "Alpha158"),
    "catboost": ("CatBoost/workflow_config_catboost_Alpha158.yaml", "Alpha158"),
    "doubleensemble": ("DoubleEnsemble/workflow_config_doubleensemble_Alpha158.yaml", "Alpha158"),
    "alstm": ("ALSTM/workflow_config_alstm_Alpha158.yaml", "Alpha158"),
    "adarnn": ("ADARNN/workflow_config_adarnn_Alpha360.yaml", "Alpha360"),
}

WINDOWS = {
    "short": ("2021-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01"),
    "mid": ("2019-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01"),
    "long": ("2016-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01"),
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(path)


def parse_market(item: str) -> tuple[str, str]:
    if ":" not in item:
        raise ValueError(f"Market must be like csi1000:SH000852, got {item}")
    market, benchmark = item.split(":", 1)
    return market.strip(), benchmark.strip()


def patch_common(
    cfg: dict,
    *,
    market: str,
    benchmark: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
) -> dict:
    cfg["qlib_init"]["provider_uri"] = PROVIDER_URI
    cfg["market"] = market
    cfg["benchmark"] = benchmark

    dh = cfg["data_handler_config"]
    dh["start_time"] = train_start
    dh["end_time"] = test_end
    dh["fit_start_time"] = train_start
    dh["fit_end_time"] = train_end
    dh["instruments"] = market

    dataset_kwargs = cfg["task"]["dataset"]["kwargs"]
    handler_kwargs = dataset_kwargs["handler"]["kwargs"]
    handler_kwargs["start_time"] = train_start
    handler_kwargs["end_time"] = test_end
    handler_kwargs["fit_start_time"] = train_start
    handler_kwargs["fit_end_time"] = train_end
    handler_kwargs["instruments"] = market

    segments = dataset_kwargs["segments"]
    segments["train"] = [train_start, train_end]
    segments["valid"] = [valid_start, valid_end]
    segments["test"] = [test_start, test_end]

    backtest = cfg["port_analysis_config"]["backtest"]
    backtest["start_time"] = test_start
    backtest["end_time"] = test_end
    backtest["benchmark"] = benchmark
    backtest.setdefault("exchange_kwargs", {})["codes"] = market

    strategy = cfg["port_analysis_config"]["strategy"]["kwargs"]
    strategy["topk"] = 20
    strategy["n_drop"] = 2
    return cfg


def tune_model(cfg: dict, model: str, profile: str, device: str) -> dict:
    kwargs = cfg["task"]["model"].setdefault("kwargs", {})
    is_smoke = profile == "smoke"

    if model == "lgb":
        kwargs["num_threads"] = 16
    elif model == "xgb":
        kwargs["nthread"] = 16
        kwargs["n_estimators"] = 50 if is_smoke else 647
    elif model == "catboost":
        kwargs["thread_count"] = 16
        kwargs["iterations"] = 80 if is_smoke else 600
        if device == "gpu":
            kwargs["task_type"] = "GPU"
    elif model == "doubleensemble":
        kwargs["num_threads"] = 16
        kwargs["num_models"] = 2 if is_smoke else 3
        kwargs["epochs"] = 2 if is_smoke else 12
    elif model in {"alstm", "adarnn"}:
        kwargs["GPU"] = 0 if device == "gpu" else -1
        kwargs["n_epochs"] = 2 if is_smoke else 120
        kwargs["early_stop"] = 2 if is_smoke else 15
        kwargs["batch_size"] = 800
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate small-cap model configs by market/window/model.")
    parser.add_argument("--markets", default="csi1000:SH000852,csi2000:SH000932")
    parser.add_argument("--models", default="lgb,xgb,linear,catboost,doubleensemble,alstm,adarnn")
    parser.add_argument("--windows", default="short,mid,long")
    parser.add_argument("--profile", choices=["prod", "smoke"], default="prod")
    parser.add_argument("--device", choices=["cpu", "gpu"], default=os.environ.get("DEVICE", "gpu"))
    parser.add_argument("--test-end", default=os.environ.get("PROD_TEST_END", "2026-05-18"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not QLIB_SRC.exists():
        raise FileNotFoundError(f"QLIB_SRC is required: {QLIB_SRC}")

    markets = [parse_market(item) for item in args.markets.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    windows = [item.strip() for item in args.windows.split(",") if item.strip()]

    for model in models:
        if model not in MODELS:
            raise ValueError(f"Unknown model {model}. Available: {', '.join(MODELS)}")
    for window in windows:
        if window not in WINDOWS:
            raise ValueError(f"Unknown window {window}. Available: {', '.join(WINDOWS)}")

    for market, benchmark in markets:
        for window in windows:
            train_start, train_end, valid_start, valid_end, test_start = WINDOWS[window]
            for model in models:
                rel_template, _feature = MODELS[model]
                cfg = load_yaml(QLIB_SRC / "examples" / "benchmarks" / rel_template)
                cfg = patch_common(
                    cfg,
                    market=market,
                    benchmark=benchmark,
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=args.test_end,
                )
                cfg = tune_model(cfg, model, args.profile, args.device)
                name = f"workflow_config_{model}_{market}_{window}_{args.profile}2026.yaml"
                write_yaml(OUT_DIR / market / name, cfg)


if __name__ == "__main__":
    main()
