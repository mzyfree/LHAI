from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


PAPER_HOME = Path(os.environ.get("PAPER_HOME", Path(__file__).resolve().parents[1])).expanduser().resolve()
QLIB_SRC = Path(os.environ.get("QLIB_SRC", "")).expanduser().resolve()
PROVIDER_URI = os.environ["QLIB_PROVIDER_URI"]
OUT_DIR = PAPER_HOME / "config" / "generated"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(path)


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

    hkw = cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    hkw["start_time"] = train_start
    hkw["end_time"] = test_end
    hkw["fit_start_time"] = train_start
    hkw["fit_end_time"] = train_end
    hkw["instruments"] = market

    seg = cfg["task"]["dataset"]["kwargs"]["segments"]
    seg["train"] = [train_start, train_end]
    seg["valid"] = [valid_start, valid_end]
    seg["test"] = [test_start, test_end]

    bt = cfg["port_analysis_config"]["backtest"]
    bt["start_time"] = test_start
    bt["end_time"] = test_end
    bt["benchmark"] = benchmark
    bt.setdefault("exchange_kwargs", {})["codes"] = market
    return cfg


def make_xgb(
    name: str,
    train_start: str,
    mode: str,
    *,
    market: str,
    benchmark: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
) -> None:
    base = QLIB_SRC / "examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml"
    cfg = patch_common(
        load_yaml(base),
        market=market,
        benchmark=benchmark,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
    )
    kwargs = cfg["task"]["model"]["kwargs"]
    kwargs["nthread"] = 8
    kwargs["n_estimators"] = 30 if mode == "smoke" else 300 if mode == "timing" else 647
    write_yaml(OUT_DIR / f"workflow_config_{name}_{mode}.yaml", cfg)


def make_adarnn(
    name: str,
    mode: str,
    *,
    market: str,
    benchmark: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
) -> None:
    base = QLIB_SRC / "examples/benchmarks/ADARNN/workflow_config_adarnn_Alpha360.yaml"
    cfg = patch_common(
        load_yaml(base),
        market=market,
        benchmark=benchmark,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
    )
    kwargs = cfg["task"]["model"]["kwargs"]
    kwargs["GPU"] = -1
    kwargs["n_epochs"] = 2 if mode == "smoke" else 30 if mode == "timing" else 200
    kwargs["early_stop"] = 2 if mode == "smoke" else 5 if mode == "timing" else 20
    kwargs["batch_size"] = 800
    write_yaml(OUT_DIR / f"workflow_config_{name}_{mode}.yaml", cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portable paper-trading Qlib workflow configs.")
    parser.add_argument("--market", default=os.environ.get("MARKET", "csi500"))
    parser.add_argument("--benchmark", default=os.environ.get("BENCHMARK", "SH000905"))
    parser.add_argument("--tag", default=os.environ.get("CONFIG_TAG", ""))
    parser.add_argument("--prod-test-end", default=os.environ.get("PROD_TEST_END", "2026-05-18"))
    return parser.parse_args()


def with_tag(name: str, tag: str) -> str:
    return f"{name}_{tag}" if tag else name


def main() -> None:
    args = parse_args()
    if not QLIB_SRC.exists():
        raise FileNotFoundError(f"QLIB_SRC is required for local timing config generation: {QLIB_SRC}")
    suffix = args.tag or args.market
    xgb_short = with_tag("xgboost_short", suffix)
    xgb_long = with_tag("xgboost_long", suffix)
    adarnn_short = with_tag("adarnn_short", suffix)
    for mode in ["smoke", "timing"]:
        make_xgb(
            xgb_short,
            "2014-01-01",
            mode,
            market=args.market,
            benchmark=args.benchmark,
            train_end="2018-12-31",
            valid_start="2019-01-01",
            valid_end="2019-12-31",
            test_start="2020-01-01",
            test_end="2020-09-25",
        )
        make_xgb(
            xgb_long,
            "2011-01-01",
            mode,
            market=args.market,
            benchmark=args.benchmark,
            train_end="2018-12-31",
            valid_start="2019-01-01",
            valid_end="2019-12-31",
            test_start="2020-01-01",
            test_end="2020-09-25",
        )
        make_adarnn(
            adarnn_short,
            mode,
            market=args.market,
            benchmark=args.benchmark,
            train_start="2014-01-01",
            train_end="2018-12-31",
            valid_start="2019-01-01",
            valid_end="2019-12-31",
            test_start="2020-01-01",
            test_end="2020-09-25",
        )

    make_xgb(
        xgb_short,
        "2019-01-01",
        "prod2026",
        market=args.market,
        benchmark=args.benchmark,
        train_end="2023-12-31",
        valid_start="2024-01-01",
        valid_end="2025-12-31",
        test_start="2026-01-01",
        test_end=args.prod_test_end,
    )
    make_xgb(
        xgb_long,
        "2016-01-01",
        "prod2026",
        market=args.market,
        benchmark=args.benchmark,
        train_end="2023-12-31",
        valid_start="2024-01-01",
        valid_end="2025-12-31",
        test_start="2026-01-01",
        test_end=args.prod_test_end,
    )
    make_adarnn(
        adarnn_short,
        "prod2026",
        market=args.market,
        benchmark=args.benchmark,
        train_start="2019-01-01",
        train_end="2023-12-31",
        valid_start="2024-01-01",
        valid_end="2025-12-31",
        test_start="2026-01-01",
        test_end=args.prod_test_end,
    )


if __name__ == "__main__":
    main()
