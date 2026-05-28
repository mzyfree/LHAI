from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("/root/autodl-tmp/llhh")
DEFAULT_SRC_ROOT = DEFAULT_PROJECT_ROOT / "src"
DEFAULT_PRED_DIR = DEFAULT_PROJECT_ROOT / "preds"
DEFAULT_REPORT_DIR = DEFAULT_PROJECT_ROOT / "reports"
DEFAULT_CONFIG_DIR = DEFAULT_SRC_ROOT / "configs/reference_rolling_xgb2_adarnn"
DEFAULT_DATA_DIR = DEFAULT_PROJECT_ROOT / "reference_cn_data/cn_data"
DEFAULT_BASE_XGB = DEFAULT_SRC_ROOT / "configs/reference_prod_like/workflow_config_xgboost_train2019_2023_2026_ytd_prod_like.yaml"
DEFAULT_BASE_ADARNN = DEFAULT_SRC_ROOT / "configs/reference_prod_like/workflow_config_adarnn_train2019_2023_2026_ytd_prod_like.yaml"


@dataclass(frozen=True)
class YearSpec:
    year: int
    xgb_short_train: tuple[str, str]
    xgb_long_train: tuple[str, str]
    adarnn_train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]


def year_spec(year: int, test_end: str | None = None) -> YearSpec:
    """Build leak-free rolling windows.

    For target year Y:
    - valid: Y-2 through Y-1
    - train end: Y-3
    - XGB short / ADARNN: 5-year train window
    - XGB long: 8-year train window
    """

    train_end_year = year - 3
    valid_start_year = year - 2
    valid_end_year = year - 1

    test_start = f"{year}-01-01"
    final_test_end = test_end if test_end is not None and year == 2026 else f"{year}-12-31"

    short_start_year = train_end_year - 4
    long_start_year = train_end_year - 7

    return YearSpec(
        year=year,
        xgb_short_train=(f"{short_start_year}-01-01", f"{train_end_year}-12-31"),
        xgb_long_train=(f"{long_start_year}-01-01", f"{train_end_year}-12-31"),
        adarnn_train=(f"{short_start_year}-01-01", f"{train_end_year}-12-31"),
        valid=(f"{valid_start_year}-01-01", f"{valid_end_year}-12-31"),
        test=(test_start, final_test_end),
    )


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, cfg: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def patch_workflow(
    cfg: dict,
    *,
    model_name: str,
    spec: YearSpec,
    train: tuple[str, str],
    provider_uri: Path,
    market: str,
    benchmark: str,
    tmp_root: Path,
) -> dict:
    cfg = dict(cfg)
    train_start, train_end = train
    valid_start, valid_end = spec.valid
    test_start, test_end = spec.test

    cfg.setdefault("qlib_init", {})["provider_uri"] = str(provider_uri)
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

    segments = cfg["task"]["dataset"]["kwargs"]["segments"]
    segments["train"] = [train_start, train_end]
    segments["valid"] = [valid_start, valid_end]
    segments["test"] = [test_start, test_end]

    backtest_cfg = cfg["port_analysis_config"]["backtest"]
    backtest_cfg["start_time"] = test_start
    backtest_cfg["end_time"] = test_end
    backtest_cfg["benchmark"] = benchmark
    backtest_cfg.setdefault("exchange_kwargs", {})["codes"] = market

    model_kwargs = cfg["task"]["model"].get("kwargs", {})
    if "save_path" in model_kwargs:
        model_kwargs["save_path"] = str(tmp_root / f"{model_name}_{spec.year}")

    return cfg


def config_path(config_dir: Path, model_name: str, year: int) -> Path:
    return config_dir / f"workflow_config_{model_name}_{year}.yaml"


def pred_path(pred_dir: Path, model_name: str, year: int) -> Path:
    # Keep compatibility with the existing safe_run_copy_prod_like helper.
    return pred_dir / f"{model_name}_{year}_2026_ytd_prod_like.pkl"


def gen_configs(args: argparse.Namespace) -> None:
    base_xgb = load_yaml(Path(args.base_xgb))
    base_adarnn = load_yaml(Path(args.base_adarnn))
    config_dir = Path(args.config_dir)
    provider_uri = Path(args.provider_uri)
    tmp_root = Path(args.tmp_root)

    for year in args.years:
        spec = year_spec(year, test_end=args.test_end)
        jobs = [
            ("xgboost_short", base_xgb, spec.xgb_short_train),
            ("xgboost_long", base_xgb, spec.xgb_long_train),
            ("adarnn_short", base_adarnn, spec.adarnn_train),
        ]
        for model_name, base_cfg, train in jobs:
            cfg = patch_workflow(
                base_cfg,
                model_name=model_name,
                spec=spec,
                train=train,
                provider_uri=provider_uri,
                market=args.market,
                benchmark=args.benchmark,
                tmp_root=tmp_root,
            )
            out = config_path(config_dir, model_name, year)
            write_yaml(out, cfg)
            print(out)


def print_train_commands(args: argparse.Namespace) -> None:
    config_dir = Path(args.config_dir)
    pred_dir = Path(args.pred_dir)
    print("cd /root/autodl-tmp/llhh/src")
    print("export PYTHONPATH=/root/autodl-tmp/llhh/qlib:$PYTHONPATH")
    print("source /root/autodl-tmp/llhh/src/prod_like_helpers.sh")
    print()
    for year in args.years:
        for model_name in ["xgboost_short", "xgboost_long", "adarnn_short"]:
            pred = pred_path(pred_dir, model_name, year)
            cfg = config_path(config_dir, model_name, year)
            print(f'if [ -f "{pred}" ]; then')
            print(f'  echo "SKIP exists: {pred}"')
            print("else")
            print(f'  safe_run_copy_prod_like "{model_name}_{year}" "{cfg}"')
            print("fi")
            print()


def print_fusion_commands(args: argparse.Namespace) -> None:
    config_dir = Path(args.config_dir)
    report_dir = Path(args.report_dir)
    pred_dir = Path(args.pred_dir)
    print("cd /root/autodl-tmp/llhh/src")
    print("export PYTHONPATH=/root/autodl-tmp/llhh/qlib:$PYTHONPATH")
    print()
    for year in args.years:
        print(f'echo "===== xgb2+adarnn fixed 8:1:1 {year} ====="')
        print("python run_prod_like_ensemble_search.py \\")
        print(f'  --config "{config_path(config_dir, "xgboost_short", year)}" \\')
        print(f'  --models "xgboost_short_{year},xgboost_long_{year},adarnn_short_{year}" \\')
        print('  --methods "zscore_mean" \\')
        print('  --weight-sets "8,1,1" \\')
        print(f'  --topk-pairs "{args.topk}:{args.n_drop}" \\')
        print(f'  --output "{report_dir / f"rolling_xgb2_adarnn_fixed_8_1_1_{year}_top{args.topk}_drop{args.n_drop}.csv"}" \\')
        print(f'  --save-best-pred "{pred_dir / f"rolling_xgb2_adarnn_fixed_8_1_1_{year}_top{args.topk}_drop{args.n_drop}.pkl"}"')
        print()

        print(f'echo "===== xgb2 fixed 8:1 {year} ====="')
        print("python run_prod_like_ensemble_search.py \\")
        print(f'  --config "{config_path(config_dir, "xgboost_short", year)}" \\')
        print(f'  --models "xgboost_short_{year},xgboost_long_{year}" \\')
        print('  --methods "zscore_mean" \\')
        print('  --weight-sets "8,1" \\')
        print(f'  --topk-pairs "{args.topk}:{args.n_drop}" \\')
        print(f'  --output "{report_dir / f"rolling_xgb2_fixed_8_1_{year}_top{args.topk}_drop{args.n_drop}.csv"}" \\')
        print(f'  --save-best-pred "{pred_dir / f"rolling_xgb2_fixed_8_1_{year}_top{args.topk}_drop{args.n_drop}.pkl"}"')
        print()


def summarize(args: argparse.Namespace) -> None:
    import pandas as pd

    report_dir = Path(args.report_dir)
    rows = []
    patterns = [
        ("xgb2_adarnn_8_1_1", f"rolling_xgb2_adarnn_fixed_8_1_1_*_top{args.topk}_drop{args.n_drop}.csv"),
        ("xgb2_8_1", f"rolling_xgb2_fixed_8_1_*_top{args.topk}_drop{args.n_drop}.csv"),
    ]
    for group, pattern in patterns:
        for path in report_dir.glob(pattern):
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            stem = path.stem
            year = stem.split("_top")[0].split("_")[-1]
            row["group"] = group
            row["year"] = int(year)
            row["file"] = path.name
            rows.append(row)

    if not rows:
        print("No result rows found.")
        return

    out = pd.DataFrame(rows).sort_values(["year", "group"])
    cols = [
        "group",
        "year",
        "models",
        "method",
        "weights",
        "topk",
        "n_drop",
        "annualized_return",
        "information_ratio",
        "max_drawdown",
        "IC",
        "Rank IC",
        "file",
    ]
    print(out[cols].to_string(index=False))
    print("\n===== by group mean =====")
    print(
        out.groupby("group")[["annualized_return", "information_ratio", "max_drawdown", "IC", "Rank IC"]]
        .mean()
        .to_string()
    )


def parse_years(value: str) -> list[int]:
    years = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XGB short + XGB long + ADARNN rolling pipeline helper.")
    parser.add_argument(
        "action",
        choices=["gen-configs", "print-train", "print-fusion", "summarize"],
        help="Pipeline action.",
    )
    parser.add_argument("--years", type=parse_years, default=parse_years("2023-2026"))
    parser.add_argument("--test-end", default="2026-05-08", help="Final test end for 2026.")
    parser.add_argument("--market", default="csi500")
    parser.add_argument("--benchmark", default="SH000905")
    parser.add_argument("--provider-uri", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--base-xgb", default=str(DEFAULT_BASE_XGB))
    parser.add_argument("--base-adarnn", default=str(DEFAULT_BASE_ADARNN))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--tmp-root", default=str(DEFAULT_PROJECT_ROOT / "tmp"))
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--n-drop", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "gen-configs":
        gen_configs(args)
    elif args.action == "print-train":
        print_train_commands(args)
    elif args.action == "print-fusion":
        print_fusion_commands(args)
    elif args.action == "summarize":
        summarize(args)
    else:
        raise ValueError(args.action)


if __name__ == "__main__":
    main()
