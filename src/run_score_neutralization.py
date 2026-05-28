from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import qlib
from qlib.backtest import backtest as normal_backtest
from qlib.constant import REG_CN
from qlib.contrib.evaluate import indicator_analysis, risk_analysis
from qlib.data import D
from qlib.utils import fill_placeholder


def load_config(config_path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    return yaml.load(config_path.read_text(encoding="utf-8"))


def load_pred(path: Path) -> pd.DataFrame:
    pred = pd.read_pickle(path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if "score" not in pred.columns:
        raise ValueError(f"`score` column not found in {path}")
    pred = pred[["score"]].copy()
    pred.index = pred.index.set_names(["datetime", "instrument"])
    return pred.sort_index()


def load_industry_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = set(df.columns)
    if {"instrument", "industry"} <= cols:
        return df[["instrument", "industry"]].dropna().drop_duplicates()
    raise ValueError("Industry map must contain columns: instrument, industry")


def load_size_proxy(index: pd.MultiIndex) -> pd.Series:
    instruments = sorted(index.get_level_values("instrument").unique())
    dates = index.get_level_values("datetime")
    start_time = dates.min()
    end_time = dates.max()
    size_df = D.features(instruments, ["$close", "$volume"], start_time, end_time, freq="day")
    size_df.columns = ["close", "volume"]
    size = np.log((size_df["close"].abs() * size_df["volume"].abs()).clip(lower=1e-12))
    size.name = "size_proxy"
    size.index = size.index.set_names(["instrument", "datetime"])
    size = size.swaplevel().sort_index()
    return size.reindex(index)


def daily_zscore(series: pd.Series) -> pd.Series:
    def _z(group: pd.Series) -> pd.Series:
        std = group.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return series.groupby(level="datetime", group_keys=False).apply(_z)


def regress_residual(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    valid = y.notna()
    if x is not None:
        valid &= x.notna().all(axis=1)
    yv = y.loc[valid]
    if len(yv) < 5:
        return y.copy()

    if x is None or x.shape[1] == 0:
        return y - yv.mean()

    xv = x.loc[valid]
    xv = xv.astype(float)
    X = np.column_stack([np.ones(len(xv)), xv.values])
    beta, *_ = np.linalg.lstsq(X, yv.values, rcond=None)
    fitted = X @ beta
    resid = pd.Series(yv.values - fitted, index=yv.index)
    out = y.copy()
    out.loc[resid.index] = resid
    return out


def neutralize_scores(
    pred: pd.DataFrame,
    neutralize_size: bool,
    industry_map: pd.DataFrame | None,
) -> pd.DataFrame:
    df = pred.copy()
    size_proxy = load_size_proxy(df.index) if neutralize_size else None

    static_industry = None
    if industry_map is not None:
        static_industry = industry_map.set_index("instrument")["industry"]

    neutralized = []
    for dt, g in df.groupby(level="datetime"):
        score = g["score"].copy()
        exposures = []

        if neutralize_size and size_proxy is not None:
            size_g = size_proxy.loc[g.index]
            exposures.append(daily_zscore(size_g).loc[g.index].rename("size_proxy"))

        if static_industry is not None:
            inst = pd.Index(g.index.get_level_values("instrument"))
            ind = inst.map(static_industry)
            if ind.notna().any():
                ind_df = pd.get_dummies(pd.Series(ind, index=g.index, name="industry"), prefix="ind", dummy_na=False)
                if ind_df.shape[1] > 1:
                    ind_df = ind_df.iloc[:, :-1]
                exposures.append(ind_df)

        if exposures:
            x = pd.concat(exposures, axis=1)
            score = regress_residual(score, x)
        else:
            score = score - score.mean()

        neutralized.append(score.to_frame("score"))

    out = pd.concat(neutralized).sort_index()
    return out


def calc_ic(pred: pd.Series, label: pd.Series) -> tuple[float, float, float, float]:
    from scipy.stats import spearmanr

    df = pd.concat([pred.rename("score"), label.rename("label")], axis=1).dropna()
    ic_vals = []
    ric_vals = []
    for _, g in df.groupby(level="datetime"):
        if g["score"].nunique() < 2 or g["label"].nunique() < 2:
            continue
        ic_vals.append(g["score"].corr(g["label"]))
        ric_vals.append(spearmanr(g["score"], g["label"]).statistic)
    ic = pd.Series(ic_vals)
    ric = pd.Series(ric_vals)
    return float(ic.mean()), float(ic.mean() / ic.std()), float(ric.mean()), float(ric.mean() / ric.std())


def backtest_metrics(pred: pd.DataFrame, port_cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    strategy_config = port_cfg["strategy"]
    executor_config = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    backtest_config = port_cfg["backtest"].copy()

    placeholder_value = {"<PRED>": pred}
    strategy_config = fill_placeholder(strategy_config, placeholder_value)
    executor_config = fill_placeholder(executor_config, placeholder_value)

    portfolio_metric_dict, indicator_dict = normal_backtest(
        executor=executor_config,
        strategy=strategy_config,
        **backtest_config,
    )
    freq_key = "day" if "day" in portfolio_metric_dict else "1day"
    report_normal, _ = portfolio_metric_dict[freq_key]
    indicators_normal = indicator_dict[freq_key][0]

    analysis = {
        "benchmark": risk_analysis(report_normal["bench"], freq="day"),
        "excess_return_without_cost": risk_analysis(report_normal["return"] - report_normal["bench"], freq="day"),
        "excess_return_with_cost": risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"],
            freq="day",
        ),
    }
    indicator_df = indicator_analysis(indicators_normal)
    return analysis, indicator_df


def load_test_label(index: pd.MultiIndex) -> pd.Series:
    instruments = sorted(index.get_level_values("instrument").unique())
    dates = index.get_level_values("datetime")
    start_time = dates.min()
    end_time = dates.max()
    label_df = D.features(instruments, ["Ref($close, -2) / Ref($close, -1) - 1"], start_time, end_time, freq="day")
    label_df.columns = ["label"]
    label_df.index = label_df.index.set_names(["instrument", "datetime"])
    label_df = label_df.swaplevel().sort_index()
    return label_df["label"].reindex(index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Neutralize a prediction score and run the same backtest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--industry-map", default=None, help="Optional csv with columns instrument,industry")
    parser.add_argument("--neutralize-size", action="store_true", default=False)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    qlib_init = config["qlib_init"]
    provider_uri = str(Path(qlib_init["provider_uri"]).expanduser())
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    pred = load_pred(Path(args.pred).expanduser().resolve())
    industry_map = load_industry_map(Path(args.industry_map).expanduser().resolve()) if args.industry_map else None
    neutralized = neutralize_scores(pred, args.neutralize_size, industry_map)

    label = load_test_label(neutralized.index)
    ic, icir, ric, ricir = calc_ic(neutralized["score"], label)
    analysis, indicator_df = backtest_metrics(neutralized, config["port_analysis_config"])

    print("Neutralization setup:")
    print(f"- config: {config_path}")
    print(f"- pred: {args.pred}")
    print(f"- neutralize_size: {args.neutralize_size}")
    print(f"- industry_map: {args.industry_map or 'none'}")
    print()
    pprint({"IC": ic, "ICIR": icir, "Rank IC": ric, "Rank ICIR": ricir})
    print()
    print("The following are analysis results of benchmark return(1day).")
    pprint(analysis["benchmark"])
    print("The following are analysis results of the excess return without cost(1day).")
    pprint(analysis["excess_return_without_cost"])
    print("The following are analysis results of the excess return with cost(1day).")
    pprint(analysis["excess_return_with_cost"])
    print("The following are analysis results of indicators(1day).")
    pprint(indicator_df)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        report = f"""# Score Neutralization Report

## Setup

- Config: `{config_path}`
- Pred: `{Path(args.pred).name}`
- Neutralize size: `{args.neutralize_size}`
- Industry map: `{args.industry_map or "none"}`

## Signal Metrics

| Metric | Value |
| --- | ---: |
| IC | {ic:.6f} |
| ICIR | {icir:.6f} |
| Rank IC | {ric:.6f} |
| Rank ICIR | {ricir:.6f} |

## Excess Return With Cost (1day)

{analysis["excess_return_with_cost"].to_markdown()}

## Indicators (1day)

{indicator_df.to_markdown()}
"""
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
