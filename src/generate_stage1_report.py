from __future__ import annotations

import argparse
from pathlib import Path


EXPERIMENT_NAME = "lhai_ashare_baseline"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def latest_run_dir(mlruns_dir: Path) -> Path:
    candidates = []
    for path in mlruns_dir.glob("*/*/meta.yaml"):
        if path.parent.name == "artifacts":
            continue
        run_dir = path.parent
        if run_dir.name == "0":
            continue
        try:
            meta = read_text(path)
        except OSError:
            continue
        if "status: 3" not in meta:
            continue
        candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError("No completed MLflow runs found in mlruns.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def metric_value(metrics_dir: Path, name: str) -> float:
    raw = read_text(metrics_dir / name).split()
    return float(raw[1])


def render_report(run_dir: Path) -> str:
    metrics_dir = run_dir / "metrics"
    run_id = run_dir.name

    metrics = {
        "IC": metric_value(metrics_dir, "IC"),
        "ICIR": metric_value(metrics_dir, "ICIR"),
        "Rank IC": metric_value(metrics_dir, "Rank IC"),
        "Rank ICIR": metric_value(metrics_dir, "Rank ICIR"),
        "ann_return_wc": metric_value(metrics_dir, "1day.excess_return_with_cost.annualized_return"),
        "ir_wc": metric_value(metrics_dir, "1day.excess_return_with_cost.information_ratio"),
        "mdd_wc": metric_value(metrics_dir, "1day.excess_return_with_cost.max_drawdown"),
        "ann_return_woc": metric_value(metrics_dir, "1day.excess_return_without_cost.annualized_return"),
        "ir_woc": metric_value(metrics_dir, "1day.excess_return_without_cost.information_ratio"),
        "mdd_woc": metric_value(metrics_dir, "1day.excess_return_without_cost.max_drawdown"),
    }

    return f"""# Stage 1 Baseline Report

## Experiment

- Name: `{EXPERIMENT_NAME}`
- Run ID: `{run_id}`
- Config: `src/configs/workflow_config_lightgbm_a_share.yaml`
- Data: `data/qlib_cn_data`
- Market: `csi300`
- Benchmark: `SH000300`
- Features: `Alpha158`
- Model: `LightGBM`
- Strategy: `TopkDropoutStrategy(topk=30, n_drop=3)`
- Train: `2008-01-01` to `2014-12-31`
- Valid: `2015-01-01` to `2016-12-31`
- Test: `2017-01-01` to `2020-08-01`

## Prediction Quality

- IC: `{metrics["IC"]:.4f}`
- ICIR: `{metrics["ICIR"]:.4f}`
- Rank IC: `{metrics["Rank IC"]:.4f}`
- Rank ICIR: `{metrics["Rank ICIR"]:.4f}`

## Portfolio Results

### Excess Return With Cost

- Annualized return: `{metrics["ann_return_wc"]:.4%}`
- Information ratio: `{metrics["ir_wc"]:.4f}`
- Max drawdown: `{metrics["mdd_wc"]:.4%}`

### Excess Return Without Cost

- Annualized return: `{metrics["ann_return_woc"]:.4%}`
- Information ratio: `{metrics["ir_woc"]:.4f}`
- Max drawdown: `{metrics["mdd_woc"]:.4%}`

## Stage 1 Interpretation

这次 baseline 已经满足第一阶段目标：

1. 数据、特征、模型、策略、回测已经完整打通。
2. 预测指标为正，说明模型打分对未来收益存在一定排序能力。
3. 扣成本后仍有正的超额年化和正的信息比率，说明 baseline 值得继续迭代。

## Stage 1 Rules

- 第一阶段只改一个维度做对比，不同时改股票池、标签、模型。
- 所有后续实验都要和这份 baseline 做同口径比较。
- 优先改任务定义和股票池，再考虑更复杂模型。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a markdown report for the latest stage-1 baseline run.")
    parser.add_argument("--mlruns-dir", default="mlruns", help="Path to the local MLflow tracking directory.")
    parser.add_argument(
        "--output",
        default="reports/stage1_baseline.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    mlruns_dir = Path(args.mlruns_dir).resolve()
    output_path = Path(args.output).resolve()

    run_dir = latest_run_dir(mlruns_dir)
    report = render_report(run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
