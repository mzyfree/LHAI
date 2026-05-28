from __future__ import annotations

from pathlib import Path


RUNS = {
    "CSI300 baseline": {
        "experiment_name": "lhai_ashare_baseline",
        "run_id": "28cf151235c342a1a7c47aa9fa6b9436",
        "market": "csi300",
        "benchmark": "SH000300",
        "config": "src/configs/workflow_config_lightgbm_a_share.yaml",
    },
    "CSI500 baseline": {
        "experiment_name": "lhai_stage1_csi500",
        "run_id": "5dabbf2716bd4b2daec693e68820da51",
        "market": "csi500",
        "benchmark": "SH000905",
        "config": "src/configs/workflow_config_lightgbm_csi500.yaml",
    },
}

METRICS = {
    "IC": "IC",
    "Rank IC": "Rank IC",
    "Ann Return (with cost)": "1day.excess_return_with_cost.annualized_return",
    "IR (with cost)": "1day.excess_return_with_cost.information_ratio",
    "Max Drawdown (with cost)": "1day.excess_return_with_cost.max_drawdown",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def metric_value(run_dir: Path, metric_name: str) -> float:
    raw = read_text(run_dir / "metrics" / metric_name).split()
    return float(raw[1])


def fmt_metric(name: str, value: float) -> str:
    if "Return" in name or "Drawdown" in name:
        return f"{value:.4%}"
    return f"{value:.4f}"


def render_table(mlruns_dir: Path) -> str:
    headers = ["Metric", *RUNS.keys()]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for label, metric_name in METRICS.items():
        values = []
        for run in RUNS.values():
            run_dir = next(mlruns_dir.glob(f"*/{run['run_id']}"))
            values.append(fmt_metric(label, metric_value(run_dir, metric_name)))
        rows.append("| " + " | ".join([label, *values]) + " |")
    return "\n".join(rows)


def render_report(mlruns_dir: Path) -> str:
    table = render_table(mlruns_dir)
    return f"""# Stage 1 CSI300 vs CSI500

## Compared Experiments

- `CSI300 baseline`: `Alpha158 + LightGBM + TopkDropout(topk=30, n_drop=3)`
- `CSI500 baseline`: same setup, only changing stock pool and benchmark

## Metric Comparison

{table}

## Interpretation

- `CSI500` shows slightly lower raw `IC`, but similar positive ranking ability.
- `CSI500` is materially better on portfolio metrics after cost.
- In the current setup, `CSI500` is the better stage-1 main pool for follow-up experiments.

## Decision

阶段 1 后续默认主股票池切到 `CSI500`，接下来的单变量实验在 `CSI500` 上继续展开。
"""


def main() -> None:
    mlruns_dir = Path("mlruns").resolve()
    output = Path("reports/stage1_csi300_vs_csi500.md").resolve()
    output.write_text(render_report(mlruns_dir), encoding="utf-8")
    print(f"Wrote report to {output}")


if __name__ == "__main__":
    main()
