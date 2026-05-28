# Stage 1 CSI300 vs CSI500

## Compared Experiments

- `CSI300 baseline`: `Alpha158 + LightGBM + TopkDropout(topk=30, n_drop=3)`
- `CSI500 baseline`: same setup, only changing stock pool and benchmark

## Metric Comparison

| Metric | CSI300 baseline | CSI500 baseline |
| --- | --- | --- |
| IC | 0.0468 | 0.0391 |
| Rank IC | 0.0490 | 0.0465 |
| Ann Return (with cost) | 7.5360% | 14.2411% |
| IR (with cost) | 0.7010 | 1.1887 |
| Max Drawdown (with cost) | -13.0154% | -12.6852% |

## Interpretation

- `CSI500` shows slightly lower raw `IC`, but similar positive ranking ability.
- `CSI500` is materially better on portfolio metrics after cost.
- In the current setup, `CSI500` is the better stage-1 main pool for follow-up experiments.

## Decision

阶段 1 后续默认主股票池切到 `CSI500`，接下来的单变量实验在 `CSI500` 上继续展开。
