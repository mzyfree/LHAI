# Stage 1 Mid Summary

## Current Best Setup

当前阶段 1 的最优配置为：

- Market: `csi500`
- Benchmark: `SH000905`
- Features: `Alpha158`
- Model: `LightGBM`
- Strategy: `TopkDropout(topk=30, n_drop=3)`
- Train: `2008-01-01` to `2014-12-31`
- Valid: `2015-01-01` to `2016-12-31`
- Test: `2017-01-01` to `2020-08-01`

## What We Learned

### 1. CSI500 is better than CSI300 in the current setup

- `CSI300` with cost annualized excess return: `7.54%`
- `CSI500` with cost annualized excess return: `14.24%`
- `CSI500` with cost IR: `1.1887`

结论：

在当前 `Alpha158 + LightGBM + TopkDropout` 设定下，`CSI500` 是更合适的阶段 1 主股票池。

### 2. More concentrated holdings did not help

- `CSI500 topk=30`: with cost annualized excess return `14.24%`
- `CSI500 topk=20`: with cost annualized excess return `11.68%`

结论：

更集中持仓并没有带来更好结果，反而降低了收益和稳定性。

### 3. Faster turnover did not help

- `CSI500 n_drop=3`: with cost annualized excess return `14.24%`
- `CSI500 n_drop=5`: with cost annualized excess return `11.65%`

结论：

更高换手没有提升表现，反而使扣成本收益下降。

## Stage 1 Decision

阶段 1 当前主配置正式固定为：

- `csi500`
- `topk=30`
- `n_drop=3`

## Next Step

下一步不再继续盲调策略参数，而是开始做稳健性验证。

建议的下一个实验方向：

### Time Window Robustness

保持以下内容不变：

- `csi500`
- `Alpha158`
- `LightGBM`
- `TopkDropout(topk=30, n_drop=3)`

只改变时间切分，验证当前结论是否只在 `2017-01-01` 到 `2020-08-01` 这段测试区间成立。

目标：

- 看结果是否能穿越不同市场环境
- 判断当前 baseline 是否只是样本内幸运，还是具有一定稳健性
