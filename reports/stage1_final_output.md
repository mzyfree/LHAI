# Stage 1 Final Output

## Stage Goal

第一阶段目标不是追求最优收益，而是建立一个稳定、可复现、可对比的 A 股量化 baseline。

## What Was Built

第一阶段已经完成了以下能力：

1. 搭建本地 `Qlib + Python 3.12` 研究环境
2. 下载并接入项目内 A 股 `cn_data`
3. 跑通 `Alpha158 + LightGBM + TopkDropout` 基线流程
4. 形成可复现的实验配置、运行脚本、报告和实验台账
5. 完成第一轮单变量实验与时间窗口稳健性验证

## Experiments Completed

### Baseline: CSI300

- Config: `src/configs/workflow_config_lightgbm_a_share.yaml`
- With cost annualized excess return: `7.54%`
- With cost IR: `0.7010`
- Max drawdown: `-13.02%`

### Experiment 1: Switch Pool to CSI500

- Config: `src/configs/workflow_config_lightgbm_csi500.yaml`
- With cost annualized excess return: `14.24%`
- With cost IR: `1.1887`
- Max drawdown: `-12.69%`

结论：

`CSI500` 明显优于 `CSI300`，成为阶段 1 的主股票池。

### Experiment 2: More Concentrated Holdings

- Config: `src/configs/workflow_config_lightgbm_csi500_top20.yaml`
- With cost annualized excess return: `11.68%`
- With cost IR: `0.7821`
- Max drawdown: `-13.38%`

结论：

把 `topk` 从 `30` 降到 `20` 没有提升表现，反而削弱了收益与稳定性。

### Experiment 3: Faster Turnover

- Config: `src/configs/workflow_config_lightgbm_csi500_drop5.yaml`
- With cost annualized excess return: `11.65%`
- With cost IR: `0.9580`
- Max drawdown: `-12.68%`

结论：

把 `n_drop` 从 `3` 提高到 `5` 没有带来更好结果，说明更高换手并不划算。

### Robustness Check: Later Time Window

- Config: `src/configs/workflow_config_lightgbm_csi500_robustness_2021.yaml`
- Test window: `2018-01-01` to `2020-09-24`
- IC: `0.0366`
- Rank IC: `0.0397`
- With cost annualized excess return: `9.39%`
- With cost IR: `0.7023`
- Max drawdown: `-12.73%`

结论：

方法在更靠后的时间窗口中仍然有效，但强度较初始窗口有所下降，说明该 baseline 具有一定稳健性，但并非在所有阶段都同样强。

## Stage 1 Final Decision

第一阶段固定主配置为：

- Market: `csi500`
- Benchmark: `SH000905`
- Features: `Alpha158`
- Model: `LightGBM`
- Strategy: `TopkDropout(topk=30, n_drop=3)`

## What Stage 1 Proved

第一阶段证明了三件事：

1. 这套 A 股机器学习选股 baseline 可以在本地完整跑通
2. `CSI500` 是当前更合适的主研究池
3. 当前 baseline 在不同时间窗口下具备一定稳健性

## Recommended Next Step

第二阶段建议从以下方向开始：

1. 调整预测周期或标签定义
2. 再尝试更强特征模板
3. 然后逐步引入第一类 AI 特征，例如公告/新闻文本特征
