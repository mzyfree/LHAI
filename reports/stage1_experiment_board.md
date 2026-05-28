# Stage 1 Experiment Board

## Goal

第一阶段目标不是追求最优收益，而是建立一个稳定、可比较、可解释的 A 股 baseline 研究流程。

## Experiment Rules

- 每次实验只改一个维度
- 保留相同的评估口径：IC、Rank IC、超额年化、IR、回撤
- 先改股票池和策略参数，再改模型超参数
- 所有实验都要和 baseline 对照

## Baseline

- Status: done
- Name: `lhai_ashare_baseline`
- Config: `src/configs/workflow_config_lightgbm_a_share.yaml`
- Market: `csi300`
- Benchmark: `SH000300`
- Notes: 作为阶段 1 的对照基线

## Planned Experiments

### Experiment 1: CSI500 Pool

- Status: done
- Suggested experiment name: `lhai_stage1_csi500`
- Config: `src/configs/workflow_config_lightgbm_csi500.yaml`
- Changed dimension: 股票池与基准
- Fixed dimensions:
  - `Alpha158`
  - `LightGBM`
  - `TopkDropout(topk=30, n_drop=3)`
  - 相同 train / valid / test 时间切分
- Question: 这套 baseline 在中盘风格股票池里是否依然有效？
- Result: `CSI500` 扣成本超额年化 `14.24%`，明显高于 `CSI300` 的 `7.54%`
- Decision: 阶段 1 后续默认主股票池切到 `CSI500`

### Experiment 2: More Concentrated Portfolio

- Status: done
- Suggested experiment name: `lhai_stage1_top20`
- Config: `src/configs/workflow_config_lightgbm_csi500_top20.yaml`
- Changed dimension: `topk=20`
- Question: 更集中持仓是否能提升超额收益，同时是否会放大回撤？
- Result: 扣成本超额年化 `11.68%`，低于 `CSI500 topk=30` 的 `14.24%`
- Result: 扣成本 IR `0.7821`，低于 `topk=30` 的 `1.1887`
- Result: 最大回撤 `-13.38%`，略差于 `topk=30` 的 `-12.69%`
- Decision: 当前阶段保留 `topk=30`，不切换到更集中持仓

### Experiment 3: Faster Turnover

- Status: done
- Suggested experiment name: `lhai_stage1_drop5`
- Config: `src/configs/workflow_config_lightgbm_csi500_drop5.yaml`
- Changed dimension: `n_drop=5`
- Question: 更高换手是否能提升信号兑现效率，还是只是让成本吃掉收益？
- Result: 扣成本超额年化 `11.65%`，低于 `CSI500 topk=30, n_drop=3` 的 `14.24%`
- Result: 扣成本 IR `0.9580`，低于 `n_drop=3` 的 `1.1887`
- Result: 最大回撤 `-12.68%`，与 `n_drop=3` 接近
- Decision: 当前阶段保留 `n_drop=3`，不切换到更高换手

## How To Use

运行实验：

```bash
python src/run_qlib_workflow.py --config src/configs/workflow_config_lightgbm_csi500.yaml --experiment-name lhai_stage1_csi500
```

生成最新一次运行报告：

```bash
python src/generate_stage1_report.py
```
