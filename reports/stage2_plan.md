# Stage 2 Plan

## Goal

第二阶段先不直接上文本或 LLM 特征，而是先调整预测任务定义，确认更适合当前研究框架的预测周期。

## Why Start Here

第一阶段已经找到了相对稳定的 baseline：

- `csi500`
- `Alpha158`
- `LightGBM`
- `TopkDropout(topk=30, n_drop=3)`

下一步最应该先确认的是：

`我们到底应该预测多长周期的收益？`

如果标签定义本身不合理，后面即使引入 AI 特征，也容易建立在错误任务上。

## Stage 2 First Task

先做标签周期实验。

### Baseline Label

Qlib `Alpha158` 默认标签：

- `Ref($close, -2)/Ref($close, -1) - 1`

可以理解为非常短周期的未来收益标签。

### New Experiment

新增一个更长持有周期的标签配置：

- `Ref($close, -6)/Ref($close, -1) - 1`

可以理解为从 `T+1` 到 `T+6` 的未来 5 个交易日收益。

## New Config

- `src/configs/workflow_config_lightgbm_csi500_h5.yaml`

## Expected Learning

这个实验会回答几个关键问题：

1. 更长持有期是否比短周期更稳定？
2. 当前 `Alpha158 + LightGBM` 是否更适合做短周期还是 5 日周期预测？
3. 第二阶段后续是更应该围绕“短周期 alpha”还是“更平滑的中短周期 alpha”继续做？

## Recommended Order

1. 跑 `csi500` baseline 对照
2. 跑 `csi500_h5`
3. 对比 `IC / Rank IC / 扣成本超额年化 / IR / 回撤`
4. 再跑 `10日` 标签
5. 最后再决定是否转向特征增强
