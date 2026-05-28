# Stage 1 Baseline Report

## Experiment

- Name: `lhai_ashare_baseline`
- Run ID: `28cf151235c342a1a7c47aa9fa6b9436`
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

- IC: `0.0468`
- ICIR: `0.3816`
- Rank IC: `0.0490`
- Rank ICIR: `0.4068`

## Portfolio Results

### Excess Return With Cost

- Annualized return: `7.5360%`
- Information ratio: `0.7010`
- Max drawdown: `-13.0154%`

### Excess Return Without Cost

- Annualized return: `12.1462%`
- Information ratio: `1.1294`
- Max drawdown: `-11.4191%`

## Stage 1 Interpretation

这次 baseline 已经满足第一阶段目标：

1. 数据、特征、模型、策略、回测已经完整打通。
2. 预测指标为正，说明模型打分对未来收益存在一定排序能力。
3. 扣成本后仍有正的超额年化和正的信息比率，说明 baseline 值得继续迭代。

## Stage 1 Rules

- 第一阶段只改一个维度做对比，不同时改股票池、标签、模型。
- 所有后续实验都要和这份 baseline 做同口径比较。
- 优先改任务定义和股票池，再考虑更复杂模型。
