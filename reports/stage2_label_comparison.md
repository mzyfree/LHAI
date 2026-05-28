# Stage 2 Label Comparison

## Goal

比较不同预测周期标签，确认第二阶段更适合围绕哪种时间尺度继续做研究。

## Compared Setups

所有实验保持以下内容不变：

- `market: csi500`
- `benchmark: SH000905`
- `features: Alpha158`
- `model: LightGBM`
- `strategy: TopkDropout(topk=30, n_drop=3)`
- 相同 train / valid / test 时间切分

唯一变化是标签定义：

- Short label: `Ref($close, -2)/Ref($close, -1) - 1`
- H5 label: `Ref($close, -6)/Ref($close, -1) - 1`
- H10 label: `Ref($close, -11)/Ref($close, -1) - 1`

## Metric Comparison

| Metric | Short | H5 | H10 |
| --- | --- | --- | --- |
| IC | 0.0391 | 0.0644 | 0.0709 |
| Rank IC | 0.0465 | 0.0771 | 0.0869 |
| Ann Return (with cost) | 14.24% | 12.37% | 12.65% |
| IR (with cost) | 1.1887 | 1.2239 | 1.2619 |
| Max Drawdown (with cost) | -12.69% | -10.34% | -12.14% |

## Interpretation

### Short Label

- 超额年化最高
- 但预测相关性明显弱于更长标签

### H5 Label

- 预测层指标明显提升
- 组合层更平滑，回撤更小

### H10 Label

- 预测层指标最好
- 扣成本后 IR 最好
- 年化仍然保持在较高水平

## Decision

第二阶段当前主标签切换为：

- `H10`

原因：

1. `IC` 和 `Rank IC` 最强
2. 扣成本后 `IR` 最好
3. 收益和回撤的综合平衡更适合单人量化继续推进

## Next Step

接下来在 `CSI500 + H10` 设定上引入第一类 AI 特征：

- 公告/新闻文本事件特征
- 情绪或事件方向特征
- 后续再逐步升级到 embedding / LLM 特征
