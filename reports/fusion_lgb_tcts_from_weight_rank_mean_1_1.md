# Prediction Fusion Report

## Setup

- Config: `/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500.yaml`
- Pred A: `pred.pkl`
- Pred B: `tcts_alpha360_csi500_from_weight.pkl`
- Method: `rank_mean`
- Weights: `1.0:1.0`

## Signal Metrics

| Metric | Value |
| --- | ---: |
| IC | 0.050954 |
| ICIR | 0.564666 |
| Rank IC | 0.062842 |
| Rank ICIR | 0.678669 |

## Excess Return With Cost (1day)

|                   |         risk |
|:------------------|-------------:|
| mean              |  0.000602355 |
| std               |  0.00533078  |
| annualized_return |  0.14336     |
| information_ratio |  1.74321     |
| max_drawdown      | -0.0646205   |

## Indicators (1day)

|     |   value |
|:----|--------:|
| ffr |       1 |
| pa  |       0 |
| pos |       0 |
