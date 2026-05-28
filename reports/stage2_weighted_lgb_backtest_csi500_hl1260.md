# Stage 2 Weighted LightGBM Backtest

## Setup

- Config: `/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500.yaml`
- Weighting scheme: `time_decay`
- Half-life days: `1260`
- Saved plain pred: `/Users/Dylan.Min/Documents/Code/learn/LHAI/preds/weighted_lgb_plain.pkl`
- Saved weighted pred: `/Users/Dylan.Min/Documents/Code/learn/LHAI/preds/weighted_lgb_time_decay_hl_1260.pkl`

## Weight Example

- Newest train sample weight: `1.000000`
- Oldest train sample weight: `0.245233`
- Median train sample weight: `0.500550`

## Signal Comparison

| Metric | Plain | Time-decay weighted |
| --- | ---: | ---: |
| IC | 0.0060 | 0.0015 |
| ICIR | 0.0568 | 0.0147 |
| Rank IC | 0.0150 | -0.0045 |
| Rank ICIR | 0.1426 | -0.0425 |

## Backtest Comparison (Excess Return With Cost, 1day)

| Metric | Plain | Time-decay weighted |
| --- | ---: | ---: |
| annualized_return | -0.063852 | -0.063852 |
| information_ratio | -0.876885 | -0.876885 |
| max_drawdown | -0.282011 | -0.282011 |

## Plain Backtest Detail

|                   |         risk |
|:------------------|-------------:|
| mean              | -0.000268286 |
| std               |  0.00472002  |
| annualized_return | -0.0638521   |
| information_ratio | -0.876885    |
| max_drawdown      | -0.282011    |

## Weighted Backtest Detail

|                   |         risk |
|:------------------|-------------:|
| mean              | -0.000268286 |
| std               |  0.00472002  |
| annualized_return | -0.0638521   |
| information_ratio | -0.876885    |
| max_drawdown      | -0.282011    |

## Plain Indicators

|     |   value |
|:----|--------:|
| ffr |       1 |
| pa  |       0 |
| pos |       0 |

## Weighted Indicators

|     |   value |
|:----|--------:|
| ffr |       1 |
| pa  |       0 |
| pos |       0 |
