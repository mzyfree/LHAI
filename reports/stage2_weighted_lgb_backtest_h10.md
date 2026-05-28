# Stage 2 Weighted LightGBM Backtest

## Setup

- Config: `/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500_h10.yaml`
- Weighting scheme: `time_decay`
- Half-life days: `252`
- Saved plain pred: `/Users/Dylan.Min/Documents/Code/learn/LHAI/preds/weighted_lgb_plain.pkl`
- Saved weighted pred: `/Users/Dylan.Min/Documents/Code/learn/LHAI/preds/weighted_lgb_time_decay_hl_252.pkl`

## Weight Example

- Newest train sample weight: `1.000000`
- Oldest train sample weight: `0.000887`
- Median train sample weight: `0.031422`

## Signal Comparison

| Metric | Plain | Time-decay weighted |
| --- | ---: | ---: |
| IC | 0.0399 | 0.0096 |
| ICIR | 0.4238 | 0.0998 |
| Rank IC | 0.0505 | 0.0110 |
| Rank ICIR | 0.5099 | 0.1059 |

## Backtest Comparison (Excess Return With Cost, 1day)

| Metric | Plain | Time-decay weighted |
| --- | ---: | ---: |
| annualized_return | 0.022949 | 0.022949 |
| information_ratio | 0.221877 | 0.221877 |
| max_drawdown | -0.151885 | -0.151885 |

## Plain Backtest Detail

|                   |         risk |
|:------------------|-------------:|
| mean              |  9.64224e-05 |
| std               |  0.00670431  |
| annualized_return |  0.0229485   |
| information_ratio |  0.221877    |
| max_drawdown      | -0.151885    |

## Weighted Backtest Detail

|                   |         risk |
|:------------------|-------------:|
| mean              |  9.64224e-05 |
| std               |  0.00670431  |
| annualized_return |  0.0229485   |
| information_ratio |  0.221877    |
| max_drawdown      | -0.151885    |

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
