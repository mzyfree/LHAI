# Qlib Data Overlap Validation

## Setup

- Old provider: `/Users/Dylan.Min/Documents/Code/learn/LHAI/data/qlib_cn_data`
- New provider: `/Users/Dylan.Min/Documents/Code/learn/LHAI/data/qlib_cn_data_yahoo_2021_snapshot`
- Validation range: `2019-01-01` to `2020-09-25`
- Symbols: `SH600000, SZ000001, SH600519, SZ300750`

## Calendar Summary

| Metric | Value |
| --- | ---: |
| old_count | 424 |
| new_count | 424 |
| overlap_count | 424 |
| old_only_count | 0 |
| new_only_count | 0 |
| overlap_start | 2019-01-02 |
| overlap_end | 2020-09-25 |

## Symbol OHLCV Comparison

| symbol   | field   |   rows |   corr |   mean_abs_diff |   mean_abs_pct_diff |
|:---------|:--------|-------:|-------:|----------------:|--------------------:|
| SH600000 | $open   |    422 |      1 |     7.04464     |         0.847284    |
| SH600000 | $high   |    422 |      1 |     7.10871     |         0.847284    |
| SH600000 | $low    |    422 |      1 |     6.98382     |         0.847284    |
| SH600000 | $close  |    422 |      1 |     7.04141     |         0.847284    |
| SH600000 | $volume |    422 |      1 |     2.96128e+08 |         5.54812     |
| SZ000001 | $open   |    422 |      1 |     1.45448     |         0.499431    |
| SZ000001 | $high   |    422 |      1 |     1.47549     |         0.499431    |
| SZ000001 | $low    |    422 |      1 |     1.43637     |         0.499431    |
| SZ000001 | $close  |    422 |      1 |     1.4566      |         0.499431    |
| SZ000001 | $volume |    422 |      1 |     5.19245e+08 |         0.997727    |
| SH600519 | $open   |    422 |      1 |   231.294       |         0.965526    |
| SH600519 | $high   |    422 |      1 |   234.31        |         0.965526    |
| SH600519 | $low    |    422 |      1 |   228.86        |         0.965526    |
| SH600519 | $close  |    422 |      1 |   231.836       |         0.965526    |
| SH600519 | $volume |    422 |      1 |     5.19135e+08 |        28.0077      |
| SZ300750 | $open   |    422 |      1 |     2.42656e-07 |         8.35228e-08 |
| SZ300750 | $high   |    422 |      1 |     2.48306e-07 |         8.19603e-08 |
| SZ300750 | $low    |    422 |      1 |     2.39831e-07 |         8.32948e-08 |
| SZ300750 | $close  |    422 |      1 |     2.44068e-07 |         8.29085e-08 |
| SZ300750 | $volume |    422 |      1 |    50.8626      |         9.64122e-08 |
