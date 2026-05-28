# Stage 3 Reference Data And 2026 YTD Production-Like Summary

Date: 2026-05-15

## Scope

This note records the current stage results for the Qlib CSI500 index-enhancement research after switching from the Tushare bundle to the community reference Qlib CN data source.

Main objective:

- Validate whether the weak Tushare results were data-source related.
- Build a production-like workflow using Qlib daily models.
- Test whether retraining windows affect 2026 YTD performance.
- Identify the current best model/window/strategy candidate.

## Data Source

Current primary data source:

```text
/root/autodl-tmp/llhh/reference_cn_data/cn_data
```

The data was downloaded from the community source recommended by Qlib:

```text
chenditc/investment_data
release: 2026-05-10
calendar: 2008-01-02 ~ 2026-05-08
```

We moved away from the previous Tushare-only bundle because model performance on 2019-2020 research tests was consistently weak and did not match earlier Qlib-like expectations.

## Dataset Splits Used

Reference long research split:

```text
train:         2008-01-01 ~ 2018-12-31
valid:         2019-01-01 ~ 2020-12-31
research test: 2021-01-01 ~ 2022-12-31
final blind:   2023-01-01 ~ 2026-05-08
```

Initial production-like 2026 YTD split:

```text
train: 2008-01-01 ~ 2023-12-31
valid: 2024-01-01 ~ 2025-12-31
test:  2026-01-01 ~ 2026-05-08
```

Short-window experiments:

```text
train: 2014-01-01 ~ 2023-12-31
train: 2016-01-01 ~ 2023-12-31
train: 2019-01-01 ~ 2023-12-31
valid: 2024-01-01 ~ 2025-12-31
test:  2026-01-01 ~ 2026-05-08
```

## Data Quality Findings

Final-period quality check on 2023-01-01 to 2026-05-08:

```text
calendar start: 2023-01-03
calendar end:   2026-05-08
trading days:   807
duplicate days: 0
CSI500 count:   exactly 500 on all 807 days
benchmark:      SH000905, no close missing
benchmark cumulative return: +45.98%
```

Feature quality:

```text
OHLCV missing rate: 0.1871%
factor missing rate: 0.0000%
label missing rate: 0.4574%
non-positive price count: 0
negative volume count: 0
high/low/open/close consistency errors: 0
```

Notable local anomalies:

```text
SH689009 close missing: 582 days
SH688615 extreme labels around 2024-09-26 to 2024-09-30
factor jumps > 50%: sparse, only 17 rows in the checked period
```

Interpretation:

- The reference data source looks usable for research.
- Current 2026 underperformance is more likely caused by regime/window/model issues than obvious data corruption.
- Production should still filter instruments with very high missing close rates.

## Reference Long Single-Model Results

Research test period: 2021-2022.

Metrics below are excess return with cost against CSI500.

```text
model                    excess_ann    IR       MaxDD
lgb_alpha158             +9.68%        0.97     -8.53%
linear                   +6.80%        0.75     -7.96%
tabnet                   +6.67%        0.58     -11.27%
lstm                     +4.52%        0.53     -11.29%
catboost                 +3.60%        0.36     -11.02%
adarnn                   +3.21%        0.31     -13.29%
xgboost                  +3.06%        0.29     -9.25%
alstm                    +1.32%        0.15     -8.10%
localformer              +1.28%        0.13     -13.02%
doubleensemble_light     +1.09%        0.13     -9.49%
mlp                      -1.93%       -0.20     -14.36%
gru                      -0.20%       -0.02     -11.00%
tcn                      -1.31%       -0.12     -17.78%
tcts_alpha360_lite       -1.45%       -0.17     -9.84%
```

The community reference data restored reasonable research performance. This supports the hypothesis that the previous Tushare-only bundle was not ideal for our current workflow.

## Reference Long Ensemble Findings

Core ensemble pool:

```text
lgb_alpha158
linear
tabnet
lstm
catboost
adarnn
xgboost
```

Important bug found and fixed:

- The ensemble search script initially reused the first prediction in all backtests because Qlib's `fill_placeholder` mutates the strategy config.
- Fixed by deep-copying `port_analysis_config` for every candidate.

Best reference-long ensemble after fix:

```text
method: zscore_mean
weights:
  lgb_alpha158:4
  linear:2
  tabnet:2
  lstm:1
  catboost:1
  adarnn:1
  xgboost:1

excess_ann: +6.10%
IR:         0.55
MaxDD:      -7.02%
```

Interpretation:

- Ensemble reduced drawdown but did not beat the best LGB single model on research return.
- First production candidate remained LGB, not the broad ensemble.

## Static Final Blind Finding

Static final test with old training window:

```text
train: 2008-2018
valid: 2019-2020
test:  2023-2026
```

This setup underperformed CSI500, especially in 2025 and 2026 YTD.

For LGB top150/drop15:

```text
2023 excess ann:  +0.19%
2024 excess ann:  -2.10%
2025 excess ann: -11.06%
2026 excess ann: -19.86%
```

Interpretation:

- A model trained only through 2018 was too stale for 2025-2026.
- This motivated the production-like and short-window experiments.

## Production-Like 2026 YTD Results

Test period:

```text
2026-01-01 ~ 2026-05-08
benchmark cumulative return: +16.46%
benchmark annualized return: +48.47%
```

Initial production-like split:

```text
train: 2008-2023
valid: 2024-2025
test:  2026 YTD
```

Top results:

```text
case                         strategy_cum  benchmark_cum  excess_cum  excess_ann  IR       MaxDD
lgb_alpha158_top30_drop3     +14.43%       +16.46%        -1.70%      -4.21%      -0.31    -6.33%
xgboost_top30_drop3          +13.29%       +16.46%        -2.68%      -7.17%      -0.53    -8.45%
xgboost_top150_drop15        +13.03%       +16.46%        -3.22%      -9.38%      -1.13    -5.15%
lgb_alpha158_top150_drop15   +11.02%       +16.46%        -4.88%      -14.54%     -1.74    -5.51%
```

Second-batch models did not help:

```text
catboost top30/drop3     excess_ann -17.58%
tabnet top30/drop3       excess_ann -31.68%
lstm top30/drop3         excess_ann -33.32%
```

Interpretation:

- Updating the training window improved LGB, but not enough.
- Deep/sequence models did not help in the 2026 YTD regime.
- Tree models remained the strongest family.

## Training Window Sensitivity

This was the key breakthrough.

LGB Alpha158 window search:

```text
case                                      strategy_cum  benchmark_cum  excess_cum  excess_ann  IR       MaxDD
lgb_train2016_2023_top30_drop3            +21.31%       +16.46%        +3.91%      +12.10%     1.02     -5.45%
lgb_train2014_2023_top30_drop3            +16.11%       +16.46%        -0.24%      -0.02%     -0.00     -3.99%
lgb_train2019_2023_top30_drop3            +15.57%       +16.46%        -0.89%      -2.13%     -0.21     -4.04%
lgb_train2008_2023_top30_drop3            +14.43%       +16.46%        -1.70%      -4.21%     -0.31     -6.33%
```

XGBoost window search:

```text
case                                      strategy_cum  benchmark_cum  excess_cum  excess_ann  IR       MaxDD
xgboost_train2019_2023_top30_drop3        +25.31%       +16.46%        +7.87%      +23.46%     1.72     -3.13%
xgboost_train2016_2023_top30_drop3        +21.32%       +16.46%        +4.08%      +12.38%     1.26     -4.15%
xgboost_train2014_2023_top30_drop3        +15.55%       +16.46%        -0.84%      -1.78%     -0.15     -6.34%
xgboost_train2008_2023_top30_drop3        +13.29%       +16.46%        -2.68%      -7.17%     -0.53     -8.45%
```

CatBoost window search:

```text
case                                      strategy_cum  benchmark_cum  excess_cum  excess_ann  IR       MaxDD
catboost_train2008_2023_top30_drop3       +10.00%       +16.46%        -5.96%      -17.58%     -1.49    -6.80%
catboost_train2008_2023_top150_drop15     +5.83%        +16.46%        -9.70%      -29.73%     -2.70    -7.71%
catboost_train2014_2023_top150_drop15     +4.02%        +16.46%        -11.37%     -35.07%     -2.74    -9.55%
catboost_train2019_2023_top150_drop15     +1.53%        +16.46%        -13.63%     -42.51%     -2.94    -12.07%
catboost_train2016_2023_top150_drop15     +1.32%        +16.46%        -13.72%     -42.94%     -3.15    -12.40%
catboost_train2016_2023_top30_drop3       +0.67%        +16.46%        -14.68%     -45.35%     -2.35    -13.77%
catboost_train2019_2023_top30_drop3       +0.56%        +16.46%        -14.74%     -45.73%     -2.48    -12.92%
catboost_train2014_2023_top30_drop3       +0.32%        +16.46%        -14.71%     -46.00%     -2.83    -12.98%
```

Interpretation:

- Training window is a first-order variable.
- Longer is not necessarily better.
- Different models prefer different windows.
- 2026 YTD favors more recent training windows, especially XGBoost 2019-2023.
- The old 2008-start expanding window diluted the current regime signal.
- CatBoost is not rescued by shorter windows in this setup. It should be excluded from the 2026 YTD production candidate pool unless its feature set or hyperparameters are changed.

Linear, ADARNN, and Localformer window search:

```text
case                                      strategy_cum  benchmark_cum  excess_cum  excess_ann  IR       MaxDD
adarnn_train2019_2023_top30_drop3         +14.67%       +16.46%        -1.98%      -5.19%      -0.42    -7.08%
adarnn_train2014_2023_top150_drop15       +7.27%        +16.46%        -8.56%      -25.90%     -2.15    -7.49%
adarnn_train2016_2023_top30_drop3         +6.59%        +16.46%        -9.52%      -28.32%     -1.67    -10.91%
localformer_train2014_2023_top30_drop3    +6.39%        +16.46%        -9.79%      -28.88%     -1.54    -11.70%
adarnn_train2014_2023_top30_drop3         +5.96%        +16.46%        -9.64%      -29.34%     -2.29    -7.39%
linear_train2014_2023_top30_drop3         +5.52%        +16.46%        -10.31%     -31.05%     -1.90    -9.32%
linear_train2016_2023_top30_drop3         +3.57%        +16.46%        -12.09%     -36.75%     -2.07    -12.12%
linear_train2019_2023_top30_drop3         +1.42%        +16.46%        -14.06%     -43.21%     -2.26    -12.91%
localformer_train2019_2023_top30_drop3    +0.31%        +16.46%        -15.08%     -46.58%     -2.32    -14.24%
localformer_train2019_2023_top150_drop15  -0.79%        +16.46%        -15.67%     -49.49%     -3.21    -14.34%
```

Interpretation:

- Linear did not benefit from shorter windows in 2026 YTD. It was strong in the 2021-2022 research split, but not useful in this 2026 production-like setup.
- ADARNN improved with the 2019-2023 window and was the best of this group, but still did not beat CSI500.
- Localformer deteriorated materially with the 2019-2023 window and should not be a 2026 candidate under the current setup.
- None of these models challenge XGBoost 2019-2023 or LGB 2016-2023.

Transformer, ALSTM, and DoubleEnsemble-light window search:

```text
case                                              strategy_cum  benchmark_cum  excess_cum  excess_ann  IR       MaxDD
doubleensemble_light_train2014_2023_top30_drop3   +5.87%        +16.46%        -10.10%     -30.08%     -1.67    -11.28%
alstm_train2014_2023_top30_drop3                  +4.98%        +16.46%        -11.07%     -32.98%     -1.68    -13.71%
alstm_train2016_2023_top30_drop3                  +4.52%        +16.46%        -11.28%     -34.02%     -1.91    -10.99%
doubleensemble_light_train2014_2023_top150_drop15 +3.82%        +16.46%        -11.54%     -35.63%     -2.78    -9.87%
alstm_train2016_2023_top150_drop15                +3.25%        +16.46%        -12.07%     -37.34%     -2.78    -10.58%
alstm_train2019_2023_top150_drop15                +2.92%        +16.46%        -12.49%     -38.56%     -2.58    -10.69%
doubleensemble_light_train2016_2023_top30_drop3   +2.94%        +16.46%        -12.71%     -38.63%     -2.04    -12.34%
doubleensemble_light_train2019_2023_top30_drop3   +2.99%        +16.46%        -12.82%     -38.72%     -1.90    -13.68%
transformer_train2016_2023_top30_drop3            +2.60%        +16.46%        -13.13%     -39.79%     -1.95    -13.09%
alstm_train2014_2023_top150_drop15                +1.78%        +16.46%        -13.39%     -41.71%     -2.91    -12.07%
alstm_train2019_2023_top30_drop3                  +1.26%        +16.46%        -14.27%     -43.75%     -2.16    -15.30%
doubleensemble_light_train2016_2023_top150_drop15 +1.00%        +16.46%        -14.00%     -43.91%     -3.20    -12.74%
doubleensemble_light_train2019_2023_top150_drop15 +0.73%        +16.46%        -14.33%     -44.88%     -3.02    -12.99%
transformer_train2014_2023_top150_drop15          +0.47%        +16.46%        -14.58%     -45.68%     -2.97    -13.32%
transformer_train2016_2023_top150_drop15          +0.43%        +16.46%        -14.66%     -45.91%     -2.93    -13.38%
transformer_train2014_2023_top30_drop3            +0.32%        +16.46%        -15.05%     -46.46%     -2.29    -13.94%
transformer_train2019_2023_top30_drop3            +0.12%        +16.46%        -15.25%     -47.12%     -2.30    -13.71%
transformer_train2019_2023_top150_drop15          -0.46%        +16.46%        -15.47%     -48.64%     -3.00    -14.29%
```

Interpretation:

- Transformer, ALSTM, and DoubleEnsemble-light all fail to approach the XGBoost/LGB leaders.
- DoubleEnsemble-light is not useful in this 2026 YTD production-like setup despite being a tree ensemble.
- Transformer and ALSTM are clearly not suitable candidates under the current Alpha360 configuration and 2026 regime.

## Production-Like Ensemble Search

After the single-model and training-window sweeps, we tested small candidate ensembles around the strongest models.

Two-model fusion: XGBoost 2019-2023 plus LGB 2016-2023.

```text
models:      xgboost_train2019_2023, lgb_alpha158_train2016_2023
method:      zscore_mean
weights:     xgboost_train2019_2023:3, lgb_alpha158_train2016_2023:1
topk:        30
n_drop:      3

IC:          0.038671
Rank IC:     0.023627
excess_ann:  +25.68%
IR:          2.33
MaxDD:       -3.66%
```

Two-model fusion: XGBoost 2019-2023 plus XGBoost 2016-2023.

```text
models:      xgboost_train2019_2023, xgboost_train2016_2023
method:      zscore_mean
weights:     xgboost_train2019_2023:2, xgboost_train2016_2023:1
topk:        20
n_drop:      2

IC:          0.038960
Rank IC:     0.028098
excess_ann:  +37.27%
IR:          2.17
MaxDD:       -5.16%
```

Three-model fusion: XGBoost 2019-2023 plus LGB 2016-2023 plus ADARNN 2019-2023.

```text
models:      xgboost_train2019_2023, lgb_alpha158_train2016_2023, adarnn_train2019_2023
method:      zscore_mean
weights:     xgboost_train2019_2023:5, lgb_alpha158_train2016_2023:2, adarnn_train2019_2023:1
topk:        20
n_drop:      2

IC:          0.037238
Rank IC:     0.026394
excess_ann:  +36.30%
IR:          2.43
MaxDD:       -4.06%
```

Interpretation:

- The highest-return candidate is `xgboost_train2019_2023 + xgboost_train2016_2023`, with excess annualized return of +37.27%.
- The best risk-adjusted candidate is the three-model fusion, with IR 2.43 and lower max drawdown than the two-XGBoost fusion.
- ADARNN is not strong alone, but it appears to help stabilize the fused portfolio when lightly weighted.
- Fusion confirms the main production candidate pool should stay focused on recent-window XGBoost plus selective LGB/ADARNN diversification.

## Stable Candidate Diagnostics

Stable candidate:

```text
models:      xgboost_train2019_2023, lgb_alpha158_train2016_2023, adarnn_train2019_2023
method:      zscore_mean
weights:     5:2:1
strategy:    top20/drop2
period:      2026-01-01 ~ 2026-05-08
```

Monthly excess with cost:

```text
month      days  excess_cum  excess_ann   IR      MaxDD
2026-01    20    -3.71%      -37.89%     -2.77   -2.53%
2026-02    14    +4.03%     +103.66%     +5.03   -2.30%
2026-03    22   +10.44%     +212.02%     +8.41   -2.13%
2026-04    21    -2.00%      -21.54%     -1.82   -3.00%
2026-05     3    +3.81%    +2218.17%    +14.30   -0.33%
```

Cost sensitivity:

```text
cost_scale  excess_cum  excess_ann  IR     MaxDD
0x          +14.24%     +52.11%     2.81   -3.79%
1x          +12.55%     +45.14%     2.51   -4.01%
2x          +10.90%     +38.51%     2.20   -4.23%
3x           +9.27%     +32.21%     1.90   -4.45%
5x           +6.10%     +20.49%     1.30   -5.31%
```

Daily contribution:

```text
total excess cumulative return: +12.55%
sum daily excess:               +12.20%
top 5 positive days sum:        +10.15%
top 10 positive days sum:       +17.61%
top 5 share of sum daily excess: 83.14%
top 10 share of sum daily excess: 144.33%
```

Turnover:

```text
average holdings:        19.28
average buys per day:     2.01
average sells per day:    1.78
average changed per day:  3.79
max changed per day:     19
```

Interpretation:

- Cost robustness is good. Even at 5x cost, the strategy remains positive with excess annualized return above +20%.
- Turnover is consistent with `top20/drop2`; the high max changed day is likely the initial portfolio build.
- Monthly behavior is mixed. February and March are very strong, while January and April underperform.
- Daily contribution is concentrated. The top five positive days explain most of the net daily excess, so the strategy should not be considered fully validated yet.
- The candidate is still attractive, but the next validation should focus on whether the edge survives broader periods and whether large positive days are repeatable rather than accidental.

## Rolling-Year Validation

After the 2026 YTD diagnostics, we tested whether similar model structures generalize across earlier blind years without using future data.

Single-model rolling check:

```text
case              year  excess_ann  IR      MaxDD
lgb_alpha158      2023  +3.94%      0.36    -8.34%
xgboost           2023  +0.40%      0.04    -6.06%
lgb_alpha158      2024 -13.31%     -1.01   -14.88%
xgboost           2024  +5.22%      0.40   -11.44%
```

Interpretation:

- XGBoost is more stable than LGB in rolling validation.
- LGB failed materially in 2024, so the original `XGB + LGB + ADARNN` stable candidate should be treated with caution.

Rolling `XGB + LGB` fusion:

```text
case              best weights  topk/drop  excess_ann  IR      MaxDD
2023              1:1           30/3       +9.48%      1.04    -7.07%
2024              2:1           30/3       +3.02%      0.25   -10.30%
```

Interpretation:

- `XGB + LGB` fusion can produce positive excess, but the 2024 result is weak.
- This combination is usable as a reference but is not the preferred production structure.

Rolling `XGB short + XGB long + ADARNN` fusion:

```text
case  best weights                 topk/drop  excess_ann  IR      MaxDD
2023  xgb_short:8,xgb_long:1,adarnn:1 20/2     +3.10%      0.37    -7.28%
2024  xgb_short:5,xgb_long:2,adarnn:1 20/2     +8.12%      0.74    -9.23%
2025  xgb_short:8,xgb_long:1,adarnn:1 20/2    +18.51%      1.85    -5.69%
```

Interpretation:

- This is the first rolling structure with positive excess in 2023, 2024, and 2025.
- The signal improves over time, with 2025 showing a strong result.
- The best weights favor short-window XGBoost heavily, with long-window XGBoost and ADARNN as small stabilizers.
- This structure appears more general than the earlier `XGB + LGB` path.
- Current preferred production research direction shifts from `XGB + LGB + ADARNN` to `XGB short + XGB long + ADARNN`.

Fixed-weight ablation:

```text
group              year  weights  excess_ann  IR      MaxDD
xgb2_8_1           2023  8:1      -5.55%     -0.60   -10.50%
xgb2_adarnn_8_1_1  2023  8:1:1    +3.10%      0.37    -7.28%
xgb2_8_1           2024  8:1      -0.05%     -0.00   -12.30%
xgb2_adarnn_8_1_1  2024  8:1:1    +6.09%      0.46   -12.01%
xgb2_8_1           2025  8:1     +24.34%      2.10    -7.92%
xgb2_adarnn_8_1_1  2025  8:1:1   +18.51%      1.85    -5.69%
xgb2_8_1           2026  8:1     +27.64%      1.68    -4.18%
xgb2_adarnn_8_1_1  2026  8:1:1    +8.52%      0.61    -4.78%
```

Four-year average:

```text
group              avg_excess_ann  avg_IR  avg_MaxDD
xgb2_8_1           +11.59%         0.79    -8.72%
xgb2_adarnn_8_1_1   +9.05%         0.82    -7.44%
```

Interpretation:

- ADARNN is valuable as a stabilizer even though it is not a strong standalone model.
- Without ADARNN, the two-XGBoost structure is negative in 2023 and nearly flat in 2024.
- With ADARNN, all four tested years are positive.
- ADARNN reduces average drawdown, from -8.72% to -7.44%, and improves average Rank IC.
- The cost of ADARNN is meaningfully lower upside in strong years, especially 2026 under the strict rolling configuration.
- For production, `xgb_short:xgb_long:adarnn = 8:1:1` is more defensive, while pure `xgb_short:xgb_long = 8:1` is the stronger return engine.

ADARNN weight sweep:

```text
ratio      2023       2024       2025       2026       interpretation
8:1:1      +3.10%     +6.09%    +18.51%     +8.52%    defensive, all years positive, but drags 2026
10:1:1     +2.17%     +3.66%    +13.36%    +26.38%    stronger 2026, weaker 2024
16:2:1     -3.73%     +9.25%     +8.43%    +21.40%    2023 negative
20:2:1     -1.57%     +5.24%    +17.85%    +22.58%    2023 negative
20:1:1     -1.74%     +0.76%    +17.16%    +24.60%    2023 weak
30:2:1     +2.27%     +7.07%    +15.43%    +20.88%    balanced, all years positive, best average candidate
```

Interpretation:

- `8:1:1` is no longer the only candidate. It is the most defensive simple ratio but over-weights ADARNN for 2026.
- `10:1:1` has almost the same average excess annualized return as `30:2:1`, but with higher average IR, smaller average drawdown, and better average Rank IC.
- The current preferred fixed production ratio shifts to `10:1:1`.
- `8:1:1` remains a conservative fallback if the priority is maximum weak-year protection.

Small-capital top/drop sweep with fixed `10:1:1`:

```text
pair       avg_excess_ann  avg_IR  avg_MaxDD  verdict
15:2       +12.83%         0.80    -11.32%    high return, higher drawdown
10:1       +12.13%         0.64    -13.85%    unstable, 2024 negative
20:2       +11.39%         0.89     -7.60%    standard main strategy
15:1       +10.06%         0.77     -7.36%    small-capital recommended version
10:2        +7.90%         0.40    -15.77%    unstable, 2025 negative
8:1         +5.71%         0.29    -14.31%    unstable, 2024 negative
5:1         -3.46%        -0.08    -17.72%    reject
```

Interpretation:

- Do not compress the portfolio below top15. `top5/drop1`, `top8/drop1`, and `top10/drop2` are unstable.
- `top20/drop2` remains the standard research strategy.
- `top15/drop1` is the preferred small-capital version because it reduces holdings and turnover while keeping all four tested years positive.
- `top15/drop2` is a more aggressive small-capital variant, but its drawdown is materially worse.

Current execution versions:

```text
standard:      XGB2 + ADARNN, 10:1:1, top20/drop2
small-capital: XGB2 + ADARNN, 10:1:1, top15/drop1
```

## Current Best Candidate

Current preferred production candidate:

```text
models:       XGB short + XGB long + ADARNN
method:       zscore_mean
weights:      10:1:1
strategy:     TopkDropoutStrategy
topk:         20
n_drop:       2
```

Production refresh schedule fixed on 2026-05-19:

```text
XGB short:     retrain daily
XGB long:      retrain monthly
ADARNN short:  retrain weekly

Daily routine:
  - update reference Qlib data
  - refresh the online prediction set
  - fuse XGB short, XGB long, and ADARNN short with 10:1:1
  - generate top20/drop2 paper-trading orders
  - update the simulated account and NAV
```

This cadence keeps the main short-window XGBoost leg highly adaptive, while limiting unnecessary churn from the long-window XGBoost stabilizer and the noisier neural ADARNN leg.

Performance:

```text
2023 excess annualized:     +2.17%
2024 excess annualized:     +3.66%
2025 excess annualized:     +13.36%
2026 excess annualized:     +26.38%
average excess annualized:  +11.39%
average IR:                 0.89
average max drawdown:       -7.60%
```

Highest-return 2026 YTD candidate:

```text
models:       XGBoost 2019-2023 + XGBoost 2016-2023
method:       zscore_mean
weights:      2:1
strategy:     top20/drop2
2026 excess annualized return: +37.27%
2026 excess IR: 2.17
2026 excess max drawdown: -5.16%
```

Single-model reference candidate:

```text
model:        XGBoost
features:     Alpha158
train window: 2019-2023
strategy:     top30/drop3
excess annualized return: +23.46%
excess IR: 1.72
excess max drawdown: -3.13%
```

## Current Working Hypotheses

1. The community reference data is adequate for research.
2. The Tushare-only bundle likely weakened previous model results.
3. The 2026 YTD problem was not primarily a model-family issue.
4. Training window length was the key driver.
5. Shorter recent windows are more suitable for the current CSI500 regime.
6. Alpha158 tree models are currently stronger than Alpha360 neural/sequence models.
7. `top30/drop3` is better than `top150/drop15` in the 2026 YTD trend regime.
8. Not all tree models benefit from shorter windows: CatBoost deteriorated materially, while XGBoost improved sharply.
9. Linear, ADARNN, and Localformer do not currently challenge the tree-model leaders. ADARNN 2019-2023 is the only one worth keeping as a weak diversifier candidate.
10. Transformer, ALSTM, and DoubleEnsemble-light also fail under the current setup. The current candidate pool should remain concentrated around XGBoost and LGB.
11. Small ensembles are better than broad ensembles in this regime. The best current mix is recent-window XGBoost plus selective LGB/ADARNN.

## Recommended Next Steps

Immediate:

```text
1. Compare the two current ensemble finalists on a longer blind period if possible:
   - xgb2019 + xgb2016, zscore 2:1, top20/drop2
   - xgb2019 + lgb2016 + adarnn2019, zscore 5:2:1, top20/drop2
2. Run per-month attribution for 2026 YTD to check whether the ensemble edge is concentrated in a few days.
3. Run turnover/cost sensitivity for top20/drop2 and top30/drop3.
4. Keep single XGBoost 2019-2023 as the simplest baseline.
5. Stop broad model expansion unless testing a materially different model family, feature set, or portfolio rule.
```

Then:

```text
1. Avoid more CatBoost runs unless changing features or hyperparameters.
2. Avoid more deep model runs until there is a clear reason; current Alpha360 neural models did not help.
3. Use daily/weekly/monthly model refresh cadence:
   - XGB short daily
   - ADARNN short weekly
   - XGB long monthly
4. Add production data filters:
   - remove instruments with excessive close missing rate
   - flag extreme labels
   - monitor factor jumps
```

Potential production default:

```text
For each prediction date:
  - use recent 5-8 year training window
  - use previous 1-2 years as validation
  - retrain XGB short daily
  - retrain ADARNN short weekly
  - retrain XGB long monthly
  - produce daily prediction and top20/drop2 signal
```

## Operational Notes

The AutoDL machine moved between GPU types several times. Key persistent files were kept under:

```text
/root/autodl-tmp/llhh
```

Helper script created on GPU:

```text
/root/autodl-tmp/llhh/src/prod_like_helpers.sh
```

Use after new shell or machine drift:

```bash
cd /root/autodl-tmp/llhh/src
source /root/autodl-tmp/llhh/src/prod_like_helpers.sh
```

Environment issue observed repeatedly:

```text
OpenSSL / cryptography mismatch:
AttributeError: module 'hashes' has no attribute 'XOFHash'
```

Working package combination:

```text
cryptography==45.0.7
pyOpenSSL==25.1.0
```

If the error returns, uninstall both, delete stale `cryptography*`, `OpenSSL*`, `pyOpenSSL*` from site-packages, and reinstall the above versions.
