# Rolling Window Evaluation

## Setup

- Config: `/Users/Dylan.Min/Documents/Code/learn/LHAI/results/2026-04-18/results_pack_2026-04-18/workflow_config_lightgbm_Alpha158_csi500_local.yaml`
- Pred: `/Users/Dylan.Min/Documents/Code/learn/LHAI/results/2026-04-18/results_pack_2026-04-18/fusion_lgb_tcts_csi500_rank_mean_1_1.pkl`
- Provider URI: `/Users/Dylan.Min/Documents/Code/learn/LHAI/data/qlib_cn_data`

## Windows

| window                   |        IC |     ICIR |   Rank IC |   Rank ICIR |   annualized_return |   information_ratio |   max_drawdown |
|:-------------------------|----------:|---------:|----------:|------------:|--------------------:|--------------------:|---------------:|
| 2017-01-01 -> 2018-12-31 | 0.0635684 | 0.722379 | 0.0713388 |    0.815381 |           0.11459   |            2.05967  |     -0.0530235 |
| 2018-01-01 -> 2019-12-31 | 0.0438989 | 0.541331 | 0.0526731 |    0.602863 |           0.0465536 |            0.739698 |     -0.0770017 |
| 2019-01-01 -> 2020-08-01 | 0.0357202 | 0.393528 | 0.0525512 |    0.541698 |           0.113002  |            1.55964  |     -0.0614075 |
