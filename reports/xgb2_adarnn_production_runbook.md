# XGB2 + ADARNN Production-Like Runbook

Date: 2026-05-19

## Current Mainline

```text
models:
  xgboost_short: recent 5-year XGBoost
  xgboost_long: recent 8-year XGBoost
  adarnn_short: recent 5-year ADARNN

fusion:
  method: zscore_mean
  weights: 10:1:1

portfolio:
  strategy: TopkDropoutStrategy
  topk: 20
  n_drop: 2
```

This is the current preferred structure because fixed-weight rolling validation was positive in 2023, 2024, 2025, and 2026 YTD, while the ADARNN leg improves weak-year behavior without overly diluting strong years.

## Production Refresh Schedule

The paper-trading production cadence is fixed as follows:

```text
daily:
  - update reference Qlib data
  - retrain XGB short
  - generate latest predictions for all online legs
  - fuse predictions with 10:1:1
  - generate top20/drop2 paper-trading orders
  - update simulated account, positions, NAV, and daily report

weekly:
  - retrain ADARNN short
  - refresh the online ADARNN prediction leg

monthly:
  - retrain XGB long
  - refresh the online long-window stabilizer leg
```

Rationale:

```text
XGB short:
  daily retraining keeps the main alpha close to the latest market regime.

XGB long:
  monthly retraining avoids unnecessary churn in the long-window stabilizer.

ADARNN short:
  weekly retraining balances regime adaptation with neural-model training noise.
```

The system should still generate a fresh daily paper-trading plan even on days when `XGB long` or `ADARNN short` are not retrained; those legs should reuse their latest accepted online predictions/models.

## GPU Paths

```text
project: /root/autodl-tmp/llhh
src:     /root/autodl-tmp/llhh/src
data:    /root/autodl-tmp/llhh/reference_cn_data/cn_data
preds:   /root/autodl-tmp/llhh/preds
reports: /root/autodl-tmp/llhh/reports
```

## One-Time Setup

Copy this helper to the GPU machine:

```bash
scp -P <PORT> \
  /Users/Dylan.Min/Documents/Code/learn/LHAI/src/run_xgb2_adarnn_pipeline.py \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/llhh/src/

scp -P <PORT> \
  /Users/Dylan.Min/Documents/Code/learn/LHAI/src/run_paper_trading_daily.py \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/llhh/src/
```

Then on the GPU machine:

```bash
cd /root/autodl-tmp/llhh/src
export PYTHONPATH=/root/autodl-tmp/llhh/qlib:$PYTHONPATH
```

## Generate Rolling Configs

```bash
python run_xgb2_adarnn_pipeline.py gen-configs --years 2023-2026 --test-end 2026-05-08
```

This writes configs to:

```text
/root/autodl-tmp/llhh/src/configs/reference_rolling_xgb2_adarnn
```

## Train Missing Models

Print the train commands:

```bash
python run_xgb2_adarnn_pipeline.py print-train --years 2023-2026
```

Run the printed commands, or pipe them to a shell after reviewing:

```bash
python run_xgb2_adarnn_pipeline.py print-train --years 2023-2026 > /tmp/train_xgb2_adarnn.sh
bash /tmp/train_xgb2_adarnn.sh
```

The commands use the existing GPU helper:

```bash
source /root/autodl-tmp/llhh/src/prod_like_helpers.sh
safe_run_copy_prod_like ...
```

## Run Fixed Fusion And Ablation

Print fusion/evaluation commands:

```bash
python run_xgb2_adarnn_pipeline.py print-fusion --years 2023-2026 --topk 20 --n-drop 2
```

Run after reviewing:

```bash
python run_xgb2_adarnn_pipeline.py print-fusion --years 2023-2026 --topk 20 --n-drop 2 > /tmp/fuse_xgb2_adarnn.sh
bash /tmp/fuse_xgb2_adarnn.sh
```

This evaluates:

```text
xgb2_adarnn_8_1_1: xgboost_short + xgboost_long + adarnn_short
xgb2_8_1:          xgboost_short + xgboost_long
```

For the current preferred `10:1:1` ratio, run:

```bash
for y in 2023 2024 2025 2026; do
  python run_prod_like_ensemble_search.py \
    --config "/root/autodl-tmp/llhh/src/configs/reference_rolling_xgb2_adarnn/workflow_config_xgboost_short_${y}.yaml" \
    --models "xgboost_short_${y},xgboost_long_${y},adarnn_short_${y}" \
    --methods "zscore_mean" \
    --weight-sets "10,1,1" \
    --topk-pairs "20:2" \
    --output "/root/autodl-tmp/llhh/reports/rolling_xgb2_adarnn_fixed_10_1_1_${y}_top20_drop2.csv" \
    --save-best-pred "/root/autodl-tmp/llhh/preds/rolling_xgb2_adarnn_fixed_10_1_1_${y}_top20_drop2.pkl"
done
```

## Summarize Results

```bash
python run_xgb2_adarnn_pipeline.py summarize --topk 20 --n-drop 2
```

Current expected headline from prior runs:

```text
xgb2_adarnn_10_1_1 average:
  excess annualized: +11.39%
  IR: 0.89
  MaxDD: -7.60%

xgb2_8_1 average:
  excess annualized: +11.59%
  IR: 0.79
  MaxDD: -8.72%
```

## Small-Capital Follow-Up

Small-capital sweep conclusion:

```text
standard:      top20/drop2
small-capital: top15/drop1
```

Observed fixed `10:1:1` results:

```text
top20/drop2 average:
  excess annualized: +11.39%
  IR: 0.89
  MaxDD: -7.60%

top15/drop1 average:
  excess annualized: +10.06%
  IR: 0.77
  MaxDD: -7.36%
```

Execution estimate:

```text
top20/drop2: 10万 RMB is usable; 15万+ RMB is more comfortable
top15/drop1: 8万~10万 RMB is usable
below 8万: pure stock top15/top20 is not recommended because A-share lot size causes low diversification
top10 or lower: not recommended from current rolling tests
```

Raw-price capital scan corrected the earlier adjusted-price estimate. Under A-share 100-share lots, 3万~5万 RMB produced too few executable names and too much concentration, so it should only be used for ETF-plus-satellite experiments or observation-only paper trading.

## Paper-Trading State Machine

The production paper-trading account uses the executable timing below:

```text
T after close:
  - update data
  - train or load the required model legs
  - generate fused prediction
  - create T+1 pending orders from the T close signal

T+1 after open:
  - load the pending orders
  - execute them at T+1 raw open
  - deduct A-share transaction cost estimate
  - update cash, positions, NAV, and execution report
```

This is intentionally different from the Qlib research benchmark. Research runs may keep `deal_price=close` for comparability, but paper trading uses `next_open` to avoid same-close look-ahead.

Current account and cost assumptions:

```text
initial capital: 100000 RMB, configurable
main strategy:   XGB short + XGB long + ADARNN, zscore_mean 10:1:1
portfolio rule:  top20/drop2
lot size:        100 shares
buy cost:        0.0003, min 5 RMB/order
sell cost:       0.0008, min 5 RMB/order
```

State files:

```text
/root/autodl-tmp/llhh/paper_trading/paper_account.json
/root/autodl-tmp/llhh/paper_trading/paper_positions.csv
/root/autodl-tmp/llhh/paper_trading/paper_nav.csv
/root/autodl-tmp/llhh/paper_trading/pending_orders_<date>.csv
/root/autodl-tmp/llhh/paper_trading/paper_trades_<date>.csv
```

After-close command with an already-fused prediction:

```bash
cd /root/autodl-tmp/llhh/src
export PYTHONPATH=/root/autodl-tmp/llhh/qlib:$PYTHONPATH

python run_paper_trading_daily.py after-close \
  --fused-pred /root/autodl-tmp/llhh/preds/live_xgb2_adarnn_10_1_1_top20_drop2.pkl \
  --capital 100000 \
  --topk 20 \
  --n-drop 2
```

After-close command when fusing three prediction legs directly:

```bash
python run_paper_trading_daily.py after-close \
  --pred-a /root/autodl-tmp/llhh/preds/xgboost_short_2026_2026_ytd_prod_like.pkl \
  --pred-b /root/autodl-tmp/llhh/preds/xgboost_long_2026_2026_ytd_prod_like.pkl \
  --pred-c /root/autodl-tmp/llhh/preds/adarnn_short_2026_2026_ytd_prod_like.pkl \
  --weights 10,1,1 \
  --capital 100000 \
  --topk 20 \
  --n-drop 2
```

After-open command:

```bash
python run_paper_trading_daily.py after-open \
  --execution-date <T_PLUS_1_DATE> \
  --capital 100000
```
