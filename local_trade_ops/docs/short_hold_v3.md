# Short Hold V3

Short Hold V3 keeps the existing short-hold return model and adds an optional execution-aware reranking layer.

The goal is not to replace the current model immediately. It gives us a controlled way to penalize names that look hard to buy at the next open, reward names that look likely to stay strong, and compare that against the current v2 ordering.

## Modes

- `SHORT_HOLD_SCORING_MODE=v2`: current behavior. Sort candidates by the short-hold return model score.
- `SHORT_HOLD_SCORING_MODE=v3`: keep the same return score, then rerank with side-model signals.

The safe default is still:

```bash
SHORT_HOLD_SCORING_MODE=v2
```

## V3 Columns

- `收益分`: raw fused score from the short-hold return model.
- `买入风险`: probability-like value from 0 to 1. Higher means the stock is more likely to be hard to buy at T+1 open.
- `强势概率`: probability-like value from 0 to 1. Higher means stronger expected T+1 open to T+2 close behavior.
- `流动性风险`: liquidity penalty. Higher means weaker liquidity in the current lightweight feature set.
- `最终分`: v3 ranking score used for ordering and softmax allocation.
- `打分源`: currently `v3` when the reranking layer produced the extra columns.

## Formula

The v3 score is:

```text
final_score = normalized(return_score)
            + alpha * strong_prob
            - beta * buyability_risk
            - gamma * liquidity_risk
```

Default knobs:

```bash
SHORT_HOLD_V3_ALPHA=1.0
SHORT_HOLD_V3_BETA=2.0
SHORT_HOLD_V3_GAMMA=0.0
```

## Side Model Artifacts

The order generator expects:

```text
model_packages/short_hold_v3_side_models/buyability_model.pkl
model_packages/short_hold_v3_side_models/strong_model.pkl
model_packages/short_hold_v3_side_models/metadata.json
```

Train them with:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python bin/train_short_hold_v3_side_models.py \
  --samples /path/to/short_hold_v3_samples.csv \
  --output-dir model_packages/short_hold_v3_side_models
```

## Enable V3

```bash
SHORT_HOLD_SCORING_MODE=v3
SHORT_HOLD_V3_SIDE_MODEL_DIR=/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/model_packages/short_hold_v3_side_models
```

Then regenerate the short-hold order list.

## Compare V2 And V3

If you have daily backtest CSVs with either `daily_return` or `nav`, compare them with:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python bin/backtest_short_hold_v3_score.py \
  --v2-daily /path/to/v2_daily.csv \
  --v3-daily /path/to/v3_daily.csv \
  --output reports/short_hold_v3_compare.csv
```

## Rollback

Set:

```bash
SHORT_HOLD_SCORING_MODE=v2
```

Restart the local backend and regenerate the short-hold order list. Existing v2 behavior is unchanged unless v3 mode is explicitly enabled.
