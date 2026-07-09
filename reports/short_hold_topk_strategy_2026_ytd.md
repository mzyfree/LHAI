# Short-Hold TOPK Strategy 2026 YTD Research Note

Date: 2026-05-30

## Background

This note records a new strategy branch that is different from the current T+1 manual production line.

The idea is to use the daily ensemble ranking as a short-hold signal:

- Generate signal after T close using data available up to T.
- Enter selected stocks on T+1 open.
- Hold for a short fixed window.
- Exit at the configured close day.
- Rebuild the queue every trading day.

This branch is still research-only. It has not replaced the current manual Eastmoney workflow.

## Current Model Inputs

Model package:

```text
/root/autodl-tmp/llhh/csi1000_main_model_packages_20260522.tar.gz
```

Daily prediction cache:

```text
/root/autodl-tmp/llhh/strict_online_intraday_2026
```

Data provider:

```text
/root/autodl-tmp/llhh/reference_cn_data/cn_data
```

Test window:

```text
2026-01-05 to 2026-05-26
```

The run uses strict online-style prediction cache: each signal day has its own cached prediction result. The grid calculation reads those caches and does not re-run model inference.

## Simulation Assumptions

The meaningful table is `strict_online_portfolio_queue_2026_summary.csv`, because it simulates portfolio-level cash occupation:

- Cash is shared across active positions.
- Positions not due for exit remain occupied.
- Due positions are sold at close.
- NAV is cash plus holdings marked to close.
- Allocation uses `softmax` over model scores.
- Current cost model includes proportional buy/sell cost, but does not yet include per-order minimum commission.

Important remaining realism gaps:

- Eastmoney minimum commission, commonly 5 CNY per order, is not yet modeled.
- High-open / limit-up buy failures are not fully modeled.
- ChiNext / STAR Market permission restrictions are not fully modeled.
- Real bid/ask spread, queue priority, and manual execution slippage are not fully modeled.

## Grid Scope

Softmax grid:

```text
topk: 3, 4, 5, 6
hold_days: 1, 2, 3
max_position_pct: 0.20, 0.25, 0.30, 0.35
reserve_cash_pct: 0.00, 0.02, 0.05, 0.10
min_amount: 10M, 20M, 30M, 50M
capital: 100K, 200K, 1M
```

## Best Observed Region

The strongest region is concentrated around:

```text
topk = 4 or 5
hold_days = 1
max_position_pct = 0.35
min_amount = 30M
allocation = softmax
```

This is encouraging because the leading results are a cluster, not a single isolated point.

## Leading Candidate

Candidate:

```text
top4 / hold_days=1 / max_position_pct=0.35 / reserve_cash_pct=0.02 / min_amount=30000000 / softmax
```

Observed results:

| capital | cum_return | max_drawdown | IR | avg_position_ratio | avg_positions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | 63.19% | -12.13% | 3.15 | 43.58% | 3.09 |
| 200K | 61.03% | -13.18% | 3.02 | 44.81% | 3.23 |
| 1M | 64.21% | -13.07% | 3.11 | 45.85% | 3.45 |

Interpretation:

- Strong across all three capital buckets.
- Drawdown remains around 12-13%.
- Average position ratio is below 50%, so capital usage is not extreme.
- Average positions are about 3-4 names, which is operationally manageable.

## Alternative Candidates

More defensive reserve:

```text
top4 / hold_days=1 / max_position_pct=0.35 / reserve_cash_pct=0.05 / min_amount=30000000 / softmax
```

Observed:

| capital | cum_return | max_drawdown | IR |
| ---: | ---: | ---: | ---: |
| 100K | 60.20% | -12.12% | 3.04 |
| 200K | 60.68% | -12.75% | 3.06 |
| 1M | 63.39% | -12.75% | 3.13 |

Slightly more diversified:

```text
top5 / hold_days=1 / max_position_pct=0.35 / reserve_cash_pct=0.05 / min_amount=30000000 / softmax
```

Observed:

| capital | cum_return | max_drawdown | IR |
| ---: | ---: | ---: | ---: |
| 100K | 57.31% | -12.00% | 3.06 |
| 200K | 59.48% | -12.18% | 3.07 |
| 1M | 60.37% | -12.28% | 3.07 |

## Preliminary Conclusion

This short-hold branch is worth continuing.

The current best candidate is:

```text
top4 / hold_days=1 / max_position_pct=0.35 / reserve_cash_pct=0.02 / min_amount=30000000 / softmax
```

However, it should remain research-only until a realism pass is completed.

## Next Validation Steps

Before considering production use:

- Add Eastmoney-style minimum commission.
- Add high-open / limit-up buy failure rules.
- Add market permission filters for ChiNext and STAR Market.
- Compare `softmax` with rank-based and equal-weight allocation near the best region.
- Validate robustness on a longer out-of-sample window.
- Check whether the strategy is concentrated in a few exceptional days.
- Review daily turnover and operational burden.

