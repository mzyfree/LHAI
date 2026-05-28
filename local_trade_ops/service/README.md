# Local Trade Ops Service

Zero-dependency TypeScript control panel for the local trading workflow.

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
node service/server.ts
```

Open:

```text
http://127.0.0.1:8787
```

The service wraps the existing, audited scripts:

- `paper_trading_system/bin/update_data.sh`
- `bin/generate_predictions.sh`
- `bin/daily_after_data_pipeline.sh`
- `bin/after_close_local.sh`
- `bin/approve_orders.py`
- `bin/trigger_live_task.sh`
- `bin/submit_live_orders.sh`
- `bin/after_open_local.sh`

By default it only creates live task files and manual broker tickets. It does
not call a broker unless `LIVE_BROKER_MODE=hook`, `LIVE_TRADING_ENABLED=1`,
and `LIVE_ORDER_HOOK` are configured. The prepared hook is
`bin/qmt_order_adapter.py` for miniQMT/QMT.

## Live Submit Step

After review, the manual execution UI flow is:

```text
Approve -> 生成实盘任务 -> 提交/导出委托 -> 人工下单 -> 应用成交回填
```

`提交/导出委托` reads `live_order_task_YYYY-MM-DD.csv`, writes:

- `live_submissions/broker_order_ticket_YYYY-MM-DD.csv`
- `live_submissions/live_submission_YYYY-MM-DD.json`
- `live_fills/manual_fill_template_YYYY-MM-DD.csv`

In `manual` mode, this is the final handoff file for operator review. In
`hook` mode, the configured broker adapter is called with:

```text
broker_order_ticket.csv live_submission.json live_order_task.csv live_order_task.json
```

For manual execution, edit `manual_fill_template_YYYY-MM-DD.csv` after broker
execution. Fill `fill_shares`, `fill_price`, optional `fee`, and
`operator_note`, then click `应用成交回填`. The local account book uses these real
fills to update cash, positions, NAV, and the fill report.

## QMT Adapter

The QMT adapter is intentionally inert until miniQMT/xtquant is installed and
configured locally. Fill these values in `config/env.local`:

- `QMT_CLIENT_PATH`: miniQMT user data path used by `XtQuantTrader`.
- `QMT_ACCOUNT_ID`: stock account id.
- `QMT_ACCOUNT_TYPE`: usually `STOCK`.
- `QMT_ORDER_TYPE`, `QMT_PRICE_TYPE`: names from `xtquant.xtconstant`.

Dry-run validation:

```bash
python bin/qmt_order_adapter.py \
  live_submissions/broker_order_ticket_YYYY-MM-DD.csv \
  live_submissions/live_submission_YYYY-MM-DD.json \
  --dry-run
```

## Scheduled Pre-Review Chain

The community release has recently appeared around `11:09-11:11` on the
GitHub releases page. The service has an internal scheduler configured in:

```text
config/scheduler.json
```

By default, it runs this chain once per day at `20:30` `Asia/Shanghai` while
the service is running:

```text
update local Qlib data -> validate local prediction pkls -> generate after-close suggestions -> wait for review
```

The UI can toggle the scheduler on or off. It will not approve orders or create
the next-day live task automatically.

Prediction handling is controlled by `config/env.local`. The default is manual:

- `PREDICTION_MODE=skip`: validate the existing pkl files only.
- `PREDICTION_MODE=sync`: pull prediction pkl files from GPU.
- `PREDICTION_MODE=command`: run `PREDICTION_CMD` to produce local pkl files.
