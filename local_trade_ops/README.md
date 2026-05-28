# Local Trade Ops

This directory is the local execution workspace.

GPU responsibilities:
- Train models.
- Generate daily prediction pkl files.

Local responsibilities:
- Update the community Qlib daily data release.
- Sync local Qlib data to GPU before training.
- Validate local prediction pkl files, or generate/sync them when explicitly configured.
- Generate after-close trade suggestions.
- Wait for human review.
- Execute only approved orders.
- Maintain local paper/live account state.

## Current Main Strategy

```text
xgb_csi1000_long_prod2026
+ doubleensemble_csi1000_short_prod2026
+ catboost_csi1000_long_prod2026

weights: 3,1,1
filter: jq_filter
topk/drop: 7/1
min_amount: 20000000
capital: 100000
```

## Daily Flow

You can run the local control panel instead of typing each command:

```bash
node service/server.ts
```

Then open `http://127.0.0.1:8787`.

The backend service has its own scheduler. It is configured in
`config/scheduler.json` and is enabled by default. The default scheduled chain
is: update data -> validate local predictions -> generate after-close review
package.

0. When data has been refreshed locally, sync Qlib data to GPU.

```bash
bash bin/sync_qlib_to_gpu.sh
```

1. Validate local predictions.

```bash
bash bin/generate_predictions.sh
```

By default, `PREDICTION_MODE=skip` only validates the existing pkl files and
does not touch GPU. Later, `PREDICTION_MODE=sync` can pull pkl files from GPU,
or `PREDICTION_MODE=command` can run `PREDICTION_CMD` for fully-local inference.

2. After close, generate suggested orders.

```bash
bash bin/after_close_local.sh
```

Or run the pre-review chain in one shot:

```bash
bash bin/daily_after_data_pipeline.sh
```

3. Review the generated `pending_orders_YYYY-MM-DD.csv`.

4. Approve or edit.

Approve all:

```bash
python bin/approve_orders.py --execution-date YYYY-MM-DD --status approved
```

Reject:

```bash
python bin/approve_orders.py --execution-date YYYY-MM-DD --status rejected --note "skip today"
```

Approve after manually editing `approved_orders_YYYY-MM-DD.csv`:

```bash
python bin/approve_orders.py --execution-date YYYY-MM-DD --status approved_with_edits
```

5. Next open, create the approved live order task.

```bash
bash bin/trigger_live_task.sh YYYY-MM-DD
```

6. Submit/export the live order task.

Default `LIVE_BROKER_MODE=manual` writes an operator ticket only:

```bash
bash bin/submit_live_orders.sh YYYY-MM-DD
```

This also writes a manual fill template:

```text
live_fills/manual_fill_template_YYYY-MM-DD.csv
```

After manually placing orders in the broker app/client, fill in:

```text
fill_status, fill_shares, fill_price, fee, operator_note
```

Then apply the real fills to the local account book:

```bash
bash bin/apply_manual_fills.sh YYYY-MM-DD
```

To call the QMT adapter, first install/configure miniQMT and fill
`QMT_CLIENT_PATH` and `QMT_ACCOUNT_ID` in `config/env.local`. Then set
`LIVE_BROKER_MODE=hook` and `LIVE_TRADING_ENABLED=1`. The adapter receives:

```text
broker_order_ticket.csv live_submission.json live_order_task.csv live_order_task.json
```

The submit step refuses non-today dates by default. Use `--allow-non-today`
only for dry-run/backfill tests.

QMT adapter dry-run:

```bash
python bin/qmt_order_adapter.py \
  live_submissions/broker_order_ticket_YYYY-MM-DD.csv \
  live_submissions/live_submission_YYYY-MM-DD.json \
  --dry-run
```

7. Optional: record simulated paper execution for audit and NAV tracking.

```bash
bash bin/after_open_local.sh YYYY-MM-DD
```

Without `approved` or `approved_with_edits`, execution is skipped.
