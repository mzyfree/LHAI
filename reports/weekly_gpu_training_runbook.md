# Weekly GPU Training Runbook

This runbook documents the weekly CSI1000 production-model refresh flow.

## Goal

Refresh the three main CSI1000 model artifacts on a GPU node, bring the model package back to the local ops platform, and make local inference use the new package through a stable `latest` symlink.

Main models:

- `xgb_csi1000_long_prod2026`
- `doubleensemble_csi1000_short_prod2026`
- `catboost_csi1000_long_prod2026`

## Training Split

The weekly script derives dates from the local Qlib calendar archive:

- `infer/test`: latest local Qlib calendar date.
- `valid_end`: latest local Qlib calendar date minus 5 trading days.
- `valid_start`: `2026-01-05`.
- `train_end`: `2025-12-31`.
- `train_start`: inherited from each base workflow config.

Example with local data ending at `2026-05-29`:

```text
train: config start -> 2025-12-31
valid: 2026-01-05 -> 2026-05-22
infer/test: 2026-05-29 -> 2026-05-29
```

The recent gap between `valid_end` and `infer/test` is intentional. It keeps the newest bars as inference features while avoiding incomplete future labels in validation.

Manual overrides are still available:

```bash
WEEKLY_VALID_END=2026-05-29 bash bin/weekly_gpu_train.sh start
WEEKLY_TRAIN_INFER_DATE=2026-06-05 bash bin/weekly_gpu_train.sh start
```

## Local Prerequisites

Run from:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
```

Expected local archive:

```text
/Users/Dylan.Min/Documents/Code/learn/LHAI/paper_trading_system/data/archives/latest_qlib_bin.tar.gz
```

The archive must contain:

```text
qlib_bin/calendars/day.txt
```

The script checks that the archive latest calendar date matches the inferred training date.

## Start Weekly Training

Use SSH key login if possible. If using password login, set the password in the shell only; do not commit it to the repository.

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops

export GPU_SSH_PASSWORD='your-password'

bash bin/weekly_gpu_train.sh start
```

The script will:

1. Validate the local Qlib archive.
2. Upload the archive to the GPU node.
3. Upload the current training helper scripts.
4. Extract the archive on GPU under `reference_cn_data/releases/<RUN_ID>/qlib_bin`.
5. Update GPU symlinks:
   - `/root/autodl-tmp/llhh/reference_cn_data/cn_data`
   - `/root/autodl-tmp/llhh/paper_trading_system/data/current`
6. Generate weekly workflow configs on GPU.
7. Start GPU training with `nohup`.

The command prints:

- `RUN_ID`
- remote log path
- a `tail -f` command
- the collect command

## Monitor Training

Use the printed command, for example:

```bash
ssh -p 37172 root@connect.westc.seetacloud.com \
  'tail -f /root/autodl-tmp/llhh/logs/weekly_train_20260531_20260529.log'
```

A successful run ends with:

```text
===== weekly training package ready =====
.../csi1000_main_model_packages_<RUN_ID>.tar.gz
.../preds/csi1000/xgb_csi1000_long_prod2026.pkl
.../preds/csi1000/doubleensemble_csi1000_short_prod2026.pkl
.../preds/csi1000/catboost_csi1000_long_prod2026.pkl
```

## Collect Artifacts

After training finishes:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops

bash bin/weekly_gpu_train.sh collect <RUN_ID>
```

The collect step downloads:

- `model_packages/csi1000_main_model_packages_<RUN_ID>.tar.gz`
- `preds/csi1000/xgb_csi1000_long_prod2026.pkl`
- `preds/csi1000/doubleensemble_csi1000_short_prod2026.pkl`
- `preds/csi1000/catboost_csi1000_long_prod2026.pkl`
- per-model training logs into `archive/weekly_train/<RUN_ID>/`

It also updates:

```text
model_packages/latest_csi1000_main_model_package.tar.gz
```

to point to the newly collected model package.

## Local Inference After Collection

The local ops config points prediction inference to:

```text
model_packages/latest_csi1000_main_model_package.tar.gz
```

Run local inference once after collecting a new package:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops

bash bin/generate_predictions.sh
```

Expected confirmation:

```text
Model package root: .../csi1000_main_<RUN_ID>
Prediction/date alignment OK.
```

This is important because the `pred.pkl` files copied from GPU are only a sanity check for the GPU `test` date. The platform should use the latest model package to regenerate full local predictions for the current local Qlib provider.

## Outputs Used By The Platform

After the collection and local inference steps, the platform uses:

- latest model package:
  `local_trade_ops/model_packages/latest_csi1000_main_model_package.tar.gz`
- local prediction pkls:
  `local_trade_ops/preds/csi1000/*.pkl`

Both the original T/T+1 page and the short-hold page share the same local prediction layer.

## Common Issues

### `qrun: command not found`

GPU `nohup` runs in a non-interactive shell, so PATH can be different. The GPU training script defaults to:

```text
/root/miniconda3/bin/qrun
```

If the GPU environment changes, override it:

```bash
QRUN_BIN=/path/to/qrun bash bin/train_weekly_main_models.sh ...
```

### New package collected but local inference uses old package

The prediction script now resolves the package root from the actual tarball instead of reusing any existing extracted directory. Verify logs include:

```text
Model package root: .../csi1000_main_<new RUN_ID>
```

If not, remove the extracted directory and rerun local inference:

```bash
rm -rf /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/model_packages/extracted
bash /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/generate_predictions.sh
```

### Password handling

Do not commit GPU passwords or tokens. Use an environment variable for a one-off session:

```bash
export GPU_SSH_PASSWORD='your-password'
```

Prefer SSH keys for recurring weekly training.
