# Paper Trading System

This directory is intended to be portable. Copy the whole `paper_trading_system/`
folder to another machine, edit `config/env.local`, and run the same commands.

## Trading Timing

```text
T after close:
  Generate T+1 pending orders from the T close signal.

T+1 after open:
  Execute pending orders at T+1 raw open.
```

Research backtests can still use Qlib `deal_price=close` for comparison, but
this paper-trading system uses `next_open` to avoid same-close look-ahead.

## Strategy

```text
models:       XGB short + XGB long + ADARNN
fusion:       zscore_mean
weights:      10:1:1
portfolio:    top20/drop2
capital:      configurable, default 100000 RMB
```

## Files

```text
config/env.example     Template runtime config
config/env.local       Machine-local config, not committed by convention
src/paper_trading_daily.py
bin/after_close.sh     Create pending orders
bin/after_open.sh      Execute pending orders
state/                 Account, positions, NAV, pending orders, trades
reports/               Markdown daily reports
logs/                  Shell logs
preds/                 Optional local prediction files
models/                Optional model artifacts
data/                  Optional local Qlib data
```

## Setup

```bash
cd /path/to/paper_trading_system
cp config/env.example config/env.local
vim config/env.local
```

Install dependencies with one command:

```bash
bash bin/setup_env.sh cpu
```

On a GPU machine:

```bash
bash bin/setup_env.sh gpu
```

If a local Qlib source tree is needed, set:

```text
QLIB_SRC=/path/to/qlib
QLIB_PROVIDER_URI=/path/to/reference_cn_data/cn_data
```

Otherwise keep `QLIB_SRC` empty.

Check the environment at any time:

```bash
set -a
source config/env.local
set +a
"$PYTHON_BIN" bin/check_env.py
```

## Run

Update data:

```bash
bash bin/update_data.sh
```

The data updater downloads the latest community Qlib binary release, extracts it
under `data/releases/<timestamp>`, switches `data/current` to the new version,
and keeps only the latest `DATA_KEEP_RELEASES` versions.

Train models:

```bash
bash bin/train_models.sh daily cpu
bash bin/train_models.sh weekly gpu
bash bin/train_models.sh monthly gpu
bash bin/train_models.sh full gpu
```

Training cadence:

```text
daily:   XGB short
weekly:  XGB short + ADARNN short
monthly: XGB short + XGB long + ADARNN short
full:    XGB short + XGB long + ADARNN short, manual full refresh
```

For local timing only, generate smoke/timing configs first:

```bash
set -a
source config/env.local
set +a
"$PYTHON_BIN" bin/gen_timing_configs.py
```

Then point `config/env.local` at the generated configs under
`config/generated/`.

The same script supports both CPU and GPU:

```text
cpu:  exports CUDA_VISIBLE_DEVICES=""
gpu:  uses CUDA_VISIBLE_DEVICES, default 0
auto: leaves the environment unchanged
```

After market close:

```bash
bash bin/after_close.sh
```

After next market open:

```bash
bash bin/after_open.sh 2026-05-20
```

## Portability Rule

Do not hardcode machine-specific paths in scripts. Put them in
`config/env.local`. The only required edit after moving machines should be this
one file.
