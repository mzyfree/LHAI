# Short Hold V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `short_hold_v3` scoring layer that keeps the current short-hold return model, then reranks candidates with explicit buyability risk and strong-stock probability.

**Architecture:** Keep `short_hold_v2` prediction pkls as the return-score source. Add focused Python modules for v3 feature building, side scoring, and side model training, then make `short_hold_generate_orders.py` choose between v2 and v3 through environment variables. The UI reads the same order/review CSVs, with extra v3 columns shown when present.

**Tech Stack:** Python, pandas, numpy, qlib daily features, pickle/joblib artifacts, Node/TypeScript local service, shell scripts, unittest.

---

## File Structure

- Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_v3_features.py`: pure feature and label builders using only T-day-visible OHLCV for inference and future OHLCV only for offline labels.
- Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_v3_scoring.py`: pure scoring functions that combine `return_score`, `buyability_risk`, `strong_prob`, and liquidity penalty into `final_score`.
- Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/train_short_hold_v3_side_models.py`: CLI that builds historical candidate samples and writes side model artifacts.
- Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/backtest_short_hold_v3_score.py`: CLI that compares v2 ordering against v3 ordering with the existing short-hold execution assumptions.
- Modify `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`: optional v3 scoring before primary/backup selection, defaulting to current v2 behavior.
- Modify `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/prepare_short_hold_orders.sh`: pass v3 env without changing default mode.
- Modify `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/service/server.ts`: show v3 columns and score explanations when CSVs contain them.
- Modify `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/config/env.local.example`: document v3 env variables with safe defaults.
- Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/tests/test_short_hold_v3_features.py`: unit tests for T-visible features and offline labels.
- Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/tests/test_short_hold_v3_scoring.py`: unit tests for score combination and sorting.
- Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/docs/short_hold_v3.md`: operator notes and rollback instructions.

---

### Task 1: Add V3 Feature And Label Builders

**Files:**
- Create: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/tests/test_short_hold_v3_features.py`
- Create: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_v3_features.py`

- [x] **Step 1: Write failing tests for feature and label builders**

Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/tests/test_short_hold_v3_features.py` with:

```python
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.short_hold_v3_features import (
    build_inference_features,
    build_training_labels,
    stock_limit_up_pct,
)


class ShortHoldV3FeatureTests(unittest.TestCase):
    def setUp(self):
        index = pd.MultiIndex.from_tuples(
            [
                ("2026-06-10", "SH600001"),
                ("2026-06-11", "SH600001"),
                ("2026-06-12", "SH600001"),
                ("2026-06-15", "SH600001"),
                ("2026-06-10", "SZ300001"),
                ("2026-06-11", "SZ300001"),
                ("2026-06-12", "SZ300001"),
                ("2026-06-15", "SZ300001"),
            ],
            names=["datetime", "instrument"],
        )
        self.ohlcv = pd.DataFrame(
            {
                "open": [10.0, 10.3, 11.0, 11.5, 20.0, 20.8, 22.9, 23.5],
                "high": [10.4, 10.8, 11.5, 11.8, 21.0, 22.8, 24.8, 24.2],
                "low": [9.9, 10.1, 10.8, 11.1, 19.8, 20.5, 22.5, 23.0],
                "close": [10.2, 10.5, 11.2, 11.6, 20.5, 22.6, 23.1, 24.0],
                "volume": [1000, 1200, 1500, 1400, 2000, 2500, 3000, 2800],
                "amount": [10_200_000, 12_600_000, 16_800_000, 16_240_000, 41_000_000, 56_500_000, 69_300_000, 67_200_000],
            },
            index=index,
        )
        self.ohlcv.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(self.ohlcv.index.get_level_values("datetime")),
                self.ohlcv.index.get_level_values("instrument"),
            ],
            names=["datetime", "instrument"],
        )

    def test_stock_limit_up_pct_handles_board_prefix(self):
        self.assertEqual(stock_limit_up_pct("SH600001"), 0.10)
        self.assertEqual(stock_limit_up_pct("SZ000001"), 0.10)
        self.assertEqual(stock_limit_up_pct("SZ300001"), 0.20)
        self.assertEqual(stock_limit_up_pct("SH688001"), 0.20)

    def test_build_inference_features_uses_signal_date_only(self):
        rows = pd.DataFrame(
            {
                "instrument": ["SH600001", "SZ300001"],
                "return_score": [2.0, 3.0],
                "model_rank": [2, 1],
            }
        )
        features = build_inference_features(rows, self.ohlcv, pd.Timestamp("2026-06-11"))
        self.assertEqual(list(features["instrument"]), ["SH600001", "SZ300001"])
        self.assertIn("ret_1d", features.columns)
        self.assertIn("amount_mean_3d", features.columns)
        self.assertIn("distance_to_limit_up", features.columns)
        self.assertAlmostEqual(float(features.loc[0, "ret_1d"]), 10.5 / 10.2 - 1.0, places=8)
        self.assertAlmostEqual(float(features.loc[1, "limit_up_pct"]), 0.20, places=8)

    def test_build_training_labels_uses_future_dates_for_labels(self):
        rows = pd.DataFrame({"instrument": ["SH600001"], "return_score": [2.0], "model_rank": [1]})
        labels = build_training_labels(
            rows,
            self.ohlcv,
            signal_date=pd.Timestamp("2026-06-11"),
            entry_date=pd.Timestamp("2026-06-12"),
            exit_date=pd.Timestamp("2026-06-15"),
        )
        self.assertEqual(labels.loc[0, "instrument"], "SH600001")
        self.assertAlmostEqual(float(labels.loc[0, "realized_return"]), 11.6 / 11.0 - 1.0, places=8)
        self.assertEqual(int(labels.loc[0, "open_gt_5pct"]), 0)
        self.assertEqual(int(labels.loc[0, "buyability_bad"]), 0)
        self.assertIn("strong_label", labels.columns)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the feature tests and verify they fail**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m unittest tests.test_short_hold_v3_features
```

Expected: fail with `ModuleNotFoundError: No module named 'bin.short_hold_v3_features'`.

- [x] **Step 3: Implement feature and label builders**

Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_v3_features.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V3FeatureColumns:
    numeric: tuple[str, ...] = (
        "return_score",
        "model_rank",
        "close",
        "amount",
        "volume",
        "ret_1d",
        "ret_3d",
        "amplitude_1d",
        "amount_mean_3d",
        "distance_to_limit_up",
        "limit_up_pct",
    )


def stock_limit_up_pct(instrument: str) -> float:
    code = instrument.upper()
    if code.startswith("SZ300") or code.startswith("SH688"):
        return 0.20
    return 0.10


def _date_slice(ohlcv: pd.DataFrame, instrument: str, end_date: pd.Timestamp, window: int) -> pd.DataFrame:
    dates = ohlcv.index.get_level_values("datetime")
    instruments = ohlcv.index.get_level_values("instrument")
    mask = (instruments == instrument) & (dates <= end_date)
    return ohlcv.loc[mask].sort_index().tail(window)


def _row_at(ohlcv: pd.DataFrame, instrument: str, date: pd.Timestamp) -> pd.Series:
    key = (pd.Timestamp(date), instrument)
    if key not in ohlcv.index:
        raise KeyError(f"missing OHLCV row for {instrument} on {pd.Timestamp(date).date()}")
    return ohlcv.loc[key]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_inference_features(candidates: pd.DataFrame, ohlcv: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signal_date = pd.Timestamp(signal_date)
    for candidate in candidates.to_dict("records"):
        instrument = str(candidate["instrument"])
        hist = _date_slice(ohlcv, instrument, signal_date, 20)
        if hist.empty:
            continue
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else last
        first3 = hist.tail(3).iloc[0]
        limit_pct = stock_limit_up_pct(instrument)
        theoretical_limit = _safe_float(prev["close"]) * (1.0 + limit_pct)
        close = _safe_float(last["close"])
        high = _safe_float(last["high"])
        low = _safe_float(last["low"])
        amount = _safe_float(last["amount"])
        rows.append(
            {
                "instrument": instrument,
                "return_score": _safe_float(candidate.get("return_score", candidate.get("score"))),
                "model_rank": _safe_float(candidate.get("model_rank", candidate.get("rank"))),
                "close": close,
                "amount": amount,
                "volume": _safe_float(last["volume"]),
                "ret_1d": close / _safe_float(prev["close"]) - 1.0 if _safe_float(prev["close"]) else np.nan,
                "ret_3d": close / _safe_float(first3["close"]) - 1.0 if _safe_float(first3["close"]) else np.nan,
                "amplitude_1d": (high - low) / close if close else np.nan,
                "amount_mean_3d": float(hist.tail(3)["amount"].mean()),
                "distance_to_limit_up": theoretical_limit / close - 1.0 if close else np.nan,
                "limit_up_pct": limit_pct,
            }
        )
    return pd.DataFrame(rows)


def build_training_labels(
    candidates: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    strong_return_threshold: float = 0.06,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        instrument = str(candidate["instrument"])
        signal_row = _row_at(ohlcv, instrument, signal_date)
        entry_row = _row_at(ohlcv, instrument, entry_date)
        exit_row = _row_at(ohlcv, instrument, exit_date)
        prev_close = _safe_float(signal_row["close"])
        entry_open = _safe_float(entry_row["open"])
        entry_high = _safe_float(entry_row["high"])
        entry_low = _safe_float(entry_row["low"])
        exit_close = _safe_float(exit_row["close"])
        limit_pct = stock_limit_up_pct(instrument)
        entry_gap = entry_open / prev_close - 1.0 if prev_close else np.nan
        realized_return = exit_close / entry_open - 1.0 if entry_open else np.nan
        one_price_limit = abs(entry_high - entry_low) < 1e-8 and entry_gap >= limit_pct * 0.95
        buyability_bad = bool(entry_gap > 0.05 or one_price_limit)
        rows.append(
            {
                "instrument": instrument,
                "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                "entry_date": pd.Timestamp(entry_date).date().isoformat(),
                "exit_date": pd.Timestamp(exit_date).date().isoformat(),
                "realized_return": realized_return,
                "entry_gap": entry_gap,
                "open_gt_5pct": int(entry_gap > 0.05),
                "one_price_limit": int(one_price_limit),
                "buyability_bad": int(buyability_bad),
                "strong_label": int(realized_return >= strong_return_threshold),
            }
        )
    return pd.DataFrame(rows)


def load_qlib_ohlcv(instruments: list[str], start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DataFrame:
    from qlib.data import D

    raw = D.features(
        instruments,
        ["$open", "$high", "$low", "$close", "$volume", "$factor"],
        start_time=str(pd.Timestamp(start_date).date()),
        end_time=str(pd.Timestamp(end_date).date()),
        freq="day",
    )
    raw = raw.rename(
        columns={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$factor": "factor",
        }
    )
    raw["amount"] = raw["close"] * raw["volume"] * raw["factor"]
    return raw
```

- [x] **Step 4: Run feature tests and verify they pass**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m unittest tests.test_short_hold_v3_features
```

Expected: `OK`.

- [x] **Step 5: Commit feature builders**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI
git add local_trade_ops/bin/short_hold_v3_features.py local_trade_ops/tests/test_short_hold_v3_features.py
git commit -m "feat: add short hold v3 feature builders"
```

---

### Task 2: Add V3 Scoring Functions

**Files:**
- Create: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/tests/test_short_hold_v3_scoring.py`
- Create: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_v3_scoring.py`

- [x] **Step 1: Write failing scoring tests**

Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/tests/test_short_hold_v3_scoring.py` with:

```python
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.short_hold_v3_scoring import score_candidates_v3, sort_candidates_for_orders


class ShortHoldV3ScoringTests(unittest.TestCase):
    def test_score_candidates_combines_return_strong_and_risk(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B", "C"],
                "score": [5.0, 4.0, 3.0],
                "model_rank": [1, 2, 3],
                "amount": [50_000_000, 40_000_000, 10_000_000],
            }
        )
        side = pd.DataFrame(
            {
                "instrument": ["A", "B", "C"],
                "buyability_risk": [0.90, 0.10, 0.20],
                "strong_prob": [0.20, 0.60, 0.10],
            }
        )
        scored = score_candidates_v3(candidates, side, alpha=1.0, beta=2.0, gamma=0.0)
        row_a = scored.loc[scored["instrument"] == "A"].iloc[0]
        row_b = scored.loc[scored["instrument"] == "B"].iloc[0]
        self.assertAlmostEqual(float(row_a["return_score"]), 5.0)
        self.assertLess(float(row_a["final_score"]), float(row_b["final_score"]))
        self.assertEqual(row_a["score_source"], "v3")

    def test_sort_candidates_uses_final_score_when_present(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B"],
                "score": [5.0, 4.0],
                "final_score": [3.0, 4.5],
            }
        )
        sorted_candidates = sort_candidates_for_orders(candidates)
        self.assertEqual(list(sorted_candidates["instrument"]), ["B", "A"])

    def test_sort_candidates_falls_back_to_score(self):
        candidates = pd.DataFrame({"instrument": ["A", "B"], "score": [1.0, 2.0]})
        sorted_candidates = sort_candidates_for_orders(candidates)
        self.assertEqual(list(sorted_candidates["instrument"]), ["B", "A"])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run scoring tests and verify they fail**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m unittest tests.test_short_hold_v3_scoring
```

Expected: fail with `ModuleNotFoundError: No module named 'bin.short_hold_v3_scoring'`.

- [x] **Step 3: Implement scoring functions**

Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_v3_scoring.py` with:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return numeric.fillna(numeric.median()).fillna(0.0) * 0.0
    return (numeric - numeric.mean()) / std


def _liquidity_risk(amount: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(amount, errors="coerce").fillna(0.0)
    risk = 1.0 - (numeric / 50_000_000.0).clip(lower=0.0, upper=1.0)
    return risk


def score_candidates_v3(
    candidates: pd.DataFrame,
    side_scores: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    merged = candidates.copy()
    if "return_score" not in merged.columns:
        merged["return_score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged = merged.merge(
        side_scores[["instrument", "buyability_risk", "strong_prob"]],
        on="instrument",
        how="left",
    )
    merged["buyability_risk"] = pd.to_numeric(merged["buyability_risk"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged["strong_prob"] = pd.to_numeric(merged["strong_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    amount = merged["amount"] if "amount" in merged.columns else pd.Series(50_000_000.0, index=merged.index)
    merged["liquidity_risk"] = _liquidity_risk(amount)
    merged["final_score"] = (
        _normalize_series(merged["return_score"])
        + alpha * merged["strong_prob"]
        - beta * merged["buyability_risk"]
        - gamma * merged["liquidity_risk"]
    )
    merged["score_source"] = "v3"
    return merged


def sort_candidates_for_orders(candidates: pd.DataFrame) -> pd.DataFrame:
    sort_col = "final_score" if "final_score" in candidates.columns else "score"
    return candidates.sort_values(sort_col, ascending=False).reset_index(drop=True)
```

- [x] **Step 4: Run scoring tests and verify they pass**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m unittest tests.test_short_hold_v3_scoring
```

Expected: `OK`.

- [x] **Step 5: Commit scoring functions**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI
git add local_trade_ops/bin/short_hold_v3_scoring.py local_trade_ops/tests/test_short_hold_v3_scoring.py
git commit -m "feat: add short hold v3 scoring"
```

---

### Task 3: Add Side Model Training CLI

**Files:**
- Create: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/train_short_hold_v3_side_models.py`

- [x] **Step 1: Add the side model trainer CLI**

Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/train_short_hold_v3_side_models.py` with the following interface:

```python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from bin.short_hold_v3_features import V3FeatureColumns


class QuantileSideModel:
    def __init__(self, column: str, direction: str):
        self.column = column
        self.direction = direction
        self.quantiles: list[float] = []

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "QuantileSideModel":
        series = pd.to_numeric(features[self.column], errors="coerce").fillna(0.0)
        positive = series[labels.astype(int) == 1]
        if positive.empty:
            positive = series
        self.quantiles = [float(positive.quantile(q)) for q in (0.25, 0.50, 0.75)]
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        series = pd.to_numeric(features[self.column], errors="coerce").fillna(0.0)
        q1, q2, q3 = self.quantiles
        if self.direction == "high":
            score = np.select([series >= q3, series >= q2, series >= q1], [0.85, 0.65, 0.45], default=0.20)
        else:
            score = np.select([series <= q1, series <= q2, series <= q3], [0.85, 0.65, 0.45], default=0.20)
        return np.vstack([1.0 - score, score]).T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train short-hold v3 side scoring models.")
    parser.add_argument("--samples", required=True, help="CSV with v3 features and labels.")
    parser.add_argument("--output-dir", required=True, help="Directory for side model artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = pd.read_csv(args.samples)
    required = set(V3FeatureColumns().numeric) | {"buyability_bad", "strong_label"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"sample file is missing columns: {missing}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buyability_model = QuantileSideModel("distance_to_limit_up", "low").fit(samples, samples["buyability_bad"])
    strong_model = QuantileSideModel("return_score", "high").fit(samples, samples["strong_label"])
    with (output_dir / "buyability_model.pkl").open("wb") as f:
        pickle.dump(buyability_model, f)
    with (output_dir / "strong_model.pkl").open("wb") as f:
        pickle.dump(strong_model, f)
    metadata = {
        "feature_columns": list(V3FeatureColumns().numeric),
        "buyability_label": "buyability_bad",
        "strong_label": "strong_label",
        "model_type": "quantile_side_model",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Wrote side models: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Run CLI syntax check**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m py_compile bin/train_short_hold_v3_side_models.py
```

Expected: no output and exit code `0`.

- [x] **Step 3: Train side models with a tiny sample**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
mkdir -p tmp
cat > tmp/short_hold_v3_side_samples.csv <<'CSV'
instrument,return_score,model_rank,close,amount,volume,ret_1d,ret_3d,amplitude_1d,amount_mean_3d,distance_to_limit_up,limit_up_pct,buyability_bad,strong_label
SH600001,5.0,1,10.2,50000000,1000,0.01,0.03,0.04,45000000,0.08,0.10,0,1
SH600002,3.0,2,12.0,20000000,800,0.08,0.12,0.01,18000000,0.01,0.10,1,0
SZ300001,4.0,3,22.6,60000000,2000,0.03,0.09,0.05,55000000,0.15,0.20,0,1
CSV
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python bin/train_short_hold_v3_side_models.py \
  --samples tmp/short_hold_v3_side_samples.csv \
  --output-dir tmp/short_hold_v3_side_models
```

Expected:

```text
Wrote side models: tmp/short_hold_v3_side_models
```

- [x] **Step 4: Verify artifacts exist**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
ls -lh tmp/short_hold_v3_side_models/buyability_model.pkl tmp/short_hold_v3_side_models/strong_model.pkl tmp/short_hold_v3_side_models/metadata.json
```

Expected: all three files are listed.

- [x] **Step 5: Commit side model trainer**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI
git add local_trade_ops/bin/train_short_hold_v3_side_models.py
git commit -m "feat: add short hold v3 side model trainer"
```

---

### Task 4: Integrate V3 Scoring Into Short-Hold Order Generation

**Files:**
- Modify: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`
- Modify: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/prepare_short_hold_orders.sh`
- Modify: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/config/env.local.example`

- [x] **Step 1: Add env variables to `env.local.example`**

Append to `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/config/env.local.example`:

```bash
# Short-hold v3 scoring. Default keeps the current v2 score path.
SHORT_HOLD_SCORING_MODE=v2
SHORT_HOLD_V3_SIDE_MODEL_DIR=/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/model_packages/short_hold_v3_side_models
SHORT_HOLD_V3_ALPHA=1.0
SHORT_HOLD_V3_BETA=1.0
SHORT_HOLD_V3_GAMMA=0.25
```

- [x] **Step 2: Import v3 helpers in the generator**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`, add imports near the existing local imports:

```python
try:
    from bin.short_hold_v3_scoring import score_candidates_v3, sort_candidates_for_orders
except ModuleNotFoundError:
    score_candidates_v3 = None
    sort_candidates_for_orders = None
```

- [x] **Step 3: Add side model directory validation in the generator**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`, add this function near existing helper functions:

```python
def validate_v3_side_model_dir(side_model_dir: Path) -> None:
    if not side_model_dir.exists():
        raise FileNotFoundError(f"SHORT_HOLD_V3_SIDE_MODEL_DIR does not exist: {side_model_dir}")
    buyability_path = side_model_dir / "buyability_model.pkl"
    strong_path = side_model_dir / "strong_model.pkl"
    if not buyability_path.exists() or not strong_path.exists():
        raise FileNotFoundError(f"missing v3 side model artifacts in {side_model_dir}")
```

- [x] **Step 4: Add v3 scoring application in the generator**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`, add this function near `load_v3_side_scores`:

```python
def apply_v3_scoring(
    candidates: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signal_date: pd.Timestamp,
    side_model_dir: Path,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    import pickle

    from bin.short_hold_v3_features import build_inference_features
    from bin.short_hold_v3_scoring import score_candidates_v3

    validate_v3_side_model_dir(side_model_dir)
    feature_rows = build_inference_features(
        candidates.rename(columns={"score": "return_score"}),
        ohlcv,
        signal_date,
    )
    with (side_model_dir / "buyability_model.pkl").open("rb") as f:
        buyability_model = pickle.load(f)
    with (side_model_dir / "strong_model.pkl").open("rb") as f:
        strong_model = pickle.load(f)
    feature_columns = [c for c in feature_rows.columns if c not in {"instrument"}]
    buyability_risk = buyability_model.predict_proba(feature_rows[feature_columns])[:, 1]
    strong_prob = strong_model.predict_proba(feature_rows[feature_columns])[:, 1]
    side_scores = pd.DataFrame(
        {
            "instrument": feature_rows["instrument"],
            "buyability_risk": buyability_risk,
            "strong_prob": strong_prob,
        }
    )
    return score_candidates_v3(candidates, side_scores, alpha=alpha, beta=beta, gamma=gamma)
```

- [x] **Step 5: Wire env parsing in `main()`**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`, read these env values after existing short-hold env parsing:

```python
scoring_mode = env.get("SHORT_HOLD_SCORING_MODE", "v2").strip().lower()
v3_side_model_dir = Path(env.get("SHORT_HOLD_V3_SIDE_MODEL_DIR", str(ROOT / "model_packages" / "short_hold_v3_side_models")))
v3_alpha = float(env.get("SHORT_HOLD_V3_ALPHA", "1.0"))
v3_beta = float(env.get("SHORT_HOLD_V3_BETA", "1.0"))
v3_gamma = float(env.get("SHORT_HOLD_V3_GAMMA", "0.25"))
if scoring_mode not in {"v2", "v3"}:
    raise ValueError(f"SHORT_HOLD_SCORING_MODE must be v2 or v3, got {scoring_mode}")
```

- [x] **Step 6: Apply v3 scoring before primary/backup selection**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`, immediately after filter/permission processing and before sorting candidates for primary/backup, add:

```python
if scoring_mode == "v3":
    from bin.short_hold_v3_features import load_qlib_ohlcv

    feature_start_date = pd.Timestamp(signal_date) - pd.Timedelta(days=45)
    ohlcv = load_qlib_ohlcv(candidates["instrument"].tolist(), feature_start_date, signal_date)
    candidates = apply_v3_scoring(
        candidates,
        ohlcv,
        pd.Timestamp(signal_date),
        v3_side_model_dir,
        alpha=v3_alpha,
        beta=v3_beta,
        gamma=v3_gamma,
    )
else:
    candidates["return_score"] = candidates["score"]
    candidates["final_score"] = candidates["score"]
    candidates["score_source"] = "v2"
```

- [x] **Step 7: Replace candidate sorting with helper**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`, replace sorting expressions that use `sort_values("score", ascending=False)` for candidate selection with:

```python
sort_candidates_for_orders(candidates) if sort_candidates_for_orders else candidates.sort_values("score", ascending=False)
```

- [x] **Step 8: Include v3 columns in outputs**

In every order/review row dictionary written by `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/short_hold_generate_orders.py`, include these keys when present:

```python
"return_score": row.get("return_score", row.get("score")),
"buyability_risk": row.get("buyability_risk"),
"strong_prob": row.get("strong_prob"),
"final_score": row.get("final_score", row.get("score")),
"score_source": row.get("score_source", scoring_mode),
```

- [x] **Step 9: Pass env through the shell wrapper**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/prepare_short_hold_orders.sh`, no command flag is required. Add this log line before running `short_hold_generate_orders.py`:

```bash
echo "Short-hold scoring mode: ${SHORT_HOLD_SCORING_MODE:-v2}"
```

- [x] **Step 10: Run syntax checks**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m py_compile bin/short_hold_generate_orders.py bin/short_hold_v3_features.py bin/short_hold_v3_scoring.py
bash -n bin/prepare_short_hold_orders.sh
```

Expected: no output from `py_compile`, no output from `bash -n`, exit code `0`.

- [ ] **Step 11: Commit generator integration**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI
git add local_trade_ops/bin/short_hold_generate_orders.py local_trade_ops/bin/prepare_short_hold_orders.sh local_trade_ops/config/env.local.example
git commit -m "feat: wire short hold v3 scoring into order generation"
```

---

### Task 5: Update Short-Hold UI To Explain V3 Scores

**Files:**
- Modify: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/service/server.ts`

- [x] **Step 1: Add v3 columns to the order table**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/service/server.ts`, update the short-hold order table column list to include:

```ts
const shortHoldScoreColumns = [
  "return_score",
  "buyability_risk",
  "strong_prob",
  "final_score",
  "score_source",
];
```

Render each column only when at least one row has a non-empty value:

```ts
function hasColumn(rows: Record<string, string>[], column: string): boolean {
  return rows.some((row) => String(row[column] ?? "").trim() !== "");
}
```

- [x] **Step 2: Add display labels**

In `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/service/server.ts`, add labels near the existing table heading:

```ts
const shortHoldColumnLabels: Record<string, string> = {
  return_score: "收益分",
  buyability_risk: "买入风险",
  strong_prob: "强势概率",
  final_score: "最终分",
  score_source: "分数来源",
};
```

- [x] **Step 3: Add concise score explanation text**

In the short-hold instruction card in `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/service/server.ts`, update the text to:

```html
<p>
  分数说明：收益分来自短持模型；买入风险越高表示越可能高开、涨停或难成交；强势概率越高表示 T+1 买入到 T+2 卖出的强收益概率越高；最终分用于排序。
</p>
```

- [x] **Step 4: Run TypeScript syntax check**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
node --check service/server.ts
```

Expected: no syntax errors.

- [ ] **Step 5: Manually verify page renders**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
node service/server.ts
```

Open:

```text
http://127.0.0.1:8787/short-hold
```

Expected:

```text
The short-hold page loads, existing v2 rows still render, and v3 score columns appear only when the CSV contains those fields.
```

- [ ] **Step 6: Commit UI update**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI
git add local_trade_ops/service/server.ts
git commit -m "feat: show short hold v3 score explanations"
```

---

### Task 6: Add V2 Versus V3 Backtest CLI

**Files:**
- Create: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/backtest_short_hold_v3_score.py`

- [x] **Step 1: Add backtest CLI skeleton**

Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/bin/backtest_short_hold_v3_score.py` with this command interface:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare short-hold v2 and v3 candidate ordering.")
    parser.add_argument("--v2-daily", required=True, help="Existing v2 daily result CSV.")
    parser.add_argument("--v3-daily", required=True, help="V3 daily result CSV.")
    parser.add_argument("--output", required=True, help="Comparison CSV output path.")
    return parser.parse_args()


def summarize(df: pd.DataFrame, label: str) -> dict[str, object]:
    returns = pd.to_numeric(df["daily_return"], errors="coerce").fillna(0.0)
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return {
        "label": label,
        "days": int(len(df)),
        "cum_return": float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
        "avg_daily_return": float(returns.mean()) if len(returns) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
    }


def main() -> int:
    args = parse_args()
    v2 = pd.read_csv(args.v2_daily)
    v3 = pd.read_csv(args.v3_daily)
    result = pd.DataFrame([summarize(v2, "v2"), summarize(v3, "v3")])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Run CLI syntax check**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m py_compile bin/backtest_short_hold_v3_score.py
```

Expected: no output and exit code `0`.

- [x] **Step 3: Run sample comparison**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
mkdir -p tmp
cat > tmp/v2_daily.csv <<'CSV'
date,daily_return
2026-01-05,0.01
2026-01-06,-0.02
CSV
cat > tmp/v3_daily.csv <<'CSV'
date,daily_return
2026-01-05,0.02
2026-01-06,-0.01
CSV
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python bin/backtest_short_hold_v3_score.py \
  --v2-daily tmp/v2_daily.csv \
  --v3-daily tmp/v3_daily.csv \
  --output tmp/short_hold_v3_compare.csv
```

Expected output contains rows labeled `v2` and `v3`, and writes `tmp/short_hold_v3_compare.csv`.

- [ ] **Step 4: Commit backtest CLI**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI
git add local_trade_ops/bin/backtest_short_hold_v3_score.py
git commit -m "feat: add short hold v3 comparison report"
```

---

### Task 7: Add Operator Documentation

**Files:**
- Create: `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/docs/short_hold_v3.md`

- [x] **Step 1: Create operator documentation**

Create `/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/docs/short_hold_v3.md` with:

```markdown
# Short Hold V3

Short Hold V3 keeps the existing short-hold return model and adds an optional execution-aware scoring layer.

## Modes

- `SHORT_HOLD_SCORING_MODE=v2`: current behavior. Sort candidates by the model return score.
- `SHORT_HOLD_SCORING_MODE=v3`: sort candidates by `final_score`.

## Score Columns

- `return_score`: raw fused score from the short-hold return model.
- `buyability_risk`: probability-like score from 0 to 1; higher means the stock is more likely to be hard to buy at T+1 open.
- `strong_prob`: probability-like score from 0 to 1; higher means the stock is more likely to have strong T+1 open to T+2 close performance.
- `final_score`: normalized return score plus strong probability minus buyability and liquidity risk.
- `score_source`: `v2` or `v3`.

## Safe Default

The safe default remains:

```bash
SHORT_HOLD_SCORING_MODE=v2
```

Switch to v3 only after a backtest and at least one observation run:

```bash
SHORT_HOLD_SCORING_MODE=v3
SHORT_HOLD_V3_SIDE_MODEL_DIR=/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/model_packages/short_hold_v3_side_models
SHORT_HOLD_V3_ALPHA=1.0
SHORT_HOLD_V3_BETA=1.0
SHORT_HOLD_V3_GAMMA=0.25
```

## Rollback

Set:

```bash
SHORT_HOLD_SCORING_MODE=v2
```

Restart the local backend and regenerate the short-hold order list.
```

- [ ] **Step 2: Commit docs**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI
git add local_trade_ops/docs/short_hold_v3.md
git commit -m "docs: explain short hold v3 scoring"
```

---

### Task 8: Final Verification

**Files:**
- Verify repository-wide touched files.

- [x] **Step 1: Run Python unit tests**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m unittest \
  tests.test_short_hold_v3_features \
  tests.test_short_hold_v3_scoring
```

Expected: `OK`.

- [x] **Step 2: Run syntax checks**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
/Users/Dylan.Min/Documents/Code/learn/LHAI/.venv/bin/python -m py_compile \
  bin/short_hold_generate_orders.py \
  bin/short_hold_v3_features.py \
  bin/short_hold_v3_scoring.py \
  bin/train_short_hold_v3_side_models.py \
  bin/backtest_short_hold_v3_score.py
bash -n bin/prepare_short_hold_orders.sh
node --check service/server.ts
```

Expected: all commands exit `0`.

- [ ] **Step 3: Run default v2 smoke test**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
SHORT_HOLD_SCORING_MODE=v2 bash bin/prepare_short_hold_orders.sh
```

Expected:

```text
Short-hold scoring mode: v2
Orders:
Ticket:
Fill template:
```

- [ ] **Step 4: Run v3 smoke test with local side artifacts**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops
SHORT_HOLD_SCORING_MODE=v3 \
SHORT_HOLD_V3_SIDE_MODEL_DIR=/Users/Dylan.Min/Documents/Code/learn/LHAI/local_trade_ops/tmp/short_hold_v3_side_models \
bash bin/prepare_short_hold_orders.sh
```

Expected:

```text
Short-hold scoring mode: v3
Orders:
Ticket:
Fill template:
```

The generated short-hold order CSV contains `return_score`, `buyability_risk`, `strong_prob`, `final_score`, and `score_source`.

- [ ] **Step 5: Inspect git diff**

Run:

```bash
cd /Users/Dylan.Min/Documents/Code/learn/LHAI
git status --short
git log --oneline -8
```

Expected: working tree is clean after commits; recent commits include the short-hold v3 feature, scoring, generator, UI, backtest, and docs commits.

---

## Self-Review

- Spec coverage: this plan covers sidecar v3 scoring, T-visible features, buyability risk, strong probability, final score sorting, primary/backup reuse, UI explanation, backtest comparison, docs, and rollback.
- Default safety: `SHORT_HOLD_SCORING_MODE=v2` preserves current short-hold behavior until v3 is explicitly enabled.
- Data leakage guard: inference features use signal-date and prior rows; future entry/exit rows are only used by `build_training_labels` for offline side model training.
- Operational guard: v3 columns are additive in CSV/UI outputs, so existing fill and report flows continue reading old fields.
