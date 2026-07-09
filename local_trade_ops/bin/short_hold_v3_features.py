#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


LABEL_COLUMNS: tuple[str, ...] = (
    "signal_date",
    "entry_date",
    "exit_date",
    "entry_open",
    "exit_close",
    "entry_gap",
    "realized_return",
    "open_gt_5pct",
    "open_limit_up",
    "buyability_bad",
    "strong_label",
)


def stock_limit_up_pct(instrument: str) -> float:
    code = str(instrument).upper()
    if code.startswith("SZ300") or code.startswith("SH688"):
        return 0.20
    return 0.10


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _safe_ratio(numerator: object, denominator: object) -> float:
    top = _safe_float(numerator)
    bottom = _safe_float(denominator)
    if not np.isfinite(top) or not np.isfinite(bottom) or bottom == 0.0:
        return float("nan")
    return top / bottom - 1.0


def _as_ohlcv_index(ohlcv: pd.DataFrame) -> pd.DataFrame:
    frame = ohlcv.copy()
    if {"datetime", "instrument"}.issubset(frame.columns):
        frame = frame.set_index(["datetime", "instrument"])
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels < 2:
        raise ValueError("ohlcv must be indexed by datetime and instrument")

    names = list(frame.index.names)
    if "datetime" not in names or "instrument" not in names:
        frame.index = frame.index.set_names(["datetime", "instrument"] + names[2:])

    dates = pd.to_datetime(frame.index.get_level_values("datetime"))
    instruments = frame.index.get_level_values("instrument").astype(str)
    frame.index = pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"])
    if frame.index.duplicated().any():
        duplicate = frame.index[frame.index.duplicated()][0]
        raise ValueError(f"duplicate OHLCV rows for {duplicate[1]} on {duplicate[0].date()}")
    return frame.sort_index()


def _as_candidate_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    if "instrument" not in frame.columns:
        if frame.empty:
            frame["instrument"] = pd.Series(dtype="object")
            return frame.reset_index(drop=True)
        if isinstance(frame.index, pd.MultiIndex) and "instrument" in frame.index.names:
            frame = frame.reset_index("instrument")
        elif frame.index.name == "instrument":
            frame = frame.reset_index()
        else:
            raise KeyError("candidates must include an instrument column")
    frame["instrument"] = frame["instrument"].astype(str)
    return frame.reset_index(drop=True)


def _history_through(ohlcv: pd.DataFrame, instrument: str, end_date: pd.Timestamp, window: int) -> pd.DataFrame:
    dates = ohlcv.index.get_level_values("datetime")
    instruments = ohlcv.index.get_level_values("instrument")
    mask = (instruments == instrument) & (dates <= pd.Timestamp(end_date))
    return ohlcv.loc[mask].sort_index().tail(window)


def _row_at(ohlcv: pd.DataFrame, instrument: str, date: pd.Timestamp) -> pd.Series:
    key = (pd.Timestamp(date), str(instrument))
    if key not in ohlcv.index:
        raise KeyError(f"missing OHLCV row for {instrument} on {pd.Timestamp(date).date()}")
    row = ohlcv.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _candidate_score(candidate: dict[str, object], primary: str, fallback: str) -> float:
    primary_value = _safe_float(candidate.get(primary))
    if np.isfinite(primary_value):
        return primary_value
    return _safe_float(candidate.get(fallback))


def _unique_columns(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        result.append(column)
    return result


def _feature_columns(candidate_columns: Iterable[str] = ()) -> list[str]:
    return _unique_columns(["instrument", *V3FeatureColumns().numeric, *candidate_columns])


def build_inference_features(candidates: pd.DataFrame, ohlcv: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    candidate_frame = _as_candidate_frame(candidates)
    columns = _feature_columns(candidate_frame.columns)
    if candidate_frame.empty:
        return pd.DataFrame(columns=columns)

    ohlcv_frame = _as_ohlcv_index(ohlcv)
    signal_ts = pd.Timestamp(signal_date)
    rows: list[dict[str, object]] = []

    for candidate in candidate_frame.to_dict("records"):
        instrument = str(candidate["instrument"])
        hist = _history_through(ohlcv_frame, instrument, signal_ts, 20)
        if hist.empty:
            continue

        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else None
        first_in_3d = hist.tail(3).iloc[0]
        close = _safe_float(last.get("close"))
        high = _safe_float(last.get("high"))
        low = _safe_float(last.get("low"))
        prev_close = _safe_float(prev.get("close")) if prev is not None else float("nan")
        limit_pct = stock_limit_up_pct(instrument)
        theoretical_limit = prev_close * (1.0 + limit_pct) if np.isfinite(prev_close) else float("nan")

        row = dict(candidate)
        row.update(
            {
                "instrument": instrument,
                "return_score": _candidate_score(candidate, "return_score", "score"),
                "model_rank": _candidate_score(candidate, "model_rank", "rank"),
                "close": close,
                "amount": _safe_float(last.get("amount")),
                "volume": _safe_float(last.get("volume")),
                "ret_1d": _safe_ratio(close, prev_close),
                "ret_3d": _safe_ratio(close, first_in_3d.get("close")),
                "amplitude_1d": (high - low) / close if np.isfinite(high) and np.isfinite(low) and close else float("nan"),
                "amount_mean_3d": _safe_float(pd.to_numeric(hist.tail(3).get("amount"), errors="coerce").mean()),
                "distance_to_limit_up": _safe_ratio(theoretical_limit, close),
                "limit_up_pct": limit_pct,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def build_training_labels(
    candidates: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    strong_return_threshold: float = 0.06,
) -> pd.DataFrame:
    candidate_frame = _as_candidate_frame(candidates)
    if candidate_frame.empty:
        return pd.DataFrame(columns=_unique_columns([*_feature_columns(candidate_frame.columns), *LABEL_COLUMNS]))

    ohlcv_frame = _as_ohlcv_index(ohlcv)
    signal_ts = pd.Timestamp(signal_date)
    features = build_inference_features(candidate_frame, ohlcv_frame, signal_ts)
    columns = _unique_columns([*features.columns, *LABEL_COLUMNS])
    rows: list[dict[str, object]] = []

    for feature in features.to_dict("records"):
        instrument = str(feature["instrument"])
        signal_close = _safe_float(feature.get("close"))
        entry = _row_at(ohlcv_frame, instrument, pd.Timestamp(entry_date))
        exit_row = _row_at(ohlcv_frame, instrument, pd.Timestamp(exit_date))

        entry_open = _safe_float(entry.get("open"))
        exit_close = _safe_float(exit_row.get("close"))
        entry_gap = _safe_ratio(entry_open, signal_close)
        realized_return = _safe_ratio(exit_close, entry_open)
        limit_pct = stock_limit_up_pct(instrument)
        open_gt_5pct = bool(np.isfinite(entry_gap) and entry_gap > 0.05)
        open_limit_up = bool(np.isfinite(entry_gap) and entry_gap >= limit_pct * 0.999)
        buyability_bad = bool(open_gt_5pct or open_limit_up or not np.isfinite(entry_gap))

        row = dict(feature)
        row.update(
            {
                "signal_date": signal_ts,
                "entry_date": pd.Timestamp(entry_date),
                "exit_date": pd.Timestamp(exit_date),
                "entry_open": entry_open,
                "exit_close": exit_close,
                "entry_gap": entry_gap,
                "realized_return": realized_return,
                "open_gt_5pct": int(open_gt_5pct),
                "open_limit_up": int(open_limit_up),
                "buyability_bad": int(buyability_bad),
                "strong_label": int(np.isfinite(realized_return) and realized_return >= strong_return_threshold),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def load_qlib_ohlcv(instruments: Iterable[str], start_date: object, end_date: object) -> pd.DataFrame:
    from qlib.data import D

    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
    data = D.features(
        list(instruments),
        fields,
        start_time=pd.Timestamp(start_date),
        end_time=pd.Timestamp(end_date),
        freq="day",
    )
    frame = data.rename(
        columns={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$factor": "factor",
        }
    )
    frame = _as_ohlcv_index(frame)
    frame["amount"] = frame["close"] * frame["volume"] * frame["factor"]
    return frame
