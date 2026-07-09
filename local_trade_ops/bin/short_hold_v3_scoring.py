#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd


SIDE_SCORE_COLUMNS: tuple[str, str] = ("buyability_risk", "strong_prob")
SCORING_COLUMNS: tuple[str, ...] = (
    "instrument",
    "return_score",
    "buyability_risk",
    "strong_prob",
    "liquidity_risk",
    "final_score",
    "score_source",
)
LIQUIDITY_AMOUNT_CAP = 50_000_000


def _numeric_series(values: object, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, index=index)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalize_series(values: object) -> pd.Series:
    series = _numeric_series(values)
    if series.empty:
        return pd.Series(dtype="float64", index=series.index)

    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index, dtype="float64")

    return ((series - series.mean()) / std).fillna(0.0).astype(float)


def _liquidity_risk(amount: object) -> pd.Series:
    series = _numeric_series(amount)
    return (1.0 - (series / LIQUIDITY_AMOUNT_CAP).clip(lower=0.0, upper=1.0)).astype(float)


def _frame_from(value: object) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _ensure_instrument_column(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if "instrument" in frame.columns:
        frame["instrument"] = frame["instrument"].astype(str)
        return frame

    if frame.empty:
        frame["instrument"] = pd.Series(dtype="object")
        return frame.reset_index(drop=True)

    if isinstance(frame.index, pd.MultiIndex) and "instrument" in frame.index.names:
        frame = frame.reset_index("instrument")
    elif frame.index.name == "instrument":
        frame = frame.reset_index()
    else:
        raise ValueError(f"{name} must include an instrument column")

    frame["instrument"] = frame["instrument"].astype(str)
    return frame


def _candidate_frame(candidates: object) -> pd.DataFrame:
    frame = _ensure_instrument_column(_frame_from(candidates), "candidates")
    if "return_score" not in frame.columns:
        if "score" in frame.columns:
            frame["return_score"] = _numeric_series(frame["score"], index=frame.index)
        elif frame.empty:
            frame["return_score"] = pd.Series(dtype="float64")
        else:
            raise ValueError("candidates must include return_score or score column")
    else:
        frame["return_score"] = _numeric_series(frame["return_score"], index=frame.index)
    return frame.reset_index(drop=True)


def _side_score_frame(side_scores: object) -> pd.DataFrame:
    frame = _ensure_instrument_column(_frame_from(side_scores), "side_scores")
    for column in SIDE_SCORE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = _numeric_series(frame[column], index=frame.index).fillna(0.0).clip(lower=0.0, upper=1.0)

    return frame[["instrument", *SIDE_SCORE_COLUMNS]].drop_duplicates(subset=["instrument"], keep="first")


def score_candidates_v3(
    candidates: object,
    side_scores: object,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.0,
) -> pd.DataFrame:
    frame = _candidate_frame(candidates)
    frame = frame.drop(columns=[column for column in SIDE_SCORE_COLUMNS if column in frame.columns], errors="ignore")
    side_frame = _side_score_frame(side_scores)
    scored = frame.merge(side_frame, on="instrument", how="left")

    for column in SIDE_SCORE_COLUMNS:
        scored[column] = _numeric_series(scored[column], index=scored.index).fillna(0.0).clip(lower=0.0, upper=1.0)

    if "amount" in scored.columns:
        amount = _numeric_series(scored["amount"], index=scored.index).fillna(float(LIQUIDITY_AMOUNT_CAP))
    else:
        amount = pd.Series(float(LIQUIDITY_AMOUNT_CAP), index=scored.index, dtype="float64")
    scored["liquidity_risk"] = _liquidity_risk(amount)

    scored["final_score"] = (
        _normalize_series(scored["return_score"])
        + float(alpha) * scored["strong_prob"]
        - float(beta) * scored["buyability_risk"]
        - float(gamma) * scored["liquidity_risk"]
    )
    scored["score_source"] = "v3"

    for column in SCORING_COLUMNS:
        if column not in scored.columns:
            scored[column] = pd.Series(dtype="object")

    return scored.reset_index(drop=True)


def sort_candidates_for_orders(candidates: object) -> pd.DataFrame:
    frame = _frame_from(candidates)
    if frame.empty:
        return frame.reset_index(drop=True)

    if "final_score" in frame.columns:
        sort_column = "final_score"
    elif "score" in frame.columns:
        sort_column = "score"
    else:
        raise ValueError("candidates must include final_score or score column")

    return frame.sort_values(
        by=sort_column,
        ascending=False,
        kind="mergesort",
        na_position="last",
        key=lambda values: _numeric_series(values),
    ).reset_index(drop=True)
