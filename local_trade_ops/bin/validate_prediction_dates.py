#!/usr/bin/env python3
"""Validate that local prediction pkls are aligned with the latest Qlib data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def latest_calendar_date(provider_uri: Path) -> str:
    calendar_path = provider_uri / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise SystemExit(f"Missing Qlib day calendar: {calendar_path}")

    dates = [line.strip() for line in calendar_path.read_text().splitlines() if line.strip()]
    if not dates:
        raise SystemExit(f"Empty Qlib day calendar: {calendar_path}")
    return dates[-1]


def prediction_range(pred_path: Path) -> tuple[str, str, int]:
    if not pred_path.exists() or pred_path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty prediction pkl: {pred_path}")

    df = pd.read_pickle(pred_path)
    if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
        raise SystemExit(f"Prediction pkl has no MultiIndex level named datetime: {pred_path}")

    dates = pd.DatetimeIndex(sorted(df.index.get_level_values("datetime").unique()))
    if dates.empty:
        raise SystemExit(f"Prediction pkl has no dates: {pred_path}")
    return str(dates[0].date()), str(dates[-1].date()), len(dates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--pred", action="append", required=True)
    parser.add_argument("--allow-stale", action="store_true")
    args = parser.parse_args()

    provider_uri = Path(args.provider_uri)
    calendar_latest = latest_calendar_date(provider_uri)

    latest_dates: list[str] = []
    print("Prediction date ranges:")
    for pred in args.pred:
        path = Path(pred)
        start, end, n_dates = prediction_range(path)
        latest_dates.append(end)
        print(f"- {path.name}: {start} -> {end} ({n_dates} dates)")

    print(f"Latest Qlib calendar date: {calendar_latest}")

    if len(set(latest_dates)) != 1:
        raise SystemExit(
            "Prediction pkls are not aligned with each other: "
            + ", ".join(f"{Path(p).name}={d}" for p, d in zip(args.pred, latest_dates))
        )

    pred_latest = latest_dates[0]
    if pred_latest != calendar_latest:
        message = (
            f"Stale predictions: latest common prediction date {pred_latest} "
            f"!= latest Qlib calendar date {calendar_latest}. "
            "Run local inference or sync fresh pkls before generating Review."
        )
        if args.allow_stale:
            print(f"WARNING: {message}")
        else:
            raise SystemExit(message)

    print("Prediction/date alignment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
