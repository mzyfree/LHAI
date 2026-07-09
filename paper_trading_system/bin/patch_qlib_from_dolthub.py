#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_API = "https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master"
FIELDS = ["open", "high", "low", "close", "vwap", "change", "adjclose", "factor", "volume", "amount"]
INDEX_INSTRUMENTS = ["csi300", "csi500", "csi800", "csi1000", "csiall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch a Qlib cn_data directory with fresh DoltHub EOD rows.")
    parser.add_argument("--qlib-dir", required=True, help="Extracted Qlib cn_data directory to patch in place.")
    parser.add_argument("--work-dir", required=True, help="Directory for temporary DoltHub patch CSVs.")
    parser.add_argument("--api-url", default=DEFAULT_API)
    parser.add_argument("--qlib-src", default=os.environ.get("QLIB_SRC", ""))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--page-size", type=int, default=10000)
    parser.add_argument("--max-patch-days", type=int, default=5)
    parser.add_argument(
        "--normalize-instruments-only",
        action="store_true",
        help="Only extend index instrument end dates to the latest Qlib calendar date.",
    )
    return parser.parse_args()


def dolt_query(api_url: str, sql: str, retries: int = 3) -> dict[str, Any]:
    url = f"{api_url}?{urllib.parse.urlencode({'q': sql})}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("query_execution_status") not in {"Success", "RowLimit"}:
                raise RuntimeError(payload.get("query_execution_message") or payload)
            return payload
        except Exception as exc:  # noqa: BLE001 - preserve retry context for CLI output.
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"DoltHub query failed after {retries} attempts: {last_error}") from last_error


def qlib_calendar(qlib_dir: Path) -> list[str]:
    path = qlib_dir / "calendars" / "day.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Qlib day calendar: {path}")
    dates = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not dates:
        raise ValueError(f"Empty Qlib day calendar: {path}")
    return dates


def latest_dolt_date(api_url: str) -> str:
    payload = dolt_query(
        api_url,
        "SELECT tradedate FROM final_a_stock_eod_price ORDER BY tradedate DESC LIMIT 1",
    )
    return payload["rows"][0]["tradedate"]


def missing_dolt_dates(api_url: str, qlib_latest: str, dolt_latest: str, max_days: int) -> list[str]:
    sql = (
        "SELECT tradedate, COUNT(*) AS n "
        "FROM final_a_stock_eod_price "
        f"WHERE tradedate > DATE('{qlib_latest}') AND tradedate <= DATE('{dolt_latest}') "
        "GROUP BY tradedate ORDER BY tradedate"
    )
    rows = dolt_query(api_url, sql)["rows"]
    dates = [row["tradedate"] for row in rows]
    if len(dates) > max_days:
        raise RuntimeError(
            f"DoltHub has {len(dates)} missing dates ({dates[0]} -> {dates[-1]}), "
            f"exceeding --max-patch-days={max_days}. Use full Qlib rebuild instead."
        )
    return dates


def fetch_eod_rows(api_url: str, date: str, page_size: int) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        sql = (
            "SELECT tradedate, symbol, open, high, low, close, adjclose, volume, amount "
            "FROM final_a_stock_eod_price "
            f"WHERE tradedate = DATE('{date}') "
            f"ORDER BY symbol LIMIT {page_size} OFFSET {offset}"
        )
        payload = dolt_query(api_url, sql)
        rows = payload["rows"]
        all_rows.extend(rows)
        print(f"Fetched DoltHub rows for {date}: offset={offset} rows={len(rows)} status={payload.get('query_execution_status')}")
        if not rows or (payload.get("query_execution_status") != "RowLimit" and len(rows) < page_size):
            break
        offset += len(rows)
    if not all_rows:
        raise RuntimeError(f"No DoltHub EOD rows found for {date}")
    df = pd.DataFrame(all_rows)
    for col in ["open", "high", "low", "close", "adjclose", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["tradedate"]).dt.strftime("%Y-%m-%d")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def read_bin_value(path: Path, calendar_index: int) -> float | None:
    if not path.exists():
        return None
    data = np.fromfile(path, dtype="<f")
    if len(data) < 2:
        return None
    start = int(data[0])
    pos = 1 + calendar_index - start
    if pos < 1 or pos >= len(data):
        return None
    value = float(data[pos])
    if np.isnan(value):
        return None
    return value


def read_last_bin_value(path: Path, calendar_index: int) -> tuple[float | None, int | None]:
    if not path.exists():
        return None, None
    data = np.fromfile(path, dtype="<f")
    if len(data) < 2:
        return None, None
    start = int(data[0])
    end_pos = min(1 + calendar_index - start, len(data) - 1)
    if end_pos < 1:
        return None, None
    values = data[1 : end_pos + 1]
    valid = np.where(~np.isnan(values))[0]
    if len(valid) == 0:
        return None, None
    pos = int(valid[-1])
    return float(values[pos]), start + pos


def previous_state(qlib_dir: Path, calendar_index: int) -> dict[str, dict[str, float]]:
    state: dict[str, dict[str, float]] = {}
    features_dir = qlib_dir / "features"
    for inst_dir in features_dir.iterdir():
        if not inst_dir.is_dir():
            continue
        symbol = inst_dir.name.upper()
        close_adj, close_idx = read_last_bin_value(inst_dir / "close.day.bin", calendar_index)
        factor, _ = read_last_bin_value(inst_dir / "factor.day.bin", calendar_index)
        adjclose, adjclose_idx = read_last_bin_value(inst_dir / "adjclose.day.bin", calendar_index)
        if factor is None or factor <= 0:
            continue
        row = {
            "factor": factor,
        }
        if close_adj is not None and adjclose is not None and adjclose > 0 and close_idx == adjclose_idx:
            raw_close = close_adj / factor
            scale = close_adj / adjclose
            if raw_close > 0 and scale > 0:
                row["raw_close"] = raw_close
                row["scale"] = scale
        state[symbol] = row
    return state


def transform_rows(raw: pd.DataFrame, state: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = raw.copy()
    patch_rows: list[dict[str, Any]] = []
    for row in out.itertuples(index=False):
        symbol = str(row.symbol).upper()
        raw_close = float(row.close)
        adjclose = float(row.adjclose)
        volume = float(row.volume)
        amount = float(row.amount)
        prior = state.get(symbol)
        if prior and raw_close > 0 and adjclose > 0 and prior.get("scale") and prior.get("raw_close"):
            factor = adjclose * prior["scale"] / raw_close
            prev_raw_close = prior["raw_close"]
        elif prior and raw_close > 0 and prior.get("factor"):
            # For suspended/recently listed instruments, Qlib release keeps the latest valid factor.
            factor = prior["factor"]
            prev_raw_close = prior.get("raw_close")
        else:
            # Match Qlib dump_bin's first-valid-day convention: adjusted close starts at 1.
            factor = 1.0 / raw_close if raw_close > 0 else 1.0
            prev_raw_close = None

        vwap_raw = amount / volume * 10.0 if volume > 0 else np.nan
        change = raw_close / prev_raw_close - 1.0 if prev_raw_close and prev_raw_close > 0 else np.nan
        patch_rows.append(
            {
                "date": row.date,
                "symbol": symbol,
                "open": float(row.open) * factor,
                "high": float(row.high) * factor,
                "low": float(row.low) * factor,
                "close": raw_close * factor,
                "vwap": vwap_raw * factor if not np.isnan(vwap_raw) else np.nan,
                "change": change,
                "adjclose": adjclose,
                "factor": factor,
                "volume": volume / factor if factor > 0 else np.nan,
                "amount": amount,
            }
        )
        state[symbol] = {
            "raw_close": raw_close,
            "factor": factor,
            "scale": raw_close * factor / adjclose if adjclose > 0 else prior["scale"] if prior else 1.0,
        }
    return pd.DataFrame(patch_rows)


def dump_update(csv_path: Path, qlib_dir: Path, qlib_src: str, python_bin: str) -> None:
    script = Path(qlib_src).expanduser() / "scripts" / "dump_bin.py" if qlib_src else Path("qlib/scripts/dump_bin.py")
    if not script.exists():
        raise FileNotFoundError(f"Cannot find qlib dump_bin.py: {script}")
    env = os.environ.copy()
    if qlib_src:
        env["PYTHONPATH"] = f"{Path(qlib_src).expanduser()}{os.pathsep}{env.get('PYTHONPATH', '')}"
    cmd = [
        python_bin,
        str(script),
        "dump_update",
        "--data_path",
        str(csv_path),
        "--qlib_dir",
        str(qlib_dir),
        "--freq",
        "day",
        "--date_field_name",
        "date",
        "--symbol_field_name",
        "symbol",
        "--include_fields",
        ",".join(FIELDS),
        "--file_suffix",
        ".csv",
        "--max_workers",
        "1",
    ]
    subprocess.run(cmd, check=True, env=env)


def extend_index_instruments(qlib_dir: Path, old_latest: str, new_latest: str) -> None:
    instruments_dir = qlib_dir / "instruments"
    for name in INDEX_INSTRUMENTS:
        path = instruments_dir / f"{name}.txt"
        if not path.exists():
            continue
        changed = False
        out_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == old_latest:
                parts[2] = new_latest
                changed = True
                out_lines.append("\t".join(parts))
            else:
                out_lines.append(line)
        if changed:
            path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            print(f"Extended instrument calendar: {path.name} {old_latest} -> {new_latest}")


def normalize_index_instruments_to_latest(qlib_dir: Path, latest: str) -> None:
    """Make index universe files usable on the latest trading day.

    Some upstream Qlib releases can have calendars/features through the latest
    trading day while index instrument end dates stop on a weekend or the prior
    trading day. Qlib treats those universe ranges as exclusive for the missing
    day, so inference silently stops before the latest calendar date.
    """
    instruments_dir = qlib_dir / "instruments"
    for name in INDEX_INSTRUMENTS:
        path = instruments_dir / f"{name}.txt"
        if not path.exists():
            continue
        changed = False
        out_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] < latest:
                parts[2] = latest
                changed = True
                out_lines.append("\t".join(parts))
            else:
                out_lines.append(line)
        if changed:
            path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            print(f"Normalized instrument calendar: {path.name} -> {latest}")


def main() -> int:
    args = parse_args()
    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    calendar = qlib_calendar(qlib_dir)
    qlib_latest = calendar[-1]
    if args.normalize_instruments_only:
        normalize_index_instruments_to_latest(qlib_dir, qlib_latest)
        return 0

    dolt_latest = latest_dolt_date(args.api_url)
    print(f"DoltHub latest date: {dolt_latest}")
    print(f"Qlib latest date before patch: {qlib_latest}")
    if dolt_latest <= qlib_latest:
        print("Qlib release is already up to date with DoltHub. No patch needed.")
        return 0

    dates = missing_dolt_dates(args.api_url, qlib_latest, dolt_latest, args.max_patch_days)
    if not dates:
        print("No missing DoltHub dates found. No patch needed.")
        return 0
    print(f"Patching missing dates from DoltHub: {', '.join(dates)}")

    state = previous_state(qlib_dir, len(calendar) - 1)
    patch_frames = []
    for date in dates:
        raw = fetch_eod_rows(args.api_url, date, args.page_size)
        patch = transform_rows(raw, state)
        patch_path = work_dir / f"dolthub_patch_{date}.csv"
        patch.to_csv(patch_path, index=False)
        print(f"Wrote DoltHub patch CSV: {patch_path} rows={len(patch)}")
        patch_frames.append(patch)

    merged = pd.concat(patch_frames, ignore_index=True)
    merged_path = work_dir / f"dolthub_patch_{dates[0]}_to_{dates[-1]}.csv"
    merged.to_csv(merged_path, index=False)
    dump_update(merged_path, qlib_dir, args.qlib_src, args.python_bin)
    extend_index_instruments(qlib_dir, qlib_latest, dates[-1])

    patched_calendar = qlib_calendar(qlib_dir)
    print(f"Qlib latest date after patch: {patched_calendar[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
