from __future__ import annotations

import argparse
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper_trading_daily import after_close, after_open, fuse_three, load_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top5/top8 high-confidence paper trading scenarios.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--pred-a", required=True)
    parser.add_argument("--pred-b", required=True)
    parser.add_argument("--pred-c", required=True)
    parser.add_argument("--weights", default="10,1,1")
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="2026-05-18")
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scenario_args(
    provider_uri: str,
    state_dir: Path,
    report_dir: Path,
    fused_pred: Path,
    capital: float,
    topk: int,
    n_drop: int,
    signal_date: str | None = None,
    execution_date: str | None = None,
) -> Namespace:
    return Namespace(
        provider_uri=provider_uri,
        state_dir=str(state_dir),
        report_dir=str(report_dir),
        capital=capital,
        topk=topk,
        n_drop=n_drop,
        buy_scan_topk=topk,
        lot_size=100,
        reserve_cash_pct=0.02,
        max_position_pct=0.12,
        allow_one_lot_over_target=True,
        fused_pred=str(fused_pred),
        pred_a=None,
        pred_b=None,
        pred_c=None,
        weights="10,1,1",
        signal_date=signal_date,
        execution_date=execution_date,
        orders=None,
        buy_cost_rate=0.0003,
        sell_cost_rate=0.0008,
        min_cost=5.0,
    )


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Re-run with --overwrite or choose another root.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    state_root = Path(args.state_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    prepare_dir(state_root, args.overwrite)
    prepare_dir(report_root, args.overwrite)

    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) != 3:
        raise ValueError("--weights must be like 10,1,1")

    fused = fuse_three(
        load_pred(Path(args.pred_a).expanduser().resolve()),
        load_pred(Path(args.pred_b).expanduser().resolve()),
        load_pred(Path(args.pred_c).expanduser().resolve()),
        weights,  # type: ignore[arg-type]
    )
    fused_path = state_root / "fused_10_1_1.pkl"
    fused.to_pickle(fused_path)

    available = pd.DatetimeIndex(sorted(fused.index.get_level_values("datetime").unique()))
    dates = available[(available >= pd.Timestamp(args.start_date)) & (available <= pd.Timestamp(args.end_date))]
    if len(dates) < 2:
        raise ValueError("Need at least two prediction dates to replay.")

    scenarios = [
        ("10w_top5_drop1", 100000.0, 5, 1),
        ("20w_top5_drop1", 200000.0, 5, 1),
        ("10w_top8_drop1", 100000.0, 8, 1),
        ("20w_top8_drop1", 200000.0, 8, 1),
    ]

    for label, capital, topk, n_drop in scenarios:
        state_dir = state_root / label
        report_dir = report_root / label
        state_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        print(f"===== RUN {label} =====", flush=True)

        for i, signal_date in enumerate(dates):
            execution_date = dates[i + 1] if i + 1 < len(dates) else None
            close_args = scenario_args(
                args.provider_uri,
                state_dir,
                report_dir,
                fused_path,
                capital,
                topk,
                n_drop,
                signal_date=str(signal_date.date()),
                execution_date=str(execution_date.date()) if execution_date is not None else None,
            )
            after_close(close_args)
            if execution_date is None:
                continue
            open_args = scenario_args(
                args.provider_uri,
                state_dir,
                report_dir,
                fused_path,
                capital,
                topk,
                n_drop,
                execution_date=str(execution_date.date()),
            )
            after_open(open_args)

    print(f"Wrote states: {state_root}")
    print(f"Wrote reports: {report_root}")


if __name__ == "__main__":
    main()
