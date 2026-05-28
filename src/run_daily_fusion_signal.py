from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from run_prediction_fusion import fuse_scores, load_pred


def choose_trade_date(pred: pd.DataFrame, trade_date: str | None) -> pd.Timestamp:
    available = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    if available.empty:
        raise ValueError("Prediction frame is empty.")
    if trade_date is None:
        return pd.Timestamp(available[-1])
    target = pd.Timestamp(trade_date)
    if target not in available:
        raise ValueError(
            f"Trade date {target.date()} not found. Available range: {available[0].date()} -> {available[-1].date()}"
        )
    return target


def build_day_table(pred: pd.DataFrame, trade_date: pd.Timestamp, topk: int) -> pd.DataFrame:
    day_df = pred.xs(trade_date, level="datetime").reset_index()
    day_df = day_df.sort_values("score", ascending=False).reset_index(drop=True)
    day_df["rank"] = day_df.index + 1
    return day_df.loc[:, ["rank", "instrument", "score"]].head(topk)


def turnover_summary(pred: pd.DataFrame, trade_date: pd.Timestamp, topk: int) -> tuple[list[str], list[str], pd.Timestamp | None]:
    available = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    positions = available.get_indexer([trade_date])[0]
    if positions <= 0:
        return [], [], None

    prev_date = pd.Timestamp(available[positions - 1])
    today = set(build_day_table(pred, trade_date, topk)["instrument"])
    prev = set(build_day_table(pred, prev_date, topk)["instrument"])
    added = sorted(today - prev)
    removed = sorted(prev - today)
    return added, removed, prev_date


def markdown_list(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a daily top-k signal table from fused predictions.")
    parser.add_argument("--pred-a", default=None, help="First pred.pkl path.")
    parser.add_argument("--pred-b", default=None, help="Second pred.pkl path.")
    parser.add_argument("--fused-pred", default=None, help="Precomputed fused pred.pkl path.")
    parser.add_argument(
        "--method",
        default="rank_mean",
        choices=["mean", "rank_mean", "zscore_mean"],
        help="Fusion method when using pred-a + pred-b.",
    )
    parser.add_argument("--weight-a", type=float, default=1.0)
    parser.add_argument("--weight-b", type=float, default=1.0)
    parser.add_argument("--trade-date", default=None, help="Signal date, default latest available date.")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--output-md", default=None, help="Optional markdown report path.")
    args = parser.parse_args()

    if args.fused_pred:
        pred = load_pred(Path(args.fused_pred).expanduser().resolve())
        source_desc = f"fused_pred={Path(args.fused_pred).expanduser().resolve()}"
    else:
        if not args.pred_a or not args.pred_b:
            raise ValueError("Provide either --fused-pred or both --pred-a and --pred-b.")
        pred_a = load_pred(Path(args.pred_a).expanduser().resolve())
        pred_b = load_pred(Path(args.pred_b).expanduser().resolve())
        pred = fuse_scores(pred_a, pred_b, args.method, args.weight_a, args.weight_b)
        source_desc = (
            f"pred_a={Path(args.pred_a).expanduser().resolve()}, "
            f"pred_b={Path(args.pred_b).expanduser().resolve()}, "
            f"method={args.method}, weights={args.weight_a}:{args.weight_b}"
        )

    trade_date = choose_trade_date(pred, args.trade_date)
    table = build_day_table(pred, trade_date, args.topk)
    added, removed, prev_date = turnover_summary(pred, trade_date, args.topk)

    print("Daily fusion signal:")
    print(f"- source: {source_desc}")
    print(f"- trade_date: {trade_date.date()}")
    print(f"- topk: {args.topk}")
    if prev_date is not None:
        print(f"- previous_trade_date: {prev_date.date()}")
        print(f"- added: {markdown_list(added)}")
        print(f"- removed: {markdown_list(removed)}")
    print()
    print(table.to_string(index=False))

    if args.output_csv:
        csv_path = Path(args.output_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        print(f"Wrote CSV to {csv_path}")

    if args.output_md:
        md_path = Path(args.output_md).expanduser().resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        prev_text = ""
        if prev_date is not None:
            prev_text = f"""
## Turnover vs Previous Trade Date

- Previous trade date: `{prev_date.date()}`
- Added: {markdown_list(added)}
- Removed: {markdown_list(removed)}
"""
        report = f"""# Daily Fusion Signal

## Setup

- Source: `{source_desc}`
- Trade date: `{trade_date.date()}`
- TopK: `{args.topk}`

{prev_text}

## TopK Table

{table.to_markdown(index=False)}
"""
        md_path.write_text(report, encoding="utf-8")
        print(f"Wrote markdown to {md_path}")


if __name__ == "__main__":
    main()
