#!/usr/bin/env python3
"""Analyze staggered two-pool short-hold strategy stability.

The script reads offset=0/1 portfolio queue outputs and explains whether the
observed gap is broad-based or driven by a few lucky dates.  It also runs an
approximate random split test from signal-level capital rows, which is useful
for checking whether the fixed odd/even split is unusually strong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze two-pool offset performance.")
    parser.add_argument("--base-dir", required=True, help="Directory containing two_pool_offset_0/1 outputs.")
    parser.add_argument("--offset0-dir", default="two_pool_offset_0")
    parser.add_argument("--offset1-dir", default="two_pool_offset_1")
    parser.add_argument(
        "--full-capital-daily",
        default="",
        help="Optional full-signal strict_online_intraday_capital_2026_daily.csv for random split test.",
    )
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--entry-lag-days", type=float, default=1.0)
    parser.add_argument("--hold-days", type=float, default=1.0)
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    return float((nav / nav.cummax() - 1.0).min())


def load_offset_daily(path: Path, capital: float, topk: int, entry_lag_days: float, hold_days: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = (df["capital"].astype(float) == capital) & (df["topk"].astype(int) == topk)
    if "entry_lag_days" in df.columns:
        mask &= df["entry_lag_days"].astype(float) == entry_lag_days
    if "hold_days" in df.columns:
        mask &= df["hold_days"].astype(float) == hold_days
    out = df.loc[mask].copy()
    if out.empty:
        raise RuntimeError(f"No rows for capital={capital}, topk={topk} in {path}")
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    out["daily_return"] = out["nav"].astype(float).pct_change().fillna(out["nav"].iloc[0] / capital - 1.0)
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["weekday"] = out["date"].dt.day_name()
    return out


def summarize_nav(df: pd.DataFrame, capital: float, label: str) -> dict[str, float | str]:
    ret = df["daily_return"].astype(float)
    nav = df.set_index("date")["nav"].astype(float)
    std = ret.std()
    return {
        "label": label,
        "days": float(len(df)),
        "last_nav": float(nav.iloc[-1]),
        "cum_return": float(nav.iloc[-1] / capital - 1.0),
        "ann_return": float((1.0 + ret.mean()) ** 252 - 1.0),
        "ir": float(ret.mean() / std * np.sqrt(252)) if std and std > 0 else np.nan,
        "max_drawdown": max_drawdown(nav),
        "win_rate": float((ret > 0).mean()),
        "avg_daily_return": float(ret.mean()),
    }


def month_table(df: pd.DataFrame, capital: float, label: str) -> pd.DataFrame:
    rows = []
    for month, g in df.groupby("month"):
        nav = g.set_index("date")["nav"].astype(float)
        start_nav = float(nav.iloc[0] / (1.0 + g["daily_return"].iloc[0]))
        ret = g["daily_return"].astype(float)
        rows.append(
            {
                "label": label,
                "month": month,
                "days": len(g),
                "month_return": float(nav.iloc[-1] / start_nav - 1.0),
                "win_rate": float((ret > 0).mean()),
                "best_day": float(ret.max()),
                "worst_day": float(ret.min()),
                "max_drawdown": max_drawdown(nav),
            }
        )
    return pd.DataFrame(rows)


def weekday_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for weekday, g in df.groupby("weekday"):
        ret = g["daily_return"].astype(float)
        rows.append(
            {
                "label": label,
                "weekday": weekday,
                "days": len(g),
                "avg_return": float(ret.mean()),
                "win_rate": float((ret > 0).mean()),
                "sum_return": float(ret.sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["weekday"] = pd.Categorical(out["weekday"], categories=order, ordered=True)
        out = out.sort_values(["label", "weekday"])
    return out


def top_contribution_table(df: pd.DataFrame, label: str, n: int = 10) -> pd.DataFrame:
    out = df[["date", "signal_date", "daily_return", "nav", "buy_value", "sell_value", "n_buy", "n_sell", "picks"]].copy()
    out.insert(0, "label", label)
    return out.sort_values("daily_return", ascending=False).head(n)


def concentration(df: pd.DataFrame, label: str) -> dict[str, float | str]:
    ret = df["daily_return"].astype(float)
    positive = ret[ret > 0].sort_values(ascending=False)
    total_positive = float(positive.sum())
    total = float(ret.sum())
    return {
        "label": label,
        "sum_daily_return": total,
        "positive_return_sum": total_positive,
        "top1_positive_share": float(positive.head(1).sum() / total_positive) if total_positive else np.nan,
        "top3_positive_share": float(positive.head(3).sum() / total_positive) if total_positive else np.nan,
        "top5_positive_share": float(positive.head(5).sum() / total_positive) if total_positive else np.nan,
        "top10_positive_share": float(positive.head(10).sum() / total_positive) if total_positive else np.nan,
    }


def combine_two_pools(offset0: pd.DataFrame, offset1: pd.DataFrame, per_pool_capital: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
    s0 = offset0.set_index("date")["nav"].astype(float)
    s1 = offset1.set_index("date")["nav"].astype(float)
    idx = s0.index.union(s1.index).sort_values()
    total = s0.reindex(idx).ffill().fillna(per_pool_capital) + s1.reindex(idx).ffill().fillna(per_pool_capital)
    combo = pd.DataFrame({"date": idx, "nav": total.to_numpy()})
    combo["daily_return"] = combo["nav"].pct_change().fillna(combo["nav"].iloc[0] / (per_pool_capital * 2.0) - 1.0)
    combo["month"] = combo["date"].dt.to_period("M").astype(str)
    combo["weekday"] = combo["date"].dt.day_name()
    summary = summarize_nav(combo, per_pool_capital * 2.0, "combined")
    return combo, summary


def load_full_signal_rows(path: Path, capital: float, topk: int, entry_lag_days: float, hold_days: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = (df["capital"].astype(float) == capital) & (df["topk"].astype(int) == topk)
    if "entry_lag_days" in df.columns:
        mask &= df["entry_lag_days"].astype(float) == entry_lag_days
    if "hold_days" in df.columns:
        mask &= df["hold_days"].astype(float) == hold_days
    out = df.loc[mask].copy()
    if out.empty:
        raise RuntimeError(f"No rows for random split from {path}")
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out = out.sort_values("signal_date").reset_index(drop=True)
    return out


def nav_from_signal_returns(rows: pd.DataFrame, capital: float) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    nav = capital
    values = []
    for _, row in rows.sort_values("trade_date").iterrows():
        nav *= 1.0 + float(row["daily_return"])
        values.append((pd.Timestamp(row["trade_date"]), nav))
    return pd.Series([v for _, v in values], index=pd.to_datetime([d for d, _ in values])).sort_index()


def random_split_test(full_rows: pd.DataFrame, capital: float, observed_cum_return: float, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(full_rows)
    half = n // 2
    results = []
    indices = np.arange(n)

    for i in range(repeats):
        rng.shuffle(indices)
        a_idx = np.sort(indices[:half])
        b_idx = np.sort(indices[half:])
        nav_a = nav_from_signal_returns(full_rows.iloc[a_idx], capital)
        nav_b = nav_from_signal_returns(full_rows.iloc[b_idx], capital)
        idx = nav_a.index.union(nav_b.index).sort_values()
        total_nav = nav_a.reindex(idx).ffill().fillna(capital) + nav_b.reindex(idx).ffill().fillna(capital)
        ret = total_nav.pct_change().fillna(total_nav.iloc[0] / (capital * 2.0) - 1.0)
        results.append(
            {
                "sample": i,
                "cum_return": float(total_nav.iloc[-1] / (capital * 2.0) - 1.0),
                "max_drawdown": max_drawdown(total_nav),
                "ir": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() and ret.std() > 0 else np.nan,
            }
        )

    out = pd.DataFrame(results)
    out.attrs["observed_percentile"] = float((out["cum_return"] <= observed_cum_return).mean())
    return out


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    concentration_df: pd.DataFrame,
    random_df: pd.DataFrame | None,
) -> None:
    def table_block(df: pd.DataFrame) -> str:
        return "```text\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# Two-Pool Offset Diagnostic",
        "",
        f"capital per pool: {args.capital:,.0f}",
        f"topk: {args.topk}",
        f"entry_lag_days: {args.entry_lag_days}",
        f"hold_days: {args.hold_days}",
        "",
        "## Summary",
        "",
        table_block(summary),
        "",
        "## Monthly Return",
        "",
        table_block(monthly),
        "",
        "## Positive Return Concentration",
        "",
        table_block(concentration_df),
        "",
    ]
    if random_df is not None and not random_df.empty:
        q = random_df["cum_return"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        lines.extend(
            [
                "## Approximate Random Split Test",
                "",
                "This test randomly splits signal-level returns into two pools and compounds each pool independently.",
                "It is an approximation for stability diagnostics, not a replacement for full trade replay.",
                "",
                f"observed percentile: {random_df.attrs.get('observed_percentile', np.nan):.3f}",
                "",
                table_block(q.to_frame("cum_return").reset_index().rename(columns={"index": "quantile"})),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "two_pool_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    path0 = base_dir / args.offset0_dir / "strict_online_portfolio_queue_2026_daily.csv"
    path1 = base_dir / args.offset1_dir / "strict_online_portfolio_queue_2026_daily.csv"
    offset0 = load_offset_daily(path0, args.capital, args.topk, args.entry_lag_days, args.hold_days)
    offset1 = load_offset_daily(path1, args.capital, args.topk, args.entry_lag_days, args.hold_days)
    combo, combo_summary = combine_two_pools(offset0, offset1, args.capital)

    summary = pd.DataFrame(
        [
            summarize_nav(offset0, args.capital, "offset_0"),
            summarize_nav(offset1, args.capital, "offset_1"),
            combo_summary,
        ]
    )
    monthly = pd.concat(
        [
            month_table(offset0, args.capital, "offset_0"),
            month_table(offset1, args.capital, "offset_1"),
            month_table(combo, args.capital * 2.0, "combined"),
        ],
        ignore_index=True,
    )
    weekdays = pd.concat([weekday_table(offset0, "offset_0"), weekday_table(offset1, "offset_1")], ignore_index=True)
    top_days = pd.concat(
        [top_contribution_table(offset0, "offset_0"), top_contribution_table(offset1, "offset_1")],
        ignore_index=True,
    )
    concentration_df = pd.DataFrame([concentration(offset0, "offset_0"), concentration(offset1, "offset_1")])

    random_df = None
    if args.full_capital_daily:
        full_rows = load_full_signal_rows(
            Path(args.full_capital_daily), args.capital, args.topk, args.entry_lag_days, args.hold_days
        )
        random_df = random_split_test(
            full_rows,
            args.capital,
            float(combo_summary["cum_return"]),
            args.random_repeats,
            args.seed,
        )
        random_df.to_csv(output_dir / "random_split.csv", index=False)

    summary.to_csv(output_dir / "summary.csv", index=False)
    monthly.to_csv(output_dir / "monthly.csv", index=False)
    weekdays.to_csv(output_dir / "weekday.csv", index=False)
    top_days.to_csv(output_dir / "top_contribution_days.csv", index=False)
    concentration_df.to_csv(output_dir / "concentration.csv", index=False)
    combo.to_csv(output_dir / "combined_daily.csv", index=False)
    write_markdown(output_dir / "diagnostic.md", args, summary, monthly, concentration_df, random_df)

    print("===== Summary =====")
    print(summary.to_string(index=False))
    print("\n===== Monthly =====")
    print(monthly.to_string(index=False))
    print("\n===== Positive return concentration =====")
    print(concentration_df.to_string(index=False))
    if random_df is not None:
        print("\n===== Random split cum_return quantiles =====")
        print(random_df["cum_return"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
        print(f"observed percentile: {random_df.attrs['observed_percentile']:.3f}")
    print(f"\nWrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
