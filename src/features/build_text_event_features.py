from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd


POSITIVE_WORDS = {
    "增长",
    "改善",
    "回购",
    "中标",
    "盈利",
    "扩张",
    "增持",
    "突破",
    "创新高",
    "利好",
}

NEGATIVE_WORDS = {
    "下滑",
    "亏损",
    "减持",
    "处罚",
    "问询",
    "违约",
    "暴跌",
    "风险",
    "减值",
    "利空",
}


def normalize_date(raw: str) -> str:
    return raw.strip().replace("/", "-")[:10]


def count_hits(text: str, keywords: set[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def build_features(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dt = normalize_date(row["datetime"])
        instrument = row["instrument"].strip()
        grouped[(dt, instrument)].append(row)

    output_rows: list[dict[str, str | float]] = []
    for (dt, instrument), items in sorted(grouped.items()):
        event_count = len(items)
        text_lengths = []
        pos_hits = 0
        neg_hits = 0
        event_type_counter: dict[str, int] = defaultdict(int)

        for item in items:
            text = item.get("text", "").strip()
            if not text:
                text = f"{item.get('title', '').strip()} {item.get('body', '').strip()}".strip()
            text_lengths.append(len(text))
            pos_hits += count_hits(text, POSITIVE_WORDS)
            neg_hits += count_hits(text, NEGATIVE_WORDS)
            event_type = item.get("event_type", "").strip()
            if event_type:
                event_type_counter[event_type] += 1

        text_length_mean = sum(text_lengths) / event_count if event_count else 0.0
        sentiment_balance = pos_hits - neg_hits
        unique_event_type_count = len(event_type_counter)
        top_event_type_share = (
            max(event_type_counter.values()) / event_count if event_type_counter and event_count else 0.0
        )

        output_rows.append(
            {
                "datetime": dt,
                "instrument": instrument,
                "event_count": float(event_count),
                "text_length_mean": float(text_length_mean),
                "positive_keyword_hits": float(pos_hits),
                "negative_keyword_hits": float(neg_hits),
                "sentiment_balance": float(sentiment_balance),
                "unique_event_type_count": float(unique_event_type_count),
                "top_event_type_share": float(top_event_type_share),
            }
        )

    df = pd.DataFrame(output_rows)
    if df.empty:
        return []

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)

    grouped_df = df.groupby("instrument", group_keys=False)
    for window in (3, 5, 10):
        df[f"event_count_roll_{window}"] = grouped_df["event_count"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )
    for base_col, window in (
        ("positive_keyword_hits", 5),
        ("negative_keyword_hits", 5),
        ("sentiment_balance", 5),
        ("unique_event_type_count", 5),
    ):
        df[f"{base_col}_roll_{window}"] = grouped_df[base_col].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )

    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")
    return df.to_dict("records")


def load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"datetime", "instrument"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input CSV must include at least: datetime, instrument")
        return list(reader)


def write_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = ["datetime", "instrument"]
    else:
        base = ["datetime", "instrument"]
        dynamic = [k for k in rows[0].keys() if k not in base]
        fieldnames = base + dynamic
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily text event features from news/announcement csv.")
    parser.add_argument(
        "--input",
        default="data/raw/text_events.csv",
        help="Raw event csv path. Must contain datetime and instrument columns.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/text_event_features.csv",
        help="Output aggregated feature csv path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    rows = load_rows(input_path)
    features = build_features(rows)
    write_rows(output_path, features)
    print(f"Wrote {len(features)} rows to {output_path}")


if __name__ == "__main__":
    main()
