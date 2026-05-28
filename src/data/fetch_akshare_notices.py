from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

try:
    import akshare as ak
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "akshare is required for this script. Install it with `pip install akshare` "
        "or add it to your environment before running."
    ) from exc


NOTICE_TYPES = ("全部", "重大事项", "财务报告", "融资公告", "风险提示", "资产重组", "信息变更", "持股变动")


def to_qlib_instrument(code: str) -> str:
    code = str(code).strip().zfill(6)
    if code.startswith(("6", "5", "9")):
        return f"SH{code}"
    if code.startswith(("4", "8")):
        return f"BJ{code}"
    return f"SZ{code}"


def normalize_notice_frame(df: pd.DataFrame, notice_type: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["datetime", "instrument", "title", "body", "text", "source", "event_type", "published_at", "url"]
        )

    rename_map = {
        "代码": "code",
        "名称": "name",
        "公告标题": "title",
        "公告类型": "event_type",
        "公告日期": "datetime",
        "网址": "url",
    }
    frame = df.rename(columns=rename_map).copy()

    # 节假日或无数据日，AKShare 可能返回不含正常字段的空表。
    if "code" not in frame.columns and "title" not in frame.columns:
        return pd.DataFrame(
            columns=["datetime", "instrument", "title", "body", "text", "source", "event_type", "published_at", "url"]
        )

    for col in ["code", "title", "datetime", "url"]:
        if col not in frame.columns:
            raise ValueError(f"AKShare notice result missing required column: {col}")

    if "event_type" not in frame.columns:
        frame["event_type"] = notice_type

    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d")
    frame["instrument"] = frame["code"].map(to_qlib_instrument)
    frame["title"] = frame["title"].astype(str).str.strip()
    frame["body"] = ""
    # 第一版直接让 text 等于标题，后续如果抓正文再升级。
    frame["text"] = frame["title"]
    frame["source"] = "公告"
    frame["published_at"] = ""
    frame["url"] = frame["url"].astype(str).str.strip()
    frame["event_type"] = frame["event_type"].astype(str).str.strip()

    cols = ["datetime", "instrument", "title", "body", "text", "source", "event_type", "published_at", "url"]
    return frame[cols].dropna(subset=["datetime", "instrument", "title"]).reset_index(drop=True)


def fetch_notice_for_date(date: pd.Timestamp, notice_type: str) -> pd.DataFrame:
    raw = ak.stock_notice_report(symbol=notice_type, date=date.strftime("%Y%m%d"))
    return normalize_notice_frame(raw, notice_type)


def fetch_range(start_date: str, end_date: str, notice_types: list[str], sleep_seconds: float) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    frames: list[pd.DataFrame] = []

    for date in dates:
        for notice_type in notice_types:
            try:
                frame = fetch_notice_for_date(date, notice_type)
            except Exception as exc:  # pragma: no cover - network / upstream variability
                print(f"[warn] {date.date()} {notice_type}: {exc}")
                frame = pd.DataFrame()
            if not frame.empty:
                frames.append(frame)
                print(f"[ok] {date.date()} {notice_type}: {len(frame)} rows")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame(
            columns=["datetime", "instrument", "title", "body", "text", "source", "event_type", "published_at", "url"]
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["datetime", "instrument", "title", "url"]).sort_values(
        ["datetime", "instrument", "title"]
    )
    return merged.reset_index(drop=True)


def get_pool_instruments(provider_uri: str, market: str, start_date: str, end_date: str) -> set[str]:
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    instruments = D.list_instruments(
        D.instruments(market=market),
        start_time=start_date,
        end_time=end_date,
        freq="day",
        as_list=True,
    )
    return set(instruments)


def parse_notice_types(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in items if item not in NOTICE_TYPES]
    if invalid:
        raise ValueError(f"Unsupported notice types: {invalid}. Valid choices: {NOTICE_TYPES}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch A-share notices from AKShare into text_events.csv format.")
    parser.add_argument("--start-date", required=True, help="Start date, e.g. 2017-01-01")
    parser.add_argument("--end-date", required=True, help="End date, e.g. 2017-12-31")
    parser.add_argument(
        "--notice-types",
        default="全部",
        help="Comma-separated AKShare notice categories. Default: 全部",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between requests to reduce upstream pressure. Default: 0.2",
    )
    parser.add_argument(
        "--output",
        default="/Users/Dylan.Min/Documents/Code/learn/LHAI/data/raw/text_events.csv",
        help="Output csv path in project raw text-event format.",
    )
    parser.add_argument(
        "--market",
        default="",
        help="Optional Qlib instrument pool filter, e.g. csi500. Empty means keep all fetched notices.",
    )
    parser.add_argument(
        "--provider-uri",
        default="/Users/Dylan.Min/Documents/Code/learn/LHAI/data/qlib_cn_data",
        help="Qlib data path used when --market is provided.",
    )
    args = parser.parse_args()

    notice_types = parse_notice_types(args.notice_types)
    result = fetch_range(args.start_date, args.end_date, notice_types, args.sleep_seconds)

    if args.market:
        pool = get_pool_instruments(args.provider_uri, args.market, args.start_date, args.end_date)
        before = len(result)
        result = result[result["instrument"].isin(pool)].reset_index(drop=True)
        print(f"Filtered by market={args.market}: {before} -> {len(result)} rows")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Wrote {len(result)} rows to {output_path}")
    if not result.empty:
        print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
