from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import tushare as ts
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "tushare is required for this script. Install it with `pip install tushare` before running."
    ) from exc


def fmt_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def out_date(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def ts_code_to_qlib(ts_code: str) -> str:
    symbol, exchange = ts_code.split(".")
    return f"{exchange.upper()}{symbol}"


def qlib_index_symbol(index_code: str) -> str:
    return ts_code_to_qlib(index_code)


def get_token(cli_token: str | None) -> str:
    token = cli_token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("Missing Tushare token. Pass --token or set TUSHARE_TOKEN in your environment.")
    return token


def load_or_fetch_csv(path: Path, fetcher) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str)
    df = fetcher()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


@dataclass
class TushareBundleBuilder:
    pro: object
    start_date: str
    end_date: str
    raw_cache_dir: Path
    qlib_source_dir: Path
    component_dir: Path
    benchmark_indexes: list[str]
    sleep_seconds: float

    def run(self) -> None:
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.qlib_source_dir.mkdir(parents=True, exist_ok=True)
        self.component_dir.mkdir(parents=True, exist_ok=True)
        stock_basic = self.fetch_stock_basic()
        trade_cal = self.fetch_trade_calendar()
        daily = self.fetch_daily(trade_cal)
        adj = self.fetch_adj_factor(trade_cal)
        index_daily = self.fetch_index_daily()
        index_weight = self.fetch_index_weight()

        self.write_equity_csvs(stock_basic, trade_cal, daily, adj)
        self.write_index_csvs(trade_cal, index_daily)
        self.write_csi500_membership(trade_cal, index_weight)

        print("Tushare bundle build complete.")
        print(f"- qlib_source_dir: {self.qlib_source_dir}")
        print(f"- component_dir: {self.component_dir}")

    def fetch_stock_basic(self) -> pd.DataFrame:
        cache = self.raw_cache_dir / "stock_basic.csv"

        def _fetch():
            frames = []
            for status in ["L", "D", "P"]:
                df = self.pro.stock_basic(
                    exchange="",
                    list_status=status,
                    fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status",
                )
                frames.append(df)
                time.sleep(self.sleep_seconds)
            merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"])
            return merged

        df = load_or_fetch_csv(cache, _fetch)
        df["list_date"] = df["list_date"].astype(str)
        df["delist_date"] = df["delist_date"].fillna("").astype(str)
        return df

    def fetch_trade_calendar(self) -> pd.DataFrame:
        cache = self.raw_cache_dir / "trade_cal.csv"

        def _fetch():
            return self.pro.trade_cal(
                exchange="",
                start_date=self.start_date,
                end_date=self.end_date,
                fields="cal_date,is_open",
            )

        df = load_or_fetch_csv(cache, _fetch)
        df = df[df["is_open"].astype(str) == "1"].copy()
        df = df.sort_values("cal_date").reset_index(drop=True)
        return df

    def fetch_daily(self, trade_cal: pd.DataFrame) -> pd.DataFrame:
        cache = self.raw_cache_dir / "daily.csv"
        if cache.exists():
            df = pd.read_csv(cache, dtype=str)
        else:
            frames = []
            for i, trade_date in enumerate(trade_cal["cal_date"].tolist(), start=1):
                df_day = self.pro.daily(trade_date=trade_date)
                frames.append(df_day)
                if i % 50 == 0:
                    print(f"[daily] fetched {i}/{len(trade_cal)} trade dates")
                time.sleep(self.sleep_seconds)
            df = pd.concat(frames, ignore_index=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False, encoding="utf-8")

        df["trade_date"] = df["trade_date"].astype(str)
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # tushare daily vol unit is "hand", convert to shares for better consistency with common qlib usage
        df["volume"] = df["vol"] * 100.0
        return df

    def fetch_adj_factor(self, trade_cal: pd.DataFrame) -> pd.DataFrame:
        cache = self.raw_cache_dir / "adj_factor.csv"
        if cache.exists():
            df = pd.read_csv(cache, dtype=str)
        else:
            frames = []
            for i, trade_date in enumerate(trade_cal["cal_date"].tolist(), start=1):
                df_day = self.pro.adj_factor(trade_date=trade_date)
                frames.append(df_day)
                if i % 50 == 0:
                    print(f"[adj_factor] fetched {i}/{len(trade_cal)} trade dates")
                time.sleep(self.sleep_seconds)
            df = pd.concat(frames, ignore_index=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False, encoding="utf-8")

        df["trade_date"] = df["trade_date"].astype(str)
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
        return df

    def fetch_index_daily(self) -> pd.DataFrame:
        cache = self.raw_cache_dir / "index_daily.csv"
        if cache.exists():
            df = pd.read_csv(cache, dtype=str)
        else:
            frames = []
            for ts_code in self.benchmark_indexes:
                df_idx = self.pro.index_daily(ts_code=ts_code, start_date=self.start_date, end_date=self.end_date)
                frames.append(df_idx)
                time.sleep(self.sleep_seconds)
            df = pd.concat(frames, ignore_index=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False, encoding="utf-8")

        df["trade_date"] = df["trade_date"].astype(str)
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = df["vol"] * 100.0
        return df

    def fetch_index_weight(self) -> pd.DataFrame:
        cache = self.raw_cache_dir / "index_weight.csv"
        if cache.exists():
            df = pd.read_csv(cache, dtype=str)
            return df

        # Pull in 3-month windows to avoid single-query range issues.
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        windows = pd.date_range(start=start, end=end, freq="3MS").tolist()
        if not windows or windows[0] != start:
            windows = [start] + windows

        frames = []
        for i, win_start in enumerate(windows):
            win_end = min((win_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)), end)
            df_win = self.pro.index_weight(
                index_code="000905.SH",
                start_date=fmt_date(str(win_start.date())),
                end_date=fmt_date(str(win_end.date())),
            )
            frames.append(df_win)
            print(f"[index_weight] fetched window {i + 1}/{len(windows)}: {win_start.date()} -> {win_end.date()}")
            time.sleep(self.sleep_seconds)

        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["index_code", "con_code", "trade_date"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False, encoding="utf-8")
        return df

    def write_equity_csvs(
        self,
        stock_basic: pd.DataFrame,
        trade_cal: pd.DataFrame,
        daily: pd.DataFrame,
        adj: pd.DataFrame,
    ) -> None:
        self.qlib_source_dir.mkdir(parents=True, exist_ok=True)
        trade_index = pd.DatetimeIndex(pd.to_datetime(trade_cal["cal_date"]))
        daily = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")

        grouped = daily.groupby("ts_code")
        meta = stock_basic.set_index("ts_code").to_dict("index")

        for i, (ts_code, frame) in enumerate(grouped, start=1):
            info = meta.get(ts_code)
            if info is None:
                continue

            list_date = pd.Timestamp(info["list_date"])
            delist_raw = str(info.get("delist_date", "") or "").strip()
            delist_date = pd.Timestamp(delist_raw) if delist_raw else trade_index.max()

            active_cal = trade_index[(trade_index >= list_date) & (trade_index <= delist_date)]
            if active_cal.empty:
                continue

            frame = frame.copy()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            frame = frame.set_index("trade_date").reindex(active_cal).sort_index()
            frame["ts_code"] = ts_code
            if frame["adj_factor"].dropna().empty or frame["close"].dropna().empty:
                continue

            first_valid = frame[["close", "adj_factor"]].dropna().iloc[0]
            first_adj_factor = float(first_valid["adj_factor"])
            rel_factor = frame["adj_factor"] / first_adj_factor
            adj_close = frame["close"] * rel_factor
            first_adj_close = float(adj_close.dropna().iloc[0])

            out = pd.DataFrame(index=frame.index)
            out["symbol"] = ts_code_to_qlib(ts_code)
            out["date"] = out.index.map(out_date)
            out["factor"] = rel_factor / first_adj_close
            for col in ["open", "high", "low", "close"]:
                out[col] = frame[col] * rel_factor / first_adj_close
            out["volume"] = frame["volume"] / rel_factor * first_adj_close
            out.loc[frame["close"].isna(), ["open", "high", "low", "close", "volume", "factor"]] = pd.NA

            out = out[["symbol", "date", "open", "close", "high", "low", "volume", "factor"]]
            out.to_csv(self.qlib_source_dir / f"{out['symbol'].iloc[0].lower()}.csv", index=False, encoding="utf-8")

            if i % 500 == 0:
                print(f"[equity_csv] wrote {i} symbols")

        stock_basic.to_csv(self.component_dir / "stock_basic.csv", index=False, encoding="utf-8")

    def write_index_csvs(self, trade_cal: pd.DataFrame, index_daily: pd.DataFrame) -> None:
        trade_index = pd.DatetimeIndex(pd.to_datetime(trade_cal["cal_date"]))

        for ts_code in self.benchmark_indexes:
            frame = index_daily[index_daily["ts_code"] == ts_code].copy()
            if frame.empty:
                continue
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            frame = frame.set_index("trade_date").reindex(trade_index).sort_index()
            if frame["close"].dropna().empty:
                continue

            first_close = float(frame["close"].dropna().iloc[0])
            out = pd.DataFrame(index=frame.index)
            out["symbol"] = qlib_index_symbol(ts_code)
            out["date"] = out.index.map(out_date)
            out["factor"] = 1.0 / first_close
            for col in ["open", "high", "low", "close"]:
                out[col] = frame[col] / first_close
            out["volume"] = frame["volume"] * first_close
            out.loc[frame["close"].isna(), ["open", "high", "low", "close", "volume", "factor"]] = pd.NA
            out = out[["symbol", "date", "open", "close", "high", "low", "volume", "factor"]]
            out.to_csv(self.qlib_source_dir / f"{out['symbol'].iloc[0].lower()}.csv", index=False, encoding="utf-8")

    def write_csi500_membership(self, trade_cal: pd.DataFrame, index_weight: pd.DataFrame) -> None:
        self.component_dir.mkdir(parents=True, exist_ok=True)
        instruments_dir = self.component_dir / "instruments"
        instruments_dir.mkdir(parents=True, exist_ok=True)

        cal = pd.DatetimeIndex(pd.to_datetime(trade_cal["cal_date"]))
        index_weight = index_weight.copy()
        index_weight["trade_date"] = pd.to_datetime(index_weight["trade_date"])
        snapshots = {
            dt: sorted(set(group["con_code"].tolist()))
            for dt, group in index_weight.groupby("trade_date")
        }
        snapshot_dates = sorted(snapshots.keys())
        if not snapshot_dates:
            raise ValueError("No index_weight snapshots fetched for CSI500.")

        intervals: list[tuple[str, str, str]] = []
        for i, snap_date in enumerate(snapshot_dates):
            next_date = snapshot_dates[i + 1] if i + 1 < len(snapshot_dates) else None
            if next_date is None:
                end_date = cal.max()
            else:
                prev = cal[cal < next_date]
                end_date = prev.max() if len(prev) else snap_date

            for con_code in snapshots[snap_date]:
                intervals.append((ts_code_to_qlib(con_code), out_date(snap_date), out_date(end_date)))

        interval_df = pd.DataFrame(intervals, columns=["instrument", "start_datetime", "end_datetime"])
        interval_df = interval_df.sort_values(["instrument", "start_datetime", "end_datetime"]).reset_index(drop=True)
        merged_rows = []
        for instrument, group in interval_df.groupby("instrument"):
            current_start = None
            current_end = None
            for row in group.itertuples(index=False):
                start = pd.Timestamp(row.start_datetime)
                end = pd.Timestamp(row.end_datetime)
                if current_start is None:
                    current_start, current_end = start, end
                    continue
                if start <= current_end + pd.Timedelta(days=7):
                    current_end = max(current_end, end)
                else:
                    merged_rows.append((instrument, out_date(current_start), out_date(current_end)))
                    current_start, current_end = start, end
            if current_start is not None:
                merged_rows.append((instrument, out_date(current_start), out_date(current_end)))

        out_path = instruments_dir / "csi500.txt"
        pd.DataFrame(merged_rows).to_csv(out_path, sep="\t", header=False, index=False, encoding="utf-8")
        index_weight.to_csv(self.component_dir / "csi500_index_weight.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Tushare daily data and build Qlib-ready CSV source files.")
    parser.add_argument("--token", default=None, help="Optional Tushare token. Prefer TUSHARE_TOKEN env var.")
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument(
        "--raw-cache-dir",
        default=str(Path("/Users/Dylan.Min/Documents/Code/learn/LHAI/data/raw/tushare_2021_cache")),
        help="Directory for cached Tushare API responses.",
    )
    parser.add_argument(
        "--qlib-source-dir",
        default=str(Path("/Users/Dylan.Min/Documents/Code/learn/LHAI/data/tushare_qlib_source_2021")),
        help="Directory for Qlib-ready per-symbol CSV files.",
    )
    parser.add_argument(
        "--component-dir",
        default=str(Path("/Users/Dylan.Min/Documents/Code/learn/LHAI/data/tushare_components_2021")),
        help="Directory for CSI500 membership files and stock metadata.",
    )
    parser.add_argument(
        "--benchmark-indexes",
        default="000905.SH,000300.SH",
        help="Comma-separated benchmark index codes to include as CSVs.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    args = parser.parse_args()

    token = get_token(args.token)
    ts.set_token(token)
    pro = ts.pro_api()

    builder = TushareBundleBuilder(
        pro=pro,
        start_date=fmt_date(args.start_date),
        end_date=fmt_date(args.end_date),
        raw_cache_dir=Path(args.raw_cache_dir).expanduser().resolve(),
        qlib_source_dir=Path(args.qlib_source_dir).expanduser().resolve(),
        component_dir=Path(args.component_dir).expanduser().resolve(),
        benchmark_indexes=[item.strip() for item in args.benchmark_indexes.split(",") if item.strip()],
        sleep_seconds=args.sleep_seconds,
    )
    builder.run()


if __name__ == "__main__":
    main()
