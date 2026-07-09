#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta


DEFAULT_MARKET_HOLIDAYS = {"2026-06-19"}


def parse_holidays(raw: str | None) -> set[str]:
    if not raw:
        return set(DEFAULT_MARKET_HOLIDAYS)
    holidays: set[str] = set()
    for token in raw.replace(",", " ").split():
        token = token.strip()
        if token:
            # Validate early so a typo does not silently weaken freshness checks.
            datetime.strptime(token, "%Y-%m-%d")
            holidays.add(token)
    return holidays


def resolve_required_date(required_date: str, *, today: date | None = None, holidays: set[str] | None = None) -> str:
    if required_date and required_date != "auto":
        return required_date

    current = (today or date.today()) - timedelta(days=1)
    holiday_dates = holidays if holidays is not None else set(DEFAULT_MARKET_HOLIDAYS)
    while current.weekday() >= 5 or current.isoformat() in holiday_dates:
        current -= timedelta(days=1)
    return current.isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-date", default="auto")
    parser.add_argument("--today", default="")
    parser.add_argument("--holidays", default="")
    args = parser.parse_args()

    today = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else None
    print(resolve_required_date(args.required_date, today=today, holidays=parse_holidays(args.holidays)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
