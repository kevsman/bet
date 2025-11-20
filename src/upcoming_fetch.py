from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import AppConfig, get_config

FIXTURE_URL = "https://www.football-data.co.uk/fixtures.csv"
DEFAULT_LINE = 2.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch upcoming fixtures from football-data.co.uk")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to next Friday.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to following Monday.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to data/upcoming/fixtures_filtered.csv",
    )
    return parser.parse_args()


def default_window() -> tuple[date, date]:
    today = date.today()
    days_until_friday = (4 - today.weekday()) % 7
    start = today + timedelta(days=days_until_friday)
    end = start + timedelta(days=3)
    return start, end


def fetch_raw() -> pd.DataFrame:
    response = requests.get(FIXTURE_URL, timeout=30)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


def filter_leagues(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    valid = {league.code for league in cfg.leagues}
    return df[df["Div"].isin(valid)]


def filter_dates(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
    return df[mask]


def best_over_odds(row: pd.Series) -> float:
    candidates = [row.get(col) for col in ["B365>2.5", "Max>2.5", "Avg>2.5", "P>2.5", "PC>2.5", "B365C>2.5"]]
    vals = [float(val) for val in candidates if pd.notna(val)]
    return max(vals) if vals else float("nan")


def best_under_odds(row: pd.Series) -> float:
    candidates = [row.get(col) for col in ["B365<2.5", "Max<2.5", "Avg<2.5", "P<2.5", "PC<2.5", "B365C<2.5"]]
    vals = [float(val) for val in candidates if pd.notna(val)]
    return max(vals) if vals else float("nan")


def curate(df: pd.DataFrame) -> pd.DataFrame:
    curated = df.copy()
    curated["best_over_odds"] = df.apply(best_over_odds, axis=1)
    curated["best_under_odds"] = df.apply(best_under_odds, axis=1)
    curated["market_total_line"] = DEFAULT_LINE
    keep_cols = [
        "Div",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "best_over_odds",
        "best_under_odds",
        "market_total_line",
    ]
    return curated.loc[:, keep_cols]


def ensure_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    cfg = get_config()

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        start, end = default_window()

    df = fetch_raw()
    df = normalize_dates(df)
    df = filter_leagues(df, cfg)
    df = filter_dates(df, start, end)
    if df.empty:
        print("No fixtures found for selected window.")
        return
    curated = curate(df)

    output_path = args.output or (cfg.data_dir / "upcoming" / "fixtures_filtered.csv")
    output_path = ensure_output_path(output_path)
    curated.to_csv(output_path, index=False)
    print(
        f"Saved {len(curated)} fixtures between {start} and {end} to {output_path.relative_to(cfg.base_dir)}"
    )


if __name__ == "__main__":
    main()
