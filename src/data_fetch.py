from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import AppConfig, LeagueConfig, get_config

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_EXTRA = "https://www.football-data.co.uk/new"

# Current season code - only this season will be re-downloaded
CURRENT_SEASON = "2526"

# Column mapping for extra leagues (they use different column names)
EXTRA_LEAGUE_COLUMN_MAP = {
    "Country": "Country",
    "League": "Div",
    "Date": "Date",
    "Time": "Time",
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HG": "FTHG",
    "AG": "FTAG",
    "Res": "FTR",
    "PH": "PSH",
    "PD": "PSD",
    "PA": "PSA",
    "MaxH": "MaxH",
    "MaxD": "MaxD",
    "MaxA": "MaxA",
    "AvgH": "AvgH",
    "AvgD": "AvgD",
    "AvgA": "AvgA",
}


def build_download_url(season_code: str, league_code: str) -> str:
    return f"{FOOTBALL_DATA_BASE}/{season_code}/{league_code}.csv"


def build_extra_download_url(league_code: str) -> str:
    return f"{FOOTBALL_DATA_EXTRA}/{league_code}.csv"


def download_file(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)


def normalize_extra_league_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from extra league format to standard format."""
    rename_map = {k: v for k, v in EXTRA_LEAGUE_COLUMN_MAP.items() if k in df.columns}
    return df.rename(columns=rename_map)


def extract_season_code(date_str: str) -> str:
    """Extract season code from date (e.g., '2024-08-15' -> '2425')."""
    try:
        date = pd.to_datetime(date_str, dayfirst=True)
        year = date.year
        month = date.month
        # Season runs Aug-May, so Aug onwards is start of new season
        if month >= 7:  # July onwards = new season
            return f"{str(year)[-2:]}{str(year + 1)[-2:]}"
        else:
            return f"{str(year - 1)[-2:]}{str(year)[-2:]}"
    except:
        return "unknown"


def convert_season_format(season_str: str) -> str:
    """Convert season format from '2012/2013' to '1213'."""
    try:
        parts = season_str.split("/")
        if len(parts) == 2:
            return f"{parts[0][-2:]}{parts[1][-2:]}"
        return "unknown"
    except:
        return "unknown"


def download_league(cfg: AppConfig, season_code: str, league_code: str) -> Path | None:
    destination = cfg.raw_dir / f"{league_code}_{season_code}.csv"
    
    # Skip if file exists and it's not the current season
    if destination.exists() and season_code != CURRENT_SEASON:
        print(f"Skipping {destination.relative_to(cfg.base_dir)} (already exists)")
        return destination
    
    url = build_download_url(season_code, league_code)
    try:
        download_file(url, destination)
    except requests.HTTPError as exc:
        print(f"Failed to download {url}: {exc}")
        return None
    print(f"Saved {destination.relative_to(cfg.base_dir)}")
    time.sleep(1)
    return destination


def download_extra_league(cfg: AppConfig, league: LeagueConfig) -> list[Path]:
    """Download and split extra league data by season."""
    url = build_extra_download_url(league.code)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"Failed to download {url}: {exc}")
        return []
    
    # Parse and normalize the data (decode with utf-8-sig to strip BOM)
    from io import StringIO
    text = response.content.decode('utf-8-sig')
    df = pd.read_csv(StringIO(text), on_bad_lines="skip", low_memory=False)
    
    df = normalize_extra_league_columns(df)
    
    # Extract/convert season code
    if "Season" in df.columns:
        # Use existing Season column (format: '2012/2013' -> '1213')
        df["season_code"] = df["Season"].apply(convert_season_format)
    elif "season_code" not in df.columns and "Date" in df.columns:
        # Fall back to extracting from date
        df["season_code"] = df["Date"].apply(extract_season_code)
    
    # Split by season and save
    downloaded = []
    for season_code in df["season_code"].unique():
        if season_code == "unknown" or "/" in str(season_code):
            continue
        
        destination = cfg.raw_dir / f"{league.code}_{season_code}.csv"
        
        # Skip if file exists and it's not the current season
        if destination.exists() and season_code != CURRENT_SEASON:
            print(f"Skipping {destination.relative_to(cfg.base_dir)} (already exists)")
            downloaded.append(destination)
            continue
        
        season_df = df[df["season_code"] == season_code].copy()
        # Drop helper columns before saving
        season_df = season_df.drop(columns=["season_code", "Season"], errors="ignore")
        
        season_df.to_csv(destination, index=False)
        print(f"Saved {destination.relative_to(cfg.base_dir)} ({len(season_df)} matches)")
        downloaded.append(destination)
    
    time.sleep(1)
    return downloaded


def sync_all(cfg: AppConfig) -> list[Path]:
    downloaded: list[Path] = []
    for league in cfg.leagues:
        if league.is_extra:
            # Extra leagues: single file with all seasons
            try:
                files = download_extra_league(cfg, league)
                downloaded.extend(files)
            except Exception as e:
                print(f"Error processing {league.code}: {e}")
        else:
            # Main leagues: separate file per season
            for season_code in league.season_codes:
                file_path = download_league(cfg, season_code, league.code)
                if file_path:
                    downloaded.append(file_path)
    return downloaded


def main() -> None:
    cfg = get_config()
    files = sync_all(cfg)
    if files:
        print(f"Downloaded/updated {len(files)} files.")
    else:
        print("No files downloaded. Check league codes/seasons.")


if __name__ == "__main__":
    main()
