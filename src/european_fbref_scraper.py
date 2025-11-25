"""
Scrape European competition data from FBref.com
Free source with current season data for UCL, UEL, UECL.
"""
from __future__ import annotations

import re
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .european_config import EuropeanConfig, get_european_config

# FBref competition URLs
FBREF_COMPETITIONS = {
    "UCL": {
        "name": "Champions League",
        "comp_id": 8,
        "url_template": "https://fbref.com/en/comps/8/{season}/schedule/{season}-Champions-League-Scores-and-Fixtures",
    },
    "UEL": {
        "name": "Europa League", 
        "comp_id": 19,
        "url_template": "https://fbref.com/en/comps/19/{season}/schedule/{season}-Europa-League-Scores-and-Fixtures",
    },
    "UECL": {
        "name": "Conference League",
        "comp_id": 882,
        "url_template": "https://fbref.com/en/comps/882/{season}/schedule/{season}-Conference-League-Scores-and-Fixtures",
    },
}

# Seasons available on FBref (format: "2024-2025")
FBREF_SEASONS = [
    "2021-2022",
    "2022-2023", 
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

# Realistic browser headers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]

RATE_LIMIT_DELAY = 6.0  # FBref requires slower requests


def fetch_fbref_page(url: str, retry_count: int = 3) -> Optional[BeautifulSoup]:
    """Fetch and parse a FBref page with retry logic."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }
    
    for attempt in range(retry_count):
        try:
            # Add jitter to avoid detection
            time.sleep(random.uniform(1, 3))
            
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 429:  # Too many requests
                wait_time = 60 * (attempt + 1)
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                print(f"  Access forbidden - trying alternative method...")
                # Try with different headers
                headers["User-Agent"] = random.choice(USER_AGENTS)
                time.sleep(10 * (attempt + 1))
                continue
            print(f"HTTP Error: {e}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            
        if attempt < retry_count - 1:
            time.sleep(5 * (attempt + 1))
    
    return None


def parse_score(score_str: str) -> tuple[Optional[int], Optional[int]]:
    """Parse score string like '2–1' into (home, away) goals."""
    if not score_str or score_str.strip() == "":
        return None, None
    
    # Handle different dash types
    for sep in ["–", "-", "—"]:
        if sep in score_str:
            parts = score_str.split(sep)
            if len(parts) == 2:
                try:
                    return int(parts[0].strip()), int(parts[1].strip())
                except ValueError:
                    return None, None
    return None, None


def parse_fixtures_table(soup: BeautifulSoup, comp_code: str, season: str) -> List[Dict]:
    """Parse the fixtures table from FBref."""
    matches = []
    
    # Find the scores table
    table = soup.find("table", {"id": re.compile(r"sched.*all_comps")})
    if not table:
        # Try alternative table ID
        table = soup.find("table", {"class": re.compile(r".*stats_table.*")})
    
    if not table:
        print(f"  Could not find fixtures table")
        return matches
    
    tbody = table.find("tbody")
    if not tbody:
        return matches
    
    for row in tbody.find_all("tr"):
        # Skip spacer rows
        if "spacer" in row.get("class", []) or "thead" in row.get("class", []):
            continue
        
        cells = row.find_all(["td", "th"])
        if len(cells) < 6:
            continue
        
        try:
            # Extract data from cells
            data = {}
            for cell in cells:
                data_stat = cell.get("data-stat", "")
                data[data_stat] = cell.get_text(strip=True)
                
                # Get links for team names
                if data_stat in ["home_team", "away_team"]:
                    link = cell.find("a")
                    if link:
                        data[data_stat] = link.get_text(strip=True)
            
            # Skip if no date or teams
            if not data.get("date") or not data.get("home_team") or not data.get("away_team"):
                continue
            
            # Parse score
            score = data.get("score", "")
            home_goals, away_goals = parse_score(score)
            
            # Skip unplayed matches
            if home_goals is None:
                continue
            
            match = {
                "Date": data.get("date", ""),
                "Time": data.get("time", ""),
                "round": data.get("round", ""),
                "HomeTeam": data.get("home_team", ""),
                "AwayTeam": data.get("away_team", ""),
                "FTHG": home_goals,
                "FTAG": away_goals,
                "competition_code": comp_code,
                "season": season,
                "venue": data.get("venue", ""),
                "attendance": data.get("attendance", ""),
                "referee": data.get("referee", ""),
            }
            
            # Extract xG if available
            if "home_xg" in data:
                try:
                    match["home_xg"] = float(data["home_xg"]) if data["home_xg"] else None
                except ValueError:
                    match["home_xg"] = None
            
            if "away_xg" in data:
                try:
                    match["away_xg"] = float(data["away_xg"]) if data["away_xg"] else None
                except ValueError:
                    match["away_xg"] = None
            
            matches.append(match)
            
        except Exception as e:
            continue
    
    return matches


def scrape_competition_season(comp_code: str, season: str) -> List[Dict]:
    """Scrape all matches for a competition and season."""
    comp_info = FBREF_COMPETITIONS.get(comp_code)
    if not comp_info:
        return []
    
    url = comp_info["url_template"].format(season=season)
    
    soup = fetch_fbref_page(url)
    if not soup:
        return []
    
    return parse_fixtures_table(soup, comp_code, season)


def download_fbref_european_data(cfg: EuropeanConfig) -> None:
    """Download all European competition data from FBref."""
    all_matches = []
    
    for comp_code in ["UCL", "UEL", "UECL"]:
        comp_name = FBREF_COMPETITIONS[comp_code]["name"]
        print(f"\nFetching {comp_name}...")
        
        for season in FBREF_SEASONS:
            print(f"  Season {season}...", end=" ")
            
            matches = scrape_competition_season(comp_code, season)
            
            if matches:
                all_matches.extend(matches)
                print(f"{len(matches)} matches")
            else:
                print("No data")
            
            time.sleep(RATE_LIMIT_DELAY)  # Respect rate limits
    
    if all_matches:
        df = pd.DataFrame(all_matches)
        
        # Clean up the dataframe
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "FTHG", "FTAG"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Add derived columns
        df["total_goals"] = df["FTHG"] + df["FTAG"]
        df["match_id"] = range(1, len(df) + 1)
        
        # Convert season to year format for compatibility
        df["season_year"] = df["season"].str[:4].astype(int)
        
        output_path = cfg.raw_dir / "european_matches_fbref.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} matches to {output_path}")
        
        # Also save as the main file for the model
        main_path = cfg.raw_dir / "european_matches.csv"
        df.to_csv(main_path, index=False)
        print(f"Also saved to {main_path}")
        
        # Summary
        print("\n=== Summary ===")
        for comp in df["competition_code"].unique():
            comp_df = df[df["competition_code"] == comp]
            print(f"{comp}: {len(comp_df)} matches ({comp_df['season'].nunique()} seasons)")
    else:
        print("\nNo matches downloaded.")


def main() -> None:
    cfg = get_european_config()
    download_fbref_european_data(cfg)


if __name__ == "__main__":
    main()
