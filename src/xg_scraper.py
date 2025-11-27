"""
xG (Expected Goals) data scraper for FBref and Understat.

Expected goals is a powerful predictor of future scoring - it measures
the quality of chances created rather than just outcomes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# FBref league IDs and URLs
FBREF_LEAGUES = {
    "E0": {"id": "9", "name": "Premier-League", "country": "eng"},
    "D1": {"id": "20", "name": "Bundesliga", "country": "ger"},
    "SP1": {"id": "12", "name": "La-Liga", "country": "esp"},
    "I1": {"id": "11", "name": "Serie-A", "country": "ita"},
    "F1": {"id": "13", "name": "Ligue-1", "country": "fra"},
    "N1": {"id": "23", "name": "Eredivisie", "country": "ned"},
    "P1": {"id": "32", "name": "Primeira-Liga", "country": "por"},
}

# Understat league names
UNDERSTAT_LEAGUES = {
    "E0": "EPL",
    "D1": "Bundesliga", 
    "SP1": "La_liga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}


@dataclass
class XGMatch:
    """Single match with xG data."""
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float
    league_code: str
    season: str


class FBrefScraper:
    """Scrape xG data from FBref (Sports Reference)."""
    
    BASE_URL = "https://fbref.com/en/comps"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    def __init__(self, delay: float = 3.0):
        """
        Initialize scraper with rate limiting.
        
        Args:
            delay: Seconds to wait between requests (FBref has strict rate limits)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def get_season_matches(
        self, 
        league_code: str, 
        season: str,
    ) -> List[XGMatch]:
        """
        Fetch all matches with xG for a league season.
        
        Args:
            league_code: Internal league code (E0, D1, etc.)
            season: Season string (e.g., "2024-2025")
        
        Returns:
            List of XGMatch objects
        """
        if league_code not in FBREF_LEAGUES:
            print(f"League {league_code} not supported for xG scraping")
            return []
        
        league_info = FBREF_LEAGUES[league_code]
        url = f"{self.BASE_URL}/{league_info['id']}/{season}/schedule/{season}-{league_info['name']}-Scores-and-Fixtures"
        
        print(f"Fetching xG data from: {url}")
        
        try:
            time.sleep(self.delay)
            response = self.session.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []
        
        return self._parse_matches_page(response.text, league_code, season)
    
    def _parse_matches_page(
        self, 
        html: str, 
        league_code: str, 
        season: str,
    ) -> List[XGMatch]:
        """Parse FBref schedule page for xG data."""
        soup = BeautifulSoup(html, "html.parser")
        matches = []
        
        # Find the scores-fixtures table
        table = soup.find("table", {"id": re.compile(r"sched_\d+_\d+")})
        if not table:
            # Try alternative table ID
            table = soup.find("table", class_="stats_table")
        
        if not table:
            print("Could not find matches table")
            return []
        
        tbody = table.find("tbody")
        if not tbody:
            return []
        
        for row in tbody.find_all("tr"):
            # Skip header rows
            if row.get("class") and "thead" in row.get("class", []):
                continue
            
            cells = row.find_all(["td", "th"])
            if len(cells) < 8:
                continue
            
            try:
                match = self._parse_match_row(cells, league_code, season)
                if match:
                    matches.append(match)
            except Exception as e:
                continue
        
        print(f"Parsed {len(matches)} matches with xG data")
        return matches
    
    def _parse_match_row(
        self, 
        cells: list, 
        league_code: str, 
        season: str,
    ) -> Optional[XGMatch]:
        """Parse a single match row."""
        # FBref table structure varies, find columns by data-stat attribute
        data = {}
        for cell in cells:
            stat = cell.get("data-stat", "")
            data[stat] = cell.get_text(strip=True)
        
        # Required fields
        date = data.get("date", "")
        home_team = data.get("home_team", "") or data.get("squad_a", "")
        away_team = data.get("away_team", "") or data.get("squad_b", "")
        
        # Score (format: "2-1" or "2–1")
        score = data.get("score", "")
        if not score or score == "":
            return None  # Match not played yet
        
        score_match = re.match(r"(\d+)[–-](\d+)", score)
        if not score_match:
            return None
        
        home_goals = int(score_match.group(1))
        away_goals = int(score_match.group(2))
        
        # xG values
        home_xg_str = data.get("home_xg", "") or data.get("xg_a", "")
        away_xg_str = data.get("away_xg", "") or data.get("xg_b", "")
        
        try:
            home_xg = float(home_xg_str) if home_xg_str else None
            away_xg = float(away_xg_str) if away_xg_str else None
        except ValueError:
            home_xg = None
            away_xg = None
        
        if home_xg is None or away_xg is None:
            return None  # No xG data available
        
        if not all([date, home_team, away_team]):
            return None
        
        return XGMatch(
            date=date,
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            home_xg=home_xg,
            away_xg=away_xg,
            league_code=league_code,
            season=season,
        )


class UnderstatScraper:
    """Scrape xG data from Understat (JSON API)."""
    
    BASE_URL = "https://understat.com/league"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def get_season_matches(
        self, 
        league_code: str, 
        season: str,
    ) -> List[XGMatch]:
        """
        Fetch all matches with xG for a league season from Understat.
        
        Args:
            league_code: Internal league code (E0, D1, etc.)
            season: Season year (e.g., "2024" for 2024-25)
        """
        if league_code not in UNDERSTAT_LEAGUES:
            print(f"League {league_code} not supported on Understat")
            return []
        
        league_name = UNDERSTAT_LEAGUES[league_code]
        url = f"{self.BASE_URL}/{league_name}/{season}"
        
        print(f"Fetching Understat data from: {url}")
        
        try:
            time.sleep(self.delay)
            response = self.session.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []
        
        return self._parse_understat_page(response.text, league_code, season)
    
    def _parse_understat_page(
        self, 
        html: str, 
        league_code: str, 
        season: str,
    ) -> List[XGMatch]:
        """Parse Understat page - data is embedded as JSON in script tags."""
        import json
        
        matches = []
        
        # Find the datesData JSON in script tags
        pattern = r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        
        if not match:
            print("Could not find match data in Understat page")
            return []
        
        try:
            # Decode escaped JSON
            json_str = match.group(1).encode().decode('unicode_escape')
            data = json.loads(json_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error parsing Understat JSON: {e}")
            return []
        
        for match_data in data:
            try:
                xg_match = XGMatch(
                    date=match_data.get("datetime", "")[:10],  # YYYY-MM-DD
                    home_team=match_data.get("h", {}).get("title", ""),
                    away_team=match_data.get("a", {}).get("title", ""),
                    home_goals=int(match_data.get("goals", {}).get("h", 0)),
                    away_goals=int(match_data.get("goals", {}).get("a", 0)),
                    home_xg=float(match_data.get("xG", {}).get("h", 0)),
                    away_xg=float(match_data.get("xG", {}).get("a", 0)),
                    league_code=league_code,
                    season=season,
                )
                matches.append(xg_match)
            except (KeyError, ValueError, TypeError):
                continue
        
        print(f"Parsed {len(matches)} matches from Understat")
        return matches


def matches_to_dataframe(matches: List[XGMatch]) -> pd.DataFrame:
    """Convert list of XGMatch to DataFrame."""
    if not matches:
        return pd.DataFrame()
    
    records = [
        {
            "Date": m.date,
            "HomeTeam": m.home_team,
            "AwayTeam": m.away_team,
            "FTHG": m.home_goals,
            "FTAG": m.away_goals,
            "HomeXG": m.home_xg,
            "AwayXG": m.away_xg,
            "league_code": m.league_code,
            "season": m.season,
        }
        for m in matches
    ]
    
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    
    # Derived xG features
    df["TotalXG"] = df["HomeXG"] + df["AwayXG"]
    df["XGDiff"] = df["HomeXG"] - df["AwayXG"]
    
    return df


def fetch_xg_data(
    leagues: List[str] = None,
    seasons: List[str] = None,
    source: str = "understat",
    output_dir: Path = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Main function to fetch xG data for specified leagues and seasons.
    
    Uses smart caching: past seasons are only fetched once, current season
    is always refreshed to get latest matches.
    
    Args:
        leagues: List of league codes (default: top 5 leagues)
        seasons: List of seasons (default: last 3 seasons)
        source: "fbref" or "understat"
        output_dir: Directory to save CSV files
        force_refresh: If True, re-fetch all seasons regardless of cache
    
    Returns:
        Combined DataFrame with all xG data
    """
    from datetime import datetime
    
    if leagues is None:
        leagues = ["E0", "D1", "SP1", "I1", "F1"]
    
    if seasons is None:
        # Default to recent seasons
        if source == "understat":
            seasons = ["2022", "2023", "2024"]
        else:
            seasons = ["2022-2023", "2023-2024", "2024-2025"]
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "xg"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing cached data
    cache_path = output_dir / f"xg_data_{source}.csv"
    cached_df = None
    cached_seasons = set()
    
    if cache_path.exists() and not force_refresh:
        try:
            cached_df = pd.read_csv(cache_path, parse_dates=["Date"])
            cached_seasons = set(cached_df["season"].astype(str).unique())
            print(f"Found cached xG data with seasons: {sorted(cached_seasons)}")
        except Exception as e:
            print(f"Warning: Could not read cache ({e}), will fetch fresh data")
            cached_df = None
    
    # Determine current season (for Understat, this is the year the season started)
    # Football seasons run Aug-May, so:
    # - In Jan-July 2025, current season is "2024" (2024-25 season)
    # - In Aug-Dec 2025, current season is "2025" (2025-26 season)
    now = datetime.now()
    if source == "understat":
        if now.month <= 7:
            current_season = str(now.year - 1)
        else:
            current_season = str(now.year)
    else:
        # FBref uses "2024-2025" format
        if now.month <= 7:
            current_season = f"{now.year - 1}-{now.year}"
        else:
            current_season = f"{now.year}-{now.year + 1}"
    
    print(f"Current season: {current_season}")
    
    # Determine which seasons need fetching
    # - Past/completed seasons: use cache if available
    # - Current season: always re-fetch to get latest matches
    seasons_to_fetch = []
    for season in seasons:
        if str(season) == current_season:
            # Always fetch current season
            seasons_to_fetch.append(season)
            print(f"  {season}: Will fetch (current season)")
        elif str(season) in cached_seasons and not force_refresh:
            # Use cached data for past seasons
            print(f"  {season}: Using cached data")
        else:
            # Need to fetch this past season
            seasons_to_fetch.append(season)
            print(f"  {season}: Will fetch (not in cache)")
    
    # Fetch the seasons we need
    new_matches = []
    if seasons_to_fetch:
        # Select scraper
        if source == "understat":
            scraper = UnderstatScraper(delay=2.0)
        else:
            scraper = FBrefScraper(delay=4.0)  # FBref needs longer delays
        
        for league in leagues:
            for season in seasons_to_fetch:
                print(f"\nFetching {league} {season}...")
                matches = scraper.get_season_matches(league, season)
                new_matches.extend(matches)
    
    # Convert new matches to dataframe
    new_df = matches_to_dataframe(new_matches) if new_matches else pd.DataFrame()
    
    # Combine with cached data (excluding re-fetched seasons)
    if cached_df is not None and not cached_df.empty:
        # Keep cached data for seasons we didn't re-fetch
        seasons_to_keep = [s for s in cached_seasons if s not in [str(x) for x in seasons_to_fetch]]
        if seasons_to_keep:
            cached_to_keep = cached_df[cached_df["season"].astype(str).isin(seasons_to_keep)]
            print(f"\nKeeping {len(cached_to_keep)} cached matches from seasons: {seasons_to_keep}")
            
            if not new_df.empty:
                df = pd.concat([cached_to_keep, new_df], ignore_index=True)
            else:
                df = cached_to_keep
        else:
            df = new_df
    else:
        df = new_df
    
    if df.empty:
        print("No xG data available")
        return pd.DataFrame()
    
    # Sort and clean up
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Remove duplicates (in case of overlap)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
    
    # Save combined data to cache
    df.to_csv(cache_path, index=False)
    print(f"\nSaved {len(df)} matches to {cache_path}")
    
    # Summary by season
    print("\nxG data by season:")
    for season in sorted(df["season"].astype(str).unique()):
        count = len(df[df["season"].astype(str) == season])
        print(f"  {season}: {count} matches")
    
    return df


def merge_xg_with_matches(
    match_df: pd.DataFrame,
    xg_df: pd.DataFrame,
    team_mapping: Dict[str, str] = None,
) -> pd.DataFrame:
    """
    Merge xG data with existing match dataset.
    
    Handles team name differences between sources using fuzzy matching
    or explicit mapping.
    """
    if xg_df.empty:
        return match_df
    
    # Standardize date formats
    match_df = match_df.copy()
    xg_df = xg_df.copy()
    
    match_df["Date"] = pd.to_datetime(match_df["Date"])
    xg_df["Date"] = pd.to_datetime(xg_df["Date"])
    
    # Create merge key
    match_df["_merge_key"] = (
        match_df["Date"].dt.strftime("%Y-%m-%d") + "_" +
        match_df["HomeTeam"].str.lower().str.strip() + "_" +
        match_df["AwayTeam"].str.lower().str.strip()
    )
    
    xg_df["_merge_key"] = (
        xg_df["Date"].dt.strftime("%Y-%m-%d") + "_" +
        xg_df["HomeTeam"].str.lower().str.strip() + "_" +
        xg_df["AwayTeam"].str.lower().str.strip()
    )
    
    # Merge on key
    xg_cols = ["_merge_key", "HomeXG", "AwayXG", "TotalXG", "XGDiff"]
    merged = match_df.merge(
        xg_df[xg_cols],
        on="_merge_key",
        how="left",
    )
    
    merged = merged.drop(columns=["_merge_key"])
    
    matched = merged["HomeXG"].notna().sum()
    total = len(merged)
    print(f"Matched {matched}/{total} ({100*matched/total:.1f}%) matches with xG data")
    
    return merged


# Common team name mappings between football-data.co.uk and xG sources
TEAM_NAME_MAPPING = {
    # Premier League
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield Utd",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    # Bundesliga
    "Bayern Munich": "Bayern München",
    "Leverkusen": "Bayer Leverkusen",
    "Dortmund": "Borussia Dortmund",
    "M'gladbach": "Borussia Mönchengladbach",
    "FC Koln": "FC Köln",
    # La Liga
    "Atletico Madrid": "Atlético Madrid",
    "Betis": "Real Betis",
    # Serie A
    "Inter": "Inter Milan",
    # Ligue 1
    "Paris S-G": "Paris Saint-Germain",
}


def main():
    """Example usage of xG scraper."""
    print("=" * 60)
    print("xG Data Scraper")
    print("=" * 60)
    
    # Fetch from Understat (more reliable, fewer rate limits)
    df = fetch_xg_data(
        leagues=["E0", "D1", "SP1", "I1", "F1"],
        seasons=["2023", "2024"],
        source="understat",
    )
    
    if not df.empty:
        print("\nSample data:")
        print(df.head(10).to_string())
        print(f"\nTotal matches: {len(df)}")
        print(f"Average HomeXG: {df['HomeXG'].mean():.2f}")
        print(f"Average AwayXG: {df['AwayXG'].mean():.2f}")
        print(f"Average TotalXG: {df['TotalXG'].mean():.2f}")


if __name__ == "__main__":
    main()
