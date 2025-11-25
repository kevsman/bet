"""
Fetch European competition data from API-Football.
Requires API key set in environment variable: API_FOOTBALL_KEY

Free tier: 100 requests/day
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .european_config import EuropeanConfig, get_european_config

API_BASE_URL = "https://v3.football.api-sports.io"
RATE_LIMIT_DELAY = 1.0  # Seconds between requests to respect rate limits


def make_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Make a request to API-Football."""
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": api_key,
    }
    url = f"{API_BASE_URL}/{endpoint}"
    
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    if data.get("errors"):
        raise ValueError(f"API Error: {data['errors']}")
    
    return data


def fetch_competition_fixtures(
    competition_id: int,
    season: int,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Fetch all fixtures for a competition and season."""
    params = {
        "league": competition_id,
        "season": season,
        "status": "FT",  # Only finished matches
    }
    
    data = make_api_request("fixtures", params, api_key)
    return data.get("response", [])


def parse_fixture(fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a fixture into a flat record."""
    try:
        match_info = fixture.get("fixture", {})
        league_info = fixture.get("league", {})
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        score = fixture.get("score", {})
        
        # Skip matches without scores
        if goals.get("home") is None or goals.get("away") is None:
            return None
        
        return {
            "match_id": match_info.get("id"),
            "Date": match_info.get("date", "")[:10],  # YYYY-MM-DD
            "competition_code": league_info.get("name", ""),
            "competition_id": league_info.get("id"),
            "season": league_info.get("season"),
            "round": league_info.get("round", ""),
            "HomeTeam": teams.get("home", {}).get("name"),
            "AwayTeam": teams.get("away", {}).get("name"),
            "home_team_id": teams.get("home", {}).get("id"),
            "away_team_id": teams.get("away", {}).get("id"),
            "FTHG": goals.get("home"),
            "FTAG": goals.get("away"),
            "HT_home": score.get("halftime", {}).get("home"),
            "HT_away": score.get("halftime", {}).get("away"),
            "venue": match_info.get("venue", {}).get("name"),
            "is_neutral": "Final" in league_info.get("round", ""),
        }
    except Exception as e:
        print(f"Error parsing fixture: {e}")
        return None


def fetch_team_domestic_form(
    team_id: int,
    season: int,
    api_key: str,
    last_n: int = 5,
) -> Dict[str, float]:
    """
    Fetch a team's recent domestic league form.
    This is crucial for European predictions - use domestic form as baseline.
    """
    params = {
        "team": team_id,
        "season": season,
        "last": last_n,
    }
    
    try:
        data = make_api_request("fixtures", params, api_key)
        fixtures = data.get("response", [])
        
        goals_for = []
        goals_against = []
        
        for fix in fixtures:
            teams = fix.get("teams", {})
            goals = fix.get("goals", {})
            
            if goals.get("home") is None:
                continue
            
            if teams.get("home", {}).get("id") == team_id:
                goals_for.append(goals.get("home", 0))
                goals_against.append(goals.get("away", 0))
            else:
                goals_for.append(goals.get("away", 0))
                goals_against.append(goals.get("home", 0))
        
        if not goals_for:
            return {"avg_goals_for": 1.5, "avg_goals_against": 1.2}
        
        return {
            "avg_goals_for": sum(goals_for) / len(goals_for),
            "avg_goals_against": sum(goals_against) / len(goals_against),
        }
    except Exception:
        return {"avg_goals_for": 1.5, "avg_goals_against": 1.2}


def download_european_data(cfg: EuropeanConfig) -> None:
    """Download all European competition data."""
    if not cfg.api_key:
        print("ERROR: API_FOOTBALL_KEY environment variable not set.")
        print("Get a free API key at: https://www.api-football.com/")
        print("Then set: $env:API_FOOTBALL_KEY = 'your-key-here'")
        return
    
    all_matches = []
    
    for comp_code, comp in cfg.competitions.items():
        print(f"\nFetching {comp.name}...")
        
        for season in cfg.seasons:
            print(f"  Season {season}...", end=" ")
            
            try:
                fixtures = fetch_competition_fixtures(comp.api_id, season, cfg.api_key)
                parsed = [parse_fixture(f) for f in fixtures]
                parsed = [p for p in parsed if p is not None]
                
                for match in parsed:
                    match["competition_code"] = comp_code
                
                all_matches.extend(parsed)
                print(f"{len(parsed)} matches")
                
                time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                
            except Exception as e:
                print(f"Error: {e}")
    
    if all_matches:
        df = pd.DataFrame(all_matches)
        output_path = cfg.raw_dir / "european_matches.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} matches to {output_path}")
    else:
        print("\nNo matches downloaded.")


def load_cached_data(cfg: EuropeanConfig) -> Optional[pd.DataFrame]:
    """Load cached European data if available."""
    cache_path = cfg.raw_dir / "european_matches.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["Date"])
    return None


def main() -> None:
    cfg = get_european_config()
    download_european_data(cfg)


if __name__ == "__main__":
    main()
