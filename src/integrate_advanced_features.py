"""
Advanced Features Integration Module

This module integrates scraped advanced features (FBref stats, weather, injuries, manager data)
into the main match dataset for improved model predictions.

Usage:
    # From command line
    python -m src.integrate_advanced_features --all
    
    # From code
    from src.integrate_advanced_features import integrate_all_features
    df = integrate_all_features(match_df)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig, get_config


# =============================================================================
# FEATURE LOADING FUNCTIONS
# =============================================================================

def load_advanced_stats(cfg: AppConfig) -> Optional[pd.DataFrame]:
    """Load FBref advanced team statistics."""
    path = cfg.processed_dir / "advanced_team_stats.csv"
    if not path.exists():
        print(f"[INFO] Advanced stats not found at {path}. Skipping.")
        return None
    
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} advanced stat records from {path.name}")
    return df


def load_weather_data(cfg: AppConfig) -> Optional[pd.DataFrame]:
    """Load weather data for matches."""
    path = cfg.processed_dir / "weather_data.csv"
    if not path.exists():
        print(f"[INFO] Weather data not found at {path}. Skipping.")
        return None
    
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    print(f"[INFO] Loaded {len(df)} weather records from {path.name}")
    return df


def load_injury_data(cfg: AppConfig) -> Optional[pd.DataFrame]:
    """Load player injury/suspension data."""
    path = cfg.processed_dir / "injury_data.csv"
    if not path.exists():
        print(f"[INFO] Injury data not found at {path}. Skipping.")
        return None
    
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    print(f"[INFO] Loaded {len(df)} injury records from {path.name}")
    return df


def load_manager_data(cfg: AppConfig) -> Optional[pd.DataFrame]:
    """Load manager tenure data."""
    path = cfg.processed_dir / "manager_data.csv"
    if not path.exists():
        print(f"[INFO] Manager data not found at {path}. Skipping.")
        return None
    
    df = pd.read_csv(path)
    if "scrape_date" in df.columns:
        df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    print(f"[INFO] Loaded {len(df)} manager records from {path.name}")
    return df


# =============================================================================
# TEAM NAME NORMALIZATION
# =============================================================================

# Map from various sources to canonical names used in match_dataset
TEAM_NAME_NORMALIZER: Dict[str, str] = {
    # Premier League
    "Manchester Utd": "Man United",
    "Manchester City": "Man City",
    "Newcastle Utd": "Newcastle",
    "Tottenham": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Nott'ham Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Sheffield Utd": "Sheffield United",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Aston Villa": "Aston Villa",
    
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alavés": "Alaves",
    "Celta Vigo": "Celta",
    "Real Sociedad": "Sociedad",
    
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M'gladbach": "M'gladbach",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "SC Freiburg": "Freiburg",
    "1. FSV Mainz 05": "Mainz",
    "TSG Hoffenheim": "Hoffenheim",
    "1. FC Köln": "Koln",
    "1. FC Union Berlin": "Union Berlin",
    "FC Augsburg": "Augsburg",
    "Werder Bremen": "Werder Bremen",
    "VfL Bochum 1848": "Bochum",
    
    # Serie A
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "SSC Napoli": "Napoli",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Hellas Verona": "Verona",
    
    # Ligue 1
    "Paris Saint-Germain": "Paris SG",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille",
    "AS Monaco": "Monaco",
    "OGC Nice": "Nice",
    "Stade Rennais": "Rennes",
    "RC Lens": "Lens",
    "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes",
    "Stade Brestois 29": "Brest",
    "RC Strasbourg": "Strasbourg",
    "Toulouse FC": "Toulouse",
    "Le Havre AC": "Le Havre",
    "Stade de Reims": "Reims",
    "FC Lorient": "Lorient",
    "FC Metz": "Metz",
    "Clermont Foot": "Clermont",
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to match dataset conventions."""
    return TEAM_NAME_NORMALIZER.get(name, name)


# =============================================================================
# ADVANCED STATS INTEGRATION
# =============================================================================

def merge_advanced_stats(
    match_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Merge FBref advanced stats into match dataset.
    
    The stats are season-level aggregates (e.g., possession %, xG, etc.)
    that represent the team's current season profile.
    """
    if stats_df is None or stats_df.empty:
        return match_df
    
    # Normalize team names in stats
    stats_df = stats_df.copy()
    stats_df["team"] = stats_df["team"].apply(normalize_team_name)
    
    # Define which columns to use as features
    stat_columns = [
        "possession_pct", "progressive_passes", "progressive_carries",
        "passes_into_final_third", "xg", "npxg", "xg_per_shot",
        "shots_on_target_pct", "pass_completion_pct"
    ]
    
    # Filter to only existing columns
    available_stats = [c for c in stat_columns if c in stats_df.columns]
    
    if not available_stats:
        print("[WARN] No usable stat columns found in advanced stats")
        return match_df
    
    print(f"[INFO] Using advanced stat columns: {available_stats}")
    
    result_df = match_df.copy()
    
    # Create a lookup by team name
    stats_lookup = stats_df.set_index("team")[available_stats].to_dict("index")
    
    # Add features for home and away teams
    for prefix, team_col in [("home", "HomeTeam"), ("away", "AwayTeam")]:
        for stat in available_stats:
            col_name = f"{prefix}_adv_{stat}"
            result_df[col_name] = result_df[team_col].apply(
                lambda t, s=stat: stats_lookup.get(t, {}).get(s, np.nan)
            )
    
    # Create differential features
    for stat in available_stats:
        result_df[f"adv_{stat}_diff"] = (
            result_df[f"home_adv_{stat}"] - result_df[f"away_adv_{stat}"]
        )
    
    # Count how many matches got stats
    home_has_stats = result_df["home_adv_possession_pct"].notna().sum()
    away_has_stats = result_df["away_adv_possession_pct"].notna().sum()
    print(f"[INFO] Matched stats: {home_has_stats} home, {away_has_stats} away")
    
    return result_df


# =============================================================================
# WEATHER DATA INTEGRATION
# =============================================================================

def merge_weather_data(
    match_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge weather data into match dataset.
    
    Uses current weather data for each team's stadium.
    Weather is matched by home team name.
    """
    if weather_df is None or weather_df.empty:
        return match_df
    
    result_df = match_df.copy()
    weather_df = weather_df.copy()
    weather_df["team"] = weather_df["team"].apply(normalize_team_name)
    
    # Create lookup by team
    weather_lookup = weather_df.set_index("team").to_dict("index")
    
    def get_weather_for_team(team: str) -> Dict:
        """Get weather for a team's venue."""
        data = weather_lookup.get(team, {})
        if not data:
            return {
                "weather_temp": np.nan,
                "weather_wind_speed": np.nan,
                "weather_is_cold": np.nan,
                "weather_is_hot": np.nan,
            }
        
        temp = data.get("temperature", 15)
        return {
            "weather_temp": temp,
            "weather_wind_speed": data.get("wind_speed", np.nan),
            "weather_is_cold": 1 if temp < 5 else 0,
            "weather_is_hot": 1 if temp > 30 else 0,
        }
    
    print("[INFO] Merging weather data...")
    for col in ["weather_temp", "weather_wind_speed", "weather_is_cold", "weather_is_hot"]:
        result_df[col] = result_df["HomeTeam"].apply(
            lambda t: get_weather_for_team(t).get(col, np.nan)
        )
    
    matched = result_df["weather_temp"].notna().sum()
    print(f"[INFO] Matched weather for {matched} matches")
    
    return result_df


# =============================================================================
# INJURY DATA INTEGRATION
# =============================================================================

def merge_injury_data(
    match_df: pd.DataFrame,
    injury_df: pd.DataFrame,
    lookback_days: int = 7
) -> pd.DataFrame:
    """
    Merge injury/suspension data into match dataset.
    
    Creates features for:
    - Number of injured players (by severity)
    - Total injury severity score
    - Key player injuries flag
    """
    if injury_df is None or injury_df.empty:
        return match_df
    
    result_df = match_df.copy()
    injury_df = injury_df.copy()
    injury_df["team"] = injury_df["team"].apply(normalize_team_name)
    
    # Severity weights
    SEVERITY_WEIGHTS = {
        "minor": 1,
        "moderate": 2,
        "major": 4,
        "long-term": 5,
        "unknown": 2
    }
    
    def count_injuries(row: pd.Series, team_col: str, prefix: str) -> Dict:
        """Count active injuries for a team before match date."""
        team = row[team_col]
        match_date = row["Date"]
        
        # Get injuries active around match date
        # (injured before match, expected return after match or unknown)
        active_injuries = injury_df[
            (injury_df["team"] == team) &
            (injury_df["date"] <= match_date) &
            (injury_df["date"] >= match_date - timedelta(days=lookback_days))
        ]
        
        if active_injuries.empty:
            return {
                f"{prefix}_injury_count": 0,
                f"{prefix}_injury_severity": 0,
                f"{prefix}_suspended_count": 0
            }
        
        # Count injuries vs suspensions
        injuries = active_injuries[active_injuries["injury_type"] != "suspension"]
        suspensions = active_injuries[active_injuries["injury_type"] == "suspension"]
        
        # Calculate severity score
        severity_score = sum(
            SEVERITY_WEIGHTS.get(row.get("severity", "unknown"), 2)
            for _, row in injuries.iterrows()
        )
        
        return {
            f"{prefix}_injury_count": len(injuries),
            f"{prefix}_injury_severity": severity_score,
            f"{prefix}_suspended_count": len(suspensions)
        }
    
    print("[INFO] Merging injury data...")
    home_injuries = result_df.apply(
        lambda row: pd.Series(count_injuries(row, "HomeTeam", "home")),
        axis=1
    )
    away_injuries = result_df.apply(
        lambda row: pd.Series(count_injuries(row, "AwayTeam", "away")),
        axis=1
    )
    
    result_df = pd.concat([result_df, home_injuries, away_injuries], axis=1)
    
    # Add differential features
    result_df["injury_count_diff"] = (
        result_df.get("home_injury_count", 0) - result_df.get("away_injury_count", 0)
    )
    result_df["injury_severity_diff"] = (
        result_df.get("home_injury_severity", 0) - result_df.get("away_injury_severity", 0)
    )
    
    return result_df


# =============================================================================
# MANAGER DATA INTEGRATION
# =============================================================================

def merge_manager_data(
    match_df: pd.DataFrame,
    manager_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge manager data into match dataset.
    
    Creates features for:
    - Manager tenure (days)
    - New manager flag (< 30 days)
    - Experienced manager flag (> 365 days)
    """
    if manager_df is None or manager_df.empty:
        return match_df
    
    result_df = match_df.copy()
    manager_df = manager_df.copy()
    manager_df["team"] = manager_df["team"].apply(normalize_team_name)
    
    def get_manager_features(row: pd.Series, team_col: str, prefix: str) -> Dict:
        """Get manager features for a team."""
        team = row[team_col]
        match_date = row["Date"]
        
        # Get most recent manager info before match
        team_managers = manager_df[
            (manager_df["team"] == team) &
            (manager_df["scrape_date"] <= match_date)
        ].sort_values("scrape_date", ascending=False)
        
        if team_managers.empty:
            return {
                f"{prefix}_manager_tenure_days": np.nan,
                f"{prefix}_new_manager": np.nan,
                f"{prefix}_experienced_manager": np.nan
            }
        
        latest = team_managers.iloc[0]
        tenure_days = latest.get("tenure_days", np.nan)
        
        return {
            f"{prefix}_manager_tenure_days": tenure_days,
            f"{prefix}_new_manager": 1 if tenure_days is not np.nan and tenure_days < 30 else 0,
            f"{prefix}_experienced_manager": 1 if tenure_days is not np.nan and tenure_days > 365 else 0
        }
    
    print("[INFO] Merging manager data...")
    home_manager = result_df.apply(
        lambda row: pd.Series(get_manager_features(row, "HomeTeam", "home")),
        axis=1
    )
    away_manager = result_df.apply(
        lambda row: pd.Series(get_manager_features(row, "AwayTeam", "away")),
        axis=1
    )
    
    result_df = pd.concat([result_df, home_manager, away_manager], axis=1)
    
    # Add differential features
    result_df["manager_tenure_diff"] = (
        result_df.get("home_manager_tenure_days", 0) - 
        result_df.get("away_manager_tenure_days", 0)
    )
    
    return result_df


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def integrate_all_features(
    match_df: pd.DataFrame,
    cfg: Optional[AppConfig] = None,
    include_advanced: bool = True,
    include_weather: bool = True,
    include_injuries: bool = True,
    include_manager: bool = True
) -> pd.DataFrame:
    """
    Integrate all available advanced features into the match dataset.
    
    Parameters
    ----------
    match_df : pd.DataFrame
        The base match dataset
    cfg : AppConfig, optional
        App configuration. If None, uses default config.
    include_advanced : bool
        Whether to include FBref advanced stats
    include_weather : bool
        Whether to include weather data
    include_injuries : bool
        Whether to include injury data
    include_manager : bool
        Whether to include manager data
        
    Returns
    -------
    pd.DataFrame
        Match dataset with all available advanced features merged
    """
    if cfg is None:
        cfg = get_config()
    
    result_df = match_df.copy()
    original_cols = len(result_df.columns)
    
    print(f"\n{'='*60}")
    print("INTEGRATING ADVANCED FEATURES")
    print(f"{'='*60}")
    print(f"Starting with {len(result_df)} matches, {original_cols} columns")
    
    # Ensure Date is datetime
    if "Date" in result_df.columns:
        result_df["Date"] = pd.to_datetime(result_df["Date"])
    
    # 1. Advanced Stats (FBref)
    if include_advanced:
        advanced_stats = load_advanced_stats(cfg)
        if advanced_stats is not None:
            result_df = merge_advanced_stats(
                result_df, 
                advanced_stats,
                windows=cfg.model.rolling_windows
            )
    
    # 2. Weather Data
    if include_weather:
        weather_data = load_weather_data(cfg)
        if weather_data is not None:
            result_df = merge_weather_data(result_df, weather_data)
    
    # 3. Injury Data
    if include_injuries:
        injury_data = load_injury_data(cfg)
        if injury_data is not None:
            result_df = merge_injury_data(result_df, injury_data)
    
    # 4. Manager Data
    if include_manager:
        manager_data = load_manager_data(cfg)
        if manager_data is not None:
            result_df = merge_manager_data(result_df, manager_data)
    
    new_cols = len(result_df.columns) - original_cols
    print(f"\n{'='*60}")
    print(f"INTEGRATION COMPLETE")
    print(f"Added {new_cols} new feature columns")
    print(f"Final dataset: {len(result_df)} matches, {len(result_df.columns)} columns")
    print(f"{'='*60}\n")
    
    return result_df


def create_enhanced_dataset(cfg: Optional[AppConfig] = None) -> pd.DataFrame:
    """
    Create enhanced dataset with all advanced features.
    Loads existing match_dataset.csv and enhances it.
    """
    if cfg is None:
        cfg = get_config()
    
    # Load base dataset
    base_path = cfg.processed_dir / "match_dataset.csv"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base dataset not found at {base_path}. "
            "Run prepare_dataset first."
        )
    
    print(f"[INFO] Loading base dataset from {base_path}")
    match_df = pd.read_csv(base_path, parse_dates=["Date"])
    
    # Integrate all features
    enhanced_df = integrate_all_features(match_df, cfg)
    
    # Save enhanced dataset
    output_path = cfg.processed_dir / "match_dataset_enhanced.csv"
    enhanced_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved enhanced dataset to {output_path}")
    
    return enhanced_df


def main():
    """Command-line interface for feature integration."""
    parser = argparse.ArgumentParser(
        description="Integrate advanced features into match dataset"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Create enhanced dataset with all available features"
    )
    parser.add_argument(
        "--no-advanced",
        action="store_true",
        help="Exclude FBref advanced stats"
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Exclude weather data"
    )
    parser.add_argument(
        "--no-injuries",
        action="store_true",
        help="Exclude injury data"
    )
    parser.add_argument(
        "--no-manager",
        action="store_true",
        help="Exclude manager data"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Create enhanced dataset
        create_enhanced_dataset()
    else:
        print("Use --all to create enhanced dataset with available features")
        print("\nAvailable options:")
        print("  --no-advanced   Exclude FBref advanced stats")
        print("  --no-weather    Exclude weather data")
        print("  --no-injuries   Exclude injury data")
        print("  --no-manager    Exclude manager data")


if __name__ == "__main__":
    main()
