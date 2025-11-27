"""
Prepare dataset for European competition model.
Handles cross-competition team tracking and competition-specific features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .european_config import EuropeanConfig, get_european_config


def load_european_data(cfg: EuropeanConfig) -> pd.DataFrame:
    """Load European match data.
    
    Prefers openfootball data (combined with Wikipedia UEL/UECL) if available,
    falls back to FBref scraped data.
    """
    # Prefer openfootball data (includes Wikipedia UEL/UECL)
    openfootball_path = cfg.raw_dir / "openfootball_european.csv"
    if openfootball_path.exists():
        print(f"Loading openfootball data from {openfootball_path}")
        df = pd.read_csv(openfootball_path, low_memory=False)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        
        # Standardize column names if needed
        if "competition_code" in df.columns and "competition" not in df.columns:
            df["competition"] = df["competition_code"]
        
        # Ensure required columns exist
        required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        if not all(c in df.columns for c in required_cols):
            print(f"  Missing columns in openfootball data, falling back to FBref")
        else:
            df = df.sort_values("Date").reset_index(drop=True)
            print(f"  Loaded {len(df)} matches from openfootball + Wikipedia")
            return df
    
    # Fallback to FBref scraped data
    data_path = cfg.raw_dir / "european_matches.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"European data not found at {data_path}. Run european_fetch first."
        )
    
    print(f"Loading FBref data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_domestic_data(cfg: EuropeanConfig) -> Optional[pd.DataFrame]:
    """
    Load domestic league data to supplement European team features.
    Teams' domestic form is a strong predictor in European matches.
    """
    domestic_path = cfg.base_dir / "data" / "processed" / "match_dataset.csv"
    if domestic_path.exists():
        return pd.read_csv(domestic_path, parse_dates=["Date"], low_memory=False)
    return None


def get_domestic_form_for_team(
    team_name: str,
    match_date: pd.Timestamp,
    domestic_df: pd.DataFrame,
    n_games: int = 5,
) -> Dict[str, float]:
    """
    Get a team's domestic form leading up to a European match.
    Returns rolling averages from their domestic league.
    """
    # Find team's domestic matches before this date
    home_matches = domestic_df[
        (domestic_df["HomeTeam"] == team_name) & 
        (domestic_df["Date"] < match_date)
    ].copy()
    away_matches = domestic_df[
        (domestic_df["AwayTeam"] == team_name) & 
        (domestic_df["Date"] < match_date)
    ].copy()
    
    # Combine and sort
    home_matches["goals_for"] = home_matches["FTHG"]
    home_matches["goals_against"] = home_matches["FTAG"]
    home_matches["is_home"] = True
    
    away_matches["goals_for"] = away_matches["FTAG"]
    away_matches["goals_against"] = away_matches["FTHG"]
    away_matches["is_home"] = False
    
    all_matches = pd.concat([
        home_matches[["Date", "goals_for", "goals_against", "is_home"]],
        away_matches[["Date", "goals_for", "goals_against", "is_home"]],
    ]).sort_values("Date")
    
    if len(all_matches) < 3:
        return {}  # Not enough domestic data
    
    # Get last n games
    recent = all_matches.tail(n_games)
    
    return {
        "dom_avg_for": recent["goals_for"].mean(),
        "dom_avg_against": recent["goals_against"].mean(),
        "dom_total_games": len(all_matches),
        "dom_home_pct": recent["is_home"].mean(),
        "dom_ema_for": recent["goals_for"].ewm(span=n_games).mean().iloc[-1],
        "dom_ema_against": recent["goals_against"].ewm(span=n_games).mean().iloc[-1],
    }


def add_domestic_features(
    df: pd.DataFrame,
    domestic_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Add domestic form features for both home and away teams.
    """
    if domestic_df is None or domestic_df.empty:
        print("No domestic data available - skipping domestic features")
        return df
    
    print("Adding domestic form features...")
    
    # Create team name mapping for common variations
    # European names often differ from domestic (e.g., "Bayern Munich" vs "Bayern")
    team_mappings = {
        # German
        "Bayern Munich": ["Bayern Munich", "Bayern"],
        "Borussia Dortmund": ["Borussia Dortmund", "Dortmund"],
        "RB Leipzig": ["RB Leipzig", "Leipzig"],
        "Bayer Leverkusen": ["Bayer Leverkusen", "Leverkusen"],
        # English
        "Manchester City": ["Manchester City", "Man City"],
        "Manchester United": ["Manchester United", "Man United"],
        "Tottenham": ["Tottenham", "Tottenham Hotspur", "Spurs"],
        # Spanish
        "Atletico Madrid": ["Atletico Madrid", "Ath Madrid"],
        "Real Sociedad": ["Real Sociedad", "Sociedad"],
        "Athletic Bilbao": ["Athletic Bilbao", "Ath Bilbao"],
        # Italian
        "AC Milan": ["AC Milan", "Milan"],
        "Inter Milan": ["Inter Milan", "Inter"],
        # French
        "PSG": ["PSG", "Paris SG", "Paris Saint-Germain"],
    }
    
    def find_team_in_domestic(team_name: str, domestic_df: pd.DataFrame) -> Optional[str]:
        """Find the matching team name in domestic data."""
        # Direct match
        all_teams = set(domestic_df["HomeTeam"].unique()) | set(domestic_df["AwayTeam"].unique())
        if team_name in all_teams:
            return team_name
        
        # Check mappings
        for euro_name, variants in team_mappings.items():
            if team_name in variants or team_name == euro_name:
                for v in variants:
                    if v in all_teams:
                        return v
        
        # Fuzzy match - check if team name is contained
        for t in all_teams:
            if team_name.lower() in t.lower() or t.lower() in team_name.lower():
                return t
        
        return None
    
    # Add domestic features for each match
    home_dom_features = []
    away_dom_features = []
    
    for idx, row in df.iterrows():
        match_date = row["Date"]
        
        # Home team domestic form
        home_name = find_team_in_domestic(row["HomeTeam"], domestic_df)
        if home_name:
            home_form = get_domestic_form_for_team(home_name, match_date, domestic_df)
        else:
            home_form = {}
        home_dom_features.append(home_form)
        
        # Away team domestic form
        away_name = find_team_in_domestic(row["AwayTeam"], domestic_df)
        if away_name:
            away_form = get_domestic_form_for_team(away_name, match_date, domestic_df)
        else:
            away_form = {}
        away_dom_features.append(away_form)
    
    # Convert to DataFrames and add prefixes
    home_dom_df = pd.DataFrame(home_dom_features).add_prefix("home_")
    away_dom_df = pd.DataFrame(away_dom_features).add_prefix("away_")
    
    # Merge with main DataFrame
    df = pd.concat([df.reset_index(drop=True), home_dom_df, away_dom_df], axis=1)
    
    # Fill missing domestic features with competition averages
    dom_cols = [c for c in df.columns if "_dom_" in c]
    for col in dom_cols:
        if "for" in col or "avg_for" in col or "ema_for" in col:
            df[col] = df[col].fillna(df["comp_avg_home_goals"])
        elif "against" in col:
            df[col] = df[col].fillna(df["comp_avg_away_goals"])
        else:
            df[col] = df[col].fillna(0)
    
    matched = home_dom_df["home_dom_avg_for"].notna().sum()
    print(f"  Matched {matched}/{len(df)} home teams to domestic data")
    matched = away_dom_df["away_dom_avg_for"].notna().sum()
    print(f"  Matched {matched}/{len(df)} away teams to domestic data")
    
    return df


def extract_round_stage(round_str: str) -> Dict[str, any]:
    """Parse round string into stage features."""
    round_lower = round_str.lower() if round_str else ""
    
    return {
        "is_group_stage": "group" in round_lower,
        "is_knockout": any(k in round_lower for k in ["16", "quarter", "semi", "final"]),
        "is_final": "final" in round_lower and "semi" not in round_lower,
        "stage_importance": (
            1 if "group" in round_lower else
            2 if "16" in round_lower else
            3 if "quarter" in round_lower else
            4 if "semi" in round_lower else
            5 if "final" in round_lower else 1
        ),
    }


def build_team_european_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build team-level European competition history.
    Tracks performance across all European games (not just one competition).
    """
    records = []
    
    for _, row in df.iterrows():
        # Home team record
        records.append({
            "date": row["Date"],
            "team": row["HomeTeam"],
            "team_id": row.get("home_team_id"),
            "is_home": True,
            "goals_for": row["FTHG"],
            "goals_against": row["FTAG"],
            "competition": row["competition_code"],
            "match_id": row.get("match_id"),
        })
        # Away team record
        records.append({
            "date": row["Date"],
            "team": row["AwayTeam"],
            "team_id": row.get("away_team_id"),
            "is_home": False,
            "goals_for": row["FTAG"],
            "goals_against": row["FTHG"],
            "competition": row["competition_code"],
            "match_id": row.get("match_id"),
        })
    
    team_df = pd.DataFrame(records)
    team_df = team_df.sort_values("date").reset_index(drop=True)
    
    # Calculate European-specific metrics
    team_df["european_matches"] = team_df.groupby("team").cumcount()
    team_df["goal_diff"] = team_df["goals_for"] - team_df["goals_against"]
    
    return team_df


def add_european_form_features(
    team_df: pd.DataFrame,
    windows: List[int],
) -> pd.DataFrame:
    """Add rolling form features for European matches."""
    df = team_df.copy()
    
    grouped_for = df.groupby("team")["goals_for"]
    grouped_against = df.groupby("team")["goals_against"]
    
    for window in windows:
        # Rolling averages with smaller windows (fewer European games)
        df[f"euro_avg_for_{window}"] = grouped_for.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"euro_avg_against_{window}"] = grouped_against.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        
        # EMA for recency
        df[f"euro_ema_for_{window}"] = grouped_for.transform(
            lambda s, w=window: s.shift(1).ewm(span=w, min_periods=1).mean()
        )
        df[f"euro_ema_against_{window}"] = grouped_against.transform(
            lambda s, w=window: s.shift(1).ewm(span=w, min_periods=1).mean()
        )
    
    # Total European goals
    df["euro_total_goals_for"] = grouped_for.transform(
        lambda s: s.shift(1).expanding().sum()
    )
    df["euro_total_goals_against"] = grouped_against.transform(
        lambda s: s.shift(1).expanding().sum()
    )
    
    return df


def wide_features(team_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Convert team-level features to wide format for merging."""
    columns_to_keep = ["match_id", "team", "european_matches"] + [
        col for col in team_df.columns
        if col.startswith("euro_avg_") or col.startswith("euro_ema_")
        or col.startswith("euro_total_")
    ]
    
    subset = team_df.loc[:, [c for c in columns_to_keep if c in team_df.columns]]
    renamed = subset.add_prefix(prefix + "_")
    renamed.rename(columns={f"{prefix}_match_id": "match_id"}, inplace=True)
    return renamed


def compute_competition_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute competition-specific average goals.
    UCL typically has more goals than UEL/UECL.
    """
    df = df.copy()
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    
    # Expanding average per competition (no lookahead)
    df["comp_avg_home_goals"] = df.groupby("competition_code")["FTHG"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean()
    )
    df["comp_avg_away_goals"] = df.groupby("competition_code")["FTAG"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean()
    )
    df["comp_avg_total_goals"] = df.groupby("competition_code")["total_goals"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean()
    )
    
    # Fill early matches with overall average
    overall_home = df["FTHG"].mean()
    overall_away = df["FTAG"].mean()
    overall_total = df["total_goals"].mean()
    
    df["comp_avg_home_goals"] = df["comp_avg_home_goals"].fillna(overall_home)
    df["comp_avg_away_goals"] = df["comp_avg_away_goals"].fillna(overall_away)
    df["comp_avg_total_goals"] = df["comp_avg_total_goals"].fillna(overall_total)
    
    return df


def create_european_dataset(cfg: EuropeanConfig) -> pd.DataFrame:
    """Create the full European competition dataset."""
    df = load_european_data(cfg)
    
    # Add match_id if not present
    if "match_id" not in df.columns:
        df["match_id"] = range(1, len(df) + 1)
    
    # Add derived columns
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    df["season_code"] = df["season"].astype(str)
    
    # Parse round/stage features
    stage_features = df["round"].apply(extract_round_stage).apply(pd.Series)
    df = pd.concat([df, stage_features], axis=1)
    
    # Competition averages
    df = compute_competition_averages(df)
    
    # Load and add domestic form features
    domestic_df = load_domestic_data(cfg)
    df = add_domestic_features(df, domestic_df)
    
    # Build team European history
    team_df = build_team_european_history(df)
    team_df = add_european_form_features(team_df, cfg.model.rolling_windows)
    
    # Merge home/away features
    home_features = wide_features(team_df[team_df["is_home"]], "home")
    away_features = wide_features(team_df[team_df["is_home"] == False], "away")
    
    dataset = (
        df.merge(home_features, on="match_id", how="left")
        .merge(away_features, on="match_id", how="left")
    )
    
    # Fill missing European history with competition averages
    euro_cols = [c for c in dataset.columns if c.startswith("home_euro_") or c.startswith("away_euro_")]
    for col in euro_cols:
        if "for" in col:
            dataset[col] = dataset[col].fillna(dataset["comp_avg_home_goals"])
        else:
            dataset[col] = dataset[col].fillna(dataset["comp_avg_away_goals"])
    
    # Filter to matches with minimum European experience
    min_matches = cfg.model.min_matches_for_features
    dataset = dataset[
        (dataset["home_european_matches"].fillna(0) >= min_matches) |
        (dataset["away_european_matches"].fillna(0) >= min_matches)
    ]
    
    # Save
    output_path = cfg.processed_dir / "european_dataset.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Wrote European dataset to {output_path}")
    print(f"  Total matches: {len(dataset)}")
    print(f"  Competitions: {dataset['competition_code'].value_counts().to_dict()}")
    
    return dataset


def main() -> None:
    cfg = get_european_config()
    create_european_dataset(cfg)


if __name__ == "__main__":
    main()
