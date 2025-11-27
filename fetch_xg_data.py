"""
Fetch and integrate xG data into the betting pipeline.

This script:
1. Downloads xG data from Understat/FBref
2. Merges it with existing match data
3. Adds xG-based rolling features for improved predictions
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import get_config
from src.xg_scraper import fetch_xg_data, TEAM_NAME_MAPPING


def standardize_team_name(name: str, mapping: dict = None) -> str:
    """Standardize team name for matching across sources."""
    if mapping is None:
        mapping = TEAM_NAME_MAPPING
    
    name = name.strip()
    # Try direct mapping
    if name in mapping:
        return mapping[name]
    # Try reverse mapping
    for k, v in mapping.items():
        if v == name:
            return k
    return name


def load_existing_dataset(cfg) -> pd.DataFrame:
    """Load the processed match dataset."""
    dataset_path = cfg.processed_dir / "match_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Match dataset not found. Run prepare_dataset first."
        )
    return pd.read_csv(dataset_path, parse_dates=["Date"], low_memory=False)


def add_xg_rolling_features(df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Add xG-based rolling features to the dataset.
    
    Features added:
    - Rolling average xG for/against
    - xG overperformance (goals - xG)
    - xG difference trends
    """
    df = df.copy()
    
    # Skip if no xG data
    if "HomeXG" not in df.columns or df["HomeXG"].isna().all():
        print("No xG data available, skipping xG features")
        return df
    
    # Build team-level xG records
    home_records = df[["Date", "HomeTeam", "HomeXG", "AwayXG", "FTHG", "FTAG"]].copy()
    home_records.columns = ["Date", "team", "xg_for", "xg_against", "goals_for", "goals_against"]
    home_records["is_home"] = True
    
    away_records = df[["Date", "AwayTeam", "AwayXG", "HomeXG", "FTAG", "FTHG"]].copy()
    away_records.columns = ["Date", "team", "xg_for", "xg_against", "goals_for", "goals_against"]
    away_records["is_home"] = False
    
    team_xg = pd.concat([home_records, away_records], ignore_index=True)
    team_xg = team_xg.sort_values("Date")
    team_xg = team_xg.dropna(subset=["xg_for", "xg_against"])
    
    # Calculate xG overperformance
    team_xg["xg_overperf"] = team_xg["goals_for"] - team_xg["xg_for"]
    team_xg["xg_def_overperf"] = team_xg["xg_against"] - team_xg["goals_against"]
    
    # Calculate rolling features
    for window in windows:
        team_xg[f"avg_xg_for_{window}"] = team_xg.groupby("team")["xg_for"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        team_xg[f"avg_xg_against_{window}"] = team_xg.groupby("team")["xg_against"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        team_xg[f"avg_xg_overperf_{window}"] = team_xg.groupby("team")["xg_overperf"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        team_xg[f"avg_xg_def_overperf_{window}"] = team_xg.groupby("team")["xg_def_overperf"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    
    # Prepare features for merging back
    xg_feature_cols = [col for col in team_xg.columns if col.startswith("avg_xg_")]
    
    # Create home and away feature sets
    home_xg = team_xg[team_xg["is_home"]][["Date", "team"] + xg_feature_cols].copy()
    home_xg = home_xg.add_prefix("home_")
    home_xg = home_xg.rename(columns={"home_Date": "Date", "home_team": "HomeTeam"})
    
    away_xg = team_xg[~team_xg["is_home"]][["Date", "team"] + xg_feature_cols].copy()
    away_xg = away_xg.add_prefix("away_")
    away_xg = away_xg.rename(columns={"away_Date": "Date", "away_team": "AwayTeam"})
    
    # Merge back to main dataframe
    df = df.merge(home_xg, on=["Date", "HomeTeam"], how="left")
    df = df.merge(away_xg, on=["Date", "AwayTeam"], how="left")
    
    return df


def merge_xg_data(
    match_df: pd.DataFrame,
    xg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge xG data with match dataset using fuzzy team name matching.
    """
    if xg_df.empty:
        print("No xG data to merge")
        return match_df
    
    match_df = match_df.copy()
    xg_df = xg_df.copy()
    
    # Standardize dates
    match_df["Date"] = pd.to_datetime(match_df["Date"])
    xg_df["Date"] = pd.to_datetime(xg_df["Date"])
    
    # Standardize team names
    match_df["_home_std"] = match_df["HomeTeam"].apply(
        lambda x: standardize_team_name(str(x)).lower().strip()
    )
    match_df["_away_std"] = match_df["AwayTeam"].apply(
        lambda x: standardize_team_name(str(x)).lower().strip()
    )
    
    xg_df["_home_std"] = xg_df["HomeTeam"].apply(
        lambda x: standardize_team_name(str(x)).lower().strip()
    )
    xg_df["_away_std"] = xg_df["AwayTeam"].apply(
        lambda x: standardize_team_name(str(x)).lower().strip()
    )
    
    # Create merge key using date and standardized names
    match_df["_merge_key"] = (
        match_df["Date"].dt.strftime("%Y-%m-%d") + "_" +
        match_df["_home_std"] + "_" +
        match_df["_away_std"]
    )
    
    xg_df["_merge_key"] = (
        xg_df["Date"].dt.strftime("%Y-%m-%d") + "_" +
        xg_df["_home_std"] + "_" +
        xg_df["_away_std"]
    )
    
    # Select xG columns for merge
    xg_cols = ["_merge_key", "HomeXG", "AwayXG"]
    if "TotalXG" in xg_df.columns:
        xg_cols.append("TotalXG")
    if "XGDiff" in xg_df.columns:
        xg_cols.append("XGDiff")
    
    # Merge
    merged = match_df.merge(
        xg_df[xg_cols].drop_duplicates(subset=["_merge_key"]),
        on="_merge_key",
        how="left",
    )
    
    # Clean up temp columns
    merged = merged.drop(columns=["_merge_key", "_home_std", "_away_std"])
    
    # Report match rate
    matched = merged["HomeXG"].notna().sum()
    total = len(merged)
    print(f"xG data matched: {matched}/{total} ({100*matched/total:.1f}%)")
    
    return merged


def fetch_and_integrate_xg(
    leagues: List[str] = None,
    seasons: List[str] = None,
    source: str = "understat",
) -> pd.DataFrame:
    """
    Main function to fetch xG and integrate with existing data.
    
    Args:
        leagues: League codes to fetch (default: top 5 leagues)
        seasons: Seasons to fetch
        source: "understat" or "fbref"
    
    Returns:
        Updated match dataset with xG features
    """
    cfg = get_config()
    
    # Default leagues
    if leagues is None:
        leagues = ["E0", "D1", "SP1", "I1", "F1"]
    
    # Default seasons for Understat format
    if seasons is None:
        seasons = ["2022", "2023", "2024"]
    
    # Fetch xG data
    print("=" * 60)
    print("Fetching xG Data")
    print("=" * 60)
    
    xg_dir = cfg.data_dir / "xg"
    xg_dir.mkdir(parents=True, exist_ok=True)
    
    xg_df = fetch_xg_data(
        leagues=leagues,
        seasons=seasons,
        source=source,
        output_dir=xg_dir,
    )
    
    if xg_df.empty:
        print("Failed to fetch xG data")
        return pd.DataFrame()
    
    # Load existing match data
    print("\n" + "=" * 60)
    print("Merging with Match Dataset")
    print("=" * 60)
    
    match_df = load_existing_dataset(cfg)
    print(f"Loaded {len(match_df)} matches from dataset")
    
    # Merge xG data
    merged_df = merge_xg_data(match_df, xg_df)
    
    # Add rolling xG features
    print("\nAdding xG rolling features...")
    final_df = add_xg_rolling_features(merged_df, windows=cfg.model.rolling_windows)
    
    # Save updated dataset
    output_path = cfg.processed_dir / "match_dataset_with_xg.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved enhanced dataset to {output_path}")
    
    # Summary statistics
    xg_feature_cols = [col for col in final_df.columns if "xg" in col.lower()]
    print(f"\nAdded {len(xg_feature_cols)} xG-related columns")
    
    if "HomeXG" in final_df.columns:
        valid_xg = final_df["HomeXG"].notna()
        print(f"Matches with xG data: {valid_xg.sum()}")
        if valid_xg.sum() > 0:
            print(f"Average HomeXG: {final_df.loc[valid_xg, 'HomeXG'].mean():.2f}")
            print(f"Average AwayXG: {final_df.loc[valid_xg, 'AwayXG'].mean():.2f}")
    
    return final_df


def main():
    """Run xG data integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch and integrate xG data")
    parser.add_argument(
        "--source",
        choices=["understat", "fbref"],
        default="understat",
        help="Data source (default: understat)",
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=["E0", "D1", "SP1", "I1", "F1"],
        help="League codes to fetch",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2023", "2024"],
        help="Seasons to fetch (year format for understat)",
    )
    
    args = parser.parse_args()
    
    fetch_and_integrate_xg(
        leagues=args.leagues,
        seasons=args.seasons,
        source=args.source,
    )


if __name__ == "__main__":
    main()
