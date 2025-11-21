from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import AppConfig, get_config

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
]

OPTIONAL_COLUMNS = [
    "HST", "AST",  # Shots on Target
    "HC", "AC",    # Corners
]


def load_raw_frames(cfg: AppConfig) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(cfg.raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False)
        if df.empty:
            continue
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            print(f"Skipping {csv_path.name} (missing columns: {missing})")
            continue
        league_code, season_code = csv_path.stem.split("_", maxsplit=1)
        df = df.copy()
        df["league_code"] = league_code
        df["season_code"] = season_code
        frames.append(df)
    return frames


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df.copy()
    parsed["Date"] = pd.to_datetime(parsed["Date"], dayfirst=True, errors="coerce")
    return parsed.dropna(subset=["Date"])


def attach_match_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["match_id"] = (
        df["league_code"]
        + "_"
        + df["season_code"]
        + "_"
        + df.index.astype(str)
    )
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    return df


def build_team_level(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        # Extract optional stats with defaults
        hst = row.get("HST", np.nan)
        ast = row.get("AST", np.nan)
        hc = row.get("HC", np.nan)
        ac = row.get("AC", np.nan)
        
        records.append(
            {
                "match_id": row["match_id"],
                "team": row["HomeTeam"],
                "opponent": row["AwayTeam"],
                "is_home": True,
                "date": row["Date"],
                "goals_for": row["FTHG"],
                "goals_against": row["FTAG"],
                "shots_on_target_for": hst,
                "shots_on_target_against": ast,
                "corners_for": hc,
                "corners_against": ac,
                "league_code": row["league_code"],
                "season_code": row["season_code"],
            }
        )
        records.append(
            {
                "match_id": row["match_id"],
                "team": row["AwayTeam"],
                "opponent": row["HomeTeam"],
                "is_home": False,
                "date": row["Date"],
                "goals_for": row["FTAG"],
                "goals_against": row["FTHG"],
                "shots_on_target_for": ast,
                "shots_on_target_against": hst,
                "corners_for": ac,
                "corners_against": hc,
                "league_code": row["league_code"],
                "season_code": row["season_code"],
            }
        )
    team_df = pd.DataFrame.from_records(records)
    team_df.sort_values("date", inplace=True)
    team_df["matches_played"] = team_df.groupby("team").cumcount()
    team_df["goal_diff"] = team_df["goals_for"] - team_df["goals_against"]
    return team_df


def rolling_average(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_form_features(team_df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = team_df.copy()
    grouped_for = df.groupby("team")["goals_for"]
    grouped_against = df.groupby("team")["goals_against"]
    grouped_diff = df.groupby("team")["goal_diff"]
    
    # New stats groups
    grouped_sot_for = df.groupby("team")["shots_on_target_for"]
    grouped_sot_against = df.groupby("team")["shots_on_target_against"]
    grouped_corners_for = df.groupby("team")["corners_for"]
    grouped_corners_against = df.groupby("team")["corners_against"]

    for window in windows:
        # Goals
        df[f"avg_for_{window}"] = grouped_for.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"avg_against_{window}"] = grouped_against.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"avg_diff_{window}"] = grouped_diff.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        
        # Shots on Target
        df[f"avg_sot_for_{window}"] = grouped_sot_for.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"avg_sot_against_{window}"] = grouped_sot_against.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        
        # Corners
        df[f"avg_corners_for_{window}"] = grouped_corners_for.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"avg_corners_against_{window}"] = grouped_corners_against.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )

    df["recent_goals_for"] = grouped_for.transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    df["recent_goals_against"] = grouped_against.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    return df


def wide_features(team_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns_to_keep = [
        "match_id",
        "team",
        "matches_played",
        "goal_diff",
        "recent_goals_for",
        "recent_goals_against",
    ] + [col for col in team_df.columns if any(col.startswith(key) for key in [
        "avg_for_", "avg_against_", "avg_diff_",
        "avg_sot_for_", "avg_sot_against_",
        "avg_corners_for_", "avg_corners_against_"
    ])]
    subset = team_df.loc[:, columns_to_keep]
    renamed = subset.add_prefix(prefix + "_")
    renamed.rename(columns={f"{prefix}_match_id": "match_id"}, inplace=True)
    return renamed


def best_available_odds(df: pd.DataFrame, suffix: str) -> pd.Series:
    candidates = [col for col in df.columns if col.endswith(suffix)]
    if not candidates:
        return pd.Series(np.nan, index=df.index)
    return df[candidates].apply(pd.to_numeric, errors="coerce").max(axis=1)


def create_dataset(cfg: AppConfig) -> pd.DataFrame:
    frames = load_raw_frames(cfg)
    if not frames:
        raise FileNotFoundError("No raw CSV files found. Run data_fetch first.")
    combined = parse_dates(pd.concat(frames, ignore_index=True))
    combined = attach_match_ids(combined)
    combined.sort_values("Date", inplace=True)

    combined["best_over_odds"] = best_available_odds(combined, ">2.5")
    combined["best_under_odds"] = best_available_odds(combined, "<2.5")
    combined["market_total_line"] = cfg.strategy.target_total_line

    team_df = build_team_level(combined)
    team_df = add_form_features(team_df, cfg.model.rolling_windows)

    home_features = wide_features(team_df[team_df["is_home"]], "home")
    away_features = wide_features(team_df[team_df["is_home"] == False], "away")

    dataset = (
        combined.merge(home_features, on="match_id", how="left")
        .merge(away_features, on="match_id", how="left")
    )

    min_matches = cfg.model.min_matches_for_features
    feature_cols = [col for col in dataset.columns if col.startswith("home_avg") or col.startswith("away_avg")]
    dataset = dataset.dropna(subset=feature_cols, how="any")
    dataset = dataset[(dataset["home_matches_played"] >= min_matches) & (dataset["away_matches_played"] >= min_matches)]

    output_path = cfg.processed_dir / "match_dataset.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Wrote processed dataset to {output_path.relative_to(cfg.base_dir)}")
    return dataset


def main() -> None:
    cfg = get_config()
    create_dataset(cfg)


if __name__ == "__main__":
    main()
