"""
Score upcoming European matches using the trained European model.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .european_config import EuropeanConfig, get_european_config
from .european_models import (
    EuropeanModelBundle,
    poisson_over_probability,
    poisson_under_probability,
)


def load_european_model(cfg: EuropeanConfig) -> EuropeanModelBundle:
    """Load trained European model."""
    home_model = joblib.load(cfg.models_dir / "euro_home_poisson.joblib")
    away_model = joblib.load(cfg.models_dir / "euro_away_poisson.joblib")
    feature_cols = (cfg.models_dir / "euro_features.txt").read_text().strip().split("\n")
    
    calibrator = None
    cal_path = cfg.models_dir / "euro_calibrator.joblib"
    if cal_path.exists():
        calibrator = joblib.load(cal_path)
    
    return EuropeanModelBundle(
        home_model=home_model,
        away_model=away_model,
        feature_columns=feature_cols,
        calibrator=calibrator,
    )


def fetch_upcoming_fixtures(
    cfg: EuropeanConfig,
    days_ahead: int = 14,
) -> List[Dict[str, Any]]:
    """
    Fetch upcoming European fixtures.
    First tries to load from local file, then falls back to API.
    """
    # Try local fixtures file first
    fixtures_file = cfg.raw_dir / "upcoming_fixtures.csv"
    if fixtures_file.exists():
        print(f"Loading fixtures from {fixtures_file}")
        df = pd.read_csv(fixtures_file)
        return df.to_dict("records")
    
    # Try JSON file  
    json_file = cfg.raw_dir / "upcoming_fixtures.json"
    if json_file.exists():
        print(f"Loading fixtures from {json_file}")
        with open(json_file) as f:
            return json.load(f)
    
    # Fall back to API (may not work for current season on free tier)
    if not cfg.api_key:
        print("No local fixtures file and no API key set.")
        print(f"Create {fixtures_file} with columns: Date,HomeTeam,AwayTeam,competition_code,round")
        return []
    
    try:
        from .european_fetch import make_api_request, RATE_LIMIT_DELAY
        import time
    except ImportError:
        print("API fetching not available.")
        return []
    
    all_fixtures = []
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    for comp_code, comp in cfg.competitions.items():
        try:
            params = {
                "league": comp.api_id,
                "season": datetime.now().year,
                "from": today,
                "to": end_date,
            }
            
            data = make_api_request("fixtures", params, cfg.api_key)
            
            for fix in data.get("response", []):
                fixture_info = fix.get("fixture", {})
                teams = fix.get("teams", {})
                league = fix.get("league", {})
                
                all_fixtures.append({
                    "match_id": fixture_info.get("id"),
                    "Date": fixture_info.get("date", "")[:10],
                    "Time": fixture_info.get("date", "")[11:16],
                    "competition_code": comp_code,
                    "competition_name": comp.name,
                    "round": league.get("round", ""),
                    "HomeTeam": teams.get("home", {}).get("name"),
                    "AwayTeam": teams.get("away", {}).get("name"),
                    "home_team_id": teams.get("home", {}).get("id"),
                    "away_team_id": teams.get("away", {}).get("id"),
                    "venue": fixture_info.get("venue", {}).get("name"),
                })
            
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"Error fetching {comp.name}: {e}")
    
    return all_fixtures


def get_team_european_features(
    team_id: int,
    team_name: str,
    cfg: EuropeanConfig,
    historical_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Get European features for a team based on historical data.
    Falls back to competition averages if team has no European history.
    """
    # Find team in historical data
    team_matches = historical_df[
        (historical_df["HomeTeam"] == team_name) | 
        (historical_df["AwayTeam"] == team_name)
    ].sort_values("Date")
    
    if len(team_matches) < 2:
        # No European history - use competition averages
        return None
    
    # Calculate recent form from last matches
    goals_for = []
    goals_against = []
    
    for _, row in team_matches.tail(5).iterrows():
        if row["HomeTeam"] == team_name:
            goals_for.append(row["FTHG"])
            goals_against.append(row["FTAG"])
        else:
            goals_for.append(row["FTAG"])
            goals_against.append(row["FTHG"])
    
    return {
        "european_matches": len(team_matches),
        "euro_avg_for_3": np.mean(goals_for[-3:]) if len(goals_for) >= 3 else np.mean(goals_for),
        "euro_avg_for_5": np.mean(goals_for[-5:]) if len(goals_for) >= 5 else np.mean(goals_for),
        "euro_avg_against_3": np.mean(goals_against[-3:]) if len(goals_against) >= 3 else np.mean(goals_against),
        "euro_avg_against_5": np.mean(goals_against[-5:]) if len(goals_against) >= 5 else np.mean(goals_against),
        "euro_ema_for_3": pd.Series(goals_for).ewm(span=3).mean().iloc[-1],
        "euro_ema_for_5": pd.Series(goals_for).ewm(span=5).mean().iloc[-1],
        "euro_ema_against_3": pd.Series(goals_against).ewm(span=3).mean().iloc[-1],
        "euro_ema_against_5": pd.Series(goals_against).ewm(span=5).mean().iloc[-1],
        "euro_total_goals_for": sum(goals_for),
        "euro_total_goals_against": sum(goals_against),
    }


def get_team_domestic_features(
    team_name: str,
    match_date: datetime,
    domestic_df: Optional[pd.DataFrame],
) -> Dict[str, float]:
    """
    Get domestic form features for a team for upcoming predictions.
    """
    if domestic_df is None or domestic_df.empty:
        return {}
    
    # Team name mappings (European -> Domestic)
    team_mappings = {
        "Bayern Munich": ["Bayern Munich", "Bayern"],
        "Borussia Dortmund": ["Borussia Dortmund", "Dortmund"],
        "RB Leipzig": ["RB Leipzig", "Leipzig"],
        "Bayer Leverkusen": ["Bayer Leverkusen", "Leverkusen"],
        "Manchester City": ["Manchester City", "Man City"],
        "Manchester United": ["Manchester United", "Man United"],
        "Tottenham": ["Tottenham", "Tottenham Hotspur", "Spurs"],
        "Atletico Madrid": ["Atletico Madrid", "Ath Madrid"],
        "Athletic Bilbao": ["Athletic Bilbao", "Ath Bilbao"],
        "AC Milan": ["AC Milan", "Milan"],
        "Inter Milan": ["Inter Milan", "Inter"],
        "PSG": ["PSG", "Paris SG", "Paris Saint-Germain"],
    }
    
    # Find team in domestic data
    all_teams = set(domestic_df["HomeTeam"].unique()) | set(domestic_df["AwayTeam"].unique())
    domestic_name = None
    
    if team_name in all_teams:
        domestic_name = team_name
    else:
        for euro_name, variants in team_mappings.items():
            if team_name in variants or team_name == euro_name:
                for v in variants:
                    if v in all_teams:
                        domestic_name = v
                        break
        if not domestic_name:
            for t in all_teams:
                if team_name.lower() in t.lower() or t.lower() in team_name.lower():
                    domestic_name = t
                    break
    
    if not domestic_name:
        return {}
    
    # Get recent domestic matches
    home_matches = domestic_df[
        (domestic_df["HomeTeam"] == domestic_name) & 
        (domestic_df["Date"] < match_date)
    ].copy()
    away_matches = domestic_df[
        (domestic_df["AwayTeam"] == domestic_name) & 
        (domestic_df["Date"] < match_date)
    ].copy()
    
    home_matches["goals_for"] = home_matches["FTHG"]
    home_matches["goals_against"] = home_matches["FTAG"]
    away_matches["goals_for"] = away_matches["FTAG"]
    away_matches["goals_against"] = away_matches["FTHG"]
    
    all_matches = pd.concat([
        home_matches[["Date", "goals_for", "goals_against"]],
        away_matches[["Date", "goals_for", "goals_against"]],
    ]).sort_values("Date")
    
    if len(all_matches) < 3:
        return {}
    
    recent = all_matches.tail(5)
    
    return {
        "dom_avg_for": recent["goals_for"].mean(),
        "dom_avg_against": recent["goals_against"].mean(),
        "dom_total_games": len(all_matches),
        "dom_home_pct": 0.5,  # Not tracked for simplicity
        "dom_ema_for": recent["goals_for"].ewm(span=5).mean().iloc[-1],
        "dom_ema_against": recent["goals_against"].ewm(span=5).mean().iloc[-1],
    }


def score_upcoming_european(cfg: EuropeanConfig) -> pd.DataFrame:
    """Score upcoming European fixtures."""
    bundle = load_european_model(cfg)
    
    # Load historical data for features
    historical_path = cfg.processed_dir / "european_dataset.csv"
    if not historical_path.exists():
        raise FileNotFoundError("European dataset not found. Run european_prepare first.")
    
    historical_df = pd.read_csv(historical_path, parse_dates=["Date"])
    
    # Load domestic data for supplementary features
    domestic_path = cfg.base_dir / "data" / "processed" / "match_dataset.csv"
    domestic_df = None
    if domestic_path.exists():
        domestic_df = pd.read_csv(domestic_path, parse_dates=["Date"], low_memory=False)
        print("Loaded domestic data for supplementary features")
    
    # Get upcoming fixtures
    fixtures = fetch_upcoming_fixtures(cfg)
    
    if not fixtures:
        print("No upcoming fixtures found.")
        return pd.DataFrame()
    
    print(f"Found {len(fixtures)} upcoming fixtures")
    
    # Calculate competition averages for fallback
    comp_avgs = historical_df.groupby("competition_code").agg({
        "FTHG": "mean",
        "FTAG": "mean",
        "total_goals": "mean",
    }).to_dict()
    
    predictions = []
    match_date = datetime.now()  # Use current date for upcoming predictions
    
    for fix in fixtures:
        # Get team features
        home_features = get_team_european_features(
            fix.get("home_team_id"),
            fix["HomeTeam"],
            cfg,
            historical_df,
        )
        away_features = get_team_european_features(
            fix.get("away_team_id"),
            fix["AwayTeam"],
            cfg,
            historical_df,
        )
        
        # Get domestic form features
        home_dom_features = get_team_domestic_features(fix["HomeTeam"], match_date, domestic_df)
        away_dom_features = get_team_domestic_features(fix["AwayTeam"], match_date, domestic_df)
        
        # Build feature row
        comp_code = fix["competition_code"]
        feature_row = {
            "comp_avg_home_goals": comp_avgs["FTHG"].get(comp_code, 1.5),
            "comp_avg_away_goals": comp_avgs["FTAG"].get(comp_code, 1.2),
            "comp_avg_total_goals": comp_avgs["total_goals"].get(comp_code, 2.7),
            "is_group_stage": "group" in fix.get("round", "").lower(),
            "is_knockout": any(k in fix.get("round", "").lower() for k in ["16", "quarter", "semi", "final"]),
            "is_final": "final" in fix.get("round", "").lower() and "semi" not in fix.get("round", "").lower(),
            "stage_importance": 1,  # Default
        }
        
        # Add home team European features
        if home_features:
            for key, val in home_features.items():
                feature_row[f"home_{key}"] = val
        else:
            # Use competition average
            feature_row["home_european_matches"] = 0
            for w in [3, 5]:
                feature_row[f"home_euro_avg_for_{w}"] = feature_row["comp_avg_home_goals"]
                feature_row[f"home_euro_avg_against_{w}"] = feature_row["comp_avg_away_goals"]
                feature_row[f"home_euro_ema_for_{w}"] = feature_row["comp_avg_home_goals"]
                feature_row[f"home_euro_ema_against_{w}"] = feature_row["comp_avg_away_goals"]
        
        # Add away team European features
        if away_features:
            for key, val in away_features.items():
                feature_row[f"away_{key}"] = val
        else:
            feature_row["away_european_matches"] = 0
            for w in [3, 5]:
                feature_row[f"away_euro_avg_for_{w}"] = feature_row["comp_avg_away_goals"]
                feature_row[f"away_euro_avg_against_{w}"] = feature_row["comp_avg_home_goals"]
                feature_row[f"away_euro_ema_for_{w}"] = feature_row["comp_avg_away_goals"]
                feature_row[f"away_euro_ema_against_{w}"] = feature_row["comp_avg_home_goals"]
        
        # Add domestic form features for both teams
        if home_dom_features:
            for key, val in home_dom_features.items():
                feature_row[f"home_{key}"] = val
        else:
            # Fallback to competition averages
            feature_row["home_dom_avg_for"] = feature_row["comp_avg_home_goals"]
            feature_row["home_dom_avg_against"] = feature_row["comp_avg_away_goals"]
            feature_row["home_dom_total_games"] = 0
            feature_row["home_dom_home_pct"] = 0.5
            feature_row["home_dom_ema_for"] = feature_row["comp_avg_home_goals"]
            feature_row["home_dom_ema_against"] = feature_row["comp_avg_away_goals"]
        
        if away_dom_features:
            for key, val in away_dom_features.items():
                feature_row[f"away_{key}"] = val
        else:
            feature_row["away_dom_avg_for"] = feature_row["comp_avg_away_goals"]
            feature_row["away_dom_avg_against"] = feature_row["comp_avg_home_goals"]
            feature_row["away_dom_total_games"] = 0
            feature_row["away_dom_home_pct"] = 0.5
            feature_row["away_dom_ema_for"] = feature_row["comp_avg_away_goals"]
            feature_row["away_dom_ema_against"] = feature_row["comp_avg_home_goals"]
        
        # Create feature DataFrame
        try:
            X = pd.DataFrame([feature_row])[bundle.feature_columns]
            
            pred_home = bundle.home_model.predict(X)[0]
            pred_away = bundle.away_model.predict(X)[0]
            pred_total = pred_home + pred_away
            
            raw_over = poisson_over_probability(pred_total, 2.5)
            
            if bundle.calibrator:
                cal_over = bundle.calibrator.predict([raw_over])[0]
            else:
                cal_over = raw_over
            
            predictions.append({
                **fix,
                "pred_home_goals": round(pred_home, 2),
                "pred_away_goals": round(pred_away, 2),
                "pred_total_goals": round(pred_total, 2),
                "over_2_5_prob": round(cal_over, 3),
                "under_2_5_prob": round(1 - cal_over, 3),
                "home_euro_experience": feature_row.get("home_european_matches", 0),
                "away_euro_experience": feature_row.get("away_european_matches", 0),
            })
            
        except Exception as e:
            print(f"Error predicting {fix['HomeTeam']} vs {fix['AwayTeam']}: {e}")
    
    result_df = pd.DataFrame(predictions)
    
    if len(result_df) > 0:
        output_path = cfg.processed_dir / "upcoming_european.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Saved {len(result_df)} predictions to {output_path}")
    
    return result_df


def main() -> None:
    cfg = get_european_config()
    df = score_upcoming_european(cfg)
    
    if len(df) > 0:
        print("\n=== Upcoming European Fixtures ===")
        for _, row in df.iterrows():
            print(f"\n{row['Date']} - {row['competition_code']}")
            print(f"  {row['HomeTeam']} vs {row['AwayTeam']}")
            print(f"  Predicted: {row['pred_home_goals']:.1f} - {row['pred_away_goals']:.1f}")
            print(f"  Over 2.5: {row['over_2_5_prob']:.1%}")


if __name__ == "__main__":
    main()
