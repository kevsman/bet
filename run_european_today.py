#!/usr/bin/env python3
"""
Generate European competition value bets for today using Norsk Tipping odds.

This script:
1. Loads European matches from Norsk Tipping scraper
2. Uses the trained European model to predict over/under probabilities
3. Compares with bookmaker odds to find value bets
4. Uses domestic form data as fallback for teams without European history
5. Generates a report

Usage:
    python run_european_today.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.european_config import get_european_config


# Domestic team name mappings: Norsk Tipping -> football-data.co.uk format
DOMESTIC_TEAM_MAPPINGS = {
    # English teams
    "Nottingham Forest": "Nott'm Forest",
    "Crystal Palace": "Crystal Palace",
    # German teams
    "Mainz 05": "Mainz",
    "Stuttgart": "Stuttgart",
    "Freiburg": "Freiburg",
    # Spanish teams
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    # French teams
    "Strasbourg": "Strasbourg",
    "Nice": "Nice",
    "Lyon": "Lyon",
    "Lille": "Lille",
    # Dutch teams
    "FC Utrecht": "Utrecht",
    "Go Ahead Eagles": "Go Ahead",
    "AZ Alkmaar": "AZ Alkmaar",
    # Norwegian teams
    "Brann": "Brann",
    # Scottish teams
    "Aberdeen": "Aberdeen",
    "Celtic": "Celtic",
    "Rangers": "Rangers",
    # Italian teams
    "Fiorentina": "Fiorentina",
    "Roma": "Roma",
    "Bologna": "Bologna",
    "Atalanta": "Atalanta",
    # Portuguese teams
    "Porto": "Porto",
    "Braga": "Sp Braga",
}


# Team name mappings: Norsk Tipping -> European Model
EURO_TEAM_MAPPINGS = {
    # Champions League / Europa League teams
    "Feyenoord": "Feyenoord",
    "Celtic": "Celtic",
    "Viktoria Plzen": "Viktoria Plzen",
    "Freiburg": "Freiburg",
    "PAOK": "PAOK Saloniki",
    "Brann": "Brann",
    "Porto": "Porto",
    "Nice": "Nice",
    "Ludogorets 1947": "PFC Ludogorets Razgrad",
    "Ludogorets": "PFC Ludogorets Razgrad",
    "Celta Vigo": "Celta Vigo",
    "Fenerbahce Istanbul": "Fenerbahce",
    "Fenerbahce": "Fenerbahce",
    "Ferencvaros": "Ferencvaros",
    "Lille": "Lille",
    "Dinamo Zagreb": "Dinamo Zagreb",
    "Roma": "Roma",
    "FC Midtjylland": "Midtjylland",
    "Midtjylland": "Midtjylland",
    "Aston Villa": "Aston Villa",
    "Young Boys": "Young Boys",
    "Panathinaikos": "Panathinaikos",
    "Sturm Graz": "Sturm Graz",
    "Real Betis": "Real Betis",
    "FC Utrecht": "FC Utrecht",
    "Utrecht": "FC Utrecht",
    "Røde Stjerne": "Red Star Belgrade",
    "Red Star Belgrade": "Red Star Belgrade",
    "FCSB": "FCSB",
    "Maccabi Tel Aviv": "Maccabi Tel Aviv",
    "Lyon": "Lyon",
    "Nottingham Forest": "Nottingham Forest",
    "Malmö FF": "Malmo",
    "Malmo": "Malmo",
    "Go Ahead Eagles": "Go Ahead Eagles",
    "Stuttgart": "Stuttgart",
    "Rangers": "Rangers",
    "Braga": "Sporting Braga",
    "Sporting Braga": "Sporting Braga",
    "Bologna": "Bologna",
    "Red Bull Salzburg": "Salzburg",
    "Salzburg": "Salzburg",
    "RB Salzburg": "Salzburg",
    "Genk": "Genk",
    "FC Basel": "Basel",
    "Basel": "Basel",
    # Conference League teams
    "AZ Alkmaar": "AZ Alkmaar",
    "Lech Poznan": "Lech Poznan",
    "Dinamo Kyiv": "Dinamo Kiev",
    "Omonia Nicosia": "Omonia Nikosia",
    "Slovan Bratislava": "Slovan Bratislava",
    "Legia Warszawa": "Legia Warsaw",
    "Legia Warsaw": "Legia Warsaw",
    "Sparta Praha": "Sparta Prague",
    "Sparta Prague": "Sparta Prague",
    "Fiorentina": "Fiorentina",
    "Rapid Wien": "Rapid Wien",
    "Shakhtar Donetsk": "Shakhtar Donetsk",
    "Crystal Palace": "Crystal Palace",
    "Strasbourg": "Strasbourg",
    "Aberdeen": "Aberdeen",
    "HNK Rijeka": "Dinamo Zagreb",  # Use similar Croatian team
    "Rijeka": "Dinamo Zagreb",
    "Häcken": "Hacken",
    "Hacken": "Hacken",
    "Zrinjski Mostar": "Zrinjski Mostar",
    "AEK Athens": "AEK Athens",
    "Shamrock Rovers": "Shamrock Rovers",
    "FC Noah Yerevan": "FC Noah",
    "FC Noah": "FC Noah",
    # Additional mappings for missing teams
    "Jagiellonia Bialystok": "Jagiellonia",
    "Jagiellonia": "Jagiellonia",
    "NK Celje": "Celje",
    "Celje": "Celje",
    "SK Sigma Olomouc": "1. FC Slovácko",  # Use similar Czech team
    "CS Universitatea Craiova": "CFR Cluj",  # Use similar Romanian team
    "KS Rakow Czestochowa": "Lech Poznan",  # Use similar Polish team
    "Rakow Czestochowa": "Lech Poznan",
    "Mainz 05": "Mainz",
    "Mainz": "Mainz",
    "AEK Larnaca": "AEK Larnaca",
    "AEK Larnaka": "AEK Larnaca",
    "Samsunspor": "Trabzonspor",  # Use similar Turkish team
    "Breidablik": "Breiðablik",
    "KuPS": "HJK Helsinki",  # Use similar Finnish team
    "Kuopio PS": "HJK Helsinki",
    "Hamrun Spartans": "Hibernians",  # Use similar Maltese team
    "Lincoln Red Imps FC": "Lincoln Red Imps",
    "Lincoln Red Imps": "Lincoln Red Imps",
    "KF Drita": "KF Shkëndija",  # Use similar Kosovo team
    "Shkendija": "KF Shkëndija",
    # Additional mappings
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Tottenham": "Tottenham",
    "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Inter Milan": "Inter Milan",
    "AC Milan": "AC Milan",
    "Juventus": "Juventus",
    "PSG": "PSG",
    "Ajax": "Ajax",
    "Benfica": "Benfica",
    "Sporting CP": "Sporting CP",
    "Atalanta": "Atalanta",
    "Lazio": "Lazio",
    "Napoli": "Napoli",
}

def map_team_name(name: str) -> str | None:
    """Map Norsk Tipping team name to European model team name."""
    if name in EURO_TEAM_MAPPINGS:
        return EURO_TEAM_MAPPINGS[name]
    # Try exact match
    return name


def load_european_model(cfg):
    """Load the trained European model."""
    models_dir = cfg.models_dir
    
    home_path = models_dir / "euro_home_poisson.joblib"
    away_path = models_dir / "euro_away_poisson.joblib"
    features_path = models_dir / "euro_features.txt"
    
    if not home_path.exists() or not away_path.exists():
        print("European model not found. Run 'python run_european.py all' first.")
        return None, None, None, None
    
    home_model = joblib.load(home_path)
    away_model = joblib.load(away_path)
    feature_cols = features_path.read_text().strip().split("\n")
    
    # Try to load Dixon-Coles model for ensemble
    dc_model = None
    dc_path = Path("models/dixon_coles.joblib")
    if dc_path.exists():
        try:
            from src.dixon_coles import DixonColesModel, DixonColesParams
            loaded = joblib.load(dc_path)
            if isinstance(loaded, dict):
                dc_model = DixonColesModel()
                dc_model.params = loaded.get('params')
                dc_model.teams = loaded.get('teams', [])
                dc_model._fitted = loaded.get('_fitted', True)
            else:
                dc_model = loaded
            print("✓ Dixon-Coles model loaded for ensemble")
        except Exception as e:
            print(f"Note: Dixon-Coles model not available ({e})")
    
    return home_model, away_model, feature_cols, dc_model


def load_domestic_data() -> pd.DataFrame | None:
    """Load all domestic league data for fallback form calculations."""
    raw_dir = Path("data/raw")
    frames = []
    
    for csv_path in sorted(raw_dir.glob("*.csv")):
        try:
            try:
                df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False, encoding="latin-1")
            
            if df.empty or "HomeTeam" not in df.columns:
                continue
            
            df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date"])
            frames.append(df)
        except Exception:
            continue
    
    if not frames:
        return None
    
    return pd.concat(frames, ignore_index=True).sort_values("Date")


def get_domestic_team_features(team_name: str, domestic_df: pd.DataFrame, is_home: bool) -> dict | None:
    """Get domestic form features for a team (used as fallback for teams without European history)."""
    # Try multiple name variations
    name_variations = [team_name]
    if team_name in DOMESTIC_TEAM_MAPPINGS:
        name_variations.append(DOMESTIC_TEAM_MAPPINGS[team_name])
    
    team_matches = None
    for name in name_variations:
        matches = domestic_df[
            (domestic_df["HomeTeam"] == name) | 
            (domestic_df["AwayTeam"] == name)
        ].sort_values("Date")
        if len(matches) > 0:
            team_matches = matches
            break
    
    if team_matches is None or len(team_matches) < 3:
        return None
    
    # Calculate recent form from last 5 matches
    goals_for = []
    goals_against = []
    
    for _, row in team_matches.tail(5).iterrows():
        if row["HomeTeam"] in name_variations:
            goals_for.append(row["FTHG"])
            goals_against.append(row["FTAG"])
        else:
            goals_for.append(row["FTAG"])
            goals_against.append(row["FTHG"])
    
    prefix = "home" if is_home else "away"
    
    # Return features that mimic European features but from domestic data
    return {
        f"{prefix}_european_matches": 3,  # Minimum to pass threshold
        f"{prefix}_euro_avg_for_3": np.mean(goals_for[-3:]) if len(goals_for) >= 3 else np.mean(goals_for),
        f"{prefix}_euro_avg_for_5": np.mean(goals_for[-5:]) if len(goals_for) >= 5 else np.mean(goals_for),
        f"{prefix}_euro_avg_against_3": np.mean(goals_against[-3:]) if len(goals_against) >= 3 else np.mean(goals_against),
        f"{prefix}_euro_avg_against_5": np.mean(goals_against[-5:]) if len(goals_against) >= 5 else np.mean(goals_against),
        f"{prefix}_euro_ema_for_3": pd.Series(goals_for).ewm(span=3).mean().iloc[-1],
        f"{prefix}_euro_ema_for_5": pd.Series(goals_for).ewm(span=5).mean().iloc[-1],
        f"{prefix}_euro_ema_against_3": pd.Series(goals_against).ewm(span=3).mean().iloc[-1],
        f"{prefix}_euro_ema_against_5": pd.Series(goals_against).ewm(span=5).mean().iloc[-1],
        f"{prefix}_euro_total_goals_for": sum(goals_for),
        f"{prefix}_euro_total_goals_against": sum(goals_against),
        f"_used_domestic_fallback_{prefix}": True,  # Flag to track fallback usage
    }


def get_team_features(team_name: str, historical_df: pd.DataFrame, is_home: bool) -> dict | None:
    """Get European features for a team from historical data."""
    team_matches = historical_df[
        (historical_df["HomeTeam"] == team_name) | 
        (historical_df["AwayTeam"] == team_name)
    ].sort_values("Date")
    
    if len(team_matches) < 2:
        return None
    
    # Calculate recent form
    goals_for = []
    goals_against = []
    
    for _, row in team_matches.tail(5).iterrows():
        if row["HomeTeam"] == team_name:
            goals_for.append(row["FTHG"])
            goals_against.append(row["FTAG"])
        else:
            goals_for.append(row["FTAG"])
            goals_against.append(row["FTHG"])
    
    prefix = "home" if is_home else "away"
    
    return {
        f"{prefix}_european_matches": len(team_matches),
        f"{prefix}_euro_avg_for_3": np.mean(goals_for[-3:]) if len(goals_for) >= 3 else np.mean(goals_for),
        f"{prefix}_euro_avg_for_5": np.mean(goals_for[-5:]) if len(goals_for) >= 5 else np.mean(goals_for),
        f"{prefix}_euro_avg_against_3": np.mean(goals_against[-3:]) if len(goals_against) >= 3 else np.mean(goals_against),
        f"{prefix}_euro_avg_against_5": np.mean(goals_against[-5:]) if len(goals_against) >= 5 else np.mean(goals_against),
        f"{prefix}_euro_ema_for_3": pd.Series(goals_for).ewm(span=3).mean().iloc[-1],
        f"{prefix}_euro_ema_for_5": pd.Series(goals_for).ewm(span=5).mean().iloc[-1],
        f"{prefix}_euro_ema_against_3": pd.Series(goals_against).ewm(span=3).mean().iloc[-1],
        f"{prefix}_euro_ema_against_5": pd.Series(goals_against).ewm(span=5).mean().iloc[-1],
        f"{prefix}_euro_total_goals_for": sum(goals_for),
        f"{prefix}_euro_total_goals_against": sum(goals_against),
    }


def bivariate_poisson_over_prob(home_xg: float, away_xg: float, threshold: float = 2.5) -> float:
    """Calculate over probability using bivariate Poisson."""
    from scipy.stats import poisson
    
    prob_under = 0.0
    for h in range(int(threshold) + 1):
        for a in range(int(threshold) + 1 - h):
            if h + a <= threshold:
                prob_under += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
    
    return 1.0 - prob_under


def main():
    print("=" * 60)
    print("EUROPEAN COMPETITION VALUE BETS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    cfg = get_european_config()
    
    # Load model
    home_model, away_model, feature_cols, dc_model = load_european_model(cfg)
    if home_model is None:
        return
    
    # Load historical European data
    dataset_path = cfg.processed_dir / "european_dataset.csv"
    if not dataset_path.exists():
        print("European dataset not found. Run 'python run_european.py prepare' first.")
        return
    
    historical_df = pd.read_csv(dataset_path)
    print(f"Loaded {len(historical_df)} historical European matches")
    
    # Get teams in dataset
    euro_teams = set(historical_df["HomeTeam"].unique()) | set(historical_df["AwayTeam"].unique())
    print(f"Teams in European dataset: {len(euro_teams)}")
    
    # Load domestic data for fallback
    print("Loading domestic league data for fallback...")
    domestic_df = load_domestic_data()
    if domestic_df is not None:
        domestic_teams = set(domestic_df["HomeTeam"].unique()) | set(domestic_df["AwayTeam"].unique())
        print(f"Loaded domestic data: {len(domestic_df)} matches, {len(domestic_teams)} teams")
    else:
        domestic_df = pd.DataFrame()
        print("Warning: Could not load domestic data for fallback")
    
    # Load Norsk Tipping odds
    odds_path = Path("data/upcoming/norsk_tipping_odds.csv")
    if not odds_path.exists():
        print("Norsk Tipping odds not found. Run scraper first.")
        return
    
    odds_df = pd.read_csv(odds_path)
    
    # Fetch weather forecasts for upcoming matches
    try:
        from src.weather_forecast import fetch_weather_for_matches
        odds_df = fetch_weather_for_matches(odds_df)
        print("✓ Weather forecasts loaded")
    except Exception as e:
        print(f"Warning: Weather fetch failed ({e}), continuing without weather data")
    
    # Filter to European matches with over/under odds
    euro_matches = odds_df[
        odds_df["league"].str.contains("Europa|Conference", case=False, na=False) &
        odds_df["over_2_5"].notna()
    ].copy()
    
    print(f"\nFound {len(euro_matches)} European matches from Norsk Tipping")
    
    # Process each match
    results = []
    skipped_no_data = []
    
    for _, row in euro_matches.iterrows():
        home_nt = row["home_team"]
        away_nt = row["away_team"]
        
        # Map team names
        home_mapped = map_team_name(home_nt)
        away_mapped = map_team_name(away_nt)
        
        # Check if teams exist in European dataset
        home_in_euro = home_mapped in euro_teams if home_mapped else False
        away_in_euro = away_mapped in euro_teams if away_mapped else False
        
        # Get features - try European first, then domestic fallback
        home_features = None
        away_features = None
        used_domestic_home = False
        used_domestic_away = False
        
        if home_in_euro and home_mapped:
            home_features = get_team_features(home_mapped, historical_df, is_home=True)
        
        if home_features is None and domestic_df is not None and len(domestic_df) > 0:
            # Try domestic fallback
            home_features = get_domestic_team_features(home_nt, domestic_df, is_home=True)
            if home_features:
                used_domestic_home = True
        
        if away_in_euro and away_mapped:
            away_features = get_team_features(away_mapped, historical_df, is_home=False)
        
        if away_features is None and domestic_df is not None and len(domestic_df) > 0:
            # Try domestic fallback
            away_features = get_domestic_team_features(away_nt, domestic_df, is_home=False)
            if away_features:
                used_domestic_away = True
        
        if home_features is None or away_features is None:
            missing = []
            if home_features is None:
                missing.append(f"{home_nt}")
            if away_features is None:
                missing.append(f"{away_nt}")
            skipped_no_data.append(f"{home_nt} vs {away_nt}: No data for {', '.join(missing)}")
            continue
        
        # Build feature vector
        try:
            # Combine features (remove fallback flags)
            features = {}
            for k, v in home_features.items():
                if not k.startswith("_"):
                    features[k] = v
            for k, v in away_features.items():
                if not k.startswith("_"):
                    features[k] = v
            
            # Add competition-level features (use averages from dataset)
            features["comp_avg_home_goals"] = historical_df["FTHG"].mean()
            features["comp_avg_away_goals"] = historical_df["FTAG"].mean()
            features["comp_avg_total_goals"] = (historical_df["FTHG"] + historical_df["FTAG"]).mean()
            features["is_group_stage"] = 1  # Assume group/league stage
            features["is_knockout"] = 0
            features["is_final"] = 0
            features["stage_importance"] = 1
            
            # Add domestic features (use zeros if not available)
            for col in feature_cols:
                if col not in features:
                    if "dom_" in col:
                        features[col] = 0.0
                    elif col not in features:
                        features[col] = 0.0
            
            # Inject actual weather forecast (if available)
            if 'weather_temp' in row and pd.notna(row.get('weather_temp')):
                if 'weather_temp' in feature_cols:
                    features['weather_temp'] = row['weather_temp']
                if 'weather_wind_speed' in feature_cols:
                    features['weather_wind_speed'] = row.get('weather_wind_speed', 0)
                if 'weather_is_cold' in feature_cols:
                    features['weather_is_cold'] = row.get('weather_is_cold', 0)
            
            # Create feature vector
            X = pd.DataFrame([features])[feature_cols]
            
            # Predict with European Poisson model
            pred_home = home_model.predict(X)[0]
            pred_away = away_model.predict(X)[0]
            
            # Try Dixon-Coles ensemble if available
            dc_home, dc_away = None, None
            dc_used = False
            if dc_model and dc_model._fitted:
                try:
                    # Map to DC team names (use original mapped names)
                    dc_home_name = home_mapped or home_nt
                    dc_away_name = away_mapped or away_nt
                    dc_home, dc_away = dc_model.predict_goals(dc_home_name, dc_away_name)
                    dc_used = True
                except Exception:
                    # Team not in DC model, try variations
                    for h_name in [home_nt, home_mapped]:
                        for a_name in [away_nt, away_mapped]:
                            if h_name and a_name:
                                try:
                                    dc_home, dc_away = dc_model.predict_goals(h_name, a_name)
                                    dc_used = True
                                    break
                                except Exception:
                                    continue
                        if dc_used:
                            break
            
            # Ensemble: average Poisson and Dixon-Coles if both available
            if dc_used and dc_home is not None and dc_away is not None:
                final_home = 0.5 * pred_home + 0.5 * dc_home
                final_away = 0.5 * pred_away + 0.5 * dc_away
            else:
                final_home = pred_home
                final_away = pred_away
            
            pred_total = final_home + final_away
            
            # Calculate over probability using bivariate Poisson
            over_prob = bivariate_poisson_over_prob(final_home, final_away, 2.5)
            under_prob = 1.0 - over_prob
            
            # Get odds
            over_odds = row["over_2_5"]
            under_odds = row["under_2_5"]
            
            # Calculate edge
            over_edge = (over_prob * over_odds) - 1
            under_edge = (under_prob * under_odds) - 1
            
            # Track data source
            data_source = "Euro"
            if used_domestic_home and used_domestic_away:
                data_source = "Dom"
            elif used_domestic_home or used_domestic_away:
                data_source = "Mix"
            
            # Add DC indicator
            if dc_used:
                data_source += "+DC"
            
            results.append({
                "home_team": home_nt,
                "away_team": away_nt,
                "league": row["league"],
                "pred_home": final_home,
                "pred_away": final_away,
                "pred_total": pred_total,
                "poisson_home": pred_home,
                "poisson_away": pred_away,
                "dc_home": dc_home if dc_used else None,
                "dc_away": dc_away if dc_used else None,
                "over_prob": over_prob,
                "under_prob": under_prob,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "over_ev": over_edge,
                "under_ev": under_edge,
                "data_source": data_source,
            })
            
        except Exception as e:
            print(f"  Error predicting {home_nt} vs {away_nt}: {e}")
            continue
    
    # Print skipped matches summary
    if skipped_no_data:
        print(f"\nSkipped {len(skipped_no_data)} matches (no data available):")
        for skip in skipped_no_data[:10]:  # Show first 10
            print(f"  {skip}")
        if len(skipped_no_data) > 10:
            print(f"  ... and {len(skipped_no_data) - 10} more")
    
    if not results:
        print("\nNo matches could be predicted.")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find value bets (EV > 0)
    print("\n" + "=" * 60)
    print("VALUE BETS (EV > 0%)")
    print("=" * 60)
    print("(Source: Euro=European data, Dom=Domestic fallback, Mix=Both)")
    
    value_bets = []
    
    for _, row in results_df.iterrows():
        if row["over_ev"] > 0:
            value_bets.append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "bet": "Over 2.5",
                "prob": row["over_prob"],
                "odds": row["over_odds"],
                "ev": row["over_ev"],
                "pred_goals": row["pred_total"],
                "source": row["data_source"],
            })
        
        if row["under_ev"] > 0:
            value_bets.append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "bet": "Under 2.5",
                "prob": row["under_prob"],
                "odds": row["under_odds"],
                "ev": row["under_ev"],
                "pred_goals": row["pred_total"],
                "source": row["data_source"],
            })
    
    if not value_bets:
        print("\nNo value bets found in European competitions today.")
    else:
        # Sort by EV
        value_bets_df = pd.DataFrame(value_bets).sort_values("ev", ascending=False)
        
        print(f"\n{'Match':<45} {'Bet':<12} {'Prob':>8} {'Odds':>6} {'EV':>8} {'Pred':>6} {'Src':>5}")
        print("-" * 95)
        
        for _, bet in value_bets_df.iterrows():
            print(f"{bet['match']:<45} {bet['bet']:<12} {bet['prob']*100:>7.1f}% {bet['odds']:>6.2f} {bet['ev']*100:>7.1f}% {bet['pred_goals']:>6.2f} {bet['source']:>5}")
        
        print(f"\nTotal value bets: {len(value_bets)}")
    
    # Show all predictions
    print("\n" + "=" * 60)
    print("ALL EUROPEAN PREDICTIONS")
    print("=" * 60)
    
    print(f"\n{'Match':<45} {'Pred':>10} {'Over%':>8} {'Under%':>8} {'Src':>5}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        match = f"{row['home_team']} vs {row['away_team']}"
        pred = f"{row['pred_home']:.1f}-{row['pred_away']:.1f}"
        print(f"{match:<45} {pred:>10} {row['over_prob']*100:>7.1f}% {row['under_prob']*100:>7.1f}% {row['data_source']:>5}")
    
    # Save results
    output_path = Path("data/processed/european/today_predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()