#!/usr/bin/env python3
"""
Generate European competition value bets for today using Norsk Tipping odds.

This script:
1. Loads European matches from Norsk Tipping scraper
2. Uses the trained European model to predict over/under probabilities
3. Compares with bookmaker odds to find value bets
4. Generates a report

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


# Team name mappings: Norsk Tipping -> European Model
EURO_TEAM_MAPPINGS = {
    # Champions League / Europa League teams
    "Feyenoord": "Feyenoord",
    "Celtic": "Celtic",
    "Viktoria Plzen": "Viktoria Plzen",
    "Freiburg": "Freiburg",
    "PAOK": "PAOK Saloniki",
    "Brann": "Brann",  # Not in dataset - Norwegian team
    "Porto": "Porto",
    "Nice": "Nice",
    "Ludogorets 1947": "PFC Ludogorets Razgrad",
    "Ludogorets": "PFC Ludogorets Razgrad",
    "Celta Vigo": "Celta Vigo",  # Not in dataset
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
    "FC Utrecht": "FC Utrecht",  # Not in dataset
    "Utrecht": "FC Utrecht",
    "Røde Stjerne": "Red Star Belgrade",
    "Red Star Belgrade": "Red Star Belgrade",
    "FCSB": "FCSB",
    "Maccabi Tel Aviv": "Maccabi Tel Aviv",
    "Lyon": "Lyon",
    "Nottingham Forest": "Nottingham Forest",  # Not in dataset
    "Malmö FF": "Malmo",
    "Malmo": "Malmo",
    "Go Ahead Eagles": "Go Ahead Eagles",  # Not in dataset
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
    "Crystal Palace": "Crystal Palace",  # Not in dataset
    "Strasbourg": "Strasbourg",  # Not in dataset
    "Aberdeen": "Aberdeen",
    "HNK Rijeka": "Rijeka",  # Not in dataset
    "Rijeka": "Rijeka",
    "Häcken": "Hacken",
    "Hacken": "Hacken",
    "Zrinjski Mostar": "Zrinjski Mostar",
    "AEK Athens": "AEK Athens",
    "Shamrock Rovers": "Shamrock Rovers",
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
        return None, None, None
    
    home_model = joblib.load(home_path)
    away_model = joblib.load(away_path)
    feature_cols = features_path.read_text().strip().split("\n")
    
    return home_model, away_model, feature_cols


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
    home_model, away_model, feature_cols = load_european_model(cfg)
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
    
    # Load Norsk Tipping odds
    odds_path = Path("data/upcoming/norsk_tipping_odds.csv")
    if not odds_path.exists():
        print("Norsk Tipping odds not found. Run scraper first.")
        return
    
    odds_df = pd.read_csv(odds_path)
    
    # Filter to European matches with over/under odds
    euro_matches = odds_df[
        odds_df["league"].str.contains("Europa|Conference", case=False, na=False) &
        odds_df["over_2_5"].notna()
    ].copy()
    
    print(f"\nFound {len(euro_matches)} European matches from Norsk Tipping")
    
    # Process each match
    results = []
    
    for _, row in euro_matches.iterrows():
        home_nt = row["home_team"]
        away_nt = row["away_team"]
        
        # Map team names
        home_mapped = map_team_name(home_nt)
        away_mapped = map_team_name(away_nt)
        
        # Check if teams exist in dataset
        home_in_data = home_mapped in euro_teams if home_mapped else False
        away_in_data = away_mapped in euro_teams if away_mapped else False
        
        if not home_in_data or not away_in_data:
            missing = []
            if not home_in_data:
                missing.append(f"{home_nt} (mapped: {home_mapped})")
            if not away_in_data:
                missing.append(f"{away_nt} (mapped: {away_mapped})")
            print(f"  Skipping {home_nt} vs {away_nt}: Missing teams: {', '.join(missing)}")
            continue
        
        # Get team features
        home_features = get_team_features(home_mapped, historical_df, is_home=True)
        away_features = get_team_features(away_mapped, historical_df, is_home=False)
        
        if home_features is None or away_features is None:
            print(f"  Skipping {home_nt} vs {away_nt}: Insufficient European history")
            continue
        
        # Build feature vector
        try:
            # Combine features
            features = {**home_features, **away_features}
            
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
            
            # Create feature vector
            X = pd.DataFrame([features])[feature_cols]
            
            # Predict
            pred_home = home_model.predict(X)[0]
            pred_away = away_model.predict(X)[0]
            pred_total = pred_home + pred_away
            
            # Calculate over probability using bivariate Poisson
            over_prob = bivariate_poisson_over_prob(pred_home, pred_away, 2.5)
            under_prob = 1.0 - over_prob
            
            # Get odds
            over_odds = row["over_2_5"]
            under_odds = row["under_2_5"]
            
            # Calculate edge
            over_edge = (over_prob * over_odds) - 1
            under_edge = (under_prob * under_odds) - 1
            
            results.append({
                "home_team": home_nt,
                "away_team": away_nt,
                "league": row["league"],
                "pred_home": pred_home,
                "pred_away": pred_away,
                "pred_total": pred_total,
                "over_prob": over_prob,
                "under_prob": under_prob,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "over_ev": over_edge,
                "under_ev": under_edge,
            })
            
        except Exception as e:
            print(f"  Error predicting {home_nt} vs {away_nt}: {e}")
            continue
    
    if not results:
        print("\nNo matches could be predicted.")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find value bets (EV > 0)
    print("\n" + "=" * 60)
    print("VALUE BETS (EV > 0%)")
    print("=" * 60)
    
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
            })
        
        if row["under_ev"] > 0:
            value_bets.append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "bet": "Under 2.5",
                "prob": row["under_prob"],
                "odds": row["under_odds"],
                "ev": row["under_ev"],
                "pred_goals": row["pred_total"],
            })
    
    if not value_bets:
        print("\nNo value bets found in European competitions today.")
    else:
        # Sort by EV
        value_bets_df = pd.DataFrame(value_bets).sort_values("ev", ascending=False)
        
        print(f"\n{'Match':<45} {'Bet':<12} {'Prob':>8} {'Odds':>6} {'EV':>8} {'Pred':>6}")
        print("-" * 90)
        
        for _, bet in value_bets_df.iterrows():
            print(f"{bet['match']:<45} {bet['bet']:<12} {bet['prob']*100:>7.1f}% {bet['odds']:>6.2f} {bet['ev']*100:>7.1f}% {bet['pred_goals']:>6.2f}")
        
        print(f"\nTotal value bets: {len(value_bets)}")
    
    # Show all predictions
    print("\n" + "=" * 60)
    print("ALL EUROPEAN PREDICTIONS")
    print("=" * 60)
    
    print(f"\n{'Match':<45} {'Pred':>10} {'Over%':>8} {'Under%':>8}")
    print("-" * 75)
    
    for _, row in results_df.iterrows():
        match = f"{row['home_team']} vs {row['away_team']}"
        pred = f"{row['pred_home']:.1f}-{row['pred_away']:.1f}"
        print(f"{match:<45} {pred:>10} {row['over_prob']*100:>7.1f}% {row['under_prob']*100:>7.1f}%")
    
    # Save results
    output_path = Path("data/processed/european/today_predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()
