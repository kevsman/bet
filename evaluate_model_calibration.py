#!/usr/bin/env python3
"""
Comprehensive Model Calibration Analysis

This script provides deep insights into model performance:
1. Probability calibration curves - do predicted probabilities match actual outcomes?
2. Reliability diagrams - which probability ranges are most/least accurate?
3. Brier score and log loss by probability bucket
4. Edge analysis - at what edge levels are bets profitable?
5. Line-specific performance (1.5, 2.5, 3.5 goals)
6. ROI analysis by predicted probability range
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.config import get_config
from src.dixon_coles import DixonColesModel, dixon_coles_probability


def load_models():
    """Load both Poisson and Dixon-Coles models."""
    cfg = get_config()
    
    home_model = joblib.load(cfg.models_dir / "home_poisson.joblib")
    away_model = joblib.load(cfg.models_dir / "away_poisson.joblib")
    feature_cols = (cfg.models_dir / "features.txt").read_text().strip().split("\n")
    
    dc_path = cfg.models_dir / "dixon_coles.joblib"
    dc_model = None
    if dc_path.exists():
        loaded = joblib.load(dc_path)
        if isinstance(loaded, dict):
            dc_model = DixonColesModel()
            dc_model.params = loaded.get('params')
            dc_model.teams = loaded.get('teams', [])
            dc_model._fitted = loaded.get('_fitted', True)
        else:
            dc_model = loaded
        print("[OK] Dixon-Coles model loaded")
    
    return home_model, away_model, feature_cols, dc_model


def calculate_goal_probs(home_exp: float, away_exp: float, max_goals: int = 10) -> np.ndarray:
    """Calculate goal probability matrix using independent Poisson."""
    probs = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            probs[h, a] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)
    return probs


def calculate_dc_goal_probs(home_exp: float, away_exp: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Calculate goal probability matrix using Dixon-Coles."""
    probs = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            probs[h, a] = dixon_coles_probability(h, a, home_exp, away_exp, rho)
    return probs


def probs_to_outcomes(probs: np.ndarray) -> Dict[str, float]:
    """Convert goal probability matrix to outcome probabilities."""
    home_win = np.sum(np.tril(probs, -1))
    draw = np.sum(np.diag(probs))
    away_win = np.sum(np.triu(probs, 1))
    
    total_probs = {}
    for line in [1.5, 2.5, 3.5, 4.5]:
        over = 0.0
        under = 0.0
        for h in range(probs.shape[0]):
            for a in range(probs.shape[1]):
                total = h + a
                if total > line:
                    over += probs[h, a]
                else:
                    under += probs[h, a]
        total_probs[f'over_{line}'] = over
        total_probs[f'under_{line}'] = under
    
    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win,
        **total_probs
    }


def generate_predictions(df: pd.DataFrame, home_model, away_model, feature_cols, dc_model) -> pd.DataFrame:
    """Generate predictions for all matches."""
    predictions = []
    
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row.get('FTHG')
        away_goals = row.get('FTAG')
        
        if pd.isna(home_goals) or pd.isna(away_goals):
            continue
            
        home_goals = int(home_goals)
        away_goals = int(away_goals)
        total_goals = home_goals + away_goals
        
        # Build feature vector for Poisson
        try:
            features = []
            for col in feature_cols:
                if col in row:
                    val = row[col]
                    features.append(0.0 if pd.isna(val) else float(val))
                else:
                    features.append(0.0)
            
            X = np.array(features).reshape(1, -1)
            poisson_home_exp = float(home_model.predict(X)[0])
            poisson_away_exp = float(away_model.predict(X)[0])
            
            poisson_probs = calculate_goal_probs(poisson_home_exp, poisson_away_exp)
            poisson_outcomes = probs_to_outcomes(poisson_probs)
        except Exception:
            continue
        
        # Dixon-Coles predictions
        dc_home_exp, dc_away_exp, dc_outcomes = None, None, None
        if dc_model and dc_model._fitted:
            try:
                dc_home_exp, dc_away_exp = dc_model.predict_goals(home_team, away_team)
                dc_probs = calculate_dc_goal_probs(dc_home_exp, dc_away_exp, dc_model.params.rho)
                dc_outcomes = probs_to_outcomes(dc_probs)
            except Exception:
                pass
        
        # Get odds if available
        over_25_odds = row.get('BbAv>2.5', row.get('Avg>2.5', np.nan))
        under_25_odds = row.get('BbAv<2.5', row.get('Avg<2.5', np.nan))
        over_35_odds = row.get('BbAv>3.5', row.get('Avg>3.5', np.nan))
        under_35_odds = row.get('BbAv<3.5', row.get('Avg<3.5', np.nan))
        
        pred = {
            'date': row.get('Date'),
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'total_goals': total_goals,
            # Poisson predictions
            'poisson_home_xg': poisson_home_exp,
            'poisson_away_xg': poisson_away_exp,
            'poisson_over_15': poisson_outcomes['over_1.5'],
            'poisson_over_25': poisson_outcomes['over_2.5'],
            'poisson_over_35': poisson_outcomes['over_3.5'],
            'poisson_under_25': poisson_outcomes['under_2.5'],
            'poisson_under_35': poisson_outcomes['under_3.5'],
            # Actual outcomes
            'actual_over_15': 1 if total_goals > 1.5 else 0,
            'actual_over_25': 1 if total_goals > 2.5 else 0,
            'actual_over_35': 1 if total_goals > 3.5 else 0,
            # Odds
            'over_25_odds': over_25_odds,
            'under_25_odds': under_25_odds,
            'over_35_odds': over_35_odds,
            'under_35_odds': under_35_odds,
        }
        
        if dc_outcomes:
            pred['dc_home_xg'] = dc_home_exp
            pred['dc_away_xg'] = dc_away_exp
            pred['dc_over_15'] = dc_outcomes['over_1.5']
            pred['dc_over_25'] = dc_outcomes['over_2.5']
            pred['dc_over_35'] = dc_outcomes['over_3.5']
            pred['dc_under_25'] = dc_outcomes['under_2.5']
            pred['dc_under_35'] = dc_outcomes['under_3.5']
        
        predictions.append(pred)
    
    return pd.DataFrame(predictions)


def analyze_calibration(preds: pd.DataFrame, prob_col: str, outcome_col: str, n_bins: int = 10) -> Dict:
    """Analyze probability calibration - do predicted probabilities match reality?"""
    valid = preds[[prob_col, outcome_col]].dropna()
    if len(valid) == 0:
        return None
    
    probs = valid[prob_col].values
    outcomes = valid[outcome_col].values
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_results = []
    
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if i == n_bins - 1:  # Include upper bound in last bin
            mask = (probs >= bins[i]) & (probs <= bins[i+1])
        
        if mask.sum() > 0:
            bin_probs = probs[mask]
            bin_outcomes = outcomes[mask]
            
            bin_results.append({
                'bin_center': (bins[i] + bins[i+1]) / 2,
                'bin_low': bins[i],
                'bin_high': bins[i+1],
                'mean_pred_prob': np.mean(bin_probs),
                'actual_hit_rate': np.mean(bin_outcomes),
                'count': len(bin_probs),
                'calibration_error': abs(np.mean(bin_probs) - np.mean(bin_outcomes)),
            })
    
    # Overall metrics
    brier_score = np.mean((probs - outcomes) ** 2)
    log_loss = -np.mean(outcomes * np.log(np.clip(probs, 1e-10, 1)) + 
                        (1-outcomes) * np.log(np.clip(1-probs, 1e-10, 1)))
    
    # Expected Calibration Error (ECE)
    ece = sum(r['count'] * r['calibration_error'] for r in bin_results) / len(probs)
    
    return {
        'bins': bin_results,
        'brier_score': brier_score,
        'log_loss': log_loss,
        'ece': ece,
        'total_matches': len(probs),
        'hit_rate': np.mean(outcomes),
        'mean_predicted': np.mean(probs),
    }


def analyze_edge_performance(preds: pd.DataFrame, prob_col: str, outcome_col: str, 
                             odds_col: str, selection: str) -> Dict:
    """Analyze ROI by edge level (predicted prob vs implied prob from odds)."""
    valid = preds[[prob_col, outcome_col, odds_col]].dropna()
    if len(valid) == 0:
        return None
    
    results_by_edge = []
    
    # Edge buckets: 0-2%, 2-5%, 5-10%, 10-15%, 15-20%, 20%+
    edge_buckets = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]
    
    for low, high in edge_buckets:
        # Calculate edge for each row
        implied_prob = 1 / valid[odds_col]
        edge = valid[prob_col] - implied_prob
        
        mask = (edge >= low) & (edge < high)
        if mask.sum() < 5:
            continue
        
        subset = valid[mask]
        odds = subset[odds_col].values
        outcomes = subset[outcome_col].values
        probs = subset[prob_col].values
        
        # Calculate ROI
        profits = np.where(outcomes == 1, odds - 1, -1)
        roi = np.sum(profits) / len(profits)
        hit_rate = np.mean(outcomes)
        
        results_by_edge.append({
            'edge_range': f"{low*100:.0f}%-{high*100:.0f}%",
            'edge_low': low,
            'edge_high': high,
            'count': len(subset),
            'hit_rate': hit_rate,
            'avg_odds': np.mean(odds),
            'avg_pred_prob': np.mean(probs),
            'avg_implied_prob': np.mean(implied_prob[mask]),
            'roi': roi,
            'total_profit': np.sum(profits),
        })
    
    return results_by_edge


def print_calibration_report(calibration: Dict, title: str):
    """Print calibration analysis results."""
    if not calibration:
        print(f"\n{title}: No data")
        return
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  - Total Matches: {calibration['total_matches']}")
    print(f"  - Actual Hit Rate: {calibration['hit_rate']*100:.1f}%")
    print(f"  - Mean Predicted Prob: {calibration['mean_predicted']*100:.1f}%")
    print(f"  - Brier Score: {calibration['brier_score']:.4f} (lower is better, 0.25 is random)")
    print(f"  - Log Loss: {calibration['log_loss']:.4f}")
    print(f"  - ECE (Expected Calibration Error): {calibration['ece']*100:.2f}%")
    
    print(f"\nCalibration by Probability Bucket:")
    print(f"  {'Predicted':<12} {'Actual':<10} {'Count':<8} {'Error':<8} {'Assessment'}")
    print(f"  {'-'*60}")
    
    for b in calibration['bins']:
        error_pct = b['calibration_error'] * 100
        if error_pct < 3:
            assessment = "[EXCELLENT]"
        elif error_pct < 5:
            assessment = "[GOOD]"
        elif error_pct < 10:
            assessment = "[OK]"
        else:
            assessment = "[POOR]"
        
        print(f"  {b['mean_pred_prob']*100:5.1f}%-{b['bin_high']*100:4.0f}%  "
              f"{b['actual_hit_rate']*100:5.1f}%    "
              f"{b['count']:5d}    "
              f"{error_pct:5.1f}%   {assessment}")


def print_edge_report(edge_results: List[Dict], title: str):
    """Print edge analysis results."""
    if not edge_results:
        print(f"\n{title}: No data")
        return
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    print(f"\n  {'Edge Range':<12} {'Count':<8} {'Hit Rate':<10} {'Avg Odds':<10} {'ROI':<10} {'Result'}")
    print(f"  {'-'*70}")
    
    for r in edge_results:
        roi_pct = r['roi'] * 100
        result = "[PROFITABLE]" if roi_pct > 0 else "[LOSS]"
        
        print(f"  {r['edge_range']:<12} "
              f"{r['count']:<8} "
              f"{r['hit_rate']*100:5.1f}%     "
              f"{r['avg_odds']:<10.2f} "
              f"{roi_pct:+6.1f}%    {result}")


def run_comprehensive_evaluation():
    """Run comprehensive model evaluation."""
    print("="*70)
    print("COMPREHENSIVE MODEL CALIBRATION ANALYSIS")
    print("="*70)
    
    # Load models
    home_model, away_model, feature_cols, dc_model = load_models()
    
    # Load dataset
    cfg = get_config()
    dataset_path = cfg.processed_dir / "match_dataset_with_xg.csv"
    if not dataset_path.exists():
        dataset_path = cfg.processed_dir / "match_dataset.csv"
    
    print(f"\nLoading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use different test periods
    test_periods = [
        ('Last 6 months', 180),
        ('Last 12 months', 365),
    ]
    
    for period_name, days in test_periods:
        cutoff_date = df['Date'].max() - timedelta(days=days)
        test_df = df[df['Date'] > cutoff_date].copy()
        
        print(f"\n{'#'*70}")
        print(f"# EVALUATION PERIOD: {period_name}")
        print(f"# {len(test_df)} matches from {cutoff_date.strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"{'#'*70}")
        
        # Generate predictions
        print("\nGenerating predictions...")
        preds = generate_predictions(test_df, home_model, away_model, feature_cols, dc_model)
        print(f"Generated predictions for {len(preds)} matches")
        
        # Actual goal distribution
        print(f"\n[ACTUAL GOAL DISTRIBUTION]")
        print(f"  Average total goals: {preds['total_goals'].mean():.2f}")
        print(f"  Over 1.5: {preds['actual_over_15'].mean()*100:.1f}%")
        print(f"  Over 2.5: {preds['actual_over_25'].mean()*100:.1f}%")
        print(f"  Over 3.5: {preds['actual_over_35'].mean()*100:.1f}%")
        
        # Poisson calibration analysis
        print_calibration_report(
            analyze_calibration(preds, 'poisson_over_25', 'actual_over_25', n_bins=10),
            "POISSON MODEL - OVER 2.5 GOALS CALIBRATION"
        )
        
        print_calibration_report(
            analyze_calibration(preds, 'poisson_over_35', 'actual_over_35', n_bins=10),
            "POISSON MODEL - OVER 3.5 GOALS CALIBRATION"
        )
        
        # Actually need to compute under outcome first
        preds['actual_under_25'] = 1 - preds['actual_over_25']
        preds['actual_under_35'] = 1 - preds['actual_over_35']
        
        if 'poisson_under_25' in preds.columns:
            print_calibration_report(
                analyze_calibration(preds, 'poisson_under_25', 'actual_under_25', n_bins=10),
                "POISSON MODEL - UNDER 2.5 GOALS CALIBRATION"
            )
        
        # Dixon-Coles calibration (if available)
        if 'dc_over_25' in preds.columns:
            print_calibration_report(
                analyze_calibration(preds, 'dc_over_25', 'actual_over_25', n_bins=10),
                "DIXON-COLES MODEL - OVER 2.5 GOALS CALIBRATION"
            )
            
            print_calibration_report(
                analyze_calibration(preds, 'dc_over_35', 'actual_over_35', n_bins=10),
                "DIXON-COLES MODEL - OVER 3.5 GOALS CALIBRATION"
            )
        
        # Edge analysis - where are value bets profitable?
        print_edge_report(
            analyze_edge_performance(preds, 'poisson_over_25', 'actual_over_25', 'over_25_odds', 'Over 2.5'),
            "EDGE ANALYSIS - OVER 2.5 GOALS (Poisson vs Market Odds)"
        )
        
        print_edge_report(
            analyze_edge_performance(preds, 'poisson_under_25', 'actual_under_25', 'under_25_odds', 'Under 2.5'),
            "EDGE ANALYSIS - UNDER 2.5 GOALS (Poisson vs Market Odds)"
        )
        
        print_edge_report(
            analyze_edge_performance(preds, 'poisson_over_35', 'actual_over_35', 'over_35_odds', 'Over 3.5'),
            "EDGE ANALYSIS - OVER 3.5 GOALS (Poisson vs Market Odds)"
        )
        
        # xG calibration
        print(f"\n{'='*70}")
        print("EXPECTED GOALS CALIBRATION")
        print(f"{'='*70}")
        
        poisson_total_xg = preds['poisson_home_xg'] + preds['poisson_away_xg']
        actual_total = preds['total_goals']
        
        print(f"\nPoisson Model:")
        print(f"  Mean Predicted Total xG: {poisson_total_xg.mean():.2f}")
        print(f"  Mean Actual Total Goals: {actual_total.mean():.2f}")
        print(f"  Bias (Pred - Actual): {(poisson_total_xg.mean() - actual_total.mean()):+.2f}")
        print(f"  MAE: {np.mean(np.abs(poisson_total_xg - actual_total)):.2f}")
        print(f"  RMSE: {np.sqrt(np.mean((poisson_total_xg - actual_total)**2)):.2f}")
        
        if 'dc_home_xg' in preds.columns:
            dc_total_xg = preds['dc_home_xg'] + preds['dc_away_xg']
            valid_mask = ~dc_total_xg.isna()
            
            print(f"\nDixon-Coles Model:")
            print(f"  Mean Predicted Total xG: {dc_total_xg[valid_mask].mean():.2f}")
            print(f"  Mean Actual Total Goals: {actual_total[valid_mask].mean():.2f}")
            print(f"  Bias (Pred - Actual): {(dc_total_xg[valid_mask].mean() - actual_total[valid_mask].mean()):+.2f}")
            print(f"  MAE: {np.mean(np.abs(dc_total_xg[valid_mask] - actual_total[valid_mask])):.2f}")
            print(f"  RMSE: {np.sqrt(np.mean((dc_total_xg[valid_mask] - actual_total[valid_mask])**2)):.2f}")
    
    # Summary recommendations
    print(f"\n{'='*70}")
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
1. PROBABILITY CALIBRATION:
   - Look at calibration buckets where predicted % closely matches actual %
   - Buckets with <5% error are well-calibrated
   - Use models with good calibration for those probability ranges

2. EDGE ANALYSIS:
   - Positive ROI edge buckets indicate profitable betting opportunities
   - Larger edges (10%+) typically have smaller sample sizes - use caution
   - The sweet spot is usually 5-10% edge with reasonable volume

3. EXPECTED GOALS:
   - Bias > 0 means model overestimates goals (too many Over predictions)
   - Bias < 0 means model underestimates goals (too many Under predictions)
   - Lower MAE/RMSE = better predictions

4. RECOMMENDATIONS:
   - Focus on probability ranges where calibration is best
   - Set minimum edge thresholds based on profitable edge buckets
   - Consider ensemble approaches where models complement each other
""")
    
    return preds


if __name__ == "__main__":
    preds = run_comprehensive_evaluation()
