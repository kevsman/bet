#!/usr/bin/env python3
"""
Evaluate Poisson and Dixon-Coles models on historical data.

Tests model accuracy on:
1. Match outcome (1X2) predictions
2. Over/Under goal line predictions (1.5, 2.5, 3.5)
3. Expected goals calibration
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.config import get_config
from src.dixon_coles import DixonColesModel, dixon_coles_probability


def load_models():
    """Load both Poisson and Dixon-Coles models."""
    cfg = get_config()
    
    # Load Poisson models
    home_model = joblib.load(cfg.models_dir / "home_poisson.joblib")
    away_model = joblib.load(cfg.models_dir / "away_poisson.joblib")
    feature_cols = (cfg.models_dir / "features.txt").read_text().strip().split("\n")
    
    # Load Dixon-Coles model
    dc_path = cfg.models_dir / "dixon_coles.joblib"
    dc_model = None
    if dc_path.exists():
        loaded = joblib.load(dc_path)
        # Handle both dict and object formats
        if isinstance(loaded, dict):
            dc_model = DixonColesModel()
            dc_model.params = loaded.get('params')
            dc_model.teams = loaded.get('teams', [])
            dc_model._fitted = loaded.get('_fitted', True)
        else:
            dc_model = loaded
        print("✓ Dixon-Coles model loaded")
    else:
        print("✗ Dixon-Coles model not found")
    
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
    home_win = np.sum(np.tril(probs, -1))  # Below diagonal
    draw = np.sum(np.diag(probs))
    away_win = np.sum(np.triu(probs, 1))  # Above diagonal
    
    # Goal line probabilities
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


def evaluate_on_test_set(test_df: pd.DataFrame, home_model, away_model, feature_cols, dc_model) -> Dict:
    """Evaluate models on a test set."""
    
    results = {
        'poisson': {'1x2': [], 'ou_15': [], 'ou_25': [], 'ou_35': [], 'xg_home': [], 'xg_away': []},
        'dixon_coles': {'1x2': [], 'ou_15': [], 'ou_25': [], 'ou_35': [], 'xg_home': [], 'xg_away': []},
        'actual': {'home_goals': [], 'away_goals': [], 'total_goals': [], 'result': []}
    }
    
    skipped = 0
    evaluated = 0
    
    for _, row in test_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        
        if pd.isna(home_goals) or pd.isna(away_goals):
            skipped += 1
            continue
        
        home_goals = int(home_goals)
        away_goals = int(away_goals)
        total_goals = home_goals + away_goals
        
        # Actual result
        if home_goals > away_goals:
            actual_result = 'H'
        elif home_goals < away_goals:
            actual_result = 'A'
        else:
            actual_result = 'D'
        
        # Get Poisson predictions (need features)
        try:
            # Build feature vector
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
            
            # Calculate Poisson probabilities
            poisson_probs = calculate_goal_probs(poisson_home_exp, poisson_away_exp)
            poisson_outcomes = probs_to_outcomes(poisson_probs)
            
            results['poisson']['xg_home'].append(poisson_home_exp)
            results['poisson']['xg_away'].append(poisson_away_exp)
            
            # Predicted result
            if poisson_outcomes['home_win'] > poisson_outcomes['away_win'] and poisson_outcomes['home_win'] > poisson_outcomes['draw']:
                poisson_pred = 'H'
            elif poisson_outcomes['away_win'] > poisson_outcomes['home_win'] and poisson_outcomes['away_win'] > poisson_outcomes['draw']:
                poisson_pred = 'A'
            else:
                poisson_pred = 'D'
            
            results['poisson']['1x2'].append(1 if poisson_pred == actual_result else 0)
            results['poisson']['ou_15'].append(1 if (poisson_outcomes['over_1.5'] > 0.5) == (total_goals > 1.5) else 0)
            results['poisson']['ou_25'].append(1 if (poisson_outcomes['over_2.5'] > 0.5) == (total_goals > 2.5) else 0)
            results['poisson']['ou_35'].append(1 if (poisson_outcomes['over_3.5'] > 0.5) == (total_goals > 3.5) else 0)
            
        except Exception as e:
            skipped += 1
            continue
        
        # Get Dixon-Coles predictions
        if dc_model and dc_model._fitted:
            try:
                dc_home_exp, dc_away_exp = dc_model.predict_goals(home_team, away_team)
                dc_probs = calculate_dc_goal_probs(dc_home_exp, dc_away_exp, dc_model.params.rho)
                dc_outcomes = probs_to_outcomes(dc_probs)
                
                results['dixon_coles']['xg_home'].append(dc_home_exp)
                results['dixon_coles']['xg_away'].append(dc_away_exp)
                
                # Predicted result
                if dc_outcomes['home_win'] > dc_outcomes['away_win'] and dc_outcomes['home_win'] > dc_outcomes['draw']:
                    dc_pred = 'H'
                elif dc_outcomes['away_win'] > dc_outcomes['home_win'] and dc_outcomes['away_win'] > dc_outcomes['draw']:
                    dc_pred = 'A'
                else:
                    dc_pred = 'D'
                
                results['dixon_coles']['1x2'].append(1 if dc_pred == actual_result else 0)
                results['dixon_coles']['ou_15'].append(1 if (dc_outcomes['over_1.5'] > 0.5) == (total_goals > 1.5) else 0)
                results['dixon_coles']['ou_25'].append(1 if (dc_outcomes['over_2.5'] > 0.5) == (total_goals > 2.5) else 0)
                results['dixon_coles']['ou_35'].append(1 if (dc_outcomes['over_3.5'] > 0.5) == (total_goals > 3.5) else 0)
                
            except Exception:
                # Team not in DC model
                results['dixon_coles']['1x2'].append(np.nan)
                results['dixon_coles']['ou_15'].append(np.nan)
                results['dixon_coles']['ou_25'].append(np.nan)
                results['dixon_coles']['ou_35'].append(np.nan)
                results['dixon_coles']['xg_home'].append(np.nan)
                results['dixon_coles']['xg_away'].append(np.nan)
        
        results['actual']['home_goals'].append(home_goals)
        results['actual']['away_goals'].append(away_goals)
        results['actual']['total_goals'].append(total_goals)
        results['actual']['result'].append(actual_result)
        evaluated += 1
    
    return results, evaluated, skipped


def calculate_metrics(results: Dict) -> Dict:
    """Calculate accuracy metrics for each model."""
    metrics = {}
    
    for model in ['poisson', 'dixon_coles']:
        model_metrics = {}
        
        # 1X2 accuracy
        valid_1x2 = [x for x in results[model]['1x2'] if not np.isnan(x)]
        model_metrics['1x2_accuracy'] = np.mean(valid_1x2) if valid_1x2 else np.nan
        model_metrics['1x2_count'] = len(valid_1x2)
        
        # O/U accuracy
        for line in ['15', '25', '35']:
            key = f'ou_{line}'
            valid = [x for x in results[model][key] if not np.isnan(x)]
            model_metrics[f'{key}_accuracy'] = np.mean(valid) if valid else np.nan
            model_metrics[f'{key}_count'] = len(valid)
        
        # xG calibration (MAE)
        valid_home = [(p, a) for p, a in zip(results[model]['xg_home'], results['actual']['home_goals']) if not np.isnan(p)]
        valid_away = [(p, a) for p, a in zip(results[model]['xg_away'], results['actual']['away_goals']) if not np.isnan(p)]
        
        if valid_home:
            model_metrics['home_xg_mae'] = np.mean([abs(p - a) for p, a in valid_home])
            model_metrics['home_xg_bias'] = np.mean([p - a for p, a in valid_home])
        if valid_away:
            model_metrics['away_xg_mae'] = np.mean([abs(p - a) for p, a in valid_away])
            model_metrics['away_xg_bias'] = np.mean([p - a for p, a in valid_away])
        
        metrics[model] = model_metrics
    
    return metrics


def run_evaluation():
    """Main evaluation function."""
    print("="*70)
    print("MODEL EVALUATION ON HISTORICAL DATA")
    print("="*70)
    
    # Load models
    home_model, away_model, feature_cols, dc_model = load_models()
    
    # Load dataset
    cfg = get_config()
    
    # Try xG-enhanced dataset first
    dataset_path = cfg.processed_dir / "match_dataset_with_xg.csv"
    if not dataset_path.exists():
        dataset_path = cfg.processed_dir / "match_dataset.csv"
    
    print(f"\nLoading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use last 6 months as test set (out-of-sample)
    cutoff_date = df['Date'].max() - timedelta(days=180)
    test_df = df[df['Date'] > cutoff_date].copy()
    
    print(f"Test set: {len(test_df)} matches from {cutoff_date.strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Evaluate
    print("\nEvaluating models...")
    results, evaluated, skipped = evaluate_on_test_set(test_df, home_model, away_model, feature_cols, dc_model)
    
    print(f"\nEvaluated: {evaluated} matches, Skipped: {skipped} matches")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print("\n📊 1X2 Match Result Prediction:")
    print(f"   Poisson:      {metrics['poisson']['1x2_accuracy']*100:.1f}% ({metrics['poisson']['1x2_count']} matches)")
    if not np.isnan(metrics['dixon_coles']['1x2_accuracy']):
        print(f"   Dixon-Coles:  {metrics['dixon_coles']['1x2_accuracy']*100:.1f}% ({metrics['dixon_coles']['1x2_count']} matches)")
    
    print("\n📊 Over/Under 1.5 Goals:")
    print(f"   Poisson:      {metrics['poisson']['ou_15_accuracy']*100:.1f}%")
    if not np.isnan(metrics['dixon_coles']['ou_15_accuracy']):
        print(f"   Dixon-Coles:  {metrics['dixon_coles']['ou_15_accuracy']*100:.1f}%")
    
    print("\n📊 Over/Under 2.5 Goals:")
    print(f"   Poisson:      {metrics['poisson']['ou_25_accuracy']*100:.1f}%")
    if not np.isnan(metrics['dixon_coles']['ou_25_accuracy']):
        print(f"   Dixon-Coles:  {metrics['dixon_coles']['ou_25_accuracy']*100:.1f}%")
    
    print("\n📊 Over/Under 3.5 Goals:")
    print(f"   Poisson:      {metrics['poisson']['ou_35_accuracy']*100:.1f}%")
    if not np.isnan(metrics['dixon_coles']['ou_35_accuracy']):
        print(f"   Dixon-Coles:  {metrics['dixon_coles']['ou_35_accuracy']*100:.1f}%")
    
    print("\n📊 Expected Goals Calibration (MAE / Bias):")
    print(f"   Poisson Home:      {metrics['poisson']['home_xg_mae']:.3f} / {metrics['poisson']['home_xg_bias']:+.3f}")
    print(f"   Poisson Away:      {metrics['poisson']['away_xg_mae']:.3f} / {metrics['poisson']['away_xg_bias']:+.3f}")
    if 'home_xg_mae' in metrics['dixon_coles']:
        print(f"   Dixon-Coles Home:  {metrics['dixon_coles']['home_xg_mae']:.3f} / {metrics['dixon_coles']['home_xg_bias']:+.3f}")
        print(f"   Dixon-Coles Away:  {metrics['dixon_coles']['away_xg_mae']:.3f} / {metrics['dixon_coles']['away_xg_bias']:+.3f}")
    
    # Actual distribution
    actual_total = results['actual']['total_goals']
    print(f"\n📊 Actual Goals Distribution (test set):")
    print(f"   Average total goals: {np.mean(actual_total):.2f}")
    print(f"   Over 1.5: {100*sum(1 for g in actual_total if g > 1.5)/len(actual_total):.1f}%")
    print(f"   Over 2.5: {100*sum(1 for g in actual_total if g > 2.5)/len(actual_total):.1f}%")
    print(f"   Over 3.5: {100*sum(1 for g in actual_total if g > 3.5)/len(actual_total):.1f}%")
    
    # Result distribution
    result_counts = pd.Series(results['actual']['result']).value_counts()
    print(f"\n📊 Actual Result Distribution:")
    print(f"   Home wins: {100*result_counts.get('H', 0)/len(results['actual']['result']):.1f}%")
    print(f"   Draws:     {100*result_counts.get('D', 0)/len(results['actual']['result']):.1f}%")
    print(f"   Away wins: {100*result_counts.get('A', 0)/len(results['actual']['result']):.1f}%")
    
    return metrics


if __name__ == "__main__":
    run_evaluation()
