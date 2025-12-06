#!/usr/bin/env python3
"""
Train Gradient Boosting Models for Football Betting.

This script trains LightGBM and XGBoost models for goal prediction
and compares them with the existing Poisson regression models.

Usage:
    python train_gradient_boosting.py [--compare] [--lightgbm-only] [--xgboost-only]

Options:
    --compare       Compare with existing Poisson models
    --lightgbm-only Train only LightGBM models
    --xgboost-only  Train only XGBoost models
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.config import get_config
from src.models import load_dataset, select_feature_columns, assign_splits
from src.gradient_boosting_models import (
    GBModelConfig,
    GBModelBundle,
    train_gradient_boosting_models,
    predict_with_bundle,
    save_model_bundle,
    load_model_bundle,
    compare_models,
    get_feature_importance,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
)


def check_dependencies():
    """Check that required libraries are installed."""
    missing = []
    if not HAS_LIGHTGBM:
        missing.append("lightgbm")
    if not HAS_XGBOOST:
        missing.append("xgboost")
    
    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print(f"The following packages are not installed: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"    pip install {' '.join(missing)}")
        print("=" * 60)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train gradient boosting models for football betting"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with existing Poisson regression models",
    )
    parser.add_argument(
        "--lightgbm-only",
        action="store_true",
        help="Train only LightGBM models",
    )
    parser.add_argument(
        "--xgboost-only",
        action="store_true",
        help="Train only XGBoost models",
    )
    parser.add_argument(
        "--show-importance",
        action="store_true",
        help="Show top feature importances",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to CSV",
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        if args.lightgbm_only and not HAS_LIGHTGBM:
            print("ERROR: LightGBM requested but not installed")
            sys.exit(1)
        if args.xgboost_only and not HAS_XGBOOST:
            print("ERROR: XGBoost requested but not installed")
            sys.exit(1)
        if not args.lightgbm_only and not args.xgboost_only:
            print("ERROR: Neither LightGBM nor XGBoost is installed")
            sys.exit(1)
    
    # Load configuration
    cfg = get_config()
    
    print("=" * 60)
    print("GRADIENT BOOSTING MODEL TRAINING")
    print("=" * 60)
    
    # Load and prepare dataset
    print("\nLoading dataset...")
    df = load_dataset(cfg)
    feature_columns = select_feature_columns(df)
    
    # Separate optional features (xG, advanced stats, weather, etc.) from core features
    # Core features are required for training; optional features are filled with 0 if missing
    # Define exact names of optional features that may have missing data
    optional_feature_names = {
        # xG features (will be checked by 'xg' substring)
        # Advanced stats from FBref (start with home_adv_ or away_adv_)
        # Weather data (start with weather_)
        # Injury features
        'home_injury_count', 'away_injury_count',
        'home_injury_severity', 'away_injury_severity',
        'home_suspended_count', 'away_suspended_count',
        'injury_count_diff', 'injury_severity_diff',
        # Manager features
        'home_manager_tenure_days', 'away_manager_tenure_days',
        'home_new_manager', 'away_new_manager',
        'home_experienced_manager', 'away_experienced_manager',
        'manager_tenure_diff',
    }
    optional_prefixes = ('home_adv_', 'away_adv_', 'weather_')
    
    def is_optional(col):
        if col in optional_feature_names:
            return True
        if 'xg' in col.lower():
            return True
        for prefix in optional_prefixes:
            if col.startswith(prefix):
                return True
        return False
    
    optional_features = [c for c in feature_columns if is_optional(c)]
    core_features = [c for c in feature_columns if c not in optional_features]
    
    print(f"Total features: {len(feature_columns)} (core: {len(core_features)}, optional: {len(optional_features)})")
    
    # Drop rows with missing core features only
    df = df.dropna(subset=core_features)
    
    # Fill optional features with 0 for rows that don't have them
    for col in optional_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"Dataset size after cleaning: {len(df):,} matches")
    
    # Configure model training
    model_cfg = GBModelConfig(
        use_lightgbm=not args.xgboost_only and HAS_LIGHTGBM,
        use_xgboost=not args.lightgbm_only and HAS_XGBOOST,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50,
    )
    
    # Train models
    lgb_bundle, xgb_bundle = train_gradient_boosting_models(
        df=df,
        feature_columns=feature_columns,
        cfg=model_cfg,
        validation_season=cfg.model.validation_season,
        test_season=cfg.model.test_season,
    )
    
    # Save models
    if lgb_bundle is not None:
        save_model_bundle(lgb_bundle, cfg.models_dir)
    if xgb_bundle is not None:
        save_model_bundle(xgb_bundle, cfg.models_dir)
    
    # Show feature importance
    if args.show_importance:
        print("\n" + "=" * 60)
        print("TOP 20 FEATURE IMPORTANCES")
        print("=" * 60)
        
        bundle = lgb_bundle or xgb_bundle
        if bundle is not None:
            home_importance = get_feature_importance(
                bundle.home_model, feature_columns
            ).head(20)
            away_importance = get_feature_importance(
                bundle.away_model, feature_columns
            ).head(20)
            
            print("\nHome Goals Model:")
            print(home_importance.to_string(index=False))
            
            print("\nAway Goals Model:")
            print(away_importance.to_string(index=False))
    
    # Compare with Poisson models
    if args.compare:
        print("\n" + "=" * 60)
        print("COMPARING WITH POISSON REGRESSION")
        print("=" * 60)
        
        try:
            home_poisson = joblib.load(cfg.models_dir / "home_poisson.joblib")
            away_poisson = joblib.load(cfg.models_dir / "away_poisson.joblib")
            
            bundle = lgb_bundle or xgb_bundle
            if bundle is not None:
                compare_models(
                    df=df,
                    feature_columns=feature_columns,
                    poisson_home_model=home_poisson,
                    poisson_away_model=away_poisson,
                    gb_bundle=bundle,
                    test_season=cfg.model.test_season,
                )
        except FileNotFoundError:
            print("Poisson models not found. Run 'python -m src.models' first.")
    
    # Save predictions
    if args.save_predictions:
        print("\n" + "=" * 60)
        print("SAVING PREDICTIONS")
        print("=" * 60)
        
        bundle = lgb_bundle or xgb_bundle
        if bundle is not None:
            df_with_preds = predict_with_bundle(bundle, df, model_cfg)
            
            output_path = cfg.processed_dir / f"{bundle.model_type}_predictions.csv"
            df_with_preds.to_csv(output_path, index=False)
            print(f"Saved predictions to {output_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Summary
    print("\nTrained models saved to:", cfg.models_dir)
    if lgb_bundle is not None:
        print(f"  - LightGBM: lightgbm_home.joblib, lightgbm_away.joblib")
    if xgb_bundle is not None:
        print(f"  - XGBoost: xgboost_home.joblib, xgboost_away.joblib")
    
    print("\nNext steps:")
    print("  1. Run with --compare to see improvement over Poisson regression")
    print("  2. Run with --show-importance to see which features matter most")
    print("  3. Run with --save-predictions to generate predictions CSV")


if __name__ == "__main__":
    main()
