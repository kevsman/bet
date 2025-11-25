"""
Train and evaluate models for European competitions.
Separate from domestic model due to different data characteristics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .european_config import EuropeanConfig, get_european_config


@dataclass
class EuropeanModelBundle:
    home_model: PoissonRegressor
    away_model: PoissonRegressor
    feature_columns: List[str]
    calibrator: IsotonicRegression | None = None


def load_european_dataset(cfg: EuropeanConfig) -> pd.DataFrame:
    """Load processed European dataset."""
    dataset_path = cfg.processed_dir / "european_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"European dataset not found at {dataset_path}. Run european_prepare first."
        )
    
    df = pd.read_csv(dataset_path, parse_dates=["Date"], low_memory=False)
    df["season_code"] = df["season_code"].astype(str)
    return df


def assign_splits(df: pd.DataFrame, cfg: EuropeanConfig) -> pd.Series:
    """Assign train/validation/test splits."""
    val_mask = df["season_code"] == cfg.model.validation_season
    test_mask = df["season_code"] >= cfg.model.test_season
    splits = np.where(test_mask, "test", np.where(val_mask, "validation", "train"))
    return pd.Series(splits, index=df.index, name="dataset_split")


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select features for European model."""
    candidates = []
    
    for col in df.columns:
        # European form features
        if col.startswith("home_euro_") or col.startswith("away_euro_"):
            candidates.append(col)
        # Domestic form features (from league games)
        elif col.startswith("home_dom_") or col.startswith("away_dom_"):
            candidates.append(col)
        # Competition averages
        elif col.startswith("comp_avg_"):
            candidates.append(col)
        # Stage features
        elif col in {"is_group_stage", "is_knockout", "is_final", "stage_importance"}:
            candidates.append(col)
        # European experience
        elif col in {"home_european_matches", "away_european_matches"}:
            candidates.append(col)
    
    return sorted(candidates)


def train_poisson_model(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    max_iter: int,
) -> PoissonRegressor:
    """Train a Poisson regression model."""
    model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    model.fit(X, y)
    return model


def evaluate(model: PoissonRegressor, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model performance."""
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mse)),
    }


def poisson_over_probability(lam: float, line: float) -> float:
    """Calculate probability of over given total goals lambda."""
    lam = float(np.clip(lam, 0.1, 8.0))  # Higher cap for European games
    threshold = math.floor(line - 0.5)
    return float(poisson.sf(threshold, lam))


def poisson_under_probability(lam: float, line: float) -> float:
    """Calculate probability of under given total goals lambda."""
    lam = float(np.clip(lam, 0.1, 8.0))
    threshold = math.floor(line - 0.5)
    return float(poisson.cdf(threshold, lam))


def train_calibrator(
    raw_probs: np.ndarray,
    actuals: np.ndarray,
) -> IsotonicRegression:
    """Train isotonic calibrator for probabilities."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, actuals)
    return calibrator


def persist_models(cfg: EuropeanConfig, bundle: EuropeanModelBundle) -> None:
    """Save trained models."""
    joblib.dump(bundle.home_model, cfg.models_dir / "euro_home_poisson.joblib")
    joblib.dump(bundle.away_model, cfg.models_dir / "euro_away_poisson.joblib")
    (cfg.models_dir / "euro_features.txt").write_text("\n".join(bundle.feature_columns))
    
    if bundle.calibrator:
        joblib.dump(bundle.calibrator, cfg.models_dir / "euro_calibrator.joblib")


def train_european_model(cfg: EuropeanConfig) -> EuropeanModelBundle:
    """Train the European competition model."""
    df = load_european_dataset(cfg)
    df["dataset_split"] = assign_splits(df, cfg)
    
    feature_cols = select_feature_columns(df)
    print(f"Using {len(feature_cols)} features")
    
    # Prepare data
    train_df = df[df["dataset_split"] == "train"]
    val_df = df[df["dataset_split"] == "validation"]
    test_df = df[df["dataset_split"] == "test"]
    
    # Drop rows with NaN in features
    train_df = train_df.dropna(subset=feature_cols)
    val_df = val_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    if len(train_df) < 50:
        raise ValueError("Not enough training data. Need at least 50 matches.")
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols] if len(test_df) > 0 else None
    
    # Train home goals model
    home_model = train_poisson_model(
        X_train,
        train_df["FTHG"],
        cfg.model.regularization_alpha,
        cfg.model.max_iter,
    )
    
    # Train away goals model
    away_model = train_poisson_model(
        X_train,
        train_df["FTAG"],
        cfg.model.regularization_alpha,
        cfg.model.max_iter,
    )
    
    # Evaluate
    train_metrics = {
        "home": evaluate(home_model, X_train, train_df["FTHG"]),
        "away": evaluate(away_model, X_train, train_df["FTAG"]),
    }
    print(f"Train metrics: {train_metrics}")
    
    if X_test is not None and len(X_test) > 0:
        test_metrics = {
            "home": evaluate(home_model, X_test, test_df["FTHG"]),
            "away": evaluate(away_model, X_test, test_df["FTAG"]),
        }
        print(f"Test metrics: {test_metrics}")
    
    # Train calibrator on validation set
    calibrator = None
    if len(val_df) > 20:
        X_val = val_df[feature_cols]
        val_home_pred = home_model.predict(X_val)
        val_away_pred = away_model.predict(X_val)
        val_total_lambda = val_home_pred + val_away_pred
        
        raw_over_probs = np.array([
            poisson_over_probability(lam, 2.5) for lam in val_total_lambda
        ])
        actuals = (val_df["total_goals"] > 2.5).astype(int).values
        
        calibrator = train_calibrator(raw_over_probs, actuals)
    
    bundle = EuropeanModelBundle(
        home_model=home_model,
        away_model=away_model,
        feature_columns=feature_cols,
        calibrator=calibrator,
    )
    
    persist_models(cfg, bundle)
    print(f"Models saved to {cfg.models_dir}")
    
    # Generate predictions for all data
    generate_predictions(cfg, df, bundle)
    
    return bundle


def generate_predictions(
    cfg: EuropeanConfig,
    df: pd.DataFrame,
    bundle: EuropeanModelBundle,
) -> pd.DataFrame:
    """Generate predictions for all matches."""
    feature_cols = bundle.feature_columns
    
    # Only predict for rows with features
    pred_mask = df[feature_cols].notna().all(axis=1)
    pred_df = df[pred_mask].copy()
    
    X = pred_df[feature_cols]
    
    pred_df["pred_home_goals"] = bundle.home_model.predict(X)
    pred_df["pred_away_goals"] = bundle.away_model.predict(X)
    pred_df["pred_total_goals"] = pred_df["pred_home_goals"] + pred_df["pred_away_goals"]
    
    # Over/under probabilities
    pred_df["raw_over_prob"] = pred_df["pred_total_goals"].apply(
        lambda x: poisson_over_probability(x, 2.5)
    )
    pred_df["raw_under_prob"] = 1 - pred_df["raw_over_prob"]
    
    # Calibrated probabilities
    if bundle.calibrator:
        pred_df["cal_over_prob"] = bundle.calibrator.predict(pred_df["raw_over_prob"])
        pred_df["cal_under_prob"] = 1 - pred_df["cal_over_prob"]
    else:
        pred_df["cal_over_prob"] = pred_df["raw_over_prob"]
        pred_df["cal_under_prob"] = pred_df["raw_under_prob"]
    
    # Save predictions
    output_path = cfg.processed_dir / "european_predictions.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    
    return pred_df


def main() -> None:
    cfg = get_european_config()
    train_european_model(cfg)


if __name__ == "__main__":
    main()
