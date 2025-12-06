"""
Gradient Boosting Models for Football Match Prediction.

This module implements XGBoost and LightGBM models as alternatives to
Poisson regression for predicting match outcomes and goals.

Key advantages over Poisson regression:
1. Captures non-linear relationships between features
2. Automatic feature interactions
3. Better handling of complex patterns in team form
4. Often higher predictive accuracy with sufficient data

Reference: Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

from .config import AppConfig, get_config

# Try to import gradient boosting libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")


# File names for model persistence
XGBOOST_HOME_FILE = "xgboost_home.joblib"
XGBOOST_AWAY_FILE = "xgboost_away.joblib"
LIGHTGBM_HOME_FILE = "lightgbm_home.joblib"
LIGHTGBM_AWAY_FILE = "lightgbm_away.joblib"
GB_FEATURES_FILE = "gb_features.txt"
GB_CALIBRATOR_FILE = "gb_over_calibrator.joblib"


@dataclass
class GBModelConfig:
    """Configuration for gradient boosting models."""
    
    # Shared parameters
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_samples: int = 20  # Minimum samples per leaf
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    
    # Early stopping
    early_stopping_rounds: int = 50
    
    # Goal prediction specific
    min_goal_pred: float = 0.1
    max_goal_pred: float = 5.0
    
    # Model selection
    use_lightgbm: bool = True  # Prefer LightGBM (faster, often better)
    use_xgboost: bool = True   # Also train XGBoost for comparison


@dataclass
class GBModelBundle:
    """Bundle containing trained gradient boosting models."""
    
    home_model: Any  # LGBMRegressor or XGBRegressor
    away_model: Any
    feature_columns: List[str]
    model_type: str  # "lightgbm" or "xgboost"
    calibrator: Optional[IsotonicRegression] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def get_lgb_params(cfg: GBModelConfig, objective: str = "poisson") -> Dict[str, Any]:
    """Get LightGBM parameters for goal prediction."""
    return {
        "objective": objective,
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }


def get_xgb_params(cfg: GBModelConfig, objective: str = "count:poisson") -> Dict[str, Any]:
    """Get XGBoost parameters for goal prediction."""
    return {
        "objective": objective,
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "min_child_weight": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    }


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    cfg: Optional[GBModelConfig] = None,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM model for goal prediction.
    
    Uses Poisson objective which is ideal for count data like goals.
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM is required. Install with: pip install lightgbm")
    
    if cfg is None:
        cfg = GBModelConfig()
    
    params = get_lgb_params(cfg)
    model = lgb.LGBMRegressor(**params)
    
    # Prepare callbacks for early stopping
    callbacks = []
    if X_val is not None and y_val is not None:
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))  # Suppress logging
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    else:
        model.fit(X_train, y_train)
    
    return model


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    cfg: Optional[GBModelConfig] = None,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost model for goal prediction.
    
    Uses Poisson objective (count:poisson) for goal count prediction.
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is required. Install with: pip install xgboost")
    
    if cfg is None:
        cfg = GBModelConfig()
    
    params = get_xgb_params(cfg)
    
    # Add early stopping to model params for newer XGBoost versions
    if X_val is not None and y_val is not None:
        params["early_stopping_rounds"] = cfg.early_stopping_rounds
    
    model = xgb.XGBRegressor(**params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)
    
    return model


def evaluate_goal_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Optional[GBModelConfig] = None,
) -> Dict[str, float]:
    """Evaluate a goal prediction model."""
    if cfg is None:
        cfg = GBModelConfig()
    
    preds = model.predict(X)
    preds = np.clip(preds, cfg.min_goal_pred, cfg.max_goal_pred)
    
    mse = mean_squared_error(y, preds)
    
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mse)),
        "mean_pred": float(np.mean(preds)),
        "mean_actual": float(np.mean(y)),
    }


def get_feature_importance(
    model: Any,
    feature_columns: List[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained LightGBM or XGBoost model
        feature_columns: List of feature names
        importance_type: Type of importance ("gain", "split", or "weight")
    
    Returns:
        DataFrame with feature names and importance scores, sorted descending
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_booster"):
        # XGBoost
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type=importance_type)
        importances = [importance_dict.get(f"f{i}", 0) for i in range(len(feature_columns))]
    else:
        importances = [0] * len(feature_columns)
    
    df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def calculate_match_probabilities(
    home_exp: float,
    away_exp: float,
    max_goals: int = 8,
) -> Dict[str, float]:
    """
    Calculate match outcome probabilities from expected goals.
    
    Uses independent Poisson distribution for each team.
    """
    home_exp = np.clip(home_exp, 0.1, 5.0)
    away_exp = np.clip(away_exp, 0.1, 5.0)
    
    # Build probability matrix
    probs = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            probs[h, a] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)
    
    # Match outcomes
    home_win = float(np.sum(np.tril(probs, -1)))  # Below diagonal
    draw = float(np.sum(np.diag(probs)))
    away_win = float(np.sum(np.triu(probs, 1)))  # Above diagonal
    
    # Goal line probabilities
    over_under = {}
    for line in [1.5, 2.5, 3.5, 4.5]:
        over = 0.0
        for h in range(max_goals):
            for a in range(max_goals):
                if h + a > line:
                    over += probs[h, a]
        over_under[f"over_{line}"] = float(over)
        over_under[f"under_{line}"] = float(1.0 - over)
    
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "pred_home_goals": float(home_exp),
        "pred_away_goals": float(away_exp),
        "pred_total_goals": float(home_exp + away_exp),
        **over_under,
    }


def train_gradient_boosting_models(
    df: pd.DataFrame,
    feature_columns: List[str],
    cfg: Optional[GBModelConfig] = None,
    validation_season: str = "2425",
    test_season: str = "2526",
) -> Tuple[GBModelBundle, GBModelBundle]:
    """
    Train both LightGBM and XGBoost models on the dataset.
    
    Args:
        df: Dataset with features and target columns (FTHG, FTAG)
        feature_columns: List of feature column names
        cfg: Model configuration
        validation_season: Season code for validation set
        test_season: Season code for test set
    
    Returns:
        Tuple of (LightGBM bundle, XGBoost bundle)
    """
    if cfg is None:
        cfg = GBModelConfig()
    
    # Create train/val/test splits based on season
    df = df.copy()
    df["season_code"] = df["season_code"].astype(str)
    
    train_mask = (df["season_code"] < validation_season)
    val_mask = (df["season_code"] == validation_season)
    test_mask = (df["season_code"] >= test_season)
    
    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]
    
    print(f"Training set: {len(train_df):,} matches")
    print(f"Validation set: {len(val_df):,} matches")
    print(f"Test set: {len(test_df):,} matches")
    
    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns] if len(val_df) > 0 else None
    X_test = test_df[feature_columns]
    
    y_train_home = train_df["FTHG"]
    y_train_away = train_df["FTAG"]
    y_val_home = val_df["FTHG"] if len(val_df) > 0 else None
    y_val_away = val_df["FTAG"] if len(val_df) > 0 else None
    y_test_home = test_df["FTHG"]
    y_test_away = test_df["FTAG"]
    
    lgb_bundle = None
    xgb_bundle = None
    
    # Train LightGBM models
    if cfg.use_lightgbm and HAS_LIGHTGBM:
        print("\n" + "=" * 50)
        print("Training LightGBM models...")
        print("=" * 50)
        
        lgb_home = train_lightgbm_model(X_train, y_train_home, X_val, y_val_home, cfg)
        lgb_away = train_lightgbm_model(X_train, y_train_away, X_val, y_val_away, cfg)
        
        # Evaluate
        train_metrics = {
            "home": evaluate_goal_model(lgb_home, X_train, y_train_home, cfg),
            "away": evaluate_goal_model(lgb_away, X_train, y_train_away, cfg),
        }
        test_metrics = {
            "home": evaluate_goal_model(lgb_home, X_test, y_test_home, cfg),
            "away": evaluate_goal_model(lgb_away, X_test, y_test_away, cfg),
        }
        
        print(f"\nLightGBM Train - Home RMSE: {train_metrics['home']['rmse']:.4f}, Away RMSE: {train_metrics['away']['rmse']:.4f}")
        print(f"LightGBM Test  - Home RMSE: {test_metrics['home']['rmse']:.4f}, Away RMSE: {test_metrics['away']['rmse']:.4f}")
        
        lgb_bundle = GBModelBundle(
            home_model=lgb_home,
            away_model=lgb_away,
            feature_columns=feature_columns,
            model_type="lightgbm",
            metrics={"train": train_metrics, "test": test_metrics},
        )
    
    # Train XGBoost models
    if cfg.use_xgboost and HAS_XGBOOST:
        print("\n" + "=" * 50)
        print("Training XGBoost models...")
        print("=" * 50)
        
        xgb_home = train_xgboost_model(X_train, y_train_home, X_val, y_val_home, cfg)
        xgb_away = train_xgboost_model(X_train, y_train_away, X_val, y_val_away, cfg)
        
        # Evaluate
        train_metrics = {
            "home": evaluate_goal_model(xgb_home, X_train, y_train_home, cfg),
            "away": evaluate_goal_model(xgb_away, X_train, y_train_away, cfg),
        }
        test_metrics = {
            "home": evaluate_goal_model(xgb_home, X_test, y_test_home, cfg),
            "away": evaluate_goal_model(xgb_away, X_test, y_test_away, cfg),
        }
        
        print(f"\nXGBoost Train - Home RMSE: {train_metrics['home']['rmse']:.4f}, Away RMSE: {train_metrics['away']['rmse']:.4f}")
        print(f"XGBoost Test  - Home RMSE: {test_metrics['home']['rmse']:.4f}, Away RMSE: {test_metrics['away']['rmse']:.4f}")
        
        xgb_bundle = GBModelBundle(
            home_model=xgb_home,
            away_model=xgb_away,
            feature_columns=feature_columns,
            model_type="xgboost",
            metrics={"train": train_metrics, "test": test_metrics},
        )
    
    return lgb_bundle, xgb_bundle


def predict_with_bundle(
    bundle: GBModelBundle,
    df: pd.DataFrame,
    cfg: Optional[GBModelConfig] = None,
) -> pd.DataFrame:
    """
    Generate predictions using a trained model bundle.
    
    Args:
        bundle: Trained GBModelBundle
        df: DataFrame with feature columns
        cfg: Model configuration
    
    Returns:
        DataFrame with predictions added
    """
    if cfg is None:
        cfg = GBModelConfig()
    
    df = df.copy()
    X = df[bundle.feature_columns]
    
    # Predict goals
    df["pred_home_goals"] = np.clip(
        bundle.home_model.predict(X),
        cfg.min_goal_pred,
        cfg.max_goal_pred,
    )
    df["pred_away_goals"] = np.clip(
        bundle.away_model.predict(X),
        cfg.min_goal_pred,
        cfg.max_goal_pred,
    )
    df["pred_total_goals"] = df["pred_home_goals"] + df["pred_away_goals"]
    
    # Calculate match probabilities for each row
    probs_list = []
    for _, row in df.iterrows():
        probs = calculate_match_probabilities(
            row["pred_home_goals"],
            row["pred_away_goals"],
        )
        probs_list.append(probs)
    
    probs_df = pd.DataFrame(probs_list, index=df.index)
    
    # Merge probabilities (avoiding duplicates)
    for col in probs_df.columns:
        if col not in df.columns:
            df[col] = probs_df[col]
    
    return df


def save_model_bundle(bundle: GBModelBundle, models_dir: Path) -> None:
    """Save a model bundle to disk."""
    if bundle.model_type == "lightgbm":
        home_file = LIGHTGBM_HOME_FILE
        away_file = LIGHTGBM_AWAY_FILE
    else:
        home_file = XGBOOST_HOME_FILE
        away_file = XGBOOST_AWAY_FILE
    
    joblib.dump(bundle.home_model, models_dir / home_file)
    joblib.dump(bundle.away_model, models_dir / away_file)
    
    # Save features
    (models_dir / f"{bundle.model_type}_features.txt").write_text(
        "\n".join(bundle.feature_columns)
    )
    
    # Save calibrator if present
    if bundle.calibrator is not None:
        joblib.dump(
            bundle.calibrator,
            models_dir / f"{bundle.model_type}_calibrator.joblib",
        )
    
    # Save metrics
    metrics_path = models_dir / f"{bundle.model_type}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Model Type: {bundle.model_type}\n")
        f.write(f"Features: {len(bundle.feature_columns)}\n\n")
        for split, split_metrics in bundle.metrics.items():
            f.write(f"{split.upper()} Metrics:\n")
            for target, target_metrics in split_metrics.items():
                f.write(f"  {target}:\n")
                for metric, value in target_metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
    
    print(f"Saved {bundle.model_type} models to {models_dir}")


def load_model_bundle(models_dir: Path, model_type: str = "lightgbm") -> Optional[GBModelBundle]:
    """Load a model bundle from disk."""
    if model_type == "lightgbm":
        home_file = models_dir / LIGHTGBM_HOME_FILE
        away_file = models_dir / LIGHTGBM_AWAY_FILE
    else:
        home_file = models_dir / XGBOOST_HOME_FILE
        away_file = models_dir / XGBOOST_AWAY_FILE
    
    if not home_file.exists() or not away_file.exists():
        return None
    
    home_model = joblib.load(home_file)
    away_model = joblib.load(away_file)
    
    # Load features
    features_file = models_dir / f"{model_type}_features.txt"
    if features_file.exists():
        feature_columns = features_file.read_text().strip().split("\n")
    else:
        feature_columns = []
    
    # Load calibrator if present
    calibrator_file = models_dir / f"{model_type}_calibrator.joblib"
    calibrator = None
    if calibrator_file.exists():
        calibrator = joblib.load(calibrator_file)
    
    return GBModelBundle(
        home_model=home_model,
        away_model=away_model,
        feature_columns=feature_columns,
        model_type=model_type,
        calibrator=calibrator,
    )


def compare_models(
    df: pd.DataFrame,
    feature_columns: List[str],
    poisson_home_model: Any,
    poisson_away_model: Any,
    gb_bundle: GBModelBundle,
    test_season: str = "2526",
) -> pd.DataFrame:
    """
    Compare Poisson regression with gradient boosting models.
    
    Returns a DataFrame with comparative metrics.
    """
    df = df.copy()
    df["season_code"] = df["season_code"].astype(str)
    test_df = df[df["season_code"] >= test_season].copy()
    
    if len(test_df) == 0:
        print("No test data available for comparison")
        return pd.DataFrame()
    
    X_test = test_df[feature_columns]
    y_home = test_df["FTHG"]
    y_away = test_df["FTAG"]
    
    results = []
    
    # Poisson model predictions
    poisson_home_pred = np.clip(poisson_home_model.predict(X_test), 0.1, 5.0)
    poisson_away_pred = np.clip(poisson_away_model.predict(X_test), 0.1, 5.0)
    
    poisson_home_mae = mean_absolute_error(y_home, poisson_home_pred)
    poisson_home_rmse = np.sqrt(mean_squared_error(y_home, poisson_home_pred))
    poisson_away_mae = mean_absolute_error(y_away, poisson_away_pred)
    poisson_away_rmse = np.sqrt(mean_squared_error(y_away, poisson_away_pred))
    
    results.append({
        "model": "Poisson Regression",
        "home_mae": poisson_home_mae,
        "home_rmse": poisson_home_rmse,
        "away_mae": poisson_away_mae,
        "away_rmse": poisson_away_rmse,
        "total_rmse": (poisson_home_rmse + poisson_away_rmse) / 2,
    })
    
    # Gradient boosting predictions
    gb_home_pred = np.clip(gb_bundle.home_model.predict(X_test), 0.1, 5.0)
    gb_away_pred = np.clip(gb_bundle.away_model.predict(X_test), 0.1, 5.0)
    
    gb_home_mae = mean_absolute_error(y_home, gb_home_pred)
    gb_home_rmse = np.sqrt(mean_squared_error(y_home, gb_home_pred))
    gb_away_mae = mean_absolute_error(y_away, gb_away_pred)
    gb_away_rmse = np.sqrt(mean_squared_error(y_away, gb_away_pred))
    
    results.append({
        "model": f"{gb_bundle.model_type.title()}",
        "home_mae": gb_home_mae,
        "home_rmse": gb_home_rmse,
        "away_mae": gb_away_mae,
        "away_rmse": gb_away_rmse,
        "total_rmse": (gb_home_rmse + gb_away_rmse) / 2,
    })
    
    comparison_df = pd.DataFrame(results)
    
    # Calculate improvement
    poisson_total = comparison_df.loc[0, "total_rmse"]
    gb_total = comparison_df.loc[1, "total_rmse"]
    improvement = (poisson_total - gb_total) / poisson_total * 100
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print(f"\nGradient Boosting improvement: {improvement:.2f}%")
    
    return comparison_df
