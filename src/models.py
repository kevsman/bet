from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import AppConfig, get_config

PREDICTIONS_FILE = "model_predictions.csv"
CALIBRATOR_FILE = "over_calibrator.joblib"


@dataclass
class ModelBundle:
    home_model: Union[PoissonRegressor, Pipeline]
    away_model: Union[PoissonRegressor, Pipeline]
    feature_columns: List[str]


def assign_splits(df: pd.DataFrame, cfg: AppConfig) -> pd.Series:
    val_mask = df["season_code"].eq(cfg.model.validation_season)
    test_mask = df["season_code"] >= cfg.model.test_season
    splits = np.where(test_mask, "test", np.where(val_mask, "validation", "train"))
    return pd.Series(splits, index=df.index, name="dataset_split")


def load_dataset(cfg: AppConfig) -> pd.DataFrame:
    # Prefer enhanced dataset (with advanced features) if available
    enhanced_path = cfg.processed_dir / "match_dataset_enhanced.csv"
    xg_dataset_path = cfg.processed_dir / "match_dataset_with_xg.csv"
    dataset_path = cfg.processed_dir / "match_dataset.csv"
    
    if enhanced_path.exists():
        print(f"Loading enhanced dataset (with advanced features) from {enhanced_path.name}")
        df = pd.read_csv(enhanced_path, parse_dates=["Date"], low_memory=False)
    elif xg_dataset_path.exists():
        print(f"Loading xG-enhanced dataset from {xg_dataset_path.name}")
        df = pd.read_csv(xg_dataset_path, parse_dates=["Date"], low_memory=False)
    elif dataset_path.exists():
        print(f"Loading standard dataset from {dataset_path.name}")
        df = pd.read_csv(dataset_path, parse_dates=["Date"], low_memory=False)
    else:
        raise FileNotFoundError("Processed dataset not found. Run prepare_dataset first.")
    
    df["season_code"] = df["season_code"].astype(str)
    return df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for col in df.columns:
        # Rolling averages (goals, shots, corners)
        if col.startswith("home_avg_") or col.startswith("away_avg_"):
            candidates.append(col)
        # Exponential weighted averages (recency weighting)
        elif col.startswith("home_ema_") or col.startswith("away_ema_"):
            candidates.append(col)
        # Recent sums
        elif col.startswith("home_recent_") or col.startswith("away_recent_"):
            candidates.append(col)
        # League-specific averages
        elif col.startswith("league_avg_"):
            candidates.append(col)
        # xG features (if available)
        elif col.startswith("home_avg_xg_") or col.startswith("away_avg_xg_"):
            candidates.append(col)
        # Match count features
        elif col in {
            "home_matches_played",
            "away_matches_played",
        }:
            candidates.append(col)
        # ============ ADVANCED FEATURES (NEW) ============
        # FBref advanced stats (possession, passing, pressing, etc.)
        elif col.startswith("home_adv_") or col.startswith("away_adv_"):
            candidates.append(col)
        # Weather features
        elif col.startswith("weather_"):
            candidates.append(col)
        # Injury features
        elif col in {
            "home_injury_count", "away_injury_count",
            "home_injury_severity", "away_injury_severity",
            "home_suspended_count", "away_suspended_count",
            "injury_count_diff", "injury_severity_diff",
        }:
            candidates.append(col)
        # Manager features
        elif col in {
            "home_manager_tenure_days", "away_manager_tenure_days",
            "home_new_manager", "away_new_manager",
            "home_experienced_manager", "away_experienced_manager",
            "manager_tenure_diff",
        }:
            candidates.append(col)
    
    # Ensure we don't include NaN columns if some leagues don't have shot data
    # But for now, we assume they do or we handle NaNs (PoissonRegressor handles NaNs? No, need imputation or drop)
    # The prepare_dataset script drops rows with NaNs in feature cols.
    
    return sorted(candidates)


def train_poisson_model(X: pd.DataFrame, y: pd.Series, alpha: float, max_iter: int) -> Pipeline:
    """Train a Poisson model with StandardScaler for better coefficient learning."""
    model = make_pipeline(
        StandardScaler(),
        PoissonRegressor(alpha=alpha, max_iter=max_iter)
    )
    model.fit(X, y)
    return model


def evaluate(model: Union[PoissonRegressor, Pipeline], X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mse)),
    }


def persist_models(cfg: AppConfig, bundle: ModelBundle) -> None:
    joblib.dump(bundle.home_model, cfg.models_dir / "home_poisson.joblib")
    joblib.dump(bundle.away_model, cfg.models_dir / "away_poisson.joblib")
    (cfg.models_dir / "features.txt").write_text("\n".join(bundle.feature_columns))


def save_predictions(cfg: AppConfig, df: pd.DataFrame) -> None:
    output_path = cfg.processed_dir / PREDICTIONS_FILE
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path.relative_to(cfg.base_dir)}")


def poisson_over_probability(lam: float, line: float) -> float:
    lam = float(np.clip(lam, 0.1, 6.0))
    threshold = math.floor(line - 0.5)
    return float(poisson.sf(threshold, lam))


def poisson_under_probability(lam: float, line: float) -> float:
    lam = float(np.clip(lam, 0.1, 6.0))
    threshold = math.floor(line - 0.5)
    return float(poisson.cdf(threshold, lam))


def calibrate_over_probabilities(raw_probs: pd.Series, totals: pd.Series, lines: pd.Series) -> IsotonicRegression | None:
    outcomes = (totals > lines).astype(float)
    if outcomes.nunique() <= 1:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_probs.to_numpy(), outcomes.to_numpy())
    return iso


def main() -> None:
    cfg = get_config()
    data = load_dataset(cfg)
    feature_columns = select_feature_columns(data)
    
    # Separate optional features (xG, advanced stats, weather, etc.) from core features
    # Core features are required for training; optional features are filled with 0 if missing
    optional_feature_names = {
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
    
    # Drop rows missing core features
    data = data.dropna(subset=core_features)
    
    # Fill optional features with 0 (will have no effect if model learns they're missing)
    for col in optional_features:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    # Remove zero-variance features (provide no predictive value)
    variances = data[feature_columns].var()
    zero_var_features = variances[variances == 0].index.tolist()
    if zero_var_features:
        print(f"Removing {len(zero_var_features)} zero-variance features")
        feature_columns = [f for f in feature_columns if f not in zero_var_features]
    
    print(f"Final feature count: {len(feature_columns)}")
    print(f"Dataset size after cleaning: {len(data)}")

    data["dataset_split"] = assign_splits(data, cfg)
    train_df = data[data["dataset_split"] == "train"]
    val_df = data[data["dataset_split"] == "validation"]
    test_df = data[data["dataset_split"] == "test"]
    if train_df.empty or test_df.empty:
        raise ValueError("Need data from at least two seasons to train/test.")
    if val_df.empty:
        print("Warning: validation split empty; calibration disabled.")

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    home_model = train_poisson_model(
        X_train, train_df["FTHG"], cfg.model.alpha, cfg.model.max_iter
    )
    away_model = train_poisson_model(
        X_train, train_df["FTAG"], cfg.model.alpha, cfg.model.max_iter
    )

    train_metrics = {
        "home": evaluate(home_model, X_train, train_df["FTHG"]),
        "away": evaluate(away_model, X_train, train_df["FTAG"]),
    }
    test_metrics = {
        "home": evaluate(home_model, X_test, test_df["FTHG"]),
        "away": evaluate(away_model, X_test, test_df["FTAG"]),
    }

    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    data["pred_home_goals"] = np.clip(
        home_model.predict(data[feature_columns]), 0.1, 4.5
    )
    data["pred_away_goals"] = np.clip(
        away_model.predict(data[feature_columns]), 0.1, 4.5
    )
    data["pred_total_goals"] = data["pred_home_goals"] + data["pred_away_goals"]

    lines = data.get("market_total_line", pd.Series(cfg.strategy.target_total_line, index=data.index))
    data["raw_over_prob"] = [
        poisson_over_probability(lam, line)
        for lam, line in zip(data["pred_total_goals"], lines)
    ]
    data["raw_under_prob"] = [
        poisson_under_probability(lam, line)
        for lam, line in zip(data["pred_total_goals"], lines)
    ]

    calibrator = None
    if not val_df.empty:
        val_mask = data["dataset_split"] == "validation"
        calibrator = calibrate_over_probabilities(
            data.loc[val_mask, "raw_over_prob"],
            data.loc[val_mask, "total_goals"],
            data.loc[val_mask, "market_total_line"],
        )
    if calibrator is not None:
        data["cal_over_prob"] = calibrator.predict(data["raw_over_prob"])
        data["cal_under_prob"] = 1.0 - data["cal_over_prob"]
        joblib.dump(calibrator, cfg.models_dir / CALIBRATOR_FILE)
    else:
        data["cal_over_prob"] = data["raw_over_prob"]
        data["cal_under_prob"] = data["raw_under_prob"]

    persist_models(cfg, ModelBundle(home_model, away_model, feature_columns))
    save_predictions(cfg, data)


if __name__ == "__main__":
    main()
