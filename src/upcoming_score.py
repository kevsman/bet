from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor

from .config import AppConfig, get_config
from .strategy import pick_signal, probability_over, probability_under

PREDICTIONS_OUT = "upcoming_predictions.csv"
RECOMMENDATIONS_OUT = "upcoming_recommendations.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score upcoming fixtures with trained models")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Path to curated fixtures CSV (defaults to data/upcoming/fixtures_filtered.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store predictions/recommendations (defaults to data/processed)",
    )
    return parser.parse_args()


def load_fixtures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True)
    if df.empty:
        raise ValueError("Fixture file is empty.")
    numeric_cols = [
        "best_over_odds",
        "best_under_odds",
        "market_total_line",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_models(cfg: AppConfig) -> Tuple[PoissonRegressor, PoissonRegressor, List[str], Optional[IsotonicRegression]]:
    home_model: PoissonRegressor = joblib.load(cfg.models_dir / "home_poisson.joblib")
    away_model: PoissonRegressor = joblib.load(cfg.models_dir / "away_poisson.joblib")
    features = (cfg.models_dir / "features.txt").read_text().splitlines()
    features = [f for f in features if f]
    calibrator_path = cfg.models_dir / "over_calibrator.joblib"
    calibrator: Optional[IsotonicRegression]
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
    else:
        calibrator = None
    return home_model, away_model, features, calibrator


def build_feature_maps(dataset: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], pd.Series], Dict[Tuple[str, str], pd.Series]]:
    dataset_sorted = dataset.sort_values("Date")
    home_map: Dict[Tuple[str, str], pd.Series] = {}
    away_map: Dict[Tuple[str, str], pd.Series] = {}
    for _, row in dataset_sorted.iterrows():
        key_home = (row["league_code"], row["HomeTeam"])
        key_away = (row["league_code"], row["AwayTeam"])
        home_map[key_home] = row
        away_map[key_away] = row
    return home_map, away_map


def assemble_features(
    league_code: str,
    home_team: str,
    away_team: str,
    feature_cols: List[str],
    home_map: Dict[Tuple[str, str], pd.Series],
    away_map: Dict[Tuple[str, str], pd.Series],
) -> Optional[pd.DataFrame]:
    home_row = home_map.get((league_code, home_team))
    away_row = away_map.get((league_code, away_team))
    if home_row is None or away_row is None:
        return None
    feature_values = []
    for col in feature_cols:
        if col.startswith("home_"):
            value = home_row.get(col)
        elif col.startswith("away_"):
            value = away_row.get(col)
        else:
            value = home_row.get(col, np.nan)
        feature_values.append(value)
    if any(pd.isna(feature_values)):
        return None
    return pd.DataFrame([feature_values], columns=feature_cols)


def compute_probabilities(
    total_lambda: float,
    line: float,
    calibrator: Optional[IsotonicRegression],
    use_calibration: bool,
) -> Tuple[float, float]:
    raw_over = probability_over(line, total_lambda)
    raw_under = probability_under(line, total_lambda)
    if calibrator is not None and use_calibration:
        cal_over = float(calibrator.predict([raw_over])[0])
        cal_over = float(np.clip(cal_over, 1e-4, 1 - 1e-4))
        return cal_over, 1.0 - cal_over
    return raw_over, raw_under


def main() -> None:
    args = parse_args()
    cfg = get_config()

    fixtures_path = args.fixtures or (cfg.data_dir / "upcoming" / "fixtures_filtered.csv")
    output_dir = args.output_dir or cfg.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = load_fixtures(fixtures_path)
    dataset = pd.read_csv(cfg.processed_dir / "match_dataset.csv", parse_dates=["Date"], low_memory=False)
    home_model, away_model, feature_cols, calibrator = load_models(cfg)
    home_map, away_map = build_feature_maps(dataset)

    predictions = []
    signals = []

    for _, fixture in fixtures.iterrows():
        league_code = fixture["Div"]
        home_team = fixture["HomeTeam"]
        away_team = fixture["AwayTeam"]
        features_df = assemble_features(league_code, home_team, away_team, feature_cols, home_map, away_map)
        if features_df is None:
            continue
        pred_home = float(np.clip(home_model.predict(features_df)[0], 0.1, 5.0))
        pred_away = float(np.clip(away_model.predict(features_df)[0], 0.1, 5.0))
        total = pred_home + pred_away
        line = float(fixture.get("market_total_line", cfg.strategy.target_total_line))
        use_cal = abs(line - cfg.strategy.target_total_line) < 1e-6
        over_prob, under_prob = compute_probabilities(total, line, calibrator, use_cal)

        odds_over = fixture.get("best_over_odds", np.nan)
        odds_under = fixture.get("best_under_odds", np.nan)

        prediction_row = {
            "match_id": f"{league_code}_{fixture['Date'].date()}_{home_team}_{away_team}",
            "date": fixture["Date"],
            "league_code": league_code,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "pred_home_goals": pred_home,
            "pred_away_goals": pred_away,
            "pred_total_goals": total,
            "market_total_line": line,
            "best_over_odds": odds_over,
            "best_under_odds": odds_under,
            "prob_over": over_prob,
            "prob_under": under_prob,
        }
        predictions.append(prediction_row)

        row = pd.Series(
            {
                "match_id": prediction_row["match_id"],
                "Date": prediction_row["date"],
                "league_code": league_code,
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "best_over_odds": odds_over,
                "best_under_odds": odds_under,
                "market_total_line": line,
                "pred_total_goals": total,
                "cal_over_prob": over_prob,
                "cal_under_prob": under_prob,
            }
        )
        signal = pick_signal(row, cfg)
        if signal:
            signals.append(signal)

    predictions_df = pd.DataFrame(predictions)
    predictions_path = output_dir / PREDICTIONS_OUT
    predictions_df.sort_values("date", inplace=True)
    predictions_df.to_csv(predictions_path, index=False)

    if signals:
        recs_df = pd.DataFrame(signals)
        recs_path = output_dir / RECOMMENDATIONS_OUT
        recs_df.sort_values("date", inplace=True)
        recs_df.to_csv(recs_path, index=False)
        print(
            f"Scored {len(predictions_df)} fixtures; {len(recs_df)} passed filters (saved to {recs_path.relative_to(cfg.base_dir)})"
        )
    else:
        print(f"Scored {len(predictions_df)} fixtures; no bets met the filters.")


if __name__ == "__main__":
    main()
