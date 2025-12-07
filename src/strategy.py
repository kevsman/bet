from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import AppConfig, get_config

PREDICTIONS_FILE = "model_predictions.csv"


@dataclass
class SignalOption:
    selection: str
    prob: float
    odds: float
    edge: float
    kelly: float


@dataclass
class SignalRecommendation:
    match_id: str
    date: pd.Timestamp
    league_code: str
    home_team: str
    away_team: str
    line: float
    model_total: float
    selection: str
    probability: float
    odds: float
    edge: float
    stake: float
    stake_fraction: float
    over_prob: float
    under_prob: float


def load_predictions(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.processed_dir / PREDICTIONS_FILE
    if not path.exists():
        raise FileNotFoundError("Prediction file missing. Run src.models first.")
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    if "dataset_split" in df.columns:
        df = df[df["dataset_split"].eq("test")]
    for col in ("best_over_odds", "best_under_odds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["best_over_odds", "best_under_odds"], how="all")
    return df


def kelly_fraction(prob: float, odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1 - prob
    return max(0.0, (prob * b - q) / b)


def _sanitize_lambda(lam: float) -> float:
    return float(np.clip(lam, 0.1, 6.0))


def probability_over(line: float, lam: float) -> float:
    lam = _sanitize_lambda(lam)
    threshold = math.floor(line - 0.5)
    return float(poisson.sf(threshold, lam))


def probability_under(line: float, lam: float) -> float:
    lam = _sanitize_lambda(lam)
    threshold = math.floor(line - 0.5)
    return float(poisson.cdf(threshold, lam))


def pick_signal(row: pd.Series, cfg: AppConfig) -> Optional[SignalRecommendation]:
    lam_total = row["pred_total_goals"]
    line = row.get("market_total_line", cfg.strategy.target_total_line)
    odds_over = row.get("best_over_odds", float("nan"))
    odds_under = row.get("best_under_odds", float("nan"))
    options: List[SignalOption] = []
    
    # Get probability thresholds from config
    min_prob = getattr(cfg.strategy, 'min_probability', 0.35)
    max_prob = getattr(cfg.strategy, 'max_probability', 0.75)

    calibrated_line = abs(line - cfg.strategy.target_total_line) < 1e-6
    cal_over = row.get("cal_over_prob") if calibrated_line else np.nan
    cal_under = row.get("cal_under_prob") if calibrated_line else np.nan

    if pd.notna(odds_over):
        prob_over = (
            float(cal_over)
            if pd.notna(cal_over)
            else probability_over(line, lam_total)
        )
        prob_over = float(np.clip(prob_over, 1e-4, 1 - 1e-4))
        # Only consider if probability is in well-calibrated range
        if min_prob <= prob_over <= max_prob:
            edge = prob_over * odds_over - 1
            frac = kelly_fraction(prob_over, odds_over)
            options.append(SignalOption("Over", prob_over, odds_over, edge, frac))
    else:
        prob_over = float("nan")

    if pd.notna(odds_under):
        prob_under = (
            float(cal_under)
            if pd.notna(cal_under)
            else probability_under(line, lam_total)
        )
        prob_under = float(np.clip(prob_under, 1e-4, 1 - 1e-4))
        # Only consider if probability is in well-calibrated range
        if min_prob <= prob_under <= max_prob:
            edge = prob_under * odds_under - 1
            frac = kelly_fraction(prob_under, odds_under)
            options.append(SignalOption("Under", prob_under, odds_under, edge, frac))
    else:
        prob_under = float("nan")

    if not options:
        return None

    best = max(options, key=lambda o: o.edge)
    if best.edge < cfg.strategy.min_edge:
        return None

    stake_fraction = min(
        best.kelly * cfg.strategy.kelly_fraction,
        cfg.strategy.max_bet_fraction,
    )
    stake_amount = stake_fraction * cfg.strategy.bankroll

    return SignalRecommendation(
        match_id=row["match_id"],
        date=row["Date"],
        league_code=row["league_code"],
        home_team=row["HomeTeam"],
        away_team=row["AwayTeam"],
        line=float(line),
        model_total=float(lam_total),
        selection=best.selection,
        probability=float(best.prob),
        odds=float(best.odds),
        edge=float(best.edge),
        stake=float(stake_amount),
        stake_fraction=float(stake_fraction),
        over_prob=float(prob_over),
        under_prob=float(prob_under),
    )


def generate_signals(cfg: AppConfig) -> pd.DataFrame:
    predictions = load_predictions(cfg)
    signals = [sig for _, row in predictions.iterrows() if (sig := pick_signal(row, cfg))]
    if not signals:
        print("No signals passed the edge filter. Adjust config if needed.")
        return pd.DataFrame()
    df = pd.DataFrame([asdict(sig) for sig in signals])
    output_path = cfg.processed_dir / "recommendations.csv"
    df.sort_values("date", inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} recommendations to {output_path.relative_to(cfg.base_dir)}")
    return df


def main() -> None:
    cfg = get_config()
    generate_signals(cfg)


if __name__ == "__main__":
    main()
