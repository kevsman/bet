"""
Dixon-Coles Model Implementation for Football Match Prediction.

The Dixon-Coles model improves upon independent Poisson by:
1. Adding a correlation parameter (rho) for low-scoring games
2. Using time-decay weighting (recent matches matter more)
3. Estimating team-specific attack/defense strengths

Reference: Dixon & Coles (1997) "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .config import AppConfig, get_config


@dataclass
class DixonColesParams:
    """Parameters for the Dixon-Coles model."""
    attack: Dict[str, float] = field(default_factory=dict)  # Team attack strengths
    defense: Dict[str, float] = field(default_factory=dict)  # Team defense strengths
    home_advantage: float = 0.25  # Home field advantage
    rho: float = -0.13  # Correlation parameter for low-scoring games
    league_avg_goals: Dict[str, float] = field(default_factory=dict)  # League-specific baseline
    
    def get_attack(self, team: str, league: str = None) -> float:
        """Get attack strength, defaulting to league average if unknown."""
        if team in self.attack:
            return self.attack[team]
        return 0.0  # Average attack
    
    def get_defense(self, team: str, league: str = None) -> float:
        """Get defense strength, defaulting to league average if unknown."""
        if team in self.defense:
            return self.defense[team]
        return 0.0  # Average defense


def tau(home_goals: int, away_goals: int, home_exp: float, away_exp: float, rho: float) -> float:
    """
    Dixon-Coles correlation adjustment factor for low-scoring games.
    
    This corrects for the over-prediction of 0-0, 1-0, 0-1, 1-1 scores
    that occurs with independent Poisson distributions.
    
    Args:
        home_goals: Actual home goals (0 or 1 for adjustment)
        away_goals: Actual away goals (0 or 1 for adjustment)
        home_exp: Expected home goals (lambda)
        away_exp: Expected away goals (mu)
        rho: Correlation parameter (typically -0.1 to -0.2)
    
    Returns:
        Adjustment factor to multiply with Poisson probability
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - home_exp * away_exp * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + home_exp * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + away_exp * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


def dixon_coles_probability(home_goals: int, away_goals: int, 
                            home_exp: float, away_exp: float, 
                            rho: float) -> float:
    """
    Calculate probability of a specific scoreline using Dixon-Coles model.
    
    P(X=x, Y=y) = tau(x,y,λ,μ,ρ) × P(X=x|λ) × P(Y=y|μ)
    
    Args:
        home_goals, away_goals: The scoreline
        home_exp: Expected home goals (λ)
        away_exp: Expected away goals (μ)
        rho: Correlation parameter
    
    Returns:
        Probability of this exact scoreline
    """
    # Poisson probabilities
    p_home = poisson.pmf(home_goals, max(0.01, home_exp))
    p_away = poisson.pmf(away_goals, max(0.01, away_exp))
    
    # Apply Dixon-Coles correction
    adj = tau(home_goals, away_goals, home_exp, away_exp, rho)
    
    return max(0.0, adj * p_home * p_away)


def time_decay_weight(days_old: int, xi: float = 0.0018) -> float:
    """
    Calculate time decay weight for a match.
    
    More recent matches get higher weights. Default xi=0.0018 gives
    half-life of about 385 days (roughly 1 season).
    
    Args:
        days_old: Number of days since the match
        xi: Decay rate parameter (higher = faster decay)
    
    Returns:
        Weight between 0 and 1
    """
    return math.exp(-xi * days_old)


class DixonColesModel:
    """
    Dixon-Coles model for football match prediction.
    
    Estimates team-specific attack and defense parameters using MLE,
    with time decay weighting and a correlation adjustment for low scores.
    """
    
    def __init__(self, xi: float = 0.0018, max_iter: int = 1000):
        """
        Initialize the model.
        
        Args:
            xi: Time decay parameter (0 = no decay, higher = faster decay)
            max_iter: Maximum iterations for optimization
        """
        self.xi = xi
        self.max_iter = max_iter
        self.params: Optional[DixonColesParams] = None
        self.teams: List[str] = []
        self.leagues: Dict[str, List[str]] = {}  # League -> teams mapping
        self._fitted = False
    
    def _prepare_data(self, df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
        """Prepare match data for fitting."""
        df = df.copy()
        
        # Ensure we have required columns
        required = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert date and calculate weights
        df['Date'] = pd.to_datetime(df['Date'])
        if reference_date is None:
            reference_date = df['Date'].max()
        
        df['days_old'] = (reference_date - df['Date']).dt.days
        df['weight'] = df['days_old'].apply(lambda d: time_decay_weight(max(0, d), self.xi))
        
        # Get unique teams
        self.teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        
        # Build league -> teams mapping
        if 'league_code' in df.columns:
            for league in df['league_code'].unique():
                league_df = df[df['league_code'] == league]
                self.leagues[league] = sorted(
                    set(league_df['HomeTeam'].unique()) | set(league_df['AwayTeam'].unique())
                )
        
        return df
    
    def _build_parameter_vector(self, params: DixonColesParams) -> np.ndarray:
        """Convert DixonColesParams to optimization vector."""
        n_teams = len(self.teams)
        # Vector: [attack_1, ..., attack_n, defense_1, ..., defense_n, home_adv, rho]
        x = np.zeros(2 * n_teams + 2)
        
        for i, team in enumerate(self.teams):
            x[i] = params.attack.get(team, 0.0)
            x[n_teams + i] = params.defense.get(team, 0.0)
        
        x[-2] = params.home_advantage
        x[-1] = params.rho
        
        return x
    
    def _unpack_parameter_vector(self, x: np.ndarray) -> DixonColesParams:
        """Convert optimization vector back to DixonColesParams."""
        n_teams = len(self.teams)
        params = DixonColesParams()
        
        for i, team in enumerate(self.teams):
            params.attack[team] = x[i]
            params.defense[team] = x[n_teams + i]
        
        params.home_advantage = x[-2]
        params.rho = x[-1]
        
        return params
    
    def _negative_log_likelihood(self, x: np.ndarray, _data: pd.DataFrame,
                                    home_idx: np.ndarray, away_idx: np.ndarray,
                                    home_goals: np.ndarray, away_goals: np.ndarray,
                                    weights: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for optimization (VECTORIZED).
        
        The likelihood is:
        L = Σ_i w_i × log(P(home_i, away_i | params))
        
        where w_i is the time decay weight for match i.
        """
        n_teams = len(self.teams)
        
        # Extract parameters
        attack = x[:n_teams]
        defense = x[n_teams:2*n_teams]
        home_adv = x[-2]
        rho = x[-1]
        
        # Vectorized expected goals calculation
        home_exp = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        away_exp = np.exp(attack[away_idx] + defense[home_idx])
        
        # Clip to reasonable range
        home_exp = np.clip(home_exp, 0.01, 6.0)
        away_exp = np.clip(away_exp, 0.01, 6.0)
        
        # Poisson PMF (vectorized)
        from scipy.stats import poisson
        p_home = poisson.pmf(home_goals, home_exp)
        p_away = poisson.pmf(away_goals, away_exp)
        
        # Dixon-Coles adjustment (vectorized)
        adj = np.ones(len(home_goals))
        
        # Case: 0-0
        mask_00 = (home_goals == 0) & (away_goals == 0)
        adj[mask_00] = 1 - home_exp[mask_00] * away_exp[mask_00] * rho
        
        # Case: 0-1
        mask_01 = (home_goals == 0) & (away_goals == 1)
        adj[mask_01] = 1 + home_exp[mask_01] * rho
        
        # Case: 1-0
        mask_10 = (home_goals == 1) & (away_goals == 0)
        adj[mask_10] = 1 + away_exp[mask_10] * rho
        
        # Case: 1-1
        mask_11 = (home_goals == 1) & (away_goals == 1)
        adj[mask_11] = 1 - rho
        
        # Ensure adjustment is positive
        adj = np.maximum(adj, 0.001)
        
        # Calculate probabilities
        probs = adj * p_home * p_away
        probs = np.maximum(probs, 1e-10)  # Avoid log(0)
        
        # Weighted negative log-likelihood
        nll = -np.sum(weights * np.log(probs))
        
        # Add L2 regularization
        reg_strength = 0.001
        attack_reg = reg_strength * np.sum(attack ** 2)
        defense_reg = reg_strength * np.sum(defense ** 2)
        
        return nll + attack_reg + defense_reg
    
    def fit(self, df: pd.DataFrame, reference_date: datetime = None) -> 'DixonColesModel':
        """
        Fit the Dixon-Coles model to match data.
        
        Args:
            df: DataFrame with columns HomeTeam, AwayTeam, FTHG, FTAG, Date
            reference_date: Date to calculate time decay from (default: most recent match)
        
        Returns:
            self for chaining
        """
        print("Preparing data for Dixon-Coles model...")
        data = self._prepare_data(df, reference_date)
        
        print(f"Fitting Dixon-Coles model on {len(data)} matches, {len(self.teams)} teams...")
        
        # Initialize parameters
        initial_params = DixonColesParams()
        for team in self.teams:
            initial_params.attack[team] = 0.0
            initial_params.defense[team] = 0.0
        initial_params.home_advantage = 0.25
        initial_params.rho = -0.1
        
        x0 = self._build_parameter_vector(initial_params)
        
        # Pre-compute team indices and goal arrays for vectorized optimization
        team_to_idx = {team: i for i, team in enumerate(self.teams)}
        home_idx = data['HomeTeam'].map(team_to_idx).values.astype(int)
        away_idx = data['AwayTeam'].map(team_to_idx).values.astype(int)
        home_goals = data['FTHG'].values.astype(int)
        away_goals = data['FTAG'].values.astype(int)
        weights = data['weight'].values
        
        # Constraints: sum of attack params = 0, sum of defense params = 0 (identifiability)
        n_teams = len(self.teams)
        
        def attack_constraint(x):
            return np.sum(x[:n_teams])
        
        def defense_constraint(x):
            return np.sum(x[n_teams:2*n_teams])
        
        constraints = [
            {'type': 'eq', 'fun': attack_constraint},
            {'type': 'eq', 'fun': defense_constraint}
        ]
        
        # Bounds: rho should be between -1 and 1, home advantage reasonable
        bounds = [(None, None)] * n_teams  # Attack
        bounds += [(None, None)] * n_teams  # Defense
        bounds += [(-0.5, 1.5)]  # Home advantage
        bounds += [(-0.5, 0.5)]  # Rho
        
        # Progress tracking
        self._iter_count = 0
        self._last_nll = None
        
        def callback(xk):
            """Print progress every 10 iterations."""
            self._iter_count += 1
            if self._iter_count % 10 == 0 or self._iter_count == 1:
                nll = self._negative_log_likelihood(xk, data, home_idx, away_idx, home_goals, away_goals, weights)
                home_adv = xk[-2]
                rho = xk[-1]
                print(f"  Iteration {self._iter_count}: NLL={nll:.1f}, home_adv={home_adv:.3f}, rho={rho:.3f}")
                self._last_nll = nll
        
        # Optimize using L-BFGS-B (faster than SLSQP for large problems)
        print(f"Running optimization (this may take a minute)...")
        print(f"  Parameters to optimize: {len(x0)} ({n_teams} attack + {n_teams} defense + 2)")
        
        # Calculate initial NLL
        initial_nll = self._negative_log_likelihood(x0, data, home_idx, away_idx, home_goals, away_goals, weights)
        print(f"  Initial NLL: {initial_nll:.1f}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._negative_log_likelihood,
                x0,
                args=(data, home_idx, away_idx, home_goals, away_goals, weights),
                method='L-BFGS-B',
                bounds=bounds,
                callback=callback,
                options={'maxiter': self.max_iter, 'disp': False}
            )
        
        print(f"  Optimization finished after {self._iter_count} iterations")
        
        # Apply constraint manually (normalize attack/defense to sum to 0)
        x_opt = result.x.copy()
        attack_mean = np.mean(x_opt[:n_teams])
        defense_mean = np.mean(x_opt[n_teams:2*n_teams])
        x_opt[:n_teams] -= attack_mean
        x_opt[n_teams:2*n_teams] -= defense_mean
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
        
        self.params = self._unpack_parameter_vector(x_opt)
        self._fitted = True
        
        print(f"Dixon-Coles fitted: home_adv={self.params.home_advantage:.3f}, rho={self.params.rho:.3f}")
        
        # Print top attack/defense teams
        attack_sorted = sorted(self.params.attack.items(), key=lambda x: -x[1])
        defense_sorted = sorted(self.params.defense.items(), key=lambda x: x[1])
        
        print("\nTop 5 attack ratings:")
        for team, rating in attack_sorted[:5]:
            print(f"  {team}: {rating:.3f}")
        
        print("\nTop 5 defense ratings (lower = better):")
        for team, rating in defense_sorted[:5]:
            print(f"  {team}: {rating:.3f}")
        
        return self
    
    def predict_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Predict expected goals for a match.
        
        Returns:
            (home_expected_goals, away_expected_goals)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        home_attack = self.params.get_attack(home_team)
        home_defense = self.params.get_defense(home_team)
        away_attack = self.params.get_attack(away_team)
        away_defense = self.params.get_defense(away_team)
        
        home_exp = math.exp(home_attack + away_defense + self.params.home_advantage)
        away_exp = math.exp(away_attack + home_defense)
        
        # Clip to reasonable range
        home_exp = max(0.1, min(home_exp, 5.0))
        away_exp = max(0.1, min(away_exp, 5.0))
        
        return home_exp, away_exp
    
    def predict_scoreline_probabilities(self, home_team: str, away_team: str, 
                                        max_goals: int = 8) -> np.ndarray:
        """
        Calculate probability matrix for all scorelines up to max_goals.
        
        Returns:
            (max_goals+1) x (max_goals+1) matrix of probabilities
        """
        home_exp, away_exp = self.predict_goals(home_team, away_team)
        
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                probs[h, a] = dixon_coles_probability(h, a, home_exp, away_exp, self.params.rho)
        
        # Normalize to ensure probabilities sum to ~1
        probs = probs / probs.sum()
        
        return probs
    
    def predict_1x2(self, home_team: str, away_team: str, max_goals: int = 8) -> Dict[str, float]:
        """
        Predict 1X2 (home/draw/away) probabilities.
        
        Returns:
            Dict with keys 'home', 'draw', 'away'
        """
        probs = self.predict_scoreline_probabilities(home_team, away_team, max_goals)
        
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if h > a:
                    home_win += probs[h, a]
                elif h == a:
                    draw += probs[h, a]
                else:
                    away_win += probs[h, a]
        
        return {'home': home_win, 'draw': draw, 'away': away_win}
    
    def predict_over_under(self, home_team: str, away_team: str, line: float,
                           max_goals: int = 10) -> Dict[str, float]:
        """
        Predict over/under probabilities for a given goal line.
        
        Args:
            home_team, away_team: Team names
            line: Goal line (1.5, 2.5, 3.5, etc.)
            max_goals: Maximum goals to consider in calculation
        
        Returns:
            Dict with keys 'over', 'under'
        """
        probs = self.predict_scoreline_probabilities(home_team, away_team, max_goals)
        
        threshold = int(line)  # Under is ≤ floor(line), over is > floor(line)
        
        under_prob = 0.0
        over_prob = 0.0
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                total = h + a
                if total <= threshold:
                    under_prob += probs[h, a]
                else:
                    over_prob += probs[h, a]
        
        # Normalize
        total_prob = under_prob + over_prob
        if total_prob > 0:
            under_prob /= total_prob
            over_prob /= total_prob
        
        return {'over': over_prob, 'under': under_prob}
    
    def predict_btts(self, home_team: str, away_team: str, max_goals: int = 8) -> Dict[str, float]:
        """
        Predict Both Teams To Score (BTTS) probabilities.
        
        Returns:
            Dict with keys 'yes', 'no'
        """
        probs = self.predict_scoreline_probabilities(home_team, away_team, max_goals)
        
        btts_yes = 0.0
        btts_no = 0.0
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if h >= 1 and a >= 1:
                    btts_yes += probs[h, a]
                else:
                    btts_no += probs[h, a]
        
        return {'yes': btts_yes, 'no': btts_no}
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        joblib.dump({
            'params': self.params,
            'teams': self.teams,
            'leagues': self.leagues,
            'xi': self.xi,
            'fitted': self._fitted
        }, path)
        print(f"Dixon-Coles model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'DixonColesModel':
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(xi=data['xi'])
        model.params = data['params']
        model.teams = data['teams']
        model.leagues = data['leagues']
        model._fitted = data['fitted']
        return model


def train_dixon_coles(cfg: AppConfig = None) -> DixonColesModel:
    """
    Train Dixon-Coles model on historical data.
    
    Args:
        cfg: App configuration (default: load from config)
    
    Returns:
        Trained DixonColesModel
    """
    if cfg is None:
        cfg = get_config()
    
    # Load dataset (prefer xG-enhanced)
    xg_path = cfg.processed_dir / "match_dataset_with_xg.csv"
    std_path = cfg.processed_dir / "match_dataset.csv"
    
    if xg_path.exists():
        print(f"Loading xG-enhanced dataset for Dixon-Coles training...")
        df = pd.read_csv(xg_path, parse_dates=['Date'], low_memory=False)
    else:
        print(f"Loading standard dataset for Dixon-Coles training...")
        df = pd.read_csv(std_path, parse_dates=['Date'], low_memory=False)
    
    # Filter to recent seasons for faster training and better recency
    df = df.sort_values('Date')
    cutoff_date = df['Date'].max() - pd.Timedelta(days=3*365)  # Last 3 years
    df = df[df['Date'] >= cutoff_date]
    
    print(f"Training on {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Train model
    model = DixonColesModel(xi=0.0018)  # ~1 year half-life
    model.fit(df)
    
    # Save model
    model_path = cfg.models_dir / "dixon_coles.joblib"
    model.save(model_path)
    
    return model


def main():
    """Train and evaluate Dixon-Coles model."""
    cfg = get_config()
    model = train_dixon_coles(cfg)
    
    # Test predictions on a few matches
    print("\n" + "="*60)
    print("Sample Predictions (Dixon-Coles)")
    print("="*60)
    
    test_matches = [
        ("Man City", "Arsenal"),
        ("Liverpool", "Chelsea"),
        ("Bayern Munich", "Dortmund"),
        ("Real Madrid", "Barcelona"),
        ("Juventus", "Inter"),
    ]
    
    for home, away in test_matches:
        if home in model.teams and away in model.teams:
            home_exp, away_exp = model.predict_goals(home, away)
            result_1x2 = model.predict_1x2(home, away)
            
            print(f"\n{home} vs {away}")
            print(f"  Expected: {home_exp:.2f} - {away_exp:.2f}")
            print(f"  1X2: H={result_1x2['home']:.1%}, D={result_1x2['draw']:.1%}, A={result_1x2['away']:.1%}")
            
            for line in [1.5, 2.5, 3.5]:
                ou = model.predict_over_under(home, away, line)
                print(f"  O/U {line}: Over={ou['over']:.1%}, Under={ou['under']:.1%}")


if __name__ == "__main__":
    main()
