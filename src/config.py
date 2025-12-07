from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class LeagueConfig:
    code: str
    name: str
    season_codes: List[str]
    is_extra: bool = False  # True for leagues using the /new/ URL structure


@dataclass
class ModelConfig:
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    min_matches_for_features: int = 5
    validation_season: str = "2425"
    test_season: str = "2526"
    alpha: float = 0.01  # Reduced for better learning with StandardScaler
    max_iter: int = 2000


@dataclass
class StrategyConfig:
    target_total_line: float = 2.5
    min_edge: float = 0.10  # Based on calibration analysis: edges <10% have negative ROI
    bankroll: float = 10_000.0
    kelly_fraction: float = 0.15
    max_bet_fraction: float = 0.05
    # Probability filters based on calibration analysis
    min_probability: float = 0.35  # Below this, sample sizes too small
    max_probability: float = 0.75  # Above this, calibration degrades
    # Model preference: Poisson is better calibrated than Dixon-Coles
    use_poisson_primary: bool = True


@dataclass
class BacktestConfig:
    starting_bankroll: float = 10_000.0
    max_bet_fraction: float = 0.1


@dataclass
class AppConfig:
    base_dir: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    leagues: List[LeagueConfig] = field(
        default_factory=lambda: [
            # Main leagues (seasonal URL structure)
            LeagueConfig("E0", "Premier League", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("E1", "Championship", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("E2", "League One", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("E3", "League Two", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("EC", "National League", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("SC0", "Scottish Premiership", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("SC1", "Scottish Championship", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("SC2", "Scottish League One", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("SC3", "Scottish League Two", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("D1", "Bundesliga", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("D2", "2. Bundesliga", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("SP1", "La Liga", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("SP2", "La Liga 2", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("I1", "Serie A", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("I2", "Serie B", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("F1", "Ligue 1", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("F2", "Ligue 2", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("N1", "Eredivisie", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("B1", "Jupiler Pro League", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("P1", "Primeira Liga", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("T1", "Turkish Super Lig", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("G1", "Greek Super League", ["2122", "2223", "2324", "2425", "2526"]),
            # Extra leagues (single-file URL structure with different column names)
            LeagueConfig("AUT", "Austrian Bundesliga", [], is_extra=True),
            LeagueConfig("SWZ", "Swiss Super League", [], is_extra=True),
            LeagueConfig("DNK", "Danish Superliga", [], is_extra=True),
            LeagueConfig("SWE", "Swedish Allsvenskan", [], is_extra=True),
            LeagueConfig("NOR", "Norwegian Eliteserien", [], is_extra=True),
            LeagueConfig("FIN", "Finnish Veikkausliiga", [], is_extra=True),
            LeagueConfig("POL", "Polish Ekstraklasa", [], is_extra=True),
            LeagueConfig("ROU", "Romanian Liga 1", [], is_extra=True),
            LeagueConfig("RUS", "Russian Premier League", [], is_extra=True),
            LeagueConfig("JPN", "Japanese J-League", [], is_extra=True),
            LeagueConfig("ARG", "Argentina Primera Division", [], is_extra=True),
            LeagueConfig("BRA", "Brazil Serie A", [], is_extra=True),
            LeagueConfig("MEX", "Mexican Liga MX", [], is_extra=True),
            LeagueConfig("USA", "MLS", [], is_extra=True),
            LeagueConfig("CHN", "Chinese Super League", [], is_extra=True),
            LeagueConfig("IRL", "Irish Premier Division", [], is_extra=True),
        ]
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        self.data_dir = (self.base_dir / self.data_dir).resolve()
        self.raw_dir = (self.base_dir / self.raw_dir).resolve()
        self.processed_dir = (self.base_dir / self.processed_dir).resolve()
        self.models_dir = (self.base_dir / self.models_dir).resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


CONFIG = AppConfig()


def get_config() -> AppConfig:
    """Return a singleton-style configuration object."""
    return CONFIG


# Patterns to identify women's league matches (case-insensitive)
WOMENS_LEAGUE_PATTERNS = [
    "women",       # English
    "kvinner",     # Norwegian  
    "feminine",    # French
    "femminile",   # Italian
    "femenino",    # Spanish
    "frauen",      # German
    "wfc",         # Women's Football Club suffix
    "wsl",         # Women's Super League
]


def is_womens_match(league: str = "", home_team: str = "", away_team: str = "") -> bool:
    """
    Check if a match is from a women's league.
    
    Args:
        league: League name
        home_team: Home team name
        away_team: Away team name
        
    Returns:
        True if this appears to be a women's match
    """
    # Combine all text to check, convert to lowercase
    text_to_check = f"{league} {home_team} {away_team}".lower()
    
    for pattern in WOMENS_LEAGUE_PATTERNS:
        if pattern in text_to_check:
            return True
    return False


def filter_womens_matches(df, league_col: str = "league", 
                          home_col: str = "home_team", 
                          away_col: str = "away_team",
                          verbose: bool = True):
    """
    Filter out women's league matches from a DataFrame.
    
    Args:
        df: DataFrame with match data
        league_col: Column name for league
        home_col: Column name for home team
        away_col: Column name for away team
        verbose: Print info about filtered matches
        
    Returns:
        Filtered DataFrame with only men's matches
    """
    import pandas as pd
    
    if df.empty:
        return df
    
    # Get column names safely (may not exist)
    league_vals = df[league_col].fillna("") if league_col in df.columns else pd.Series([""] * len(df))
    home_vals = df[home_col].fillna("") if home_col in df.columns else pd.Series([""] * len(df))
    away_vals = df[away_col].fillna("") if away_col in df.columns else pd.Series([""] * len(df))
    
    # Create mask for women's matches
    is_womens = [
        is_womens_match(str(league), str(home), str(away))
        for league, home, away in zip(league_vals, home_vals, away_vals)
    ]
    
    womens_count = sum(is_womens)
    if verbose and womens_count > 0:
        womens_df = df[is_womens]
        print(f"\n[!] Filtering out {womens_count} women's league matches:")
        for _, row in womens_df.head(10).iterrows():
            home = row.get(home_col, "?")
            away = row.get(away_col, "?")
            league = row.get(league_col, "?")
            print(f"   - {home} vs {away} ({league})")
        if womens_count > 10:
            print(f"   ... and {womens_count - 10} more")
    
    # Return only non-women's matches
    return df[[not w for w in is_womens]]
