from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class LeagueConfig:
    code: str
    name: str
    season_codes: List[str]


@dataclass
class ModelConfig:
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    min_matches_for_features: int = 5
    validation_season: str = "2425"
    test_season: str = "2526"
    alpha: float = 0.5
    max_iter: int = 1000


@dataclass
class StrategyConfig:
    target_total_line: float = 2.5
    min_edge: float = 0.05
    bankroll: float = 10_000.0
    kelly_fraction: float = 0.15
    max_bet_fraction: float = 0.05


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
            LeagueConfig("E0", "Premier League", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("E1", "Championship", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("E2", "League One", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("E3", "League Two", ["2122", "2223", "2324", "2425", "2526"]),
            LeagueConfig("E4", "Conference", ["2122", "2223", "2324", "2425", "2526"]),
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
            LeagueConfig("N2", "Eerste Divisie", ["2122", "2223", "2324", "2425", "2526"]),
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
