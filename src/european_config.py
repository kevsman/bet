"""
Configuration for European competitions (Champions League, Europa League, Conference League).
Uses API-Football as data source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EuropeanCompetition:
    """Configuration for a European competition."""
    name: str
    api_id: int  # API-Football league ID
    code: str    # Short code for file naming


# API-Football competition IDs
EUROPEAN_COMPETITIONS: Dict[str, EuropeanCompetition] = {
    "UCL": EuropeanCompetition("UEFA Champions League", 2, "UCL"),
    "UEL": EuropeanCompetition("UEFA Europa League", 3, "UEL"),
    "UECL": EuropeanCompetition("UEFA Conference League", 848, "UECL"),
}


@dataclass
class EuropeanModelConfig:
    """Model parameters for European competitions."""
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5])  # Smaller windows due to fewer matches
    min_matches_for_features: int = 2  # Lower threshold for European matches
    regularization_alpha: float = 1.0
    max_iter: int = 1000
    validation_season: str = "2023"
    test_season: str = "2024"


@dataclass
class EuropeanConfig:
    """Main configuration for European competition model."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    api_key: Optional[str] = None  # Set via environment variable API_FOOTBALL_KEY
    competitions: Dict[str, EuropeanCompetition] = field(default_factory=lambda: EUROPEAN_COMPETITIONS)
    model: EuropeanModelConfig = field(default_factory=EuropeanModelConfig)
    seasons: List[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024, 2025])
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / "european"
    
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed" / "european"
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models" / "european"
    
    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def get_european_config() -> EuropeanConfig:
    """Get European config with API key from environment."""
    import os
    config = EuropeanConfig()
    config.api_key = os.environ.get("API_FOOTBALL_KEY")
    config.ensure_dirs()
    return config
