"""
Convert Norsk Tipping scraped odds to the format expected by upcoming_score.py
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config

# League code mapping
LEAGUE_MAPPINGS = {
    # English leagues
    "England - Premier League": "E0",
    "England - Championship": "E1",
    "England - League One": "E2",
    "England - League Two": "E3",
    
    # Spanish leagues
    "Spain - La Liga": "SP1",
    "Spania - La Liga": "SP1",  # Norwegian
    "Spain - Segunda Division": "SP2",
    "Spania - Segunda Division": "SP2",  # Norwegian
    
    # German leagues
    "Germany - Bundesliga": "D1",
    "Tyskland - Bundesliga": "D1",  # Norwegian
    "Germany - 2. Bundesliga": "D2",
    "Tyskland - 2. Bundesliga": "D2",  # Norwegian
    
    # Italian leagues
    "Italy - Serie A": "I1",
    "Italia - Serie A": "I1",  # Norwegian
    "Italy - Serie B": "I2",
    "Italia - Serie B": "I2",  # Norwegian
    
    # French leagues
    "France - Ligue 1": "F1",
    "Frankrike - Ligue 1": "F1",  # Norwegian
    "France - Ligue 2": "F2",
    "Frankrike - Ligue 2": "F2",  # Norwegian
    
    # Other leagues
    "Netherlands - Eredivisie": "N1",
    "Nederland - Eredivisie": "N1",  # Norwegian
    "Portugal - Primeira Liga": "P1",
    "Portugal - Liga Portugal": "P1",
    "Belgium - First Division A": "B1",
    "Belgia - First Division A": "B1",  # Norwegian
    "Scotland - Premiership": "SC0",
    "Skottland - Premiership": "SC0",  # Norwegian
    "Turkey - Super Lig": "T1",
    "Tyrkia - Super Lig": "T1",  # Norwegian
    "Greece - Super League": "G1",
    "Hellas - Super League": "G1",  # Norwegian
}


def convert_norsk_tipping_to_fixtures(
    norsk_tipping_csv: Path,
    output_csv: Path | None = None
) -> pd.DataFrame:
    """
    Convert Norsk Tipping odds CSV to fixtures format.
    
    Args:
        norsk_tipping_csv: Path to norsk_tipping_odds.csv
        output_csv: Optional output path
        
    Returns:
        DataFrame in fixtures format
    """
    # Load Norsk Tipping data
    df = pd.read_csv(norsk_tipping_csv)
    
    # Map league names to codes
    df['Div'] = df['league'].map(LEAGUE_MAPPINGS)
    
    # Check for unmapped leagues
    unmapped = df[df['Div'].isna()]
    if not unmapped.empty:
        print("WARNING: Some leagues could not be mapped:")
        print(unmapped[['league']].drop_duplicates())
        df = df.dropna(subset=['Div'])
    
    # Convert date format
    df['Date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Rename columns to match expected format
    fixtures = pd.DataFrame({
        'Div': df['Div'],
        'Date': df['Date'],
        'Time': df['time'],
        'HomeTeam': df['home_team'],
        'AwayTeam': df['away_team'],
        'best_over_odds': df['over_2_5'],  # Use 2.5 line for compatibility
        'best_under_odds': df['under_2_5'],
        'market_total_line': 2.5,
    })
    
    # Save to output
    if output_csv is None:
        cfg = get_config()
        output_csv = cfg.data_dir / "upcoming" / "norsk_tipping_fixtures.csv"
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_csv, index=False)
    
    print(f"\nConverted {len(fixtures)} matches from Norsk Tipping format")
    print(f"Saved to: {output_csv}")
    
    return fixtures


if __name__ == "__main__":
    cfg = get_config()
    norsk_tipping_path = cfg.data_dir / "upcoming" / "norsk_tipping_odds.csv"
    
    print("Converting Norsk Tipping odds to fixtures format...")
    fixtures = convert_norsk_tipping_to_fixtures(norsk_tipping_path)
    
    print("\nFixtures ready for scoring:")
    print(fixtures.to_string(index=False))
