"""
Example: Extracting odds from multiple leagues at once.

This demonstrates how to combine snapshots from different leagues
into a single extraction run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.norsk_tipping_scraper import parse_snapshot_to_csv

# Multi-league snapshot combining Premier League and La Liga
MULTI_LEAGUE_SNAPSHOT = """
Lør. 22/11 13:30
Burnley
Chelsea
England - Premier League
TV3+
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.23
Under 1.5
Under 1.5, odds 3.90
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 1.75
Under 2.5
Under 2.5, odds 2.00
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 2.90
Under 3.5
Under 3.5, odds 1.38
Lør. 22/11 16:00
Real Madrid
Barcelona
Spain - La Liga
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.15
Under 1.5
Under 1.5, odds 5.20
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 1.50
Under 2.5
Under 2.5, odds 2.55
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 2.20
Under 3.5
Under 3.5, odds 1.65
Lør. 22/11 18:30
Bayern München
Borussia Dortmund
Germany - Bundesliga
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.18
Under 1.5
Under 1.5, odds 4.80
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 1.55
Under 2.5
Under 2.5, odds 2.35
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 2.30
Under 3.5
Under 3.5, odds 1.58
"""

if __name__ == "__main__":
    print("="*80)
    print("MULTI-LEAGUE EXTRACTION EXAMPLE")
    print("="*80)
    print("\nExtracting odds from Premier League, La Liga, and Bundesliga...\n")
    
    df = parse_snapshot_to_csv(MULTI_LEAGUE_SNAPSHOT)
    
    print("\n" + "="*80)
    print("EXTRACTED MATCHES BY LEAGUE")
    print("="*80)
    
    for league in df['league'].unique():
        league_matches = df[df['league'] == league]
        print(f"\n{league}:")
        for _, row in league_matches.iterrows():
            print(f"  {row['home_team']} vs {row['away_team']}")
            print(f"    Over/Under 2.5: {row['over_2_5']:.2f} / {row['under_2_5']:.2f}")
    
    print("\n" + "="*80)
    print(f"Total matches extracted: {len(df)}")
    print(f"Leagues: {', '.join(df['league'].unique())}")
