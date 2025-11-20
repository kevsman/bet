"""
Extract odds from the current MCP Chrome snapshot.

This script takes the snapshot we just captured and extracts all the
over/under odds for 1.5, 2.5, and 3.5 goal lines.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.norsk_tipping_scraper import parse_snapshot_to_csv

# The snapshot text from the Premier League page with 3 dropdowns selected
SNAPSHOT_TEXT = """
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
Fulham
Sunderland
England - Premier League
VSport Premier L4
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.40
Under 1.5
Under 1.5, odds 2.80
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 2.20
Under 2.5
Under 2.5, odds 1.62
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 4.00
Under 3.5
Under 3.5, odds 1.22
Lør. 22/11 16:00
Bournemouth
West Ham United
England - Premier League
VSport Premier L2
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.18
Under 1.5
Under 1.5, odds 4.50
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 1.62
Under 2.5
Under 2.5, odds 2.20
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 2.55
Under 3.5
Under 3.5, odds 1.48
Lør. 22/11 16:00
Liverpool
Nottingham Forest
England - Premier League
Vsport Premier League
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.16
Under 1.5
Under 1.5, odds 4.80
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 1.52
Under 2.5
Under 2.5, odds 2.40
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 2.35
Under 3.5
Under 3.5, odds 1.55
Lør. 22/11 16:00
Wolverhampton Wanderers
Crystal Palace
England - Premier League
VSport Premier L1
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.33
Under 1.5
Under 1.5, odds 3.15
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 2.00
Under 2.5
Under 2.5, odds 1.75
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 3.55
Under 3.5
Under 3.5, odds 1.27
Lør. 22/11 16:00
Brighton and Hove Albion
Brentford
England - Premier League
VSport Premier L3
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.22
Under 1.5
Under 1.5, odds 4.10
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 1.67
Under 2.5
Under 2.5, odds 2.10
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 2.70
Under 3.5
Under 3.5, odds 1.43
Lør. 22/11 18:30
Newcastle United
Manchester City
England - Premier League
Vsport Premier League
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.22
Under 1.5
Under 1.5, odds 4.10
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 1.67
Under 2.5
Under 2.5, odds 2.10
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 2.70
Under 3.5
Under 3.5, odds 1.43
Søn. 23/11 15:00
Leeds United
Aston Villa
England - Premier League
VSport Premier L1
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.37
Under 1.5
Under 1.5, odds 2.95
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 2.10
Under 2.5
Under 2.5, odds 1.67
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 3.90
Under 3.5
Under 3.5, odds 1.23
Søn. 23/11 17:30
Arsenal
Tottenham Hotspur
England - Premier League
Vsport Premier League
TOTALT ANTALL MÅL - OVER/UNDER 1.5
Over 1.5
Over 1.5, odds 1.25
Under 1.5
Under 1.5, odds 3.70
TOTALT ANTALL MÅL - OVER/UNDER 2.5
Over 2.5
Over 2.5, odds 1.80
Under 2.5
Under 2.5, odds 1.95
TOTALT ANTALL MÅL - OVER/UNDER 3.5
Over 3.5
Over 3.5, odds 3.05
Under 3.5
Under 3.5, odds 1.35
Man. 24/11 21:00
Manchester United
Everton
England - Premier League
Vsport Premier League
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
Over 3.5, odds 2.80
Under 3.5
Under 3.5, odds 1.40
"""

if __name__ == "__main__":
    print("Extracting odds from Premier League snapshot...\n")
    df = parse_snapshot_to_csv(SNAPSHOT_TEXT)
    
    print("\n" + "="*80)
    print("EXTRACTED MATCH ODDS")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    print(f"\nTotal matches: {len(df)}")
    print(f"Output saved to: data/upcoming/norsk_tipping_odds.csv")
