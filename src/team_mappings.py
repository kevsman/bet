"""
Team name mapping utilities for Norsk Tipping scraper.

Norsk Tipping uses Norwegian/full team names while football-data.co.uk
uses shortened English names. This module provides mapping between the two.
"""

from __future__ import annotations

# Premier League team mappings (Norsk Tipping -> football-data.co.uk)
PREMIER_LEAGUE_TEAMS = {
    # Current teams
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "Watford": "Watford",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

# Add other leagues as needed
# La Liga, Bundesliga, Serie A, etc.

ALL_TEAM_MAPPINGS = {
    **PREMIER_LEAGUE_TEAMS,
    # Add other leagues here
}


def normalize_team_name(norsk_tipping_name: str) -> str:
    """
    Convert Norsk Tipping team name to football-data.co.uk format.
    
    Args:
        norsk_tipping_name: Team name from Norsk Tipping site
        
    Returns:
        Standardized team name matching training data, or original if no mapping exists
    """
    normalized = ALL_TEAM_MAPPINGS.get(norsk_tipping_name)
    if normalized is None:
        # Try fuzzy matching for unknown teams
        print(f"Warning: No mapping found for '{norsk_tipping_name}', using as-is")
        return norsk_tipping_name
    return normalized


def get_unmapped_teams(team_names: list[str]) -> list[str]:
    """
    Find team names that don't have mappings yet.
    
    Args:
        team_names: List of team names from Norsk Tipping
        
    Returns:
        List of team names without mappings
    """
    return [name for name in team_names if name not in ALL_TEAM_MAPPINGS]
