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

# Championship (E1)
CHAMPIONSHIP_TEAMS = {
    "Barnsley": "Barnsley",
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Blackpool": "Blackpool",
    "Bristol City": "Bristol City",
    "Cardiff City": "Cardiff",
    "Coventry City": "Coventry",
    "Derby County": "Derby",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Middlesbrough": "Middlesbrough",
    "Millwall": "Millwall",
    "Norwich City": "Norwich",
    "Preston North End": "Preston",
    "Queens Park Rangers": "QPR",
    "Reading": "Reading",
    "Sheffield Wednesday": "Sheff Wed",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "West Bromwich Albion": "West Brom",
}

# La Liga (SP1)
LA_LIGA_TEAMS = {
    "Athletic Club": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Barcelona": "Barcelona",
    "Deportivo Alavés": "Alaves",
    "Getafe": "Getafe",
    "Girona": "Girona",
    "Real Betis": "Betis",
    "Real Madrid": "Real Madrid",
    "Real Sociedad": "Sociedad",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Osasuna": "Osasuna",
    "Espanyol": "Espanol",
    "Las Palmas": "Las Palmas",
    "Mallorca": "Mallorca",
    "Leganés": "Leganes",
    "Valladolid": "Valladolid",
}

# Bundesliga (D1)
BUNDESLIGA_TEAMS = {
    "Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Mönchengladbach": "M'gladbach",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "SC Freiburg": "Freiburg",
    "VfB Stuttgart": "Stuttgart",
    "1. FC Union Berlin": "Union Berlin",
    "1. FC Köln": "FC Koln",
    "FSV Mainz 05": "Mainz",
    "FC Augsburg": "Augsburg",
    "VfL Bochum": "Bochum",
    "Werder Bremen": "Werder Bremen",
    "FC St. Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel",
    "Heidenheim": "Heidenheim",
}

# Serie A (I1)
SERIE_A_TEAMS = {
    "AC Milan": "Milan",
    "Inter": "Inter",
    "Juventus": "Juventus",
    "Napoli": "Napoli",
    "AS Roma": "Roma",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Udinese": "Udinese",
    "Sassuolo": "Sassuolo",
    "Empoli": "Empoli",
    "Cagliari": "Cagliari",
    "Hellas Verona": "Verona",
    "Genoa": "Genoa",
    "Lecce": "Lecce",
    "Monza": "Monza",
    "Como": "Como",
    "Parma": "Parma",
}

# Ligue 1 (F1)
LIGUE_1_TEAMS = {
    "Paris Saint-Germain": "Paris SG",
    "Marseille": "Marseille",
    "Lyon": "Lyon",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Nice": "Nice",
    "Rennes": "Rennes",
    "Lens": "Lens",
    "Montpellier": "Montpellier",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Reims": "Reims",
    "Brest": "Brest",
    "Toulouse": "Toulouse",
    "Le Havre": "Le Havre",
    "Auxerre": "Auxerre",
    "Angers": "Angers",
    "Saint-Étienne": "St Etienne",
}

ALL_TEAM_MAPPINGS = {
    **PREMIER_LEAGUE_TEAMS,
    **CHAMPIONSHIP_TEAMS,
    **LA_LIGA_TEAMS,
    **BUNDESLIGA_TEAMS,
    **SERIE_A_TEAMS,
    **LIGUE_1_TEAMS,
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
