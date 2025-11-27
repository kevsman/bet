#!/usr/bin/env python3
"""
Fetch European competition historical data from openfootball GitHub repository.

This script downloads and parses UCL/UEL/UECL data from:
https://github.com/openfootball/champions-league

Data is available in Football.TXT format and covers 2011-2025.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import urllib.request
import ssl

import pandas as pd


# URLs for openfootball data
# All three competitions are in the champions-league repo!
# Files: cl.txt (Champions League), el.txt (Europa League), conf.txt (Conference League)
UCL_BASE_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master"
UEL_BASE_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master"  # Same repo!
UECL_BASE_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master"  # Same repo!

# Seasons to fetch
SEASONS = [
    "2021-22",
    "2022-23", 
    "2023-24",
    "2024-25",
    "2025-26",  # Current season
]

# Team name standardization (openfootball -> our model)
TEAM_NAME_MAP = {
    # English
    "Liverpool FC (ENG)": "Liverpool",
    "Manchester City FC (ENG)": "Manchester City",
    "Arsenal FC (ENG)": "Arsenal",
    "Chelsea FC (ENG)": "Chelsea",
    "Aston Villa FC (ENG)": "Aston Villa",
    "Manchester United FC (ENG)": "Manchester United",
    "Tottenham Hotspur FC (ENG)": "Tottenham",
    "Newcastle United FC (ENG)": "Newcastle",
    "West Ham United FC (ENG)": "West Ham",
    "Brighton & Hove Albion FC (ENG)": "Brighton",
    
    # Spanish
    "Real Madrid CF (ESP)": "Real Madrid",
    "FC Barcelona (ESP)": "Barcelona",
    "Club Atlético de Madrid (ESP)": "Atletico Madrid",
    "Sevilla FC (ESP)": "Sevilla",
    "Real Sociedad de Fútbol (ESP)": "Real Sociedad",
    "Real Betis Balompié (ESP)": "Real Betis",
    "Villarreal CF (ESP)": "Villarreal",
    "Valencia CF (ESP)": "Valencia",
    "Girona FC (ESP)": "Girona",
    "Athletic Club (ESP)": "Athletic Bilbao",
    "Real Valladolid CF (ESP)": "Valladolid",
    "CA Osasuna (ESP)": "Osasuna",
    "Celta de Vigo (ESP)": "Celta Vigo",
    "RCD Espanyol (ESP)": "Espanyol",
    
    # German
    "FC Bayern München (GER)": "Bayern Munich",
    "Borussia Dortmund (GER)": "Borussia Dortmund",
    "RB Leipzig (GER)": "RB Leipzig",
    "Bayer 04 Leverkusen (GER)": "Bayer Leverkusen",
    "VfB Stuttgart (GER)": "Stuttgart",
    "Eintracht Frankfurt (GER)": "Eintracht Frankfurt",
    "1. FC Union Berlin (GER)": "Union Berlin",
    "VfL Wolfsburg (GER)": "Wolfsburg",
    "SC Freiburg (GER)": "Freiburg",
    "1. FSV Mainz 05 (GER)": "Mainz",
    "TSG 1899 Hoffenheim (GER)": "Hoffenheim",
    "1. FC Köln (GER)": "FC Koln",
    "Borussia Mönchengladbach (GER)": "M'gladbach",
    "1. FC Heidenheim 1846 (GER)": "Heidenheim",
    
    # Italian
    "FC Internazionale Milano (ITA)": "Inter Milan",
    "AC Milan (ITA)": "AC Milan",
    "Juventus FC (ITA)": "Juventus",
    "SSC Napoli (ITA)": "Napoli",
    "AS Roma (ITA)": "Roma",
    "SS Lazio (ITA)": "Lazio",
    "Atalanta BC (ITA)": "Atalanta",
    "ACF Fiorentina (ITA)": "Fiorentina",
    "Bologna FC 1909 (ITA)": "Bologna",
    "Torino FC (ITA)": "Torino",
    "UC Sampdoria (ITA)": "Sampdoria",
    "US Sassuolo Calcio (ITA)": "Sassuolo",
    "Udinese Calcio (ITA)": "Udinese",
    "Hellas Verona FC (ITA)": "Verona",
    
    # French
    "Paris Saint-Germain FC (FRA)": "PSG",
    "Olympique de Marseille (FRA)": "Marseille",
    "AS Monaco FC (MCO)": "Monaco",
    "Olympique Lyonnais (FRA)": "Lyon",
    "Lille OSC (FRA)": "Lille",
    "Stade Rennais FC (FRA)": "Rennes",
    "RC Lens (FRA)": "Lens",
    "OGC Nice (FRA)": "Nice",
    "Stade Brestois 29 (FRA)": "Brest",
    "RC Strasbourg Alsace (FRA)": "Strasbourg",
    "Montpellier Hérault SC (FRA)": "Montpellier",
    
    # Dutch
    "AFC Ajax (NED)": "Ajax",
    "Feyenoord Rotterdam (NED)": "Feyenoord",
    "PSV (NED)": "PSV",
    "AZ Alkmaar (NED)": "AZ Alkmaar",
    "FC Utrecht (NED)": "Utrecht",
    "FC Twente (NED)": "FC Twente",
    "Go Ahead Eagles (NED)": "Go Ahead Eagles",
    
    # Portuguese
    "Sport Lisboa e Benfica (POR)": "Benfica",
    "FC Porto (POR)": "Porto",
    "Sporting Clube de Portugal (POR)": "Sporting CP",
    "SC Braga (POR)": "Braga",
    "Vitória SC (POR)": "Guimaraes",
    
    # Belgian
    "Club Brugge KV (BEL)": "Club Brugge",
    "RSC Anderlecht (BEL)": "Anderlecht",
    "KRC Genk (BEL)": "Genk",
    "R. Antwerp FC (BEL)": "Antwerp",
    "Royale Union Saint-Gilloise (BEL)": "Union SG",
    "KAA Gent (BEL)": "Gent",
    
    # Scottish
    "Celtic FC (SCO)": "Celtic",
    "Rangers FC (SCO)": "Rangers",
    "Heart of Midlothian FC (SCO)": "Hearts",
    "Aberdeen FC (SCO)": "Aberdeen",
    
    # Austrian
    "FC Red Bull Salzburg (AUT)": "Salzburg",
    "SK Rapid Wien (AUT)": "Rapid Wien",
    "SK Sturm Graz (AUT)": "Sturm Graz",
    "LASK (AUT)": "LASK",
    "FK Austria Wien (AUT)": "Austria Wien",
    
    # Swiss
    "BSC Young Boys (SUI)": "Young Boys",
    "FC Zürich (SUI)": "Zurich",
    "FC Basel 1893 (SUI)": "Basel",
    "Servette FC (SUI)": "Servette",
    
    # Ukrainian
    "FK Shakhtar Donetsk (UKR)": "Shakhtar Donetsk",
    "FC Dynamo Kyiv (UKR)": "Dynamo Kyiv",
    
    # Russian
    "FC Spartak Moskva (RUS)": "Spartak Moscow",
    "FC Zenit Saint Petersburg (RUS)": "Zenit",
    "PFC CSKA Moskva (RUS)": "CSKA Moscow",
    "FC Lokomotiv Moskva (RUS)": "Lokomotiv Moscow",
    
    # Turkish
    "Galatasaray SK (TUR)": "Galatasaray",
    "Fenerbahçe SK (TUR)": "Fenerbahce",
    "Beşiktaş JK (TUR)": "Besiktas",
    "Trabzonspor (TUR)": "Trabzonspor",
    
    # Greek
    "Olympiacos FC (GRE)": "Olympiacos",
    "PAOK FC (GRE)": "PAOK",
    "Panathinaikos FC (GRE)": "Panathinaikos",
    "AEK Athens FC (GRE)": "AEK Athens",
    
    # Other
    "Copenhagen FC (DEN)": "Copenhagen",
    "FC København (DEN)": "Copenhagen",
    "Malmö FF (SWE)": "Malmo",
    "IF Elfsborg (SWE)": "Elfsborg",
    "BK Häcken (SWE)": "Hacken",
    "Bodo/Glimt (NOR)": "Bodo/Glimt",
    "FK Bodø/Glimt (NOR)": "Bodo/Glimt",
    "SK Brann (NOR)": "Brann",
    "Rosenborg BK (NOR)": "Rosenborg",
    "Molde FK (NOR)": "Molde",
    "Legia Warszawa (POL)": "Legia Warsaw",
    "Lech Poznań (POL)": "Lech Poznan",
    "Slavia Praha (CZE)": "Slavia Prague",
    "AC Sparta Praha (CZE)": "Sparta Prague",
    "Viktoria Plzeň (CZE)": "Viktoria Plzen",
    "GNK Dinamo Zagreb (CRO)": "Dinamo Zagreb",
    "HNK Rijeka (CRO)": "Rijeka",
    "ŠK Slovan Bratislava (SVK)": "Slovan Bratislava",
    "FK Crvena Zvezda (SRB)": "Red Star Belgrade",
    "Partizan Belgrade (SRB)": "Partizan",
    "Ferencvárosi TC (HUN)": "Ferencvaros",
    "Maccabi Tel Aviv FC (ISR)": "Maccabi Tel Aviv",
    "Maccabi Haifa FC (ISR)": "Maccabi Haifa",
    "PFC Ludogorets 1945 Razgrad (BUL)": "Ludogorets",
    "FCSB (ROU)": "FCSB",
    "CFR Cluj (ROU)": "CFR Cluj",
    "Qarabağ FK (AZE)": "Qarabag",
    "FC Sheriff Tiraspol (MDA)": "Sheriff",
    "FK Žalgiris Vilnius (LTU)": "Zalgiris",
    "HJK Helsinki (FIN)": "HJK",
    "FC Midtjylland (DEN)": "Midtjylland",
    "Brøndby IF (DEN)": "Brondby",
    "AZ (NED)": "AZ Alkmaar",
}


def standardize_team_name(name: str) -> str:
    """Standardize team name to our model format."""
    # Try direct mapping first
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    
    # Try to extract team name (remove country code)
    # Format is usually "Team Name (XXX)"
    match = re.match(r"(.+?)\s*\([A-Z]{3}\)$", name)
    if match:
        base_name = match.group(1).strip()
        # Check if base name is in mappings
        for key, value in TEAM_NAME_MAP.items():
            if base_name in key:
                return value
        return base_name
    
    return name


def parse_score(score_str: str) -> Tuple[int, int] | None:
    """Parse score from format like '3-1 (2-0)' or '1-1'."""
    # Remove extra time / penalties info
    score_str = score_str.split(" pen.")[0]
    score_str = score_str.split(" a.e.t.")[0]
    
    # Extract full time score (first score)
    match = re.match(r"(\d+)-(\d+)", score_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def fetch_url(url: str) -> str | None:
    """Fetch URL content with SSL handling."""
    try:
        # Create SSL context that doesn't verify certificates
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def parse_openfootball_txt(content: str, competition: str, season: str) -> List[dict]:
    """Parse openfootball .txt format into match records."""
    matches = []
    current_round = ""
    current_date = None
    
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Skip header lines
        if line.startswith("=") or line.startswith("#"):
            continue
        
        # Match round header (e.g., "» League, Matchday 1")
        if line.startswith("»"):
            current_round = line.replace("»", "").strip()
            continue
        
        # Match date line (e.g., "Tue Sep/17 2024" or "Wed Sep/18")
        date_match = re.match(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w+)/(\d+)(?:\s+(\d{4}))?", line)
        if date_match:
            day_name, month_str, day, year = date_match.groups()
            if year is None:
                # Use year from season
                year = season.split("-")[0] if "2025" not in season else "2025"
                # Handle year rollover (Jan-May would be next year)
                month_abbrev = month_str[:3].lower()
                if month_abbrev in ["jan", "feb", "mar", "apr", "may"]:
                    year = str(int(year) + 1) if "-" in season else year
            
            # Parse month
            month_map = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
            }
            month_abbrev = month_str[:3].lower()
            month = month_map.get(month_abbrev, 1)
            
            try:
                current_date = datetime(int(year), month, int(day))
            except:
                current_date = None
            continue
        
        # Match result line (e.g., "  18.45  Team A  v Team B  3-1 (2-0)")
        match_line = re.match(r"^\s*(\d{2}\.\d{2})?\s*(.+?)\s+v\s+(.+?)\s+(\d+-\d+.*)?$", line)
        if match_line and current_date:
            time, home_team, away_team, score = match_line.groups()
            
            # Skip if no score (future match)
            if not score:
                continue
            
            # Parse score
            parsed_score = parse_score(score)
            if not parsed_score:
                continue
            
            home_goals, away_goals = parsed_score
            
            # Standardize team names
            home_team = standardize_team_name(home_team.strip())
            away_team = standardize_team_name(away_team.strip())
            
            matches.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "FTHG": home_goals,
                "FTAG": away_goals,
                "competition_code": competition,
                "season": season,
                "round": current_round,
            })
    
    return matches


def fetch_champions_league_data(seasons: List[str] = None) -> pd.DataFrame:
    """Fetch Champions League data for specified seasons."""
    if seasons is None:
        seasons = SEASONS
    
    all_matches = []
    
    for season in seasons:
        print(f"Fetching UCL {season}...")
        url = f"{UCL_BASE_URL}/{season}/cl.txt"
        content = fetch_url(url)
        
        if content:
            matches = parse_openfootball_txt(content, "UCL", season)
            all_matches.extend(matches)
            print(f"  Found {len(matches)} UCL matches")
        else:
            print(f"  No data found for UCL {season}")
    
    return pd.DataFrame(all_matches)


def fetch_europa_league_data(seasons: List[str] = None) -> pd.DataFrame:
    """Fetch Europa League data for specified seasons."""
    if seasons is None:
        seasons = SEASONS
    
    all_matches = []
    
    for season in seasons:
        print(f"Fetching UEL {season}...")
        url = f"{UEL_BASE_URL}/{season}/el.txt"  # el.txt for Europa League
        content = fetch_url(url)
        
        if content:
            matches = parse_openfootball_txt(content, "UEL", season)
            all_matches.extend(matches)
            print(f"  Found {len(matches)} UEL matches")
        else:
            print(f"  No data found for UEL {season}")
    
    return pd.DataFrame(all_matches)


def fetch_conference_league_data(seasons: List[str] = None) -> pd.DataFrame:
    """Fetch Conference League data for specified seasons."""
    if seasons is None:
        seasons = SEASONS
    
    all_matches = []
    
    for season in seasons:
        print(f"Fetching UECL {season}...")
        url = f"{UECL_BASE_URL}/{season}/conf.txt"  # conf.txt for Conference League
        content = fetch_url(url)
        
        if content:
            matches = parse_openfootball_txt(content, "UECL", season)
            all_matches.extend(matches)
            print(f"  Found {len(matches)} UECL matches")
        else:
            print(f"  No data found for UECL {season}")
    
    return pd.DataFrame(all_matches)


def fetch_all_european_data(seasons: List[str] = None) -> pd.DataFrame:
    """Fetch all European competition data."""
    ucl_df = fetch_champions_league_data(seasons)
    uel_df = fetch_europa_league_data(seasons)
    uecl_df = fetch_conference_league_data(seasons)
    
    combined = pd.concat([ucl_df, uel_df, uecl_df], ignore_index=True)
    
    # Add derived columns
    if not combined.empty:
        combined["FTR"] = combined.apply(
            lambda r: "H" if r["FTHG"] > r["FTAG"] else ("A" if r["FTHG"] < r["FTAG"] else "D"),
            axis=1
        )
        combined["total_goals"] = combined["FTHG"] + combined["FTAG"]
    
    return combined


def main():
    """Main function to fetch and save European data."""
    print("=" * 60)
    print("FETCHING EUROPEAN COMPETITION DATA FROM OPENFOOTBALL")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("data/raw/european")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch openfootball data (primarily UCL)
    df = fetch_all_european_data()
    
    # Also fetch Wikipedia data for UEL/UECL 2025-26
    try:
        from fetch_european_wikipedia import load_wikipedia_data
        wiki_df = load_wikipedia_data()
        
        if not wiki_df.empty:
            print(f"\n📥 Adding {len(wiki_df)} UEL/UECL matches from Wikipedia")
            
            # Convert Wikipedia format to match openfootball format
            wiki_converted = wiki_df.rename(columns={
                "home_team": "HomeTeam",
                "away_team": "AwayTeam",
                "home_goals": "FTHG",
                "away_goals": "FTAG",
                "date": "Date"
            })
            
            # Add missing columns
            wiki_converted["competition_code"] = wiki_converted["competition"].apply(
                lambda x: "EL" if x == "EL" else "UECL"
            )
            
            # Set FTR (Full Time Result)
            def get_ftr(row):
                if row["FTHG"] > row["FTAG"]:
                    return "H"
                elif row["FTHG"] < row["FTAG"]:
                    return "A"
                else:
                    return "D"
            
            wiki_converted["FTR"] = wiki_converted.apply(get_ftr, axis=1)
            wiki_converted["round"] = "League Phase"
            
            # Keep only columns that exist in openfootball data
            cols_to_keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", 
                          "competition_code", "season", "round"]
            wiki_converted = wiki_converted[cols_to_keep]
            
            # Combine with openfootball data
            if df.empty:
                df = wiki_converted
            else:
                # Remove any UEL/UECL 2025-26 from openfootball (likely empty anyway)
                df_filtered = df[~((df["season"] == "2025-26") & 
                                   (df["competition_code"].isin(["EL", "UECL"])))]
                df = pd.concat([df_filtered, wiki_converted], ignore_index=True)
            
            print(f"✅ Combined dataset now has {len(df)} matches")
    except Exception as e:
        print(f"⚠️ Could not load Wikipedia data: {e}")
    
    if df.empty:
        print("\nNo data fetched!")
        return
    
    # Sort by date
    df = df.sort_values("Date")
    
    # Save to CSV
    output_path = output_dir / "openfootball_european.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total matches: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nBy competition:")
    print(df["competition_code"].value_counts())
    print(f"\nBy season:")
    print(df["season"].value_counts())
    print(f"\nUnique teams: {len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))}")
    print(f"\nSaved to: {output_path}")
    
    # Also update the European model dataset
    processed_dir = Path("data/processed/european")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    existing_path = processed_dir / "european_dataset.csv"
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        print(f"\nExisting dataset has {len(existing_df)} matches")
    
    # Use new data (it's more complete)
    df.to_csv(existing_path, index=False)
    print(f"Updated dataset with {len(df)} matches")


if __name__ == "__main__":
    main()
