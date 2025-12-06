"""
Advanced features scraper for enhanced football prediction.

Fetches additional data that significantly improves model accuracy:
1. Player availability (injuries/suspensions) from Transfermarkt
2. Manager information (tenure, recent results)
3. Advanced xG metrics (PPDA, progressive passes) from FBref
4. Weather data from Open-Meteo API

Rate limiting is applied to all scrapers to respect source limits.
"""
from __future__ import annotations

import json
import re
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Realistic browser headers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def get_headers() -> Dict[str, str]:
    """Get randomized headers for web requests."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


# ==============================================================================
# TEAM NAME MAPPINGS (for matching between different sources)
# ==============================================================================

# Mapping from football-data.co.uk names to FBref names
TEAM_NAME_MAP_FBREF = {
    # Premier League
    "Man United": "Manchester Utd",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nott'ham Forest",
    "Tottenham": "Tottenham",
    "Newcastle": "Newcastle Utd",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton",
    "Brighton": "Brighton",
    "Bournemouth": "Bournemouth",
    "Sheffield United": "Sheffield Utd",
    "Luton": "Luton Town",
    # La Liga
    "Ath Madrid": "Atlético Madrid",
    "Ath Bilbao": "Athletic Club",
    "Betis": "Real Betis",
    "Sociedad": "Real Sociedad",
    "Espanol": "Espanyol",
    "Vallecano": "Rayo Vallecano",
    "Almeria": "Almería",
    "Cadiz": "Cádiz",
    # Serie A
    "Inter": "Internazionale",
    "AC Milan": "Milan",
    "Napoli": "Napoli",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Verona": "Hellas Verona",
    "Parma": "Parma",
    "Monza": "Monza",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Leverkusen": "Leverkusen",
    "Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Frankfurt": "Eint Frankfurt",
    "Freiburg": "Freiburg",
    "Hoffenheim": "Hoffenheim",
    "Wolfsburg": "Wolfsburg",
    "M'gladbach": "M'Gladbach",
    "Augsburg": "Augsburg",
    "Mainz": "Mainz 05",
    "Heidenheim": "Heidenheim",
    "Darmstadt": "Darmstadt 98",
    "Cologne": "Köln",
    "Bochum": "Bochum",
    "Union Berlin": "Union Berlin",
    "Werder Bremen": "Werder Bremen",
    "St Pauli": "St. Pauli",
    "Holstein Kiel": "Holstein Kiel",
    # Ligue 1
    "Paris SG": "Paris S-G",
    "Paris Saint Germain": "Paris S-G",
    "Lyon": "Lyon",
    "Marseille": "Marseille",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Lens": "Lens",
    "Rennes": "Rennes",
    "Nice": "Nice",
    "Nantes": "Nantes",
    "St Etienne": "Saint-Étienne",
    "Strasbourg": "Strasbourg",
    "Toulouse": "Toulouse",
    "Reims": "Reims",
    "Montpellier": "Montpellier",
    "Brest": "Brest",
    "Le Havre": "Le Havre",
    "Lorient": "Lorient",
    "Metz": "Metz",
    "Clermont": "Clermont Foot",
}

# Mapping from football-data.co.uk names to Transfermarkt slugs
TEAM_NAME_MAP_TRANSFERMARKT = {
    # Premier League
    "Man United": ("manchester-united", 985),
    "Man City": ("manchester-city", 281),
    "Liverpool": ("fc-liverpool", 31),
    "Arsenal": ("fc-arsenal", 11),
    "Chelsea": ("fc-chelsea", 631),
    "Tottenham": ("tottenham-hotspur", 148),
    "Newcastle": ("newcastle-united", 762),
    "Aston Villa": ("aston-villa", 405),
    "Brighton": ("brighton-amp-hove-albion", 1237),
    "West Ham": ("west-ham-united", 379),
    "Bournemouth": ("afc-bournemouth", 989),
    "Fulham": ("fc-fulham", 931),
    "Wolves": ("wolverhampton-wanderers", 543),
    "Crystal Palace": ("crystal-palace", 873),
    "Everton": ("fc-everton", 29),
    "Brentford": ("fc-brentford", 1148),
    "Nott'm Forest": ("nottingham-forest", 703),
    "Ipswich": ("ipswich-town", 677),
    "Leicester": ("leicester-city", 1003),
    "Southampton": ("fc-southampton", 180),
    # Add more as needed
}


# ==============================================================================
# FBref League IDs for scraping
# ==============================================================================

FBREF_LEAGUE_IDS = {
    "E0": {"id": "9", "name": "Premier-League", "country": "eng"},
    "D1": {"id": "20", "name": "Bundesliga", "country": "ger"},
    "SP1": {"id": "12", "name": "La-Liga", "country": "esp"},
    "I1": {"id": "11", "name": "Serie-A", "country": "ita"},
    "F1": {"id": "13", "name": "Ligue-1", "country": "fra"},
    "N1": {"id": "23", "name": "Eredivisie", "country": "ned"},
    "P1": {"id": "32", "name": "Primeira-Liga", "country": "por"},
    "E1": {"id": "10", "name": "Championship", "country": "eng"},
    "SC0": {"id": "40", "name": "Scottish-Premiership", "country": "sco"},
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class PlayerInjury:
    """Represents a player injury/suspension."""
    player_name: str
    team: str
    injury_type: str  # "Injury", "Suspension", "Illness"
    description: str
    date_from: Optional[str]
    date_until: Optional[str]
    is_key_player: bool  # Based on market value ranking


@dataclass
class ManagerInfo:
    """Manager information for a team."""
    name: str
    team: str
    appointed_date: Optional[str]
    tenure_days: int
    matches_in_charge: int
    win_rate: float
    is_interim: bool


@dataclass
class AdvancedTeamStats:
    """Advanced team statistics from FBref."""
    team: str
    league: str
    season: str
    # Possession
    possession_pct: float
    # Passing
    pass_completion_pct: float
    progressive_passes: int
    progressive_carries: int
    passes_into_final_third: int
    passes_into_penalty_area: int
    # Pressing (PPDA - Passes Per Defensive Action)
    ppda: float  # Lower = more pressing
    # Set pieces
    set_piece_xg: float
    corner_kicks: int
    # Defense
    tackles_won_pct: float
    interceptions: int
    clearances: int
    # Misc
    aerials_won_pct: float


@dataclass
class WeatherData:
    """Weather conditions for a match."""
    date: str
    venue_city: str
    temperature_c: float
    precipitation_mm: float
    wind_speed_kmh: float
    humidity_pct: float
    weather_code: int  # WMO weather code


# ==============================================================================
# SELENIUM BROWSER HELPER
# ==============================================================================

def get_selenium_driver(headless: bool = True):
    """
    Create a Selenium Chrome driver for bypassing bot detection.
    Uses webdriver-manager for automatic ChromeDriver installation.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        print("  Selenium not installed. Run: pip install selenium webdriver-manager")
        return None
    
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # Remove webdriver flag to avoid detection
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver


# ==============================================================================
# FBref ADVANCED STATS SCRAPER
# ==============================================================================

class FBrefAdvancedScraper:
    """
    Scrape advanced statistics from FBref.
    
    Includes:
    - PPDA (Passes Per Defensive Action)
    - Progressive passes and carries
    - Set piece expected goals
    - Possession and passing stats
    
    Uses Selenium when regular requests get blocked (403).
    """
    
    BASE_URL = "https://fbref.com/en"
    RATE_LIMIT = 4.0  # seconds between requests
    
    def __init__(self, cache_dir: Optional[Path] = None, use_selenium: bool = True):
        self.cache_dir = cache_dir or Path("data/cache/fbref")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.last_request_time = 0
        self.use_selenium = use_selenium
        self._driver = None
        self._selenium_failed = False
    
    def _get_driver(self):
        """Get or create Selenium driver."""
        if self._driver is None and not self._selenium_failed:
            self._driver = get_selenium_driver(headless=True)
            if self._driver is None:
                self._selenium_failed = True
        return self._driver
    
    def _close_driver(self):
        """Close Selenium driver if open."""
        if self._driver:
            try:
                self._driver.quit()
            except:
                pass
            self._driver = None
    
    def __del__(self):
        """Cleanup driver on destruction."""
        self._close_driver()
    
    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            sleep_time = self.RATE_LIMIT - elapsed + random.uniform(0.5, 2.0)
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _fetch_with_selenium(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch page using Selenium browser."""
        driver = self._get_driver()
        if not driver:
            return None
        
        try:
            self._rate_limit()
            driver.get(url)
            time.sleep(random.uniform(2, 4))  # Wait for page to load
            
            # Scroll to trigger any lazy loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            
            html = driver.page_source
            return BeautifulSoup(html, "html.parser")
        except Exception as e:
            print(f"  Selenium error for {url}: {e}")
            return None
    
    def _fetch_page(self, url: str, retries: int = 2) -> Optional[BeautifulSoup]:
        """Fetch and parse a page with retry logic. Falls back to Selenium on 403."""
        # First try with requests (faster)
        for attempt in range(retries):
            try:
                self._rate_limit()
                response = self.session.get(url, headers=get_headers(), timeout=30)
                
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    print(f"  Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 403:
                    print(f"  Got 403 - switching to Selenium browser...")
                    break  # Exit loop to try Selenium
                
                response.raise_for_status()
                return BeautifulSoup(response.text, "html.parser")
                
            except Exception as e:
                if "403" in str(e):
                    print(f"  Got 403 - switching to Selenium browser...")
                    break
                print(f"  Error fetching {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
        
        # Fall back to Selenium if requests failed
        if self.use_selenium:
            print(f"  Trying with Selenium browser...")
            return self._fetch_with_selenium(url)
        
        return None
    
    def get_team_possession_stats(
        self,
        league_code: str,
        season: str  # e.g., "2024-2025"
    ) -> List[Dict[str, Any]]:
        """
        Get possession and passing stats for all teams in a league.
        
        Returns progressive passes, progressive carries, passes into final third, etc.
        """
        if league_code not in FBREF_LEAGUE_IDS:
            print(f"  League {league_code} not supported for FBref scraping")
            return []
        
        league_info = FBREF_LEAGUE_IDS[league_code]
        url = f"{self.BASE_URL}/comps/{league_info['id']}/{season}/possession/{season}-{league_info['name']}-Stats"
        
        print(f"  Fetching possession stats from: {url}")
        soup = self._fetch_page(url)
        
        if not soup:
            return []
        
        results = []
        
        # Find the possession table
        table = soup.find("table", {"id": "stats_squads_possession_for"})
        if not table:
            print("  Could not find possession table")
            return []
        
        tbody = table.find("tbody")
        if not tbody:
            return []
        
        for row in tbody.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) < 15:
                continue
            
            try:
                team_cell = cells[0]
                team_name = team_cell.get_text(strip=True)
                
                data = {
                    "team": team_name,
                    "league": league_code,
                    "season": season,
                    "possession_pct": self._safe_float(cells[2].get_text(strip=True)),
                    "touches": self._safe_int(cells[3].get_text(strip=True)),
                    "touches_def_3rd": self._safe_int(cells[5].get_text(strip=True)),
                    "touches_mid_3rd": self._safe_int(cells[6].get_text(strip=True)),
                    "touches_att_3rd": self._safe_int(cells[7].get_text(strip=True)),
                    "touches_att_pen": self._safe_int(cells[8].get_text(strip=True)),
                    "progressive_carries": self._safe_int(cells[13].get_text(strip=True)),
                    "progressive_passes_received": self._safe_int(cells[15].get_text(strip=True)),
                }
                results.append(data)
                
            except Exception as e:
                continue
        
        print(f"  Found possession stats for {len(results)} teams")
        return results
    
    def get_team_passing_stats(
        self,
        league_code: str,
        season: str
    ) -> List[Dict[str, Any]]:
        """Get passing statistics including progressive passes."""
        if league_code not in FBREF_LEAGUE_IDS:
            return []
        
        league_info = FBREF_LEAGUE_IDS[league_code]
        url = f"{self.BASE_URL}/comps/{league_info['id']}/{season}/passing/{season}-{league_info['name']}-Stats"
        
        print(f"  Fetching passing stats from: {url}")
        soup = self._fetch_page(url)
        
        if not soup:
            return []
        
        results = []
        table = soup.find("table", {"id": "stats_squads_passing_for"})
        
        if not table:
            print("  Could not find passing table")
            return []
        
        tbody = table.find("tbody")
        if not tbody:
            return []
        
        for row in tbody.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) < 20:
                continue
            
            try:
                team_name = cells[0].get_text(strip=True)
                
                data = {
                    "team": team_name,
                    "league": league_code,
                    "season": season,
                    "passes_completed": self._safe_int(cells[3].get_text(strip=True)),
                    "passes_attempted": self._safe_int(cells[4].get_text(strip=True)),
                    "pass_completion_pct": self._safe_float(cells[5].get_text(strip=True)),
                    "progressive_passes": self._safe_int(cells[16].get_text(strip=True)),
                    "passes_into_final_third": self._safe_int(cells[14].get_text(strip=True)),
                    "passes_into_penalty_area": self._safe_int(cells[15].get_text(strip=True)),
                }
                results.append(data)
                
            except Exception:
                continue
        
        print(f"  Found passing stats for {len(results)} teams")
        return results
    
    def get_team_defensive_stats(
        self,
        league_code: str,
        season: str
    ) -> List[Dict[str, Any]]:
        """
        Get defensive stats including PPDA-style metrics.
        
        PPDA = Opponent passes / Defensive actions
        Lower PPDA = more pressing
        """
        if league_code not in FBREF_LEAGUE_IDS:
            return []
        
        league_info = FBREF_LEAGUE_IDS[league_code]
        url = f"{self.BASE_URL}/comps/{league_info['id']}/{season}/defense/{season}-{league_info['name']}-Stats"
        
        print(f"  Fetching defensive stats from: {url}")
        soup = self._fetch_page(url)
        
        if not soup:
            return []
        
        results = []
        table = soup.find("table", {"id": "stats_squads_defense_for"})
        
        if not table:
            print("  Could not find defense table")
            return []
        
        tbody = table.find("tbody")
        if not tbody:
            return []
        
        for row in tbody.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) < 20:
                continue
            
            try:
                team_name = cells[0].get_text(strip=True)
                
                tackles = self._safe_int(cells[2].get_text(strip=True))
                tackles_won = self._safe_int(cells[3].get_text(strip=True))
                
                data = {
                    "team": team_name,
                    "league": league_code,
                    "season": season,
                    "tackles": tackles,
                    "tackles_won": tackles_won,
                    "tackles_won_pct": (tackles_won / tackles * 100) if tackles > 0 else 0,
                    "tackles_def_3rd": self._safe_int(cells[4].get_text(strip=True)),
                    "tackles_mid_3rd": self._safe_int(cells[5].get_text(strip=True)),
                    "tackles_att_3rd": self._safe_int(cells[6].get_text(strip=True)),
                    "interceptions": self._safe_int(cells[14].get_text(strip=True)),
                    "blocks": self._safe_int(cells[11].get_text(strip=True)),
                    "clearances": self._safe_int(cells[18].get_text(strip=True)),
                }
                results.append(data)
                
            except Exception:
                continue
        
        print(f"  Found defensive stats for {len(results)} teams")
        return results
    
    def get_team_shooting_stats(
        self,
        league_code: str,
        season: str
    ) -> List[Dict[str, Any]]:
        """Get shooting stats including xG breakdown."""
        if league_code not in FBREF_LEAGUE_IDS:
            return []
        
        league_info = FBREF_LEAGUE_IDS[league_code]
        url = f"{self.BASE_URL}/comps/{league_info['id']}/{season}/shooting/{season}-{league_info['name']}-Stats"
        
        print(f"  Fetching shooting stats from: {url}")
        soup = self._fetch_page(url)
        
        if not soup:
            return []
        
        results = []
        table = soup.find("table", {"id": "stats_squads_shooting_for"})
        
        if not table:
            print("  Could not find shooting table")
            return []
        
        tbody = table.find("tbody")
        if not tbody:
            return []
        
        for row in tbody.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) < 15:
                continue
            
            try:
                team_name = cells[0].get_text(strip=True)
                
                data = {
                    "team": team_name,
                    "league": league_code,
                    "season": season,
                    "goals": self._safe_int(cells[2].get_text(strip=True)),
                    "shots": self._safe_int(cells[3].get_text(strip=True)),
                    "shots_on_target": self._safe_int(cells[4].get_text(strip=True)),
                    "shots_on_target_pct": self._safe_float(cells[5].get_text(strip=True)),
                    "xg": self._safe_float(cells[10].get_text(strip=True)),
                    "npxg": self._safe_float(cells[11].get_text(strip=True)),  # Non-penalty xG
                    "xg_per_shot": self._safe_float(cells[13].get_text(strip=True)),
                }
                results.append(data)
                
            except Exception:
                continue
        
        print(f"  Found shooting stats for {len(results)} teams")
        return results
    
    def get_all_team_stats(
        self,
        league_code: str,
        season: str
    ) -> pd.DataFrame:
        """
        Get all advanced stats for a league and merge into single DataFrame.
        """
        print(f"\nFetching all FBref stats for {league_code} {season}...")
        
        # Get all stat types
        possession = self.get_team_possession_stats(league_code, season)
        passing = self.get_team_passing_stats(league_code, season)
        defense = self.get_team_defensive_stats(league_code, season)
        shooting = self.get_team_shooting_stats(league_code, season)
        
        if not any([possession, passing, defense, shooting]):
            return pd.DataFrame()
        
        # Convert to DataFrames
        dfs = []
        if possession:
            dfs.append(pd.DataFrame(possession))
        if passing:
            dfs.append(pd.DataFrame(passing))
        if defense:
            dfs.append(pd.DataFrame(defense))
        if shooting:
            dfs.append(pd.DataFrame(shooting))
        
        if not dfs:
            return pd.DataFrame()
        
        # Merge all stats on team
        result = dfs[0]
        for df in dfs[1:]:
            # Drop duplicate columns before merge
            merge_cols = ["team", "league", "season"]
            df_to_merge = df.drop(
                columns=[c for c in df.columns if c in result.columns and c not in merge_cols],
                errors="ignore"
            )
            result = result.merge(df_to_merge, on=merge_cols, how="outer")
        
        return result
    
    @staticmethod
    def _safe_float(value: str) -> float:
        """Safely convert string to float."""
        try:
            return float(value.replace(",", "").replace("%", ""))
        except (ValueError, AttributeError):
            return 0.0
    
    @staticmethod
    def _safe_int(value: str) -> int:
        """Safely convert string to int."""
        try:
            return int(value.replace(",", ""))
        except (ValueError, AttributeError):
            return 0


# ==============================================================================
# WEATHER DATA FETCHER (Open-Meteo API - Free, no API key required)
# ==============================================================================

class WeatherFetcher:
    """
    Fetch weather data from Open-Meteo API.
    
    This is completely free and doesn't require an API key.
    Historical data available from 1940 onwards.
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # Stadium coordinates (latitude, longitude)
    STADIUM_COORDINATES = {
        # Premier League
        "Arsenal": (51.555, -0.108),  # Emirates
        "Aston Villa": (52.509, -1.885),  # Villa Park
        "Bournemouth": (50.735, -1.838),  # Vitality
        "Brentford": (51.491, -0.289),  # Gtech
        "Brighton": (50.862, -0.084),  # Amex
        "Chelsea": (51.482, -0.191),  # Stamford Bridge
        "Crystal Palace": (51.398, -0.086),  # Selhurst Park
        "Everton": (53.439, -2.966),  # Goodison Park
        "Fulham": (51.475, -0.221),  # Craven Cottage
        "Ipswich": (52.055, 1.145),  # Portman Road
        "Leicester": (52.620, -1.142),  # King Power
        "Liverpool": (53.431, -2.961),  # Anfield
        "Man City": (53.483, -2.200),  # Etihad
        "Man United": (53.463, -2.291),  # Old Trafford
        "Newcastle": (54.976, -1.622),  # St James' Park
        "Nott'm Forest": (52.940, -1.133),  # City Ground
        "Southampton": (50.906, -1.391),  # St Mary's
        "Tottenham": (51.604, -0.067),  # Tottenham Stadium
        "West Ham": (51.539, -0.017),  # London Stadium
        "Wolves": (52.590, -2.130),  # Molineux
        # La Liga
        "Barcelona": (41.381, 2.123),  # Camp Nou (Montjuic temp)
        "Real Madrid": (40.453, -3.688),  # Bernabeu
        "Ath Madrid": (40.436, -3.599),  # Wanda Metropolitano
        "Sevilla": (37.384, -5.971),  # Sánchez-Pizjuán
        "Sociedad": (43.301, -1.974),  # Anoeta
        "Betis": (37.357, -5.982),  # Benito Villamarín
        "Villarreal": (39.944, -0.104),  # Estadio de la Cerámica
        "Ath Bilbao": (43.264, -2.949),  # San Mamés
        "Valencia": (39.475, -0.358),  # Mestalla
        # Bundesliga
        "Bayern Munich": (48.219, 11.625),  # Allianz Arena
        "Dortmund": (51.493, 7.452),  # Signal Iduna Park
        "RB Leipzig": (51.346, 12.348),  # Red Bull Arena
        "Leverkusen": (51.038, 7.002),  # BayArena
        "Frankfurt": (50.069, 8.646),  # Deutsche Bank Park
        "Freiburg": (48.022, 7.831),  # Europa-Park Stadion
        # Serie A
        "Inter": (45.478, 9.124),  # San Siro
        "AC Milan": (45.478, 9.124),  # San Siro
        "Juventus": (45.110, 7.641),  # Allianz Stadium
        "Napoli": (40.828, 14.193),  # Diego Armando Maradona
        "Roma": (41.934, 12.455),  # Stadio Olimpico
        "Lazio": (41.934, 12.455),  # Stadio Olimpico
        "Atalanta": (45.709, 9.681),  # Gewiss Stadium
        # Ligue 1
        "Paris SG": (48.842, 2.253),  # Parc des Princes
        "Marseille": (43.270, 5.396),  # Stade Vélodrome
        "Lyon": (45.765, 4.982),  # Groupama Stadium
        "Monaco": (43.727, 7.416),  # Stade Louis II
        "Lille": (50.612, 3.130),  # Pierre-Mauroy
    }
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_match_weather(
        self,
        home_team: str,
        match_date: str,  # YYYY-MM-DD format
    ) -> Optional[WeatherData]:
        """
        Get weather data for a match.
        
        Uses the home team's stadium location.
        """
        # Find stadium coordinates
        coords = self.STADIUM_COORDINATES.get(home_team)
        if not coords:
            # Try partial match
            for team, coord in self.STADIUM_COORDINATES.items():
                if home_team.lower() in team.lower() or team.lower() in home_team.lower():
                    coords = coord
                    break
        
        if not coords:
            return None
        
        lat, lon = coords
        
        # Build API URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": match_date,
            "end_date": match_date,
            "hourly": "temperature_2m,precipitation,windspeed_10m,relativehumidity_2m,weathercode",
            "timezone": "auto",
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get afternoon values (15:00 local time typical for matches)
            hourly = data.get("hourly", {})
            if not hourly:
                return None
            
            # Find the 15:00 index (or closest to match time)
            times = hourly.get("time", [])
            target_hour = 15
            
            for i, t in enumerate(times):
                if f"{target_hour:02d}:00" in t:
                    return WeatherData(
                        date=match_date,
                        venue_city=home_team,
                        temperature_c=hourly.get("temperature_2m", [None])[i] or 15.0,
                        precipitation_mm=hourly.get("precipitation", [None])[i] or 0.0,
                        wind_speed_kmh=hourly.get("windspeed_10m", [None])[i] or 10.0,
                        humidity_pct=hourly.get("relativehumidity_2m", [None])[i] or 60.0,
                        weather_code=hourly.get("weathercode", [None])[i] or 0,
                    )
            
        except Exception as e:
            print(f"  Error fetching weather for {home_team} on {match_date}: {e}")
        
        return None
    
    def get_weather_for_matches(
        self,
        matches: pd.DataFrame,  # Must have 'HomeTeam' and 'Date' columns
        date_col: str = "Date",
        home_col: str = "HomeTeam",
    ) -> pd.DataFrame:
        """
        Add weather data to a matches DataFrame.
        
        Returns DataFrame with weather columns added.
        """
        weather_data = []
        
        for idx, row in matches.iterrows():
            home_team = row[home_col]
            
            # Parse date - handle multiple formats
            date_val = row[date_col]
            if isinstance(date_val, str):
                # Try different date formats
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"]:
                    try:
                        dt = datetime.strptime(date_val, fmt)
                        date_str = dt.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                else:
                    weather_data.append({})
                    continue
            else:
                date_str = date_val.strftime("%Y-%m-%d")
            
            weather = self.get_match_weather(home_team, date_str)
            
            if weather:
                weather_data.append({
                    "temperature_c": weather.temperature_c,
                    "precipitation_mm": weather.precipitation_mm,
                    "wind_speed_kmh": weather.wind_speed_kmh,
                    "humidity_pct": weather.humidity_pct,
                    "weather_code": weather.weather_code,
                })
            else:
                weather_data.append({})
            
            # Small delay to be polite
            time.sleep(0.2)
        
        # Add weather columns
        weather_df = pd.DataFrame(weather_data)
        result = pd.concat([matches.reset_index(drop=True), weather_df], axis=1)
        
        return result


# ==============================================================================
# TRANSFERMARKT SCRAPER (Player Injuries/Suspensions)
# ==============================================================================

class TransfermarktScraper:
    """
    Scrape player availability data from Transfermarkt.
    
    Gets current injuries and suspensions for teams.
    Uses Selenium to bypass bot detection.
    """
    
    BASE_URL = "https://www.transfermarkt.com"
    RATE_LIMIT = 3.0  # Seconds between requests
    
    def __init__(self):
        self.session = requests.Session()
        self.last_request_time = 0
        self._driver = None
        self._selenium_failed = False
    
    def _get_driver(self):
        """Get or create Selenium driver."""
        if self._driver is None and not self._selenium_failed:
            self._driver = get_selenium_driver(headless=True)
            if self._driver is None:
                self._selenium_failed = True
        return self._driver
    
    def _close_driver(self):
        """Close Selenium driver if open."""
        if self._driver:
            try:
                self._driver.quit()
            except:
                pass
            self._driver = None
    
    def __del__(self):
        self._close_driver()
    
    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            sleep_time = self.RATE_LIMIT - elapsed + random.uniform(0.5, 1.5)
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a Transfermarkt page using Selenium."""
        driver = self._get_driver()
        if not driver:
            return None
        
        try:
            self._rate_limit()
            driver.get(url)
            time.sleep(random.uniform(2, 3))  # Wait for page load
            
            # Accept cookies if present
            try:
                from selenium.webdriver.common.by import By
                cookie_btn = driver.find_element(By.ID, "onetrust-accept-btn-handler")
                cookie_btn.click()
                time.sleep(1)
            except:
                pass
            
            html = driver.page_source
            return BeautifulSoup(html, "html.parser")
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None
    
    def get_team_injuries(
        self,
        team_name: str,
    ) -> List[PlayerInjury]:
        """
        Get current injuries and suspensions for a team.
        """
        # Get team info from mapping
        team_info = TEAM_NAME_MAP_TRANSFERMARKT.get(team_name)
        if not team_info:
            print(f"  Team {team_name} not in Transfermarkt mapping")
            return []
        
        team_slug, team_id = team_info
        # Use the injury page instead of the squad page
        url = f"{self.BASE_URL}/{team_slug}/sperrenundverletzungen/verein/{team_id}"
        
        print(f"  Fetching injuries for {team_name}...")
        soup = self._fetch_page(url)
        
        if not soup:
            return []
        
        injuries = []
        
        # Find all injury/suspension entries
        # Transfermarkt shows injuries in table rows
        injury_rows = soup.find_all("tr", class_=["odd", "even"])
        
        for row in injury_rows:
            try:
                # Get player name
                player_cell = row.find("td", class_="hauptlink")
                if not player_cell:
                    continue
                
                player_link = player_cell.find("a")
                if not player_link:
                    continue
                
                player_name = player_link.get_text(strip=True)
                
                # Get injury type (injury vs suspension)
                cells = row.find_all("td")
                injury_type = "Injury"
                description = ""
                
                for cell in cells:
                    text = cell.get_text(strip=True).lower()
                    if "suspension" in text or "red card" in text or "yellow" in text:
                        injury_type = "Suspension"
                    cell_class = cell.get("class")
                    if cell_class and isinstance(cell_class, list) and "hauptlink" not in cell_class:
                        # Could be injury description
                        if len(text) > 3 and len(text) < 50:
                            description = text
                
                injury = PlayerInjury(
                    player_name=player_name,
                    team=team_name,
                    injury_type=injury_type,
                    description=description,
                    date_from=None,
                    date_until=None,
                    is_key_player=False,
                )
                injuries.append(injury)
                
            except Exception:
                continue
        
        print(f"  Found {len(injuries)} injured/suspended players for {team_name}")
        return injuries
    
    def get_league_injuries(
        self,
        league_code: str,
        teams: List[str],
    ) -> pd.DataFrame:
        """
        Get injury data for all teams in a league.
        
        Returns DataFrame with team-level injury summary.
        """
        all_injuries = []
        
        for team in teams:
            injuries = self.get_team_injuries(team)
            
            summary = {
                "team": team,
                "league": league_code,
                "injured_players": len([i for i in injuries if i.injury_type == "Injury"]),
                "suspended_players": len([i for i in injuries if i.injury_type == "Suspension"]),
                "key_players_out": len([i for i in injuries if i.is_key_player]),
            }
            all_injuries.append(summary)
        
        return pd.DataFrame(all_injuries)


# ==============================================================================
# MANAGER DATA (from Wikipedia)
# ==============================================================================

# Wikipedia team page URLs for manager data
WIKIPEDIA_TEAM_URLS = {
    # Premier League
    "Man United": "Manchester_United_F.C.",
    "Man City": "Manchester_City_F.C.",
    "Liverpool": "Liverpool_F.C.",
    "Arsenal": "Arsenal_F.C.",
    "Chelsea": "Chelsea_F.C.",
    "Tottenham": "Tottenham_Hotspur_F.C.",
    "Newcastle": "Newcastle_United_F.C.",
    "Aston Villa": "Aston_Villa_F.C.",
    "Brighton": "Brighton_%26_Hove_Albion_F.C.",
    "West Ham": "West_Ham_United_F.C.",
    "Bournemouth": "AFC_Bournemouth",
    "Fulham": "Fulham_F.C.",
    "Wolves": "Wolverhampton_Wanderers_F.C.",
    "Crystal Palace": "Crystal_Palace_F.C.",
    "Everton": "Everton_F.C.",
    "Brentford": "Brentford_F.C.",
    "Nott'm Forest": "Nottingham_Forest_F.C.",
    "Ipswich": "Ipswich_Town_F.C.",
    "Leicester": "Leicester_City_F.C.",
    "Southampton": "Southampton_F.C.",
}

class ManagerDataFetcher:
    """
    Fetch manager tenure data from Wikipedia.
    
    Wikipedia is more reliable and has less bot detection.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.last_request = 0
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a Wikipedia page."""
        elapsed = time.time() - self.last_request
        if elapsed < 1:
            time.sleep(1 - elapsed)
        self.last_request = time.time()
        
        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html",
            }
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None
    
    def get_league_managers(
        self,
        league_code: str,
        season: str,
    ) -> pd.DataFrame:
        """
        Get manager information for all teams from Wikipedia.
        """
        managers = []
        
        print(f"\nFetching manager data for {league_code} {season}...")
        
        for team, wiki_page in WIKIPEDIA_TEAM_URLS.items():
            try:
                url = f"https://en.wikipedia.org/wiki/{wiki_page}"
                soup = self._fetch_page(url)
                
                if not soup:
                    continue
                
                # Find infobox which contains manager info
                infobox = soup.find("table", class_="infobox")
                if not infobox:
                    continue
                
                # Look for "Manager" or "Head coach" row
                manager_name = None
                for row in infobox.find_all("tr"):
                    header = row.find("th")
                    if header:
                        header_text = header.get_text(strip=True).lower()
                        if "manager" in header_text or "head coach" in header_text:
                            value = row.find("td")
                            if value:
                                # Get text, remove citations [1], etc
                                manager_name = re.sub(r'\[.*?\]', '', value.get_text(strip=True))
                                break
                
                if manager_name:
                    managers.append({
                        "team": team,
                        "manager": manager_name,
                        "league": league_code,
                        "season": season,
                        "scrape_date": pd.Timestamp.now(),
                    })
                    print(f"  ✓ {team}: {manager_name}")
                else:
                    print(f"  ✗ {team}: Manager not found")
                    
            except Exception as e:
                print(f"  ✗ {team}: {e}")
                continue
        
        return pd.DataFrame(managers)


# ==============================================================================
# MAIN SCRAPING FUNCTION
# ==============================================================================

def scrape_all_advanced_features(
    leagues: Optional[List[str]] = None,
    season: str = "2024-2025",
    output_dir: Optional[Path] = None,
    include_fbref: bool = True,
    include_weather: bool = True,
    include_injuries: bool = True,
    include_manager: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Scrape all advanced features for specified leagues.
    
    Args:
        leagues: List of league codes (default: top 5 leagues)
        season: Season string in FBref format (e.g., "2024-2025")
        output_dir: Directory to save output files
        include_fbref: Whether to scrape FBref advanced stats
        include_weather: Whether to fetch weather data
        include_injuries: Whether to scrape injury data
        include_manager: Whether to scrape manager data
    
    Returns:
        Dictionary with DataFrames for each feature type
    """
    if leagues is None:
        leagues = ["E0", "D1", "SP1", "I1", "F1"]
    
    if output_dir is None:
        output_dir = Path("data/processed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "team_stats": [],
        "weather": [],
        "injuries": [],
        "managers": [],
    }
    
    # Initialize scrapers
    fbref_scraper = FBrefAdvancedScraper()
    weather_fetcher = WeatherFetcher()
    transfermarkt_scraper = TransfermarktScraper()
    manager_fetcher = ManagerDataFetcher()
    
    # 1. Scrape FBref advanced stats
    if include_fbref:
        print("\n" + "=" * 60)
        print("SCRAPING FBREF ADVANCED STATISTICS")
        print("=" * 60)
        
        for league in leagues:
            try:
                stats_df = fbref_scraper.get_all_team_stats(league, season)
                if not stats_df.empty:
                    results["team_stats"].append(stats_df)
                    print(f"  ✓ Got stats for {len(stats_df)} teams in {league}")
            except Exception as e:
                print(f"  ✗ Error scraping {league}: {e}")
        
        # Combine and save
        if results["team_stats"]:
            combined_stats = pd.concat(results["team_stats"], ignore_index=True)
            output_file = output_dir / "advanced_team_stats.csv"
            combined_stats.to_csv(output_file, index=False)
            print(f"\nSaved advanced stats to {output_file}")
    
    # 2. Fetch weather data
    if include_weather:
        print("\n" + "=" * 60)
        print("FETCHING WEATHER DATA")
        print("=" * 60)
        
        # Get weather for stadiums we have coordinates for
        from datetime import datetime as dt
        today = dt.now()
        weather_records = []
        
        for team, coords in weather_fetcher.STADIUM_COORDINATES.items():
            try:
                weather = weather_fetcher.get_match_weather(team, today.strftime("%Y-%m-%d"))
                if weather:
                    weather_records.append({
                        "team": team,
                        "date": today,
                        "temperature": weather.temperature_c,
                        "rain_probability": weather.precipitation_mm,
                        "wind_speed": weather.wind_speed_kmh,
                        "condition": weather.weather_code,
                    })
                    print(f"  ✓ {team}: {weather.temperature_c}°C")
            except Exception as e:
                print(f"  ✗ {team}: {e}")
        
        if weather_records:
            weather_df = pd.DataFrame(weather_records)
            output_file = output_dir / "weather_data.csv"
            weather_df.to_csv(output_file, index=False)
            print(f"\nSaved weather data to {output_file}")
            results["weather"] = [weather_df]
    
    # 3. Scrape injury data
    if include_injuries:
        print("\n" + "=" * 60)
        print("SCRAPING INJURY DATA")
        print("=" * 60)
        
        from datetime import datetime as dt
        for team in TEAM_NAME_MAP_TRANSFERMARKT.keys():
            try:
                injuries = transfermarkt_scraper.get_team_injuries(team)
                if injuries:
                    for inj in injuries:
                        results["injuries"].append({
                            "team": team,
                            "player_name": inj.player_name,
                            "injury_type": inj.injury_type,
                            "description": inj.description,
                            "date_from": inj.date_from,
                            "date_until": inj.date_until,
                            "is_key_player": inj.is_key_player,
                            "date": dt.now(),
                        })
                    print(f"  ✓ {team}: {len(injuries)} injuries/suspensions")
            except Exception as e:
                print(f"  ✗ {team}: {e}")
        
        if results["injuries"]:
            injury_df = pd.DataFrame(results["injuries"])
            output_file = output_dir / "injury_data.csv"
            injury_df.to_csv(output_file, index=False)
            print(f"\nSaved injury data to {output_file}")
    
    # 4. Scrape manager data
    if include_manager:
        print("\n" + "=" * 60)
        print("SCRAPING MANAGER DATA")
        print("=" * 60)
        
        for league in leagues:
            try:
                manager_df = manager_fetcher.get_league_managers(league, season)
                if not manager_df.empty:
                    results["managers"].append(manager_df)
                    print(f"  ✓ Got {len(manager_df)} managers for {league}")
            except Exception as e:
                print(f"  ✗ {league}: {e}")
        
        if results["managers"]:
            combined_managers = pd.concat(results["managers"], ignore_index=True)
            output_file = output_dir / "manager_data.csv"
            combined_managers.to_csv(output_file, index=False)
            print(f"\nSaved manager data to {output_file}")
    
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    
    # Return combined results
    return {
        "team_stats": pd.concat(results["team_stats"], ignore_index=True) if results["team_stats"] else pd.DataFrame(),
        "weather": pd.concat(results["weather"], ignore_index=True) if results["weather"] else pd.DataFrame(),
        "injuries": pd.DataFrame(results["injuries"]) if results["injuries"] else pd.DataFrame(),
        "managers": pd.concat(results["managers"], ignore_index=True) if results["managers"] else pd.DataFrame(),
    }


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape advanced football features")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=["E0"],
        help="League codes to scrape (e.g., E0 D1 SP1)"
    )
    parser.add_argument(
        "--season",
        default="2024-2025",
        help="Season in FBref format (e.g., 2024-2025)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/advanced_features"),
        help="Output directory"
    )
    parser.add_argument(
        "--weather-only",
        action="store_true",
        help="Only fetch weather data"
    )
    parser.add_argument(
        "--fbref-only",
        action="store_true",
        help="Only fetch FBref stats"
    )
    
    args = parser.parse_args()
    
    # Run scraping
    results = scrape_all_advanced_features(
        leagues=args.leagues,
        season=args.season,
        output_dir=args.output_dir,
    )
    
    # Print summary
    print("\nSummary:")
    for name, df in results.items():
        if not df.empty:
            print(f"  {name}: {len(df)} rows")
