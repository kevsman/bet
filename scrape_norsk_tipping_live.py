"""
Live scraper for Norsk Tipping Oddsen using Selenium.
Scrapes over/under odds for all configured leagues.
"""
import os
import time
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Import filter function for women's leagues
try:
    from src.config import filter_womens_matches
except ImportError:
    # Fallback if run directly without src in path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.config import filter_womens_matches

# League configurations - maps country tabs to their leagues
LEAGUES_CONFIG = {
    "England": {
        "url_path": "6", # Try numeric IDs
        "leagues": ["Premier League", "Championship", "League One", "League Two"]
    },
    "Spania": {
        "url_path": "7",
        "leagues": ["La Liga", "Segunda División"]
    },
    "Tyskland": {
        "url_path": "8",
        "leagues": ["Bundesliga", "2. Bundesliga"]
    },
    "Italia": {
        "url_path": "9",
        "leagues": ["Serie A", "Serie B"]
    },
    "Frankrike": {
        "url_path": "10",
        "leagues": ["Ligue 1", "Ligue 2"]
    },
    "Skottland": {
        "url_path": "11",
        "leagues": ["Premiership", "Championship"]
    },
}

BASE_URL = "https://www.norsk-tipping.no/sport/oddsen"
FOOTBALL_URL = "https://www.norsk-tipping.no/sport/oddsen"


def setup_driver(headless: bool = False):
    """Setup Chrome driver with appropriate options."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")  # Run headless
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=no")  # Norwegian language
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def accept_cookies(driver):
    """Accept cookie consent if present."""
    try:
        # Try multiple approaches for the cookie consent button
        print("  Looking for cookie consent button...")
        
        # First, try to find all buttons and look for one with the right text
        buttons = driver.find_elements(By.TAG_NAME, "button")
        print(f"  Found {len(buttons)} buttons on page")
        
        for btn in buttons:
            try:
                btn_text = btn.text.strip()
                if btn_text and len(btn_text) < 50:
                    print(f"    Button: '{btn_text}'")
                if "Godta alle" in btn_text or "Aksepter alle" in btn_text:
                    print(f"  Clicking button: '{btn_text}'")
                    btn.click()
                    print("  Clicked cookie consent button!")
                    time.sleep(3)
                    return True
            except Exception as e:
                continue
        
        print("  No matching cookie button found")
        return False
    except Exception as e:
        print(f"  Error handling cookies: {e}")
        return False


def parse_norwegian_date(date_str: str) -> tuple[str, str]:
    """
    Parse Norwegian date format to YYYY-MM-DD and time.
    Example: 'Lør. 22/11 13:30' -> ('2025-11-22', '13:30')
    """
    # Extract day/month
    date_match = re.search(r"(\d{1,2})/(\d{1,2})", date_str)
    time_match = re.search(r"(\d{1,2}:\d{2})", date_str)
    
    if not date_match:
        return "", ""
    
    day, month = date_match.groups()
    year = datetime.now().year
    
    # If the month is less than current month, assume next year
    if int(month) < datetime.now().month:
        year += 1
    
    date = f"{year}-{int(month):02d}-{int(day):02d}"
    match_time = time_match.group(1) if time_match else ""
    
    return date, match_time


def extract_odds_value(text: str) -> float | None:
    """Extract numeric odds from text like '1.75' or 'Over 2.5 1.75'."""
    # Find decimal number pattern
    match = re.search(r"(\d+[.,]\d+)", text.replace(",", "."))
    if match:
        return float(match.group(1))
    return None


def scrape_league_page(driver, country: str, wait_time: int = 10) -> list[dict]:
    """
    Scrape all matches from the current page.
    Returns list of match dictionaries.
    """
    matches = []
    
    try:
        # Wait for matches to load
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='event-card'], .event-card, .match-row, article"))
        )
        time.sleep(2)  # Extra wait for dynamic content
        
        # Get page source and parse
        page_text = driver.page_source
        
        # Try to find match containers - Norsk Tipping uses various structures
        match_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid='event-card'], .event-card, article.match, .match-container")
        
        if not match_elements:
            # Fallback: try to parse from page text
            print(f"  No match elements found for {country}, trying text parsing...")
            return parse_page_text(driver.find_element(By.TAG_NAME, "body").text, country)
        
        for elem in match_elements:
            try:
                match_data = extract_match_from_element(elem, country)
                if match_data and match_data.get("home_team"):
                    matches.append(match_data)
            except Exception as e:
                continue
                
    except TimeoutException:
        print(f"  Timeout waiting for {country} page to load")
    except Exception as e:
        print(f"  Error scraping {country}: {e}")
    
    return matches


def extract_match_from_element(elem, country: str) -> dict | None:
    """Extract match data from a match element."""
    try:
        text = elem.text
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        if len(lines) < 4:
            return None
        
        # Parse structure - typically: date/time, home, away, league, odds...
        match_data = {
            "date": "",
            "time": "",
            "home_team": "",
            "away_team": "",
            "league": "",
            "country": country,
            "over_1_5": None,
            "under_1_5": None,
            "over_2_5": None,
            "under_2_5": None,
            "over_3_5": None,
            "under_3_5": None,
        }
        
        # Find date line (contains / and :)
        for i, line in enumerate(lines):
            if "/" in line and ":" in line:
                match_data["date"], match_data["time"] = parse_norwegian_date(line)
                # Next lines are usually teams
                if i + 1 < len(lines):
                    match_data["home_team"] = lines[i + 1]
                if i + 2 < len(lines):
                    match_data["away_team"] = lines[i + 2]
                if i + 3 < len(lines):
                    match_data["league"] = lines[i + 3]
                break
        
        # Find odds - look for Over/Under patterns
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if "over" in line_lower or "under" in line_lower:
                odds_val = extract_odds_value(line)
                if odds_val:
                    if "1.5" in line or "1,5" in line:
                        if "over" in line_lower:
                            match_data["over_1_5"] = odds_val
                        else:
                            match_data["under_1_5"] = odds_val
                    elif "2.5" in line or "2,5" in line:
                        if "over" in line_lower:
                            match_data["over_2_5"] = odds_val
                        else:
                            match_data["under_2_5"] = odds_val
                    elif "3.5" in line or "3,5" in line:
                        if "over" in line_lower:
                            match_data["over_3_5"] = odds_val
                        else:
                            match_data["under_3_5"] = odds_val
        
        return match_data
        
    except Exception as e:
        return None


def parse_page_text(text: str, country: str) -> list[dict]:
    """Fallback: parse matches from raw page text."""
    matches = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for date pattern
        if re.search(r"\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}", line):
            match_data = {
                "date": "",
                "time": "",
                "home_team": "",
                "away_team": "",
                "league": "",
                "country": country,
                "over_1_5": None,
                "under_1_5": None,
                "over_2_5": None,
                "under_2_5": None,
                "over_3_5": None,
                "under_3_5": None,
            }
            
            match_data["date"], match_data["time"] = parse_norwegian_date(line)
            
            # Get next few lines for teams and league
            if i + 1 < len(lines):
                match_data["home_team"] = lines[i + 1]
            if i + 2 < len(lines):
                match_data["away_team"] = lines[i + 2]
            if i + 3 < len(lines) and "League" in lines[i + 3] or "Liga" in lines[i + 3] or "Serie" in lines[i + 3]:
                match_data["league"] = lines[i + 3]
            
            # Look for odds in next ~20 lines
            for j in range(i, min(i + 20, len(lines))):
                odds_line = lines[j].lower()
                if "over" in odds_line or "under" in odds_line:
                    odds_val = extract_odds_value(lines[j])
                    if odds_val:
                        if "1.5" in lines[j] or "1,5" in lines[j]:
                            if "over" in odds_line:
                                match_data["over_1_5"] = odds_val
                            else:
                                match_data["under_1_5"] = odds_val
                        elif "2.5" in lines[j] or "2,5" in lines[j]:
                            if "over" in odds_line:
                                match_data["over_2_5"] = odds_val
                            else:
                                match_data["under_2_5"] = odds_val
                        elif "3.5" in lines[j] or "3,5" in lines[j]:
                            if "over" in odds_line:
                                match_data["over_3_5"] = odds_val
                            else:
                                match_data["under_3_5"] = odds_val
            
            if match_data["home_team"]:
                matches.append(match_data)
            
            i += 4
        else:
            i += 1
    
    return matches


def select_goal_markets(driver):
    """Select the over/under goal markets (1.5, 2.5, 3.5)."""
    try:
        # Look for market selector dropdowns
        market_buttons = driver.find_elements(By.CSS_SELECTOR, "[data-testid='market-selector'], .market-dropdown, button[aria-haspopup='listbox']")
        
        for btn in market_buttons:
            btn.click()
            time.sleep(0.5)
            
            # Select goal markets
            for goal_line in ["1.5", "2.5", "3.5"]:
                try:
                    option = driver.find_element(By.XPATH, f"//li[contains(text(), '{goal_line}') or contains(text(), 'mål')]")
                    option.click()
                    time.sleep(0.3)
                except:
                    pass
                    
    except Exception as e:
        print(f"  Could not select markets: {e}")


def parse_matches_with_odds(text: str) -> list[dict]:
    """
    Parse matches from page text, extracting HUB odds and over/under odds.
    The page structure shows:
    - Date line: "Tir. 25/11 18:45"
    - Home team
    - Away team  
    - League info
    - HUB odds: H X.XX U X.XX B X.XX
    - Additional markets
    """
    matches = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for date pattern like "Tir. 25/11 18:45" or "25/11 18:45"
        date_match = re.search(r"(\d{1,2})/(\d{1,2})\s+(\d{1,2}:\d{2})", line)
        
        if date_match:
            day, month, match_time = date_match.groups()
            year = datetime.now().year
            if int(month) < datetime.now().month:
                year += 1
            
            match_data = {
                "date": f"{year}-{int(month):02d}-{int(day):02d}",
                "time": match_time,
                "home_team": "",
                "away_team": "",
                "league": "",
                "country": "",
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "over_1_5": None,
                "under_1_5": None,
                "over_2_5": None,
                "under_2_5": None,
                "over_3_5": None,
                "under_3_5": None,
            }
            
            # Get next lines for teams
            if i + 1 < len(lines):
                match_data["home_team"] = lines[i + 1]
            if i + 2 < len(lines):
                match_data["away_team"] = lines[i + 2]
            
            # Look ahead for league and odds
            for j in range(i + 3, min(i + 25, len(lines))):
                current_line = lines[j]
                
                # Check for league identifiers
                if any(x in current_line for x in ["League", "Liga", "Serie", "Bundesliga", "Ligue", "Premiership", "Championship"]):
                    match_data["league"] = current_line
                
                # Look for HUB odds pattern: H followed by number, U followed by number, B followed by number
                if current_line == "H" and j + 1 < len(lines):
                    try:
                        match_data["home_odds"] = float(lines[j + 1].replace(",", "."))
                    except:
                        pass
                elif current_line == "U" and j + 1 < len(lines):
                    try:
                        match_data["draw_odds"] = float(lines[j + 1].replace(",", "."))
                    except:
                        pass
                elif current_line == "B" and j + 1 < len(lines):
                    try:
                        match_data["away_odds"] = float(lines[j + 1].replace(",", "."))
                    except:
                        pass
                
                # Look for over/under odds
                if "Over" in current_line or "over" in current_line:
                    if j + 1 < len(lines):
                        try:
                            odds_val = float(lines[j + 1].replace(",", "."))
                            if "1.5" in current_line or "1,5" in current_line:
                                match_data["over_1_5"] = odds_val
                            elif "2.5" in current_line or "2,5" in current_line:
                                match_data["over_2_5"] = odds_val
                            elif "3.5" in current_line or "3,5" in current_line:
                                match_data["over_3_5"] = odds_val
                        except:
                            pass
                
                if "Under" in current_line or "under" in current_line:
                    if j + 1 < len(lines):
                        try:
                            odds_val = float(lines[j + 1].replace(",", "."))
                            if "1.5" in current_line or "1,5" in current_line:
                                match_data["under_1_5"] = odds_val
                            elif "2.5" in current_line or "2,5" in current_line:
                                match_data["under_2_5"] = odds_val
                            elif "3.5" in current_line or "3,5" in current_line:
                                match_data["under_3_5"] = odds_val
                        except:
                            pass
                
                # Stop if we hit next match date
                if j > i + 3 and re.search(r"\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}", current_line):
                    break
            
            if match_data["home_team"] and match_data["away_team"]:
                matches.append(match_data)
            
            i += 3  # Skip past this match
        else:
            i += 1
    
    return matches


def configure_market_selects(driver, verbose=False):
    """Configure the 3 select boxes for over/under 1.5, 2.5, 3.5."""
    from selenium.webdriver.support.ui import Select
    
    selects = driver.find_elements(By.TAG_NAME, "select")
    market_values = ["1.5", "2.5", "3.5"]
    
    if verbose:
        print(f"  Found {len(selects)} select elements")
    
    for idx, select_elem in enumerate(selects[:3]):
        if idx >= len(market_values):
            break
        target_val = market_values[idx]
        try:
            sel = Select(select_elem)
            options = sel.options
            
            # Find the over/under option with the target value
            for opt in options:
                opt_text = opt.text.lower()
                if target_val in opt_text and "over/under" in opt_text:
                    sel.select_by_visible_text(opt.text)
                    if verbose:
                        print(f"    Select {idx+1}: Selected '{opt.text}'")
                    break
        except Exception as e:
            if verbose:
                print(f"    Select {idx+1}: Error - {e}")
    
    # Wait for the page to update after selecting markets
    time.sleep(1)


def scrape_all_leagues(headless: bool = False) -> pd.DataFrame:
    """Main function to scrape all configured leagues."""
    print("=" * 60)
    print("NORSK TIPPING ODDS SCRAPER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Headless' if headless else 'Visible browser'}")
    print("=" * 60)
    
    driver = setup_driver(headless=headless)
    all_matches = []
    
    try:
        # Start with football page
        print(f"\nNavigating to {FOOTBALL_URL}...")
        driver.get(FOOTBALL_URL)
        time.sleep(3)  # Wait for initial load
        
        # Save initial page for debugging
        print(f"Page title: {driver.title}")
        print(f"Current URL: {driver.current_url}")
        
        # Accept cookies
        accept_cookies(driver)
        time.sleep(2)
        
        # Check for iframes
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        print(f"\nFound {len(iframes)} iframes")
        
        # Check each iframe for content
        for i, iframe in enumerate(iframes):
            try:
                iframe_src = iframe.get_attribute("src") or "no src"
                iframe_id = iframe.get_attribute("id") or "no id"
                print(f"  iframe {i}: id='{iframe_id}', src='{iframe_src[:80]}...'")
                
                # Switch to iframe and check content
                driver.switch_to.frame(iframe)
                iframe_text = driver.find_element(By.TAG_NAME, "body").text[:200]
                iframe_source = driver.page_source
                print(f"    Text preview: {iframe_text[:100]}...")
                print(f"    Source length: {len(iframe_source)} chars")
                
                if "Fotball" in iframe_source or "fotball" in iframe_source.lower():
                    print(f"    *** FOUND 'Fotball' in iframe {i}! ***")
                    
                    # Save this iframe's HTML
                    iframe_path = Path("data/snapshots") / f"iframe_{i}.html"
                    with open(iframe_path, "w", encoding="utf-8") as f:
                        f.write(iframe_source)
                    print(f"    Saved to {iframe_path}")
                
                # Switch back to main content
                driver.switch_to.default_content()
            except Exception as e:
                print(f"    Error with iframe {i}: {e}")
                driver.switch_to.default_content()
        
        # Get page source (full HTML)
        page_source = driver.page_source
        print(f"\nPage source length: {len(page_source)} chars")
        
        # Switch to the sportsbook iframe where all the content is
        print("\nSwitching to sportsbook iframe...")
        sportsbook_iframe = driver.find_element(By.ID, "sportsbookid")
        driver.switch_to.frame(sportsbook_iframe)
        print("  Switched to iframe!")
        
        # Now look for Fotball menu item inside the iframe
        print("\nLooking for Fotball menu item...")
        fotball_clicked = False
        
        # Try to find and click Fotball link/button
        try:
            # Look for links or buttons containing "Fotball"
            elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Fotball')]")
            print(f"  Found {len(elements)} elements with 'Fotball' text")
            
            for elem in elements[:10]:  # Check first 10
                try:
                    elem_text = elem.text.strip()
                    elem_tag = elem.tag_name
                    if elem_text and len(elem_text) < 50:
                        print(f"    {elem_tag}: '{elem_text}'")
                    
                    if elem_text == "Fotball":
                        print(f"  Clicking: '{elem_text}'")
                        elem.click()
                        fotball_clicked = True
                        time.sleep(3)  # Wait for football page to load
                        break
                except Exception as e:
                    continue
                        
        except Exception as e:
            print(f"  Error finding Fotball: {e}")
        
        if fotball_clicked:
            print("  Successfully clicked Fotball menu!")
        else:
            print("  Could not find Fotball menu item")
        
        # Configure the 3 select boxes for over/under 1.5, 2.5, and 3.5
        print("\nConfiguring market select boxes for over/under odds...")
        try:
            # Find all actual select elements
            selects = driver.find_elements(By.TAG_NAME, "select")
            print(f"  Found {len(selects)} select elements")
            
            # Market options to select for each column
            market_values = [
                ("Totalt antall mål - over/under 1.5", "1.5"),
                ("Totalt antall mål - over/under 2.5", "2.5"), 
                ("Totalt antall mål - over/under 3.5", "3.5")
            ]
            
            # Configure each select box
            for idx, select_elem in enumerate(selects[:3]):
                if idx >= len(market_values):
                    break
                    
                target_text, target_val = market_values[idx]
                
                try:
                    # Get all options in this select
                    options = select_elem.find_elements(By.TAG_NAME, "option")
                    print(f"  Select {idx + 1}: {len(options)} options")
                    
                    # Find and click the right option
                    for opt in options:
                        opt_text = opt.text.strip()
                        if target_val in opt_text and "over/under" in opt_text.lower():
                            print(f"    Selecting: '{opt_text}'")
                            opt.click()
                            time.sleep(0.5)
                            break
                except Exception as e:
                    print(f"  Could not configure select {idx + 1}: {e}")
                    
        except Exception as e:
            print(f"  Error configuring market selects: {e}")
        
        time.sleep(2)  # Wait for page to update
        
        # Debug: Print the page structure to understand the menu
        print("\n--- DEBUG: Page structure ---")
        body_text = driver.find_element(By.TAG_NAME, "body").text
        lines = body_text.split("\n")
        for i, line in enumerate(lines[:80]):
            if line.strip():
                print(f"  {i}: {line.strip()[:60]}")
        print("--- END DEBUG ---\n")
        
        # Save full page for analysis
        snapshot_path = Path("data/snapshots") / "after_fotball.txt"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(body_text)
        print(f"Saved page to {snapshot_path}")
        
        # Define the leagues we want to scrape - use data-id selectors
        # Countries have IDs like: navigation_verticalsportlist_sport_selection_66748.1 (England)
        # The menu text is lowercase: "england", "spania", "tyskland", "italia", "frankrike"
        target_countries = [
            # (data_id, display_name)
            ("navigation_verticalsportlist_sport_selection_66748.1", "England"),
            ("navigation_verticalsportlist_sport_selection_66743.1", "Spania"), 
            ("navigation_verticalsportlist_sport_selection_66768.1", "Tyskland"),
            ("navigation_verticalsportlist_sport_selection_66751.1", "Italia"),
            ("navigation_verticalsportlist_sport_selection_66774.1", "Frankrike"),
            ("navigation_verticalsportlist_sport_selection_66739.1", "Skottland"),
            ("navigation_verticalsportlist_sport_selection_66760.1", "Norge"),
            # Lower divisions and women's leagues
            ("navigation_verticalsportlist_sport_selection_66729.1", "England Lower"),  # Championship, League One, etc.
            # Additional countries
            ("navigation_verticalsportlist_sport_selection_66721.1", "Nederland"),
            ("navigation_verticalsportlist_sport_selection_66731.1", "Portugal"),
            ("navigation_verticalsportlist_sport_selection_76221.1", "Tyrkia"),
            ("navigation_verticalsportlist_sport_selection_66781.1", "Belgia"),
        ]
        
        print("\n" + "=" * 60)
        print("SCRAPING LEAGUES FROM MENU")
        print("=" * 60)
        
        # STEP 1: Expand all UEFA tournaments by clicking their accordions
        # Champions League is already expanded, we need to expand Europa League and Conference League
        print("\n--- EXPANDING UEFA TOURNAMENTS ---")
        
        # Click Conference League accordion to expand
        try:
            conf_elem = driver.find_element(By.CSS_SELECTOR, "[data-id*='EuropaConferenceLeague']")
            arrow = conf_elem.find_element(By.CSS_SELECTOR, "[role='button']")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", arrow)
            time.sleep(0.3)
            driver.execute_script("arguments[0].click();", arrow)
            print("  Expanded UEFA Conference League")
            time.sleep(2)
        except Exception as e:
            print(f"  Could not expand Conference League: {e}")
        
        # Click Europa League accordion to expand
        try:
            el_elem = driver.find_element(By.CSS_SELECTOR, "[data-id*='UEFAEuropaLeague']")
            arrow = el_elem.find_element(By.CSS_SELECTOR, "[role='button']")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", arrow)
            time.sleep(0.3)
            driver.execute_script("arguments[0].click();", arrow)
            print("  Expanded UEFA Europa League")
            time.sleep(2)
        except Exception as e:
            print(f"  Could not expand Europa League: {e}")
        
        # Now configure select boxes and scrape all UEFA matches at once
        print("\n--- SCRAPING ALL UEFA TOURNAMENTS ---")
        configure_market_selects(driver)
        time.sleep(1)
        body_text = driver.find_element(By.TAG_NAME, "body").text
        uefa_matches = parse_matches_with_odds(body_text)
        for m in uefa_matches:
            m["country"] = "Europa"
        print(f"  Found {len(uefa_matches)} UEFA matches total")
        all_matches.extend(uefa_matches)
        
        # STEP 2: Now scrape countries one by one
        # For each country: click to expand, expand all sub-league accordions, scrape, move to next
        print("\n--- SCRAPING COUNTRIES ONE BY ONE ---")
        
        for data_id, country_name in target_countries:
            try:
                print(f"\n--- {country_name.upper()} ---")
                
                # Navigate back to Fotball first to reset
                try:
                    driver.switch_to.default_content()
                    time.sleep(0.3)
                    iframe = driver.find_element(By.ID, "sportsbookid")
                    driver.switch_to.frame(iframe)
                    time.sleep(0.3)
                    fotball_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Fotball')]")
                    for elem in fotball_elements:
                        if elem.text.strip() == "Fotball":
                            driver.execute_script("arguments[0].click();", elem)
                            time.sleep(1.5)
                            break
                except:
                    pass
                
                # Find and click on the country to expand it
                country_elem = driver.find_element(By.CSS_SELECTOR, f"[data-id='{data_id}']")
                arrow = country_elem.find_element(By.CSS_SELECTOR, "[role='button']")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", arrow)
                time.sleep(0.3)
                driver.execute_script("arguments[0].click();", arrow)
                print(f"  Expanded {country_name}")
                time.sleep(2)
                
                # Now find and expand ALL sub-league accordions for this country
                # After clicking the country, sub-leagues appear in a SIBLING row element
                # We need to find collapsed sub-league accordions that are now visible
                try:
                    # Get the parent container of the country element
                    parent_container = country_elem.find_element(By.XPATH, "./ancestor::div[contains(@class, 'SecondLevelMenu')]/..")
                    
                    # Find all collapsed accordion buttons that are visible within this container
                    # These should be the sub-leagues for this country
                    sub_accordions = parent_container.find_elements(By.CSS_SELECTOR, "[role='button'][aria-expanded='false']")
                    expanded_count = 0
                    for acc in sub_accordions:
                        try:
                            # Only expand if visible (meaning it belongs to the currently expanded country)
                            if acc.is_displayed():
                                # Get the parent element to check if it's a tournament/league element
                                parent = acc.find_element(By.XPATH, "./..")
                                parent_class = parent.get_attribute("class") or ""
                                # Only expand if it looks like a tournament/league element
                                if "SportTournament" in parent_class or "SportElement" in parent_class:
                                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", acc)
                                    time.sleep(0.2)
                                    driver.execute_script("arguments[0].click();", acc)
                                    expanded_count += 1
                                    time.sleep(0.5)
                        except:
                            pass
                    if expanded_count > 0:
                        print(f"  Expanded {expanded_count} sub-leagues within {country_name}")
                        time.sleep(1)
                    else:
                        print(f"  No sub-leagues to expand in {country_name}")
                except Exception as e:
                    print(f"  Error finding sub-leagues: {e}")
                
                # Configure select boxes for over/under odds
                configure_market_selects(driver)
                time.sleep(1)
                
                # Scrape all matches visible for this country
                body_text = driver.find_element(By.TAG_NAME, "body").text
                matches = parse_matches_with_odds(body_text)
                for m in matches:
                    m["country"] = country_name
                print(f"  Found {len(matches)} matches")
                all_matches.extend(matches)
                
            except Exception as e:
                print(f"  Error with {country_name}: {e}")
                continue
        
        # Remove duplicates based on date, time, home_team, away_team
        print(f"\n{'=' * 60}")
        print(f"Total matches before deduplication: {len(all_matches)}")
        seen = set()
        unique_matches = []
        for m in all_matches:
            key = (m["date"], m["time"], m["home_team"], m["away_team"])
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)
        all_matches = unique_matches
        print(f"Unique matches after deduplication: {len(all_matches)}")
        
    finally:
        driver.quit()
    
    # Create DataFrame
    if all_matches:
        df = pd.DataFrame(all_matches)
        
        # Filter out women's league matches
        original_count = len(df)
        df = filter_womens_matches(df, verbose=True)
        print(f"Matches after women's league filter: {len(df)} (removed {original_count - len(df)})")
        
        # Save to CSV
        output_path = Path("data/upcoming/norsk_tipping_odds.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print(f"SCRAPING COMPLETE")
        print(f"Total matches: {len(df)}")
        print(f"Output: {output_path}")
        print("=" * 60)
        
        # Show sample
        print("\nSample data:")
        print(df[["date", "time", "home_team", "away_team", "over_2_5", "under_2_5"]].head(10).to_string())
        
        return df
    else:
        print("\nNo matches found!")
        return pd.DataFrame()


if __name__ == "__main__":
    import sys
    headless = "--headless" in sys.argv
    df = scrape_all_leagues(headless=headless)
