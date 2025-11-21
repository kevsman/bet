"""
Interactive MCP-based scraper for all leagues on Norsk Tipping
This script will guide you through the MCP browser automation
"""

print("""
================================================================================
NORSK TIPPING - ALL LEAGUES SCRAPER (MCP Interactive Guide)
================================================================================

This script provides the exact MCP commands to scrape all supported leagues.
Follow the steps below in your VS Code Copilot chat with MCP Chrome tools active.

STEP-BY-STEP AUTOMATION:
================================================================================

STEP 1: Navigate to site and click Fotball
-------------------------------------------
1a. Navigate to Norsk Tipping (if not already there):
    > navigate to https://www.norsk-tipping.no/sport/oddsen

1b. Accept cookies if prompted:
    > click on accept cookies button

1c. Take a snapshot to find the Fotball element:
    > take a snapshot

1d. Click on "Fotball" (look for element with text "Fotball" in the left sidebar):
    > click on fotball element

1e. Wait for the page to load:
    > wait for text "Premier League"


STEP 2: Select the three Total Goals dropdowns
-----------------------------------------------
2a. Take a snapshot to find the dropdown UIDs:
    > take a snapshot

2b. Fill all three dropdowns at once using fill_form (RECOMMENDED):
    > use fill_form to fill all three dropdowns with:
      - Totalt antall mål - over/under 1.5
      - Totalt antall mål - over/under 2.5
      - Totalt antall mål - over/under 3.5

    OR individually:
    > fill dropdown 1 with "Totalt antall mål - over/under 1.5"
    > fill dropdown 2 with "Totalt antall mål - over/under 2.5"
    > fill dropdown 3 with "Totalt antall mål - over/under 3.5"

    NOTE: Using fill_form is more reliable as it fills all three at once!



STEP 3: Scrape each country's leagues
--------------------------------------

FOR EACH COUNTRY BELOW, DO THE FOLLOWING:
  1. Click the expand button for the country
  2. Wait for leagues to load
  3. Take a snapshot
  4. Save the snapshot text to a file
  5. Move to next country

COUNTRIES TO SCRAPE:
--------------------

3a. ENGLAND
   - Click on "England" expand button
   - Wait for "Premier League" text to appear
   - Take snapshot -> Save to: data/snapshots/england_snapshot.txt
   - Leagues: Premier League, Championship

3b. SPAIN (Spania)
   - Click on "Spania" expand button
   - Wait for "La Liga" text to appear
   - Take snapshot -> Save to: data/snapshots/spain_snapshot.txt
   - Leagues: La Liga, Segunda División

3c. GERMANY (Tyskland)
   - Click on "Tyskland" expand button
   - Wait for "Bundesliga" text to appear
   - Take snapshot -> Save to: data/snapshots/germany_snapshot.txt
   - Leagues: Bundesliga, 2. Bundesliga

3d. ITALY (Italia)
   - Click on "Italia" expand button
   - Wait for "Serie A" text to appear
   - Take snapshot -> Save to: data/snapshots/italy_snapshot.txt
   - Leagues: Serie A, Serie B

3e. FRANCE (Frankrike)
   - Click on "Frankrike" expand button
   - Wait for "Ligue 1" text to appear
   - Take snapshot -> Save to: data/snapshots/france_snapshot.txt
   - Leagues: Ligue 1, Ligue 2

3f. SCOTLAND (Skottland)
   - Click on "Skottland" expand button
   - Wait for "Premiership" text to appear
   - Take snapshot -> Save to: data/snapshots/scotland_snapshot.txt
   - Leagues: Premiership, Championship


STEP 4: Process all collected snapshots
----------------------------------------
After collecting all snapshots, run:
    > python scrape_all_leagues.py


EXAMPLE MCP CHAT COMMANDS:
================================================================================

# Initial navigation
"Navigate to https://www.norsk-tipping.no/sport/oddsen and accept cookies"

# Click Fotball
"Take a snapshot, then click on the element that contains the text 'Fotball'"

# Configure dropdowns
"Fill the three dropdown menus using fill_form with:
1. Totalt antall mål - over/under 1.5
2. Totalt antall mål - over/under 2.5
3. Totalt antall mål - over/under 3.5"

# Scrape England
"Click on the expand button for England, wait for Premier League to appear, then take a snapshot"

# Continue for other countries...


LEAGUE CODES FOR REFERENCE:
================================================================================
E0  = England - Premier League
E1  = England - Championship
SP1 = Spain - La Liga
SP2 = Spain - Segunda División
D1  = Germany - Bundesliga
D2  = Germany - 2. Bundesliga
I1  = Italy - Serie A
I2  = Italy - Serie B
F1  = France - Ligue 1
F2  = France - Ligue 2
SC0 = Scotland - Premiership
SC1 = Scotland - Championship

================================================================================
NOTES:
- Make sure Chrome DevTools MCP server is running
- Save each snapshot to the correct file path shown above
- After all snapshots are collected, run: python scrape_all_leagues.py
- The combined data will be saved to: data/processed/all_leagues_odds.csv
================================================================================
""")
