"""
Automated scraper for all available leagues on Norsk Tipping
Uses Chrome DevTools MCP to navigate and extract odds from all supported leagues
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from norsk_tipping_scraper import parse_snapshot_to_csv
import pandas as pd

# This script requires manual MCP browser interaction
# Follow these steps in the Chrome browser via MCP:

SCRAPING_INSTRUCTIONS = """
AUTOMATED SCRAPING PROCEDURE FOR ALL LEAGUES
============================================

1. Navigate to Norsk Tipping and accept cookies (if needed)
2. Click on "Fotball" in left sidebar (element with text "Fotball")

3. CONFIGURE THREE DROPDOWNS (CRITICAL!):
   Use fill_form to fill all three dropdowns at once:
   
   mcp_io_github_chr_fill_form(elements=[
       {"uid": "<dropdown1_uid>", "value": "Totalt antall mål - over/under 1.5"},
       {"uid": "<dropdown2_uid>", "value": "Totalt antall mål - over/under 2.5"},
       {"uid": "<dropdown3_uid>", "value": "Totalt antall mål - over/under 3.5"}
   ])
   
   This ensures all three goal line markets (O/U 1.5, 2.5, 3.5) are displayed!

4. For each country section (expand and scrape):
   - England (Premier League, Championship)
   - Spain/Spania (La Liga, Segunda División)
   - Germany/Tyskland (Bundesliga, 2. Bundesliga)
   - Italy/Italia (Serie A, Serie B)
   - France/Frankrike (Ligue 1, Ligue 2)
   - Scotland/Skottland (Premiership, Championship)

5. After clicking each country, take a snapshot and save to data/snapshots/

SUPPORTED LEAGUES IN OUR MODEL:
- England - Premier League (E0)
- England - Championship (E1)
- Spain - La Liga (SP1)
- Spain - Segunda División (SP2)
- Germany - Bundesliga (D1)
- Germany - 2. Bundesliga (D2)
- Italy - Serie A (I1)
- Italy - Serie B (I2)
- France - Ligue 1 (F1)
- France - Ligue 2 (F2)
- Scotland - Premiership (SC0)
- Scotland - Championship (SC1)
"""

# MCP Command Sequence for automation:
MCP_COMMANDS = """
# Step 1: Reload page to ensure clean state
reload_page()
wait_for(text="Fotball")

# Step 2: Configure three dropdowns (CRITICAL - ensures all O/U markets are displayed)
take_snapshot()  # Get dropdown UIDs

# Use fill_form to fill all three at once (more reliable)
fill_form(elements=[
    {"uid": "<dropdown1_uid>", "value": "Totalt antall mål - over/under 1.5"},
    {"uid": "<dropdown2_uid>", "value": "Totalt antall mål - over/under 2.5"},
    {"uid": "<dropdown3_uid>", "value": "Totalt antall mål - over/under 3.5"}
])

# Step 3: Expand each country and take snapshots
# England
click(uid="<england_expand_button_uid>")
wait_for(text="Premier League")
take_snapshot() -> save to data/snapshots/england_snapshot.txt

# Spain
click(uid="<spain_expand_button_uid>")
wait_for(text="La Liga")
take_snapshot() -> save to data/snapshots/spain_snapshot.txt

# Germany
click(uid="<germany_expand_button_uid>")
wait_for(text="Bundesliga")
take_snapshot() -> save to data/snapshots/germany_snapshot.txt

# Italy
click(uid="<italy_expand_button_uid>")
wait_for(text="Serie A")
take_snapshot() -> save to data/snapshots/italy_snapshot.txt

# France
click(uid="<france_expand_button_uid>")
wait_for(text="Ligue 1")
take_snapshot() -> save to data/snapshots/france_snapshot.txt

# Scotland
click(uid="<scotland_expand_button_uid>")
wait_for(text="Premiership")
take_snapshot() -> save to data/snapshots/scotland_snapshot.txt

# Step 4: Process all snapshots
run: python scrape_all_leagues.py
"""


def scrape_all_leagues_from_snapshots(snapshot_files):
    """
    Process multiple snapshot files and combine into one dataset
    
    Args:
        snapshot_files: List of snapshot text file paths
    
    Returns:
        Combined DataFrame with all matches
    """
    all_matches = []
    
    for snapshot_file in snapshot_files:
        if not os.path.exists(snapshot_file):
            print(f"Warning: {snapshot_file} not found, skipping...")
            continue
            
        print(f"Processing {snapshot_file}...")
        with open(snapshot_file, 'r', encoding='utf-8') as f:
            snapshot_text = f.read()
        
        df = parse_snapshot_to_csv(snapshot_text)
        all_matches.append(df)
        print(f"  Found {len(df)} matches")
    
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        print(f"\nTotal matches extracted: {len(combined)}")
        return combined
    else:
        print("No matches found!")
        return pd.DataFrame()


# Countries and their leagues to scrape
COUNTRIES_TO_SCRAPE = {
    'England': ['Premier League', 'Championship'],
    'Spania': ['La Liga', 'Segunda División'],
    'Tyskland': ['Bundesliga', '2. Bundesliga'],
    'Italia': ['Serie A', 'Serie B'],
    'Frankrike': ['Ligue 1', 'Ligue 2'],
    'Skottland': ['Premiership', 'Championship']
}


if __name__ == "__main__":
    print(SCRAPING_INSTRUCTIONS)
    print("\n" + "="*60)
    print("MCP AUTOMATION COMMANDS:")
    print("="*60)
    print(MCP_COMMANDS)
    print("\n" + "="*60)
    
    # Check if snapshot files exist
    snapshot_dir = "data/snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    
    snapshot_files = [
        os.path.join(snapshot_dir, "england_snapshot.txt"),
        os.path.join(snapshot_dir, "spain_snapshot.txt"),
        os.path.join(snapshot_dir, "germany_snapshot.txt"),
        os.path.join(snapshot_dir, "italy_snapshot.txt"),
        os.path.join(snapshot_dir, "france_snapshot.txt"),
        os.path.join(snapshot_dir, "scotland_snapshot.txt"),
    ]
    
    existing_snapshots = [f for f in snapshot_files if os.path.exists(f)]
    
    if existing_snapshots:
        print(f"\nFound {len(existing_snapshots)} snapshot files:")
        for f in existing_snapshots:
            print(f"  - {f}")
        
        print("\nProcessing snapshots...")
        combined_df = scrape_all_leagues_from_snapshots(existing_snapshots)
        
        if not combined_df.empty:
            output_file = "data/processed/all_leagues_odds.csv"
            combined_df.to_csv(output_file, index=False)
            print(f"\nSaved combined odds to: {output_file}")
            print(f"\nSample of extracted data:")
            print(combined_df.head(10))
    else:
        print("\nNo snapshot files found yet.")
        print("Please follow the MCP automation commands above to collect data.")
