"""
Fully automated league scraper using MCP Chrome tools
This script clicks through each country and retrieves odds for all leagues
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from norsk_tipping_scraper import parse_snapshot_to_csv
import pandas as pd


def save_snapshot_to_file(snapshot_text, filename):
    """Save snapshot text to file"""
    os.makedirs("data/snapshots", exist_ok=True)
    filepath = os.path.join("data/snapshots", filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(snapshot_text)
    print(f"Saved snapshot to {filepath}")
    return filepath


def process_all_snapshots():
    """Process all collected snapshots and combine into one dataset"""
    snapshot_dir = "data/snapshots"
    
    snapshot_files = [
        os.path.join(snapshot_dir, "england_snapshot.txt"),
        os.path.join(snapshot_dir, "spain_snapshot.txt"),
        os.path.join(snapshot_dir, "germany_snapshot.txt"),
        os.path.join(snapshot_dir, "italy_snapshot.txt"),
        os.path.join(snapshot_dir, "france_snapshot.txt"),
        os.path.join(snapshot_dir, "scotland_snapshot.txt"),
    ]
    
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
        
        # Save combined data
        output_file = "data/processed/all_leagues_odds.csv"
        os.makedirs("data/processed", exist_ok=True)
        combined.to_csv(output_file, index=False)
        print(f"Saved combined odds to: {output_file}")
        
        # Display sample
        print(f"\nSample of extracted data:")
        print(combined.head(10))
        
        return combined
    else:
        print("No matches found!")
        return pd.DataFrame()


# Country configurations for automated scraping
COUNTRIES = [
    {
        'name': 'England',
        'wait_text': 'Premier League',
        'snapshot_file': 'england_snapshot.txt',
        'leagues': ['Premier League', 'Championship']
    },
    {
        'name': 'Spania',
        'wait_text': 'La Liga',
        'snapshot_file': 'spain_snapshot.txt',
        'leagues': ['La Liga', 'Segunda División']
    },
    {
        'name': 'Tyskland',
        'wait_text': 'Bundesliga',
        'snapshot_file': 'germany_snapshot.txt',
        'leagues': ['Bundesliga', '2. Bundesliga']
    },
    {
        'name': 'Italia',
        'wait_text': 'Serie A',
        'snapshot_file': 'italy_snapshot.txt',
        'leagues': ['Serie A', 'Serie B']
    },
    {
        'name': 'Frankrike',
        'wait_text': 'Ligue 1',
        'snapshot_file': 'france_snapshot.txt',
        'leagues': ['Ligue 1', 'Ligue 2']
    },
    {
        'name': 'Skottland',
        'wait_text': 'Premiership',
        'snapshot_file': 'scotland_snapshot.txt',
        'leagues': ['Premiership', 'Championship']
    }
]


AUTOMATION_INSTRUCTIONS = """
================================================================================
AUTOMATED NORSK TIPPING SCRAPER - ALL LEAGUES
================================================================================

This script automates the complete scraping workflow:
1. Configure three dropdowns for O/U 1.5, 2.5, 3.5
2. Click through each country (England, Spain, Germany, Italy, France, Scotland)
3. Collect odds for all supported leagues
4. Process and combine all data into one CSV file

EXECUTION:
----------
Tell Copilot in chat:

"Run the automated Norsk Tipping scraper:
1. Reload the page and wait for Fotball
2. Take snapshot and configure all three dropdowns using fill_form with O/U 1.5, 2.5, 3.5
3. For each country (England, Spania, Tyskland, Italia, Frankrike, Skottland):
   - Take snapshot to get expand button UID
   - Click expand button
   - Wait for league text to appear
   - Take snapshot and save to data/snapshots/<country>_snapshot.txt
4. Run: python auto_scrape_leagues.py to process all snapshots"

SIMPLIFIED COMMAND:
-------------------
"Scrape all leagues from Norsk Tipping with dropdowns configured, click through all countries, and save snapshots"

COUNTRIES TO SCRAPE:
--------------------
"""

for country in COUNTRIES:
    AUTOMATION_INSTRUCTIONS += f"\n{country['name']}:"
    AUTOMATION_INSTRUCTIONS += f"\n  - Wait for: '{country['wait_text']}'"
    AUTOMATION_INSTRUCTIONS += f"\n  - Save to: data/snapshots/{country['snapshot_file']}"
    AUTOMATION_INSTRUCTIONS += f"\n  - Leagues: {', '.join(country['leagues'])}\n"

AUTOMATION_INSTRUCTIONS += """
================================================================================
MCP AUTOMATION WORKFLOW:
================================================================================

# STEP 1: Initial setup
mcp_io_github_chr_navigate_page(type="reload")
mcp_io_github_chr_wait_for(text="Fotball", timeout=5000)

# STEP 2: Configure dropdowns (get UIDs from snapshot first)
mcp_io_github_chr_take_snapshot()

# Then fill all three dropdowns at once
mcp_io_github_chr_fill_form(elements=[
    {"uid": "<first_dropdown_uid>", "value": "Totalt antall mål - over/under 1.5"},
    {"uid": "<second_dropdown_uid>", "value": "Totalt antall mål - over/under 2.5"},
    {"uid": "<third_dropdown_uid>", "value": "Totalt antall mål - over/under 3.5"}
])

# STEP 3: For each country, execute this sequence:

# --- ENGLAND ---
mcp_io_github_chr_take_snapshot()  # Get button UID
mcp_io_github_chr_click(uid="<england_button_uid>")
mcp_io_github_chr_wait_for(text="Premier League", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save this to data/snapshots/england_snapshot.txt

# --- SPANIA ---
mcp_io_github_chr_take_snapshot()  # Get button UID
mcp_io_github_chr_click(uid="<spania_button_uid>")
mcp_io_github_chr_wait_for(text="La Liga", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/spain_snapshot.txt

# --- TYSKLAND ---
mcp_io_github_chr_take_snapshot()  # Get button UID
mcp_io_github_chr_click(uid="<tyskland_button_uid>")
mcp_io_github_chr_wait_for(text="Bundesliga", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/germany_snapshot.txt

# --- ITALIA ---
mcp_io_github_chr_take_snapshot()  # Get button UID
mcp_io_github_chr_click(uid="<italia_button_uid>")
mcp_io_github_chr_wait_for(text="Serie A", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/italy_snapshot.txt

# --- FRANKRIKE ---
mcp_io_github_chr_take_snapshot()  # Get button UID
mcp_io_github_chr_click(uid="<frankrike_button_uid>")
mcp_io_github_chr_wait_for(text="Ligue 1", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/france_snapshot.txt

# --- SKOTTLAND ---
mcp_io_github_chr_take_snapshot()  # Get button UID
mcp_io_github_chr_click(uid="<skottland_button_uid>")
mcp_io_github_chr_wait_for(text="Premiership", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/scotland_snapshot.txt

# STEP 4: Process all collected snapshots
# Run this script in Python to combine all data
python auto_scrape_leagues.py

================================================================================
"""


if __name__ == "__main__":
    print(AUTOMATION_INSTRUCTIONS)
    
    # Check if we should process existing snapshots
    snapshot_dir = "data/snapshots"
    
    if os.path.exists(snapshot_dir):
        existing_files = [f for f in os.listdir(snapshot_dir) if f.endswith('_snapshot.txt')]
        
        if existing_files:
            print(f"\nFound {len(existing_files)} snapshot files in {snapshot_dir}")
            print("Processing snapshots...\n")
            
            combined_df = process_all_snapshots()
            
            if not combined_df.empty:
                print("\n" + "="*80)
                print("SUCCESS! All leagues scraped and processed.")
                print("="*80)
                print(f"\nTotal matches: {len(combined_df)}")
                print(f"Output file: data/processed/all_leagues_odds.csv")
                
                # Show breakdown by league
                if 'League' in combined_df.columns:
                    print("\nMatches by league:")
                    print(combined_df['League'].value_counts())
        else:
            print(f"\nNo snapshot files found in {snapshot_dir}")
            print("Please run the MCP automation commands above to collect data.")
    else:
        print(f"\nDirectory {snapshot_dir} does not exist yet.")
        print("Please run the MCP automation commands above to collect data.")
