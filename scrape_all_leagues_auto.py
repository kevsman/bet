"""
Fully automated scraper for all leagues on Norsk Tipping using MCP Chrome tools
This script will be called through Copilot chat to automate the entire scraping process
"""

# This script is designed to be run through VS Code Copilot with MCP Chrome tools active
# It provides the complete automation workflow with dropdown configuration

AUTOMATION_WORKFLOW = """
AUTOMATED NORSK TIPPING SCRAPER - ALL LEAGUES
==============================================

This script automates scraping of all supported leagues with proper dropdown configuration.

USAGE: Tell Copilot in chat:
"Run the automated scraper in scrape_all_leagues_auto.py"

AUTOMATION STEPS:
=================

1. INITIAL SETUP
   - Reload the page to ensure clean state
   - Wait for Fotball section to load

2. CONFIGURE DROPDOWNS (CRITICAL!)
   - Take a snapshot to get dropdown UIDs
   - Fill all three dropdowns using fill_form:
     * Dropdown 1: "Totalt antall mål - over/under 1.5"
     * Dropdown 2: "Totalt antall mål - over/under 2.5"
     * Dropdown 3: "Totalt antall mål - over/under 3.5"
   - This ensures all three goal line markets are displayed

3. SCRAPE COUNTRIES
   For each country:
   - Click expand button
   - Wait for league text to appear
   - Take snapshot
   - Save to file
   
   Countries in order:
   a) England (wait for "Premier League")
   b) Spania (wait for "La Liga")
   c) Tyskland (wait for "Bundesliga")
   d) Italia (wait for "Serie A")
   e) Frankrike (wait for "Ligue 1")
   f) Skottland (wait for "Premiership")

4. PROCESS DATA
   - Run: python scrape_all_leagues.py
   - This combines all snapshots into one CSV file

SUPPORTED LEAGUES:
==================
E0  - England Premier League
E1  - England Championship
SP1 - Spain La Liga
SP2 - Spain Segunda División
D1  - Germany Bundesliga
D2  - Germany 2. Bundesliga
I1  - Italy Serie A
I2  - Italy Serie B
F1  - France Ligue 1
F2  - France Ligue 2
SC0 - Scotland Premiership
SC1 - Scotland Championship

IMPORTANT NOTES:
================
- The three dropdowns MUST be configured before scraping
- Use fill_form to fill all three dropdowns at once (more reliable)
- Wait for page updates after dropdown changes
- Save snapshots to data/snapshots/ directory
- Run scrape_all_leagues.py after all snapshots collected
"""

# MCP command sequence for Copilot
MCP_AUTOMATION_SEQUENCE = """
# STEP 1: Reload and wait
mcp_io_github_chr_navigate_page(type="reload")
mcp_io_github_chr_wait_for(text="Fotball", timeout=5000)

# STEP 2: Configure all three dropdowns (CRITICAL STEP!)
# Take snapshot first to get UIDs
mcp_io_github_chr_take_snapshot()

# Fill all three dropdowns at once using the UIDs from snapshot
mcp_io_github_chr_fill_form(elements=[
    {"uid": "<first_combobox_uid>", "value": "Totalt antall mål - over/under 1.5"},
    {"uid": "<second_combobox_uid>", "value": "Totalt antall mål - over/under 2.5"},
    {"uid": "<third_combobox_uid>", "value": "Totalt antall mål - over/under 3.5"}
])

# STEP 3: Expand and scrape each country

# England
mcp_io_github_chr_take_snapshot()  # Get England expand button UID
mcp_io_github_chr_click(uid="<england_expand_uid>")
mcp_io_github_chr_wait_for(text="Premier League", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save this snapshot to data/snapshots/england_snapshot.txt

# Spania
mcp_io_github_chr_take_snapshot()  # Get Spania expand button UID
mcp_io_github_chr_click(uid="<spania_expand_uid>")
mcp_io_github_chr_wait_for(text="La Liga", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/spain_snapshot.txt

# Tyskland
mcp_io_github_chr_take_snapshot()  # Get Tyskland expand button UID
mcp_io_github_chr_click(uid="<tyskland_expand_uid>")
mcp_io_github_chr_wait_for(text="Bundesliga", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/germany_snapshot.txt

# Italia
mcp_io_github_chr_take_snapshot()  # Get Italia expand button UID
mcp_io_github_chr_click(uid="<italia_expand_uid>")
mcp_io_github_chr_wait_for(text="Serie A", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/italy_snapshot.txt

# Frankrike
mcp_io_github_chr_take_snapshot()  # Get Frankrike expand button UID
mcp_io_github_chr_click(uid="<frankrike_expand_uid>")
mcp_io_github_chr_wait_for(text="Ligue 1", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/france_snapshot.txt

# Skottland
mcp_io_github_chr_take_snapshot()  # Get Skottland expand button UID
mcp_io_github_chr_click(uid="<skottland_expand_uid>")
mcp_io_github_chr_wait_for(text="Premiership", timeout=5000)
mcp_io_github_chr_take_snapshot()  # Save to data/snapshots/scotland_snapshot.txt

# STEP 4: Process collected data
# Run in terminal: python scrape_all_leagues.py
"""

CHAT_COMMAND = """
SIMPLIFIED CHAT COMMAND FOR COPILOT:
====================================

Just say in chat:

"Scrape all leagues from Norsk Tipping:
1. Reload the page
2. Configure the three dropdowns with over/under 1.5, 2.5, and 3.5 using fill_form
3. Expand and scrape: England, Spania, Tyskland, Italia, Frankrike, Skottland
4. Save each snapshot to data/snapshots/
5. Then run python scrape_all_leagues.py to process"

OR even simpler:

"Run the automated Norsk Tipping scraper with all three dropdowns configured for England, Spain, Germany, Italy, France, and Scotland"
"""

if __name__ == "__main__":
    print(AUTOMATION_WORKFLOW)
    print("\n" + "="*80)
    print("MCP COMMAND SEQUENCE:")
    print("="*80)
    print(MCP_AUTOMATION_SEQUENCE)
    print("\n" + "="*80)
    print(CHAT_COMMAND)
    print("="*80)
