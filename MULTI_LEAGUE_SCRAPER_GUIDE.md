# Automated Multi-League Scraper for Norsk Tipping

Complete automation for scraping odds from all supported leagues on Norsk Tipping using Chrome DevTools MCP.

## Overview

This scraper automatically:

1. ✅ Configures three dropdown menus (O/U 1.5, 2.5, 3.5 goals)
2. ✅ Clicks through each country section
3. ✅ Collects odds for all supported leagues
4. ✅ Processes and combines data into one CSV file

## Supported Leagues

### England

- **E0** - Premier League
- **E1** - Championship

### Spain (Spania)

- **SP1** - La Liga
- **SP2** - Segunda División

### Germany (Tyskland)

- **D1** - Bundesliga
- **D2** - 2. Bundesliga

### Italy (Italia)

- **I1** - Serie A
- **I2** - Serie B

### France (Frankrike)

- **F1** - Ligue 1
- **F2** - Ligue 2

### Scotland (Skottland)

- **SC0** - Premiership
- **SC1** - Championship

**Total: 12 leagues across 6 countries**

## Quick Start

### Simple Chat Command

Just tell Copilot in chat:

```
Scrape all leagues from Norsk Tipping:
1. Reload page
2. Configure three dropdowns with O/U 1.5, 2.5, 3.5 using fill_form
3. Click through England, Spania, Tyskland, Italia, Frankrike, Skottland
4. Save each snapshot to data/snapshots/
5. Run python auto_scrape_leagues.py
```

## Detailed Workflow

### Step 1: Configure Dropdowns

**CRITICAL:** All three dropdowns must be configured before scraping!

```
Take a snapshot to get dropdown UIDs, then:

mcp_io_github_chr_fill_form(elements=[
    {"uid": "<dropdown1_uid>", "value": "Totalt antall mål - over/under 1.5"},
    {"uid": "<dropdown2_uid>", "value": "Totalt antall mål - over/under 2.5"},
    {"uid": "<dropdown3_uid>", "value": "Totalt antall mål - over/under 3.5"}
])
```

### Step 2: Scrape Each Country

For each country, execute this sequence:

1. **Take snapshot** to get expand button UID
2. **Click expand button** for the country
3. **Wait for league text** to appear
4. **Take snapshot** and save to file

#### England

```
take_snapshot() → get button UID
click(uid="<england_button>")
wait_for(text="Premier League")
take_snapshot() → save to data/snapshots/england_snapshot.txt
```

#### Spain (Spania)

```
take_snapshot() → get button UID
click(uid="<spania_button>")
wait_for(text="La Liga")
take_snapshot() → save to data/snapshots/spain_snapshot.txt
```

#### Germany (Tyskland)

```
take_snapshot() → get button UID
click(uid="<tyskland_button>")
wait_for(text="Bundesliga")
take_snapshot() → save to data/snapshots/germany_snapshot.txt
```

#### Italy (Italia)

```
take_snapshot() → get button UID
click(uid="<italia_button>")
wait_for(text="Serie A")
take_snapshot() → save to data/snapshots/italy_snapshot.txt
```

#### France (Frankrike)

```
take_snapshot() → get button UID
click(uid="<frankrike_button>")
wait_for(text="Ligue 1")
take_snapshot() → save to data/snapshots/france_snapshot.txt
```

#### Scotland (Skottland)

```
take_snapshot() → get button UID
click(uid="<skottland_button>")
wait_for(text="Premiership")
take_snapshot() → save to data/snapshots/scotland_snapshot.txt
```

### Step 3: Process All Data

After collecting all snapshots, run:

```bash
python auto_scrape_leagues.py
```

This will:

- Parse all snapshot files
- Combine odds from all leagues
- Save to `data/processed/all_leagues_odds.csv`
- Display summary statistics

## Files Structure

```
bet/
├── auto_scrape_leagues.py          # Main automation script with processing
├── scrape_all_leagues.py           # Alternative processing script
├── scrape_all_leagues_mcp.py       # Interactive MCP guide
├── scrape_all_leagues_auto.py      # Automation workflow documentation
├── data/
│   ├── snapshots/                  # Raw MCP snapshots
│   │   ├── england_snapshot.txt
│   │   ├── spain_snapshot.txt
│   │   ├── germany_snapshot.txt
│   │   ├── italy_snapshot.txt
│   │   ├── france_snapshot.txt
│   │   └── scotland_snapshot.txt
│   └── processed/
│       └── all_leagues_odds.csv    # Combined output
└── src/
    └── norsk_tipping_scraper.py    # Snapshot parser
```

## Output Format

The combined CSV file contains:

| Column    | Description                 |
| --------- | --------------------------- |
| Date      | Match date (YYYY-MM-DD)     |
| Time      | Match time (HH:MM)          |
| HomeTeam  | Home team name (normalized) |
| AwayTeam  | Away team name (normalized) |
| League    | League name                 |
| Over_1.5  | Over 1.5 goals odds         |
| Under_1.5 | Under 1.5 goals odds        |
| Over_2.5  | Over 2.5 goals odds         |
| Under_2.5 | Under 2.5 goals odds        |
| Over_3.5  | Over 3.5 goals odds         |
| Under_3.5 | Under 3.5 goals odds        |

## Important Notes

### Dropdown Configuration

- **Must be done first** - without this, only default markets will show
- Use `fill_form` to fill all three at once (more reliable)
- Dropdowns configure the page to show O/U 1.5, 2.5, 3.5 markets

### Country Clicking

- Countries are in collapsed state by default
- Must click expand button for each country
- Wait for league text to confirm expansion
- Take snapshot after expansion completes

### Snapshot Saving

- Save each snapshot to the correct filename
- Use `data/snapshots/` directory
- Filename format: `<country>_snapshot.txt`

## Troubleshooting

### Problem: Dropdowns not configured

**Solution:** Always take a snapshot first to get UIDs, then use `fill_form` to fill all three at once

### Problem: Country won't expand

**Solution:** Take a snapshot to verify button UID, ensure you're clicking the "Utvide" (expand) button

### Problem: No odds data

**Solution:** Verify dropdowns were configured before clicking countries

### Problem: Snapshot timeout

**Solution:** Increase timeout value (default 5000ms), some countries may load slower

## Next Steps

After scraping all leagues:

1. **Convert to football-data.org format:**

   ```bash
   python convert_norsk_tipping.py
   ```

2. **Score fixtures:**

   ```bash
   python score_fixtures.py
   ```

3. **View recommendations:**
   ```bash
   python show_recommendations.py
   ```

## Tips for Efficiency

1. **Batch snapshots** - Collect all 6 country snapshots before processing
2. **Save as you go** - Copy each snapshot text to its file immediately
3. **Verify configuration** - Check that all three O/U markets appear in first match
4. **Monitor progress** - Each country should show 10-50 matches depending on fixtures

## Full MCP Command Reference

See `auto_scrape_leagues.py` for complete MCP command sequence with all UIDs and parameters.
