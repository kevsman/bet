# European Competition Model

Separate model for UEFA Champions League, Europa League, and Conference League predictions.

## Setup

### 1. Get API Key
Get a free API key from [API-Football](https://www.api-football.com/):
- Free tier: 100 requests/day
- Covers all European competitions

### 2. Set Environment Variable
```powershell
$env:API_FOOTBALL_KEY = "your-api-key-here"
```

Or set permanently in Windows:
```powershell
[Environment]::SetEnvironmentVariable("API_FOOTBALL_KEY", "your-key", "User")
```

## Usage

### Full Pipeline
```bash
python run_european.py all
```

### Individual Steps
```bash
python run_european.py fetch      # Download historical data
python run_european.py prepare    # Create dataset
python run_european.py train      # Train model
python run_european.py predict    # Score upcoming matches
```

## Model Differences from Domestic

| Aspect | Domestic Model | European Model |
|--------|---------------|----------------|
| Data Source | football-data.co.uk | API-Football |
| Matches/Season | 380 per league | 125-189 per competition |
| Rolling Windows | 3, 5, 10 | 3, 5 (smaller sample) |
| Min Matches | 5 | 2 |
| Features | League averages | Competition averages + stage |

## Special Features

### Stage Importance
- Group stage matches (lower stakes)
- Round of 16, Quarter-finals, Semi-finals
- Finals (neutral venue)

### Cross-Competition Tracking
Teams' European form is tracked across all UEFA competitions, not just one.

### Competition Averages
- UCL typically has more goals than UEL/UECL
- Model adjusts expectations per competition

## Files Created

```
data/
  raw/european/
    european_matches.csv      # Historical match data
  processed/european/
    european_dataset.csv      # Processed features
    european_predictions.csv  # Historical predictions
    upcoming_european.csv     # Upcoming match predictions

models/european/
    euro_home_poisson.joblib  # Home goals model
    euro_away_poisson.joblib  # Away goals model
    euro_calibrator.joblib    # Probability calibrator
    euro_features.txt         # Feature list
```

## Competitions Covered

| Competition | Code | API ID |
|-------------|------|--------|
| Champions League | UCL | 2 |
| Europa League | UEL | 3 |
| Conference League | UECL | 848 |

## Limitations

1. **Smaller sample size** - Teams play 6-13 European matches vs 38 league
2. **Squad rotation** - Teams often rotate in group stages
3. **Knockout dynamics** - Two-legged ties have different patterns
4. **New participants** - Teams new to Europe have no historical features
