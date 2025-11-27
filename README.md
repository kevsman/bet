# Soccer Totals Betting Pipeline

A data-driven betting system for identifying value in soccer over/under totals markets. Uses Poisson regression models to predict expected goals, compares with bookmaker odds, and sizes stakes using Kelly Criterion.

## Features

-   **Two Independent Models**: Domestic leagues model and European competitions model (UCL/UEL/UECL)
-   **Automated Data Fetching**: Downloads historical data from football-data.co.uk and openfootball/Wikipedia
-   **Odds Scraping**: Selenium-based scraper for Norsk Tipping odds
-   **Value Bet Detection**: Compares model probabilities with bookmaker odds to find positive expected value
-   **HTML Reports**: Interactive reports with Top 5 picks and bet calculator
-   **Kelly Criterion Staking**: 25% fractional Kelly with 5% max cap per bet

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Full Pipeline (Recommended)

**Domestic Leagues** (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, etc.):

```bash
python run_full_pipeline.py
```

**European Competitions** (Champions League, Europa League, Conference League):

```bash
python run_full_pipeline.py --european
```

Both commands will:

1. Download latest match data
2. Prepare dataset with rolling features
3. Train Poisson models
4. Scrape current odds from Norsk Tipping
5. Calculate value bets
6. Generate HTML report (`betting_report.html`)

### Optional: Fetch xG Data (Recommended for Improved Predictions)

```bash
python fetch_xg_data.py --source understat --seasons 2023 2024
```

This fetches expected goals (xG) data from Understat for the top 5 leagues. xG measures shot quality and is a powerful predictor of future scoring.

---

## Models

### Domestic Leagues Model

**Data Source**: [football-data.co.uk](https://www.football-data.co.uk/)

**Leagues Covered**:

-   England: Premier League, Championship, League 1, League 2, National League
-   Germany: Bundesliga, 2. Bundesliga
-   Spain: La Liga, Segunda División
-   Italy: Serie A, Serie B
-   France: Ligue 1, Ligue 2
-   Scotland: Premiership
-   Belgium, Netherlands, Greece, Portugal, Turkey

**Features Used**:

-   Rolling averages (3, 5, 10 games): goals scored, goals conceded, total shots, shots on target, corners
-   Exponential moving averages (EMA) for recency weighting
-   Shot quality metrics: conversion rate (goals/shots), accuracy (SOT/shots)
-   League-specific scoring averages
-   Match count tracking
-   **xG data** (optional): Expected goals from Understat for top 5 leagues

**Training**: ~4 seasons of historical data per league

### European Competitions Model

**Data Sources**:

-   [openfootball/champions-league](https://github.com/openfootball/champions-league) - UCL data
-   Wikipedia - Europa League and Conference League match results

**Competitions**:

-   UEFA Champions League (UCL)
-   UEFA Europa League (UEL)
-   UEFA Europa Conference League (UECL)

**Features Used** (41 total):

-   12 domestic form features (from teams' league matches)
-   20 European history features (European-specific rolling averages)
-   9 competition context features (competition tier, stage, etc.)

**Special Handling**:

-   Teams without European history use domestic form as fallback
-   Cross-competition tracking (same team across UCL/UEL/UECL)
-   Competition tier weighting

---

## Pipeline Steps

### Domestic Pipeline (`run_full_pipeline.py`)

| Step | Script/Module                  | Description                             |
| ---- | ------------------------------ | --------------------------------------- |
| 1    | `src.data_fetch`               | Download CSVs from football-data.co.uk  |
| 2    | `src.prepare_dataset`          | Calculate rolling averages and features |
| 3    | `src.models`                   | Train Poisson regression models         |
| 4    | `scrape_norsk_tipping_live.py` | Scrape current odds (Selenium)          |
| 5    | `generate_report.py`           | Generate value bets + HTML report       |

### European Pipeline (`run_full_pipeline.py --european`)

| Step | Script/Module                    | Description                                  |
| ---- | -------------------------------- | -------------------------------------------- |
| 1    | `fetch_european_openfootball.py` | Download UCL + fetch UEL/UECL from Wikipedia |
| 2    | `src.european_prepare`           | Prepare features with domestic form fallback |
| 3    | `src.european_models`            | Train European Poisson models                |
| 4    | `scrape_norsk_tipping_live.py`   | Scrape current odds (shared)                 |
| 5    | `generate_report.py --european`  | Generate European value bets report          |

---

## Key Scripts

| Script                           | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `run_full_pipeline.py`           | Main entry point - runs complete pipeline   |
| `run_european_today.py`          | Generate European predictions for today     |
| `generate_report.py`             | Create HTML betting report                  |
| `scrape_norsk_tipping_live.py`   | Selenium scraper for Norsk Tipping          |
| `fetch_european_openfootball.py` | Download European competition data          |
| `fetch_european_wikipedia.py`    | Scrape UEL/UECL from Wikipedia              |
| `show_recommendations.py`        | Display current recommendations in terminal |
| `show_portfolio.py`              | Show portfolio allocation summary           |

---

## Output Files

### Data Files (`data/`)

| File                               | Description                             |
| ---------------------------------- | --------------------------------------- |
| `raw/*.csv`                        | Historical match data by league/season  |
| `processed/match_dataset.csv`      | Prepared domestic dataset with features |
| `processed/norsk_tipping_odds.csv` | Scraped odds                            |
| `processed/recommendations.csv`    | Current value bets                      |

### Model Files (`models/`)

| File                           | Description                   |
| ------------------------------ | ----------------------------- |
| `home_poisson.joblib`          | Home goals Poisson model      |
| `away_poisson.joblib`          | Away goals Poisson model      |
| `features.txt`                 | Feature columns used by model |
| `european_home_poisson.joblib` | European home goals model     |
| `european_away_poisson.joblib` | European away goals model     |

### Reports

| File                  | Description                                 |
| --------------------- | ------------------------------------------- |
| `betting_report.html` | Interactive HTML report with bet calculator |

---

## Mathematical Approach

### Goal Prediction

Uses **Poisson regression** to predict expected goals for home and away teams:

$$\lambda_{home} = \exp(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)$$
$$\lambda_{away} = \exp(\gamma_0 + \gamma_1 x_1 + ... + \gamma_n x_n)$$

### Over/Under Probability

Uses **bivariate Poisson** to calculate probability of total goals:

$$P(\text{Over } k) = 1 - \sum_{h=0}^{k} \sum_{a=0}^{k-h} P(H=h) \cdot P(A=a)$$

where $P(H=h) = \frac{\lambda_{home}^h e^{-\lambda_{home}}}{h!}$

### Expected Value

$$EV = (P_{model} \times Odds) - 1$$

Only bets with positive EV (edge > 0%) are recommended.

### Kelly Criterion Staking

$$f^* = \frac{p \cdot b - q}{b}$$

where:

-   $p$ = model probability
-   $q = 1 - p$
-   $b$ = decimal odds - 1

Uses **25% fractional Kelly** with **5% maximum cap** per bet.

---

## Configuration

Edit `src/config.py` for domestic leagues or `src/european_config.py` for European competitions:

```python
# Leagues to include
LEAGUES = ["E0", "E1", "D1", "SP1", "I1", "F1"]

# Seasons to fetch
SEASONS = ["2122", "2223", "2324", "2425"]

# Staking parameters
KELLY_FRACTION = 0.25  # 25% Kelly
MAX_STAKE = 0.05       # 5% max per bet
MIN_EDGE = 0.03        # 3% minimum edge
```

---

## Team Name Mappings

The system uses `src/team_mappings.py` with 400+ mappings to normalize team names across:

-   Norsk Tipping odds (Norwegian names)
-   football-data.co.uk (English abbreviations)
-   openfootball (full official names)
-   Wikipedia (various formats)

---

## Requirements

-   Python 3.9+
-   Chrome browser (for Selenium scraping)
-   Dependencies: pandas, numpy, scikit-learn, scipy, joblib, selenium, beautifulsoup4

---

## License

MIT
