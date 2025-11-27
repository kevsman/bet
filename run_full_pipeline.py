#!/usr/bin/env python3
"""
Full betting pipeline - runs everything from data fetch to report generation.

DOMESTIC LEAGUES PIPELINE (default):
1. Download latest results from football-data.co.uk
2. Prepare the dataset with rolling averages
2b. Fetch xG data from Understat (top 5 leagues)
3. Train the Poisson regression models
4. Scrape Norsk Tipping odds (via Selenium)
5. Calculate value bets using the trained model
6. Generate the HTML report

EUROPEAN COMPETITIONS PIPELINE (--european flag):
1. Fetch UCL/UEL/UECL data from openfootball GitHub
2. Prepare European dataset with cross-competition features
3. Train European Poisson model
4. Scrape Norsk Tipping odds (shared step)
5. Generate European value bets

Usage:
    python run_full_pipeline.py              # Run domestic leagues pipeline
    python run_full_pipeline.py --european   # Run European competitions pipeline
    python run_full_pipeline.py -e           # Short flag for European
    python run_full_pipeline.py --no-xg      # Skip xG data fetching
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def run_step(step_name: str, func, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"✓ {step_name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ {step_name} FAILED after {elapsed:.1f}s: {e}")
        raise


def step1_download_results():
    """Download latest match results from football-data.co.uk."""
    from src.data_fetch import sync_all
    from src.config import get_config
    
    cfg = get_config()
    files = sync_all(cfg)
    print(f"Downloaded/updated {len(files)} files to {cfg.raw_dir}")


def step2_prepare_dataset():
    """Prepare the dataset with rolling averages."""
    from src.prepare_dataset import main as prepare_main
    prepare_main()


def step2b_fetch_xg_data():
    """Fetch xG data from Understat and merge with dataset."""
    from fetch_xg_data import fetch_and_integrate_xg
    
    # Get current year to determine seasons
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Football seasons span two calendar years (e.g., 2024-25 season)
    # If we're in Jan-July, current season started previous year
    if current_month <= 7:
        current_season_start = current_year - 1
    else:
        current_season_start = current_year
    
    # Fetch last 3 seasons of xG data
    seasons = [current_season_start - 2, current_season_start - 1, current_season_start]
    print(f"Fetching xG data for seasons: {seasons}")
    
    try:
        result_df = fetch_and_integrate_xg(seasons=seasons)
        if result_df is not None:
            print(f"xG data integrated: {len(result_df)} matches with xG features")
        return result_df
    except Exception as e:
        print(f"Warning: xG fetch failed ({e}), continuing with core features only")
        return None


def step3_train_models():
    """Train Poisson regression models."""
    from src.models import main as train_main
    train_main()


def step4_scrape_odds(headless: bool = True, skip_if_recent: bool = True):
    """Scrape odds from Norsk Tipping using Selenium."""
    from src.config import get_config
    
    cfg = get_config()
    odds_file = cfg.processed_dir / "norsk_tipping_odds.csv"
    
    # Check if we have recent odds (less than 4 hours old)
    if skip_if_recent and odds_file.exists():
        import os
        mtime = os.path.getmtime(odds_file)
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 4:
            print(f"Odds file is {age_hours:.1f} hours old - skipping scrape")
            print(f"(Use --force-scrape to refresh)")
            return
    
    # Import and run the scraper
    from scrape_norsk_tipping_live import scrape_all_leagues
    
    print("Starting Selenium scraper (this may take a few minutes)...")
    df = scrape_all_leagues(headless=headless)
    
    if df is not None and not df.empty:
        df.to_csv(odds_file, index=False)
        print(f"Saved {len(df)} matches to {odds_file}")
    else:
        print("Warning: No odds scraped!")


def step5_generate_report():
    """Calculate value bets and generate HTML report."""
    # Import the report generator's main logic
    import pandas as pd
    import numpy as np
    import joblib
    import json
    from datetime import datetime
    from collections import defaultdict
    from scipy.stats import poisson
    from src.config import get_config
    from src.team_mappings import normalize_team_name
    
    cfg = get_config()
    
    # Load models
    print("Loading models...")
    home_model = joblib.load(cfg.models_dir / "home_poisson.joblib")
    away_model = joblib.load(cfg.models_dir / "away_poisson.joblib")
    feature_cols = (cfg.models_dir / "features.txt").read_text().strip().split("\n")
    
    # Load historical data for features
    print("Loading historical dataset...")
    dataset_path = cfg.processed_dir / "match_dataset.csv"
    hist_df = pd.read_csv(dataset_path, low_memory=False)
    
    # Load odds
    print("Loading Norsk Tipping odds...")
    odds_file = cfg.processed_dir / "norsk_tipping_odds.csv"
    if not odds_file.exists():
        raise FileNotFoundError(f"No odds file found at {odds_file}")
    
    odds_df = pd.read_csv(odds_file)
    print(f"  Loaded {len(odds_df)} matches")
    
    # Team mapping for Norsk Tipping names
    TEAM_MAPPING = {
        'Wolverhampton': 'Wolves',
        'Nottingham': 'Nott\'m Forest',
        'Man City': 'Manchester City',
        'Man Utd': 'Manchester Utd',
        'Newcastle Utd': 'Newcastle',
        'Tottenham': 'Tottenham',
        'Brighton': 'Brighton',
        'Leicester City': 'Leicester',
        'Ipswich Town': 'Ipswich',
        'Sheffield Utd': 'Sheffield United',
        'Leeds': 'Leeds',
        'Leeds United': 'Leeds',
        'West Brom': 'West Brom',
        'Norwich City': 'Norwich',
        'Stoke City': 'Stoke',
        'Swansea City': 'Swansea',
        'Cardiff City': 'Cardiff',
        'Huddersfield Town': 'Huddersfield',
        'Atlético Madrid': 'Ath Madrid',
        'Atletico Madrid': 'Ath Madrid',
        'Ath Bilbao': 'Ath Bilbao',
        'Athletic Bilbao': 'Ath Bilbao',
        'Celta Vigo': 'Celta',
        'Deportivo La Coruña': 'La Coruna',
        'Real Betis': 'Betis',
        'Real Sociedad': 'Sociedad',
        'Real Valladolid': 'Valladolid',
        'Rayo Vallecano': 'Vallecano',
        'Bayern München': 'Bayern Munich',
        'Bayern Munich': 'Bayern Munich',
        'Borussia Dortmund': 'Dortmund',
        'Borussia M\'gladbach': 'M\'gladbach',
        'Bor. Mönchengladbach': 'M\'gladbach',
        'Eintracht Frankfurt': 'Ein Frankfurt',
        'Hertha BSC': 'Hertha',
        'Hertha Berlin': 'Hertha',
        'RB Leipzig': 'RB Leipzig',
        'SC Freiburg': 'Freiburg',
        'VfB Stuttgart': 'Stuttgart',
        'VfL Wolfsburg': 'Wolfsburg',
        'Bayer Leverkusen': 'Leverkusen',
        '1. FC Köln': 'FC Koln',
        '1. FC Heidenheim': 'Heidenheim',
        'FC Augsburg': 'Augsburg',
        '1. FSV Mainz 05': 'Mainz',
        'TSG Hoffenheim': 'Hoffenheim',
        'Paris Saint-Germain': 'Paris SG',
        'Paris S-G': 'Paris SG',
        'AS Monaco': 'Monaco',
        'Olympique Marseille': 'Marseille',
        'Olympique Lyon': 'Lyon',
        'OGC Nice': 'Nice',
        'Stade Rennais': 'Rennes',
        'RC Lens': 'Lens',
        'LOSC Lille': 'Lille',
        'FC Nantes': 'Nantes',
        'Stade Brestois': 'Brest',
        'Stade de Reims': 'Reims',
        'RC Strasbourg': 'Strasbourg',
        'Juventus': 'Juventus',
        'Inter': 'Inter',
        'AC Milan': 'Milan',
        'AS Roma': 'Roma',
        'S.S. Lazio': 'Lazio',
        'SSC Napoli': 'Napoli',
        'Atalanta BC': 'Atalanta',
        'ACF Fiorentina': 'Fiorentina',
        'Torino FC': 'Torino',
        'Bologna FC': 'Bologna',
        'Udinese': 'Udinese',
        'Hellas Verona': 'Verona',
        'US Sassuolo': 'Sassuolo',
        'Empoli FC': 'Empoli',
        'Genoa CFC': 'Genoa',
        'Cagliari': 'Cagliari',
        'US Lecce': 'Lecce',
        'Parma Calcio': 'Parma',
        'Venezia FC': 'Venezia',
        'Como 1907': 'Como',
        'Celtic': 'Celtic',
        'Rangers': 'Rangers',
        'Aberdeen': 'Aberdeen',
        'Hearts': 'Hearts',
        'Hibernian': 'Hibernian',
        'St Mirren': 'St Mirren',
        'Dundee Utd': 'Dundee Utd',
        'Dundee United': 'Dundee Utd',
        'Kilmarnock': 'Kilmarnock',
        'Motherwell': 'Motherwell',
        'Ross County': 'Ross County',
        'St Johnstone': 'St Johnstone',
        'Livingston': 'Livingston',
    }
    
    def map_team(name):
        if name in TEAM_MAPPING:
            return TEAM_MAPPING[name]
        return normalize_team_name(name)
    
    def get_team_avg_features(team, is_home, hist_df, feature_cols, n=5):
        prefix = 'home_' if is_home else 'away_'
        team_col = 'HomeTeam' if is_home else 'AwayTeam'
        matches = hist_df[hist_df[team_col] == team].sort_values('Date', ascending=False).head(n)
        if len(matches) == 0:
            return None
        feats = []
        for col in feature_cols:
            if col in matches.columns:
                feats.append(matches[col].mean())
            else:
                feats.append(0)
        return feats
    
    def bivariate_prob_over(line, home_xg, away_xg, max_goals=10):
        prob = 0.0
        threshold = int(line)
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if h + a > threshold:
                    prob += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
        return prob
    
    def bivariate_prob_under(line, home_xg, away_xg, max_goals=10):
        prob = 0.0
        threshold = int(line)
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if h + a < threshold:
                    prob += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
        return prob
    
    # Calculate recommendations
    print("Calculating value bets...")
    recommendations = []
    
    for _, row in odds_df.iterrows():
        home_team_nt = row['home_team']
        away_team_nt = row['away_team']
        home_team = map_team(home_team_nt)
        away_team = map_team(away_team_nt)
        
        home_feats = get_team_avg_features(home_team, True, hist_df, feature_cols)
        away_feats = get_team_avg_features(away_team, False, hist_df, feature_cols)
        
        if home_feats is None or away_feats is None:
            continue
        
        pred_home = home_model.predict([home_feats])[0]
        pred_away = away_model.predict([away_feats])[0]
        total = pred_home + pred_away
        
        for line in [1.5, 2.5, 3.5]:
            over_col = f'over_{line}'
            under_col = f'under_{line}'
            
            if over_col not in row or under_col not in row:
                continue
            
            over_odds = row.get(over_col)
            under_odds = row.get(under_col)
            
            if pd.isna(over_odds) or pd.isna(under_odds):
                continue
            
            p_over = bivariate_prob_over(line, pred_home, pred_away)
            p_under = bivariate_prob_under(line, pred_home, pred_away)
            implied_over = 1 / over_odds
            implied_under = 1 / under_odds
            edge_over = p_over - implied_over
            edge_under = p_under - implied_under
            min_edge = 0.05
            
            if edge_over > min_edge:
                kelly = edge_over / (over_odds - 1) if over_odds > 1 else 0
                ev_per_unit = (p_over * over_odds) - 1
                recommendations.append({
                    'date': row['date'],
                    'time': row['time'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'country': row.get('country', ''),
                    'bet': f'Over {line}',
                    'odds': over_odds,
                    'model_total': total,
                    'probability': p_over,
                    'edge': edge_over,
                    'ev': ev_per_unit,
                    'kelly': min(kelly * 0.25, 0.05)
                })
            
            if edge_under > min_edge:
                kelly = edge_under / (under_odds - 1) if under_odds > 1 else 0
                ev_per_unit = (p_under * under_odds) - 1
                recommendations.append({
                    'date': row['date'],
                    'time': row['time'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'country': row.get('country', ''),
                    'bet': f'Under {line}',
                    'odds': under_odds,
                    'model_total': total,
                    'probability': p_under,
                    'edge': edge_under,
                    'ev': ev_per_unit,
                    'kelly': min(kelly * 0.25, 0.05)
                })
    
    # Sort by EV and dedupe
    recommendations = sorted(recommendations, key=lambda x: -x['ev'])
    seen = set()
    unique_recs = []
    for r in recommendations:
        key = (r['home_team'], r['away_team'], r['bet'])
        if key not in seen:
            seen.add(key)
            unique_recs.append(r)
    
    print(f"  Found {len(unique_recs)} value bets")
    
    # Save recommendations to CSV
    if unique_recs:
        recs_df = pd.DataFrame(unique_recs)
        recs_df.to_csv(cfg.processed_dir / "value_bets.csv", index=False)
    
    # Generate HTML report (using generate_value_bets_report logic)
    print("Generating HTML report...")
    
    # Import and run the report generator
    from generate_value_bets_report import main as generate_report_main
    generate_report_main()


# ============================================================
# EUROPEAN PIPELINE FUNCTIONS
# ============================================================

def step_euro_1_fetch_data():
    """Fetch European competition data from openfootball."""
    from fetch_european_openfootball import fetch_all_european_data
    from pathlib import Path
    import pandas as pd
    
    print("Fetching European data from openfootball GitHub...")
    df = fetch_all_european_data()
    
    if df.empty:
        print("No data fetched!")
        return
    
    # Save to both locations
    output_dir = Path("data/raw/european")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "openfootball_european.csv", index=False)
    df.to_csv(output_dir / "european_matches.csv", index=False)
    
    print(f"Saved {len(df)} European matches")


def step_euro_2_prepare_dataset():
    """Prepare European dataset with features."""
    from src.european_prepare import main as prepare_main
    prepare_main()


def step_euro_3_train_model():
    """Train European Poisson model."""
    from src.european_models import main as train_main
    train_main()


def step_euro_4_generate_predictions():
    """Generate European value bets using Norsk Tipping odds."""
    # Run the european today script
    from run_european_today import main as euro_main
    euro_main()


def step_euro_5_generate_html_report():
    """Generate HTML report for European bets."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "generate_report.py", "--european"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def run_european_pipeline(args):
    """Run the European competition pipeline."""
    start_time = time.time()
    
    try:
        # Step 1: Fetch European data
        if not args.no_download:
            run_step("Fetch European Data (openfootball)", step_euro_1_fetch_data)
        else:
            print("\n[Skipping European data fetch]")
        
        # Step 2: Prepare dataset
        if not args.no_train:
            run_step("Prepare European Dataset", step_euro_2_prepare_dataset)
        else:
            print("\n[Skipping European dataset preparation]")
        
        # Step 3: Train model
        if not args.no_train:
            run_step("Train European Model", step_euro_3_train_model)
        else:
            print("\n[Skipping European model training]")
        
        # Step 4: Scrape Norsk Tipping odds (same as regular pipeline)
        if not args.no_scrape:
            headless = not args.visible
            skip_recent = not args.force_scrape
            run_step("Scrape Norsk Tipping Odds", step4_scrape_odds, headless, skip_recent)
        else:
            print("\n[Skipping odds scraping]")
        
        # Step 5: Generate European predictions
        run_step("Generate European Value Bets", step_euro_4_generate_predictions)
        
        # Step 6: Generate HTML report
        run_step("Generate HTML Report", step_euro_5_generate_html_report)
        
        total_time = time.time() - start_time
        print(f"\n{'#'*60}")
        print(f"# EUROPEAN PIPELINE COMPLETE - Total time: {total_time:.1f}s")
        print(f"{'#'*60}")
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"! EUROPEAN PIPELINE FAILED: {e}")
        print(f"{'!'*60}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Run the full betting pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full betting pipeline")
    parser.add_argument("--european", "-e", action="store_true", 
                        help="Run European competitions pipeline (UCL/UEL/UECL)")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading latest results")
    parser.add_argument("--no-train", action="store_true", help="Skip model training")
    parser.add_argument("--no-xg", action="store_true", help="Skip xG data fetching")
    parser.add_argument("--no-scrape", action="store_true", help="Skip odds scraping")
    parser.add_argument("--force-scrape", action="store_true", help="Force scrape even if recent")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser in headless mode")
    parser.add_argument("--visible", action="store_true", help="Show browser window while scraping")
    parser.add_argument("--open-report", action="store_true", help="Open report in browser when done")
    args = parser.parse_args()
    
    # Determine which pipeline to run
    if args.european:
        print(f"\n{'#'*60}")
        print(f"# EUROPEAN PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"# (UCL / UEL / Conference League)")
        print(f"{'#'*60}")
        run_european_pipeline(args)
        return
    
    print(f"\n{'#'*60}")
    print(f"# BETTING PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}")
    
    start_time = time.time()
    
    try:
        # Step 1: Download latest results
        if not args.no_download:
            run_step("Download Latest Results", step1_download_results)
        else:
            print("\n[Skipping download step]")
        
        # Step 2: Prepare dataset
        if not args.no_train:
            run_step("Prepare Dataset", step2_prepare_dataset)
        else:
            print("\n[Skipping dataset preparation]")
        
        # Step 2b: Fetch xG data (optional enhancement)
        if not args.no_train and not args.no_xg:
            run_step("Fetch xG Data", step2b_fetch_xg_data)
        elif args.no_xg:
            print("\n[Skipping xG data fetch]")
        
        # Step 3: Train models
        if not args.no_train:
            run_step("Train Models", step3_train_models)
        else:
            print("\n[Skipping model training]")
        
        # Step 4: Scrape odds
        if not args.no_scrape:
            headless = not args.visible
            skip_recent = not args.force_scrape
            run_step("Scrape Norsk Tipping Odds", step4_scrape_odds, headless, skip_recent)
        else:
            print("\n[Skipping odds scraping]")
        
        # Step 5: Generate report
        run_step("Generate Value Bets Report", step5_generate_report)
        
        total_time = time.time() - start_time
        print(f"\n{'#'*60}")
        print(f"# PIPELINE COMPLETE - Total time: {total_time:.1f}s")
        print(f"{'#'*60}")
        
        # Open report if requested
        if args.open_report:
            import os
            report_path = BASE_DIR / "value_bets_report.html"
            if report_path.exists():
                os.startfile(str(report_path))
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"! PIPELINE FAILED: {e}")
        print(f"{'!'*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
