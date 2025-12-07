#!/usr/bin/env python3
"""
Full betting pipeline - runs everything from data fetch to report generation.

DOMESTIC LEAGUES PIPELINE (default):
1. Download latest results from football-data.co.uk
2. Prepare the dataset with rolling averages
2b. Fetch xG data from Understat (top 5 leagues)
2c. Integrate advanced features (FBref stats, weather, injuries, manager data)
3. Train the Poisson regression models
3b. Train Dixon-Coles model (correlation adjustment for low-scoring games)
3c. Train Gradient Boosting models (LightGBM/XGBoost)
4. Scrape Norsk Tipping odds (via Selenium)
5. Calculate value bets using ensemble of all models
6. Generate the HTML report

EUROPEAN COMPETITIONS PIPELINE (--european flag):
1. Fetch UCL/UEL/UECL data from openfootball GitHub
2. Prepare European dataset with cross-competition features
3. Train European Poisson model
4. Scrape Norsk Tipping odds (shared step)
5. Generate European value bets

Usage:
    python run_full_pipeline.py                  # Run full domestic pipeline
    python run_full_pipeline.py --european       # Run European competitions pipeline
    python run_full_pipeline.py --no-xg          # Skip xG data fetching
    python run_full_pipeline.py --no-dixon-coles # Skip Dixon-Coles model
    python run_full_pipeline.py --no-gradient    # Skip Gradient Boosting training
    python run_full_pipeline.py --no-advanced    # Skip advanced features integration
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

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
    seasons = [str(current_season_start - 2), str(current_season_start - 1), str(current_season_start)]
    print(f"Fetching xG data for seasons: {seasons}")
    
    try:
        result_df = fetch_and_integrate_xg(seasons=seasons)
        if result_df is not None:
            print(f"xG data integrated: {len(result_df)} matches with xG features")
        return result_df
    except Exception as e:
        print(f"Warning: xG fetch failed ({e}), continuing with core features only")
        return None


def step2c_integrate_advanced_features():
    """Integrate advanced features (FBref, weather, injuries, manager data)."""
    from src.integrate_advanced_features import integrate_all_features
    from src.config import get_config
    import pandas as pd
    
    cfg = get_config()
    base_dataset = cfg.processed_dir / "match_dataset.csv"
    
    if not base_dataset.exists():
        print("Warning: Base dataset not found, skipping advanced features integration")
        return None
    
    # Load base dataset
    match_df = pd.read_csv(base_dataset, parse_dates=["Date"], low_memory=False)
    
    # Run integration
    try:
        result_df = integrate_all_features(
            match_df=match_df,
            cfg=cfg,
            include_advanced=True,
            include_weather=True,
            include_injuries=True,
            include_manager=True
        )
        if result_df is not None:
            # Save enhanced dataset
            output_path = cfg.processed_dir / "match_dataset_enhanced.csv"
            result_df.to_csv(output_path, index=False)
            print(f"Advanced features integrated: {len(result_df)} matches")
            # Count new columns
            new_cols = len(result_df.columns) - len(match_df.columns)
            if new_cols > 0:
                print(f"Added {new_cols} advanced feature columns")
        return result_df
    except Exception as e:
        print(f"Warning: Advanced features integration failed ({e}), continuing without them")
        return None


def step3_train_models():
    """Train Poisson regression models."""
    from src.models import main as train_main
    train_main()


def step3b_train_dixon_coles():
    """Train Dixon-Coles model for improved predictions."""
    try:
        from src.dixon_coles import train_dixon_coles
        model = train_dixon_coles()
        return model
    except Exception as e:
        print(f"Warning: Dixon-Coles training failed ({e}), continuing with Poisson model only")
        return None


def step3c_train_gradient_boosting():
    """Train LightGBM and XGBoost models."""
    import subprocess
    import sys
    
    print("Training Gradient Boosting models (LightGBM + XGBoost)...")
    result = subprocess.run(
        [sys.executable, "train_gradient_boosting.py"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("Warning: Gradient Boosting training had issues")
        return False
    return True


def step4_scrape_odds(headless: bool = True, skip_if_recent: bool = True):
    """Scrape odds from Norsk Tipping using Selenium."""
    from src.config import get_config, filter_womens_matches
    
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
            # Still filter existing file in case it has women's matches
            df = pd.read_csv(odds_file)
            original_count = len(df)
            df = filter_womens_matches(df, verbose=True)
            if len(df) < original_count:
                df.to_csv(odds_file, index=False)
                print(f"Re-saved filtered odds: {len(df)} matches")
            return
    
    # Import and run the scraper
    from scrape_norsk_tipping_live import scrape_all_leagues
    
    print("Starting Selenium scraper (this may take a few minutes)...")
    df = scrape_all_leagues(headless=headless)
    
    if df is not None and not df.empty:
        # Filter is already applied in scraper, but double-check
        df = filter_womens_matches(df, verbose=False)
        df.to_csv(odds_file, index=False)
        print(f"Saved {len(df)} matches to {odds_file}")
    else:
        print("Warning: No odds scraped!")


def step5_generate_report():
    """Calculate value bets and generate HTML report using Dixon-Coles ensemble."""
    # Use calculate_fresh_recommendations.py which properly handles xG and Dixon-Coles
    import subprocess
    import sys
    
    print("Running calculate_fresh_recommendations.py...")
    result = subprocess.run(
        [sys.executable, "calculate_fresh_recommendations.py"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("Warning: calculate_fresh_recommendations.py had issues")
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    result = subprocess.run(
        [sys.executable, "generate_report.py"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("Warning: generate_report.py had issues")


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
    parser.add_argument("--no-advanced", action="store_true", help="Skip advanced features integration")
    parser.add_argument("--no-dixon-coles", action="store_true", help="Skip Dixon-Coles model training")
    parser.add_argument("--no-gradient", action="store_true", help="Skip Gradient Boosting model training")
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
        
        # Step 2c: Integrate advanced features (FBref, weather, injuries, manager)
        if not args.no_train and not args.no_advanced:
            run_step("Integrate Advanced Features", step2c_integrate_advanced_features)
        elif args.no_advanced:
            print("\n[Skipping advanced features integration]")
        
        # Step 3: Train models
        if not args.no_train:
            run_step("Train Poisson Models", step3_train_models)
        else:
            print("\n[Skipping model training]")
        
        # Step 3b: Train Dixon-Coles model
        if not args.no_train and not args.no_dixon_coles:
            run_step("Train Dixon-Coles Model", step3b_train_dixon_coles)
        elif args.no_dixon_coles:
            print("\n[Skipping Dixon-Coles training]")
        
        # Step 3c: Train Gradient Boosting models
        if not args.no_train and not args.no_gradient:
            run_step("Train Gradient Boosting Models", step3c_train_gradient_boosting)
        elif args.no_gradient:
            print("\n[Skipping Gradient Boosting training]")
        
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
