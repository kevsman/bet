#!/usr/bin/env python
"""
Advanced Features Pipeline

Master script to scrape all advanced features and integrate them into the dataset.

Usage:
    # Scrape all advanced features
    python run_advanced_features.py --scrape
    
    # Integrate scraped features into dataset
    python run_advanced_features.py --integrate
    
    # Both scrape and integrate
    python run_advanced_features.py --all
    
    # Scrape specific features only
    python run_advanced_features.py --scrape --fbref-only
    python run_advanced_features.py --scrape --weather-only
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config


def scrape_advanced_features(
    fbref: bool = True,
    weather: bool = True,
    injuries: bool = True,
    manager: bool = True
) -> None:
    """Run the advanced features scrapers."""
    from src.advanced_features_scraper import scrape_all_advanced_features
    
    cfg = get_config()
    
    print("\n" + "="*60)
    print("RUNNING ADVANCED FEATURES SCRAPERS")
    print("="*60)
    print(f"Output directory: {cfg.processed_dir}")
    print(f"Features to scrape:")
    print(f"  - FBref stats: {fbref}")
    print(f"  - Weather: {weather}")
    print(f"  - Injuries: {injuries}")
    print(f"  - Manager: {manager}")
    print("="*60 + "\n")
    
    scrape_all_advanced_features(
        output_dir=cfg.processed_dir,
        include_fbref=fbref,
        include_weather=weather,
        include_injuries=injuries,
        include_manager=manager
    )


def integrate_features() -> None:
    """Integrate scraped features into the match dataset."""
    from src.integrate_advanced_features import create_enhanced_dataset
    
    print("\n" + "="*60)
    print("INTEGRATING ADVANCED FEATURES")
    print("="*60 + "\n")
    
    enhanced_df = create_enhanced_dataset()
    
    print(f"\nDataset enhanced successfully!")
    print(f"Total columns: {len(enhanced_df.columns)}")
    print(f"Total matches: {len(enhanced_df)}")
    
    # Show sample of new features
    new_feature_prefixes = [
        "home_adv_", "away_adv_",  # Advanced stats
        "weather_",                 # Weather
        "home_injury", "away_injury", "home_suspended", "away_suspended",  # Injuries
        "home_manager", "away_manager", "manager_tenure"  # Manager
    ]
    
    new_features = [
        col for col in enhanced_df.columns 
        if any(col.startswith(prefix) for prefix in new_feature_prefixes)
    ]
    
    if new_features:
        print(f"\nNew feature columns added ({len(new_features)}):")
        for feat in sorted(new_features)[:20]:
            print(f"  - {feat}")
        if len(new_features) > 20:
            print(f"  ... and {len(new_features) - 20} more")


def show_status() -> None:
    """Show status of scraped data files."""
    cfg = get_config()
    
    print("\n" + "="*60)
    print("ADVANCED FEATURES STATUS")
    print("="*60 + "\n")
    
    files = [
        ("FBref Advanced Stats", "advanced_team_stats.csv"),
        ("Weather Data", "weather_data.csv"),
        ("Injury Data", "injury_data.csv"),
        ("Manager Data", "manager_data.csv"),
        ("Enhanced Dataset", "match_dataset_enhanced.csv"),
    ]
    
    for name, filename in files:
        path = cfg.processed_dir / filename
        if path.exists():
            import pandas as pd
            df = pd.read_csv(path)
            print(f"✓ {name}: {len(df)} records ({filename})")
        else:
            print(f"✗ {name}: Not found ({filename})")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Features Pipeline for Betting Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_advanced_features.py --all       # Scrape and integrate all features
  python run_advanced_features.py --scrape    # Only scrape, don't integrate
  python run_advanced_features.py --integrate # Only integrate existing data
  python run_advanced_features.py --status    # Show status of data files
  
  # Selective scraping:
  python run_advanced_features.py --scrape --fbref-only
  python run_advanced_features.py --scrape --weather-only
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all features and integrate into dataset"
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape advanced features from external sources"
    )
    parser.add_argument(
        "--integrate",
        action="store_true",
        help="Integrate scraped features into match dataset"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status of scraped data files"
    )
    
    # Selective scraping options
    parser.add_argument(
        "--fbref-only",
        action="store_true",
        help="Only scrape FBref advanced stats"
    )
    parser.add_argument(
        "--weather-only",
        action="store_true",
        help="Only scrape weather data"
    )
    parser.add_argument(
        "--injuries-only",
        action="store_true",
        help="Only scrape injury data"
    )
    parser.add_argument(
        "--manager-only",
        action="store_true",
        help="Only scrape manager data"
    )
    
    args = parser.parse_args()
    
    # Default to showing status if no action specified
    if not any([args.all, args.scrape, args.integrate, args.status]):
        show_status()
        print("Use --help to see available options")
        return
    
    if args.status:
        show_status()
        return
    
    # Determine what to scrape
    selective = args.fbref_only or args.weather_only or args.injuries_only or args.manager_only
    
    if args.all or args.scrape:
        if selective:
            scrape_advanced_features(
                fbref=args.fbref_only,
                weather=args.weather_only,
                injuries=args.injuries_only,
                manager=args.manager_only
            )
        else:
            scrape_advanced_features()
    
    if args.all or args.integrate:
        integrate_features()
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
