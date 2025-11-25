"""
Run European Competition Model Pipeline

Usage:
    python run_european.py fetch      # Download from API-Football (2021-2023, requires API key)
    python run_european.py scrape     # Scrape from FBref (2021-2026, FREE, no API key needed)
    python run_european.py prepare    # Prepare dataset
    python run_european.py train      # Train model
    python run_european.py predict    # Score upcoming matches
    python run_european.py all        # Run full pipeline (using FBref)

Data Sources:
    - FBref.com (FREE): Current season data, no API key required
    - API-Football: Requires API key, free tier limited to 2021-2023

Recommended: Use 'scrape' for free current season data
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "fetch":
        print("Fetching from API-Football (requires API key)...")
        from src.european_fetch import main as fetch_main
        fetch_main()
    
    elif command == "scrape":
        print("Scraping European data from free sources...")
        from src.european_scraper import main as scrape_main
        scrape_main()
        
    elif command == "prepare":
        from src.european_prepare import main as prepare_main
        prepare_main()
        
    elif command == "train":
        from src.european_models import main as train_main
        train_main()
        
    elif command == "predict":
        from src.european_upcoming import main as predict_main
        predict_main()
        
    elif command == "all":
        print("=== EUROPEAN MODEL PIPELINE ===\n")
        
        print("Step 1: Scraping data from free sources...")
        from src.european_scraper import main as scrape_main
        scrape_main()
        
        print("\nStep 2: Preparing dataset...")
        from src.european_prepare import main as prepare_main
        prepare_main()
        
        print("\nStep 3: Training model...")
        from src.european_models import main as train_main
        train_main()
        
        print("\n=== PIPELINE COMPLETE ===")
        
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
