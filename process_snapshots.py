"""Process all saved snapshots and extract odds to CSV files."""

from pathlib import Path
from src.norsk_tipping_scraper import parse_snapshot_to_csv

def process_all_snapshots():
    """Process all snapshot files in data/snapshots/."""
    snapshot_dir = Path("data/snapshots")
    output_dir = Path("data/processed")
    
    # Find all snapshot files
    snapshot_files = list(snapshot_dir.glob("*_snapshot.txt"))
    
    if not snapshot_files:
        print("No snapshot files found in data/snapshots/")
        return
    
    print(f"Found {len(snapshot_files)} snapshot files\n")
    
    all_dfs = []
    
    for snapshot_file in snapshot_files:
        country = snapshot_file.stem.replace("_snapshot", "")
        output_file = output_dir / f"{country}_odds.csv"
        
        print(f"Processing {snapshot_file.name}...")
        
        # Read snapshot text
        snapshot_text = snapshot_file.read_text(encoding="utf-8")
        
        # Parse and save
        df = parse_snapshot_to_csv(snapshot_text, output_file)
        all_dfs.append(df)
        print(f"  → Saved {len(df)} matches to {output_file}\n")
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_file = output_dir / "all_leagues_odds.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"✓ Combined {len(combined_df)} total matches → {combined_file}")

if __name__ == "__main__":
    import pandas as pd
    process_all_snapshots()
