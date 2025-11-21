"""
Scraper for Norsk Tipping Oddsen - automated odds extraction.

This script parses the page snapshot from Chrome DevTools MCP to extract
over/under odds for 1.5, 2.5, and 3.5 goal lines from the league view page.

USAGE:
1. Open Chrome via MCP and navigate to a league page (e.g., Premier League)
2. Use the MCP to select 3 dropdowns: Over/Under 1.5, 2.5, and 3.5
3. Take a snapshot and pass the content to this parser
4. Outputs CSV with all odds data
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import get_config
from .team_mappings import normalize_team_name


def extract_text_from_uid(uid_line: str) -> str:
    """
    Extract clean text from UID line.
    
    Example:
        'uid=25_192 heading "Burnley" level="2"' -> 'Burnley'
        'uid=25_194 StaticText "England - Premier League"' -> 'England - Premier League'
    """
    # Match text within quotes
    match = re.search(r'"([^"]+)"', uid_line)
    if match:
        return match.group(1)
    return uid_line.strip()


@dataclass
class MatchOdds:
    """Represents a football match with over/under odds."""

    date: str
    time: str
    home_team: str
    away_team: str
    league: str
    tv_channel: str = ""
    over_1_5: float | None = None
    under_1_5: float | None = None
    over_2_5: float | None = None
    under_2_5: float | None = None
    over_3_5: float | None = None
    under_3_5: float | None = None


class SnapshotParser:
    """
    Parses Chrome DevTools MCP snapshots to extract match odds.
    
    The parser expects a page with 3 dropdown markets selected:
    - Totalt antall mål - over/under 1.5
    - Totalt antall mål - over/under 2.5
    - Totalt antall mål - over/under 3.5
    
    Each match row will have 6 odds buttons:
    - Over 1.5, Under 1.5, Over 2.5, Under 2.5, Over 3.5, Under 3.5
    """

    @staticmethod
    def parse_odds_from_text(text: str) -> float | None:
        """Extract odds value from button text like 'Over 2.5, odds 1.75'."""
        match = re.search(r"odds\s+([\d.]+)", text)
        return float(match.group(1)) if match else None

    @staticmethod
    def parse_date(date_str: str) -> str:
        """
        Convert Norwegian date format to YYYY-MM-DD.
        Example: 'Lør. 22/11 13:30' -> '2025-11-22'
        """
        # Extract day/month
        match = re.search(r"(\d{1,2})/(\d{1,2})", date_str)
        if not match:
            return ""
        
        day, month = match.groups()
        # Assume current year (or next year if month has passed)
        year = 2025  # Could be made smarter
        return f"{year}-{int(month):02d}-{int(day):02d}"

    @staticmethod
    def parse_time(date_str: str) -> str:
        """Extract time from date string. Example: 'Lør. 22/11 13:30' -> '13:30'."""
        match = re.search(r"(\d{1,2}:\d{2})", date_str)
        return match.group(1) if match else ""

    def parse_snapshot_lines(self, lines: list[str]) -> list[MatchOdds]:
        """
        Parse snapshot text lines to extract match odds.
        
        Expected pattern per match:
        - Date/time line (e.g., "Lør. 22/11 13:30")
        - Home team heading
        - Away team heading
        - League name (e.g., "England - Premier League")
        - Optional TV channel
        - "TOTALT ANTALL MÅL - OVER/UNDER 1.5"
        - "Over 1.5" / button with odds
        - "Under 1.5" / button with odds
        - "TOTALT ANTALL MÅL - OVER/UNDER 2.5"
        - "Over 2.5" / button with odds
        - "Under 2.5" / button with odds
        - "TOTALT ANTALL MÅL - OVER/UNDER 3.5"
        - "Over 3.5" / button with odds
        - "Under 3.5" / button with odds
        """
        matches: list[MatchOdds] = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for date pattern (e.g., "Lør. 22/11 13:30")
            if re.search(r"\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}", line):
                try:
                    match_data = MatchOdds(
                        date=self.parse_date(line),
                        time=self.parse_time(line),
                        home_team="",
                        away_team="",
                        league="",
                    )
                    
                    # Next line should be home team (heading level 2)
                    i += 1
                    if i < len(lines) and lines[i].strip():
                        raw_home = extract_text_from_uid(lines[i].strip())
                        match_data.home_team = normalize_team_name(raw_home)
                    
                    # Next line should be away team
                    i += 1
                    if i < len(lines) and lines[i].strip():
                        raw_away = extract_text_from_uid(lines[i].strip())
                        match_data.away_team = normalize_team_name(raw_away)
                    
                    # Next line should be league
                    i += 1
                    if i < len(lines) and lines[i].strip():
                        match_data.league = extract_text_from_uid(lines[i].strip())
                    
                    # Skip empty line and potential TV channel
                    i += 1
                    while i < len(lines) and (not lines[i].strip() or "TOTALT" not in lines[i]):
                        if lines[i].strip() and not lines[i].strip().startswith("uid="):
                            match_data.tv_channel = lines[i].strip()
                        i += 1
                    
                    # Now parse the 3 over/under markets
                    # Pattern: Market header, "Over X.5", odds button, "Under X.5", odds button
                    
                    # Over/Under 1.5
                    if i < len(lines) and "1.5" in lines[i]:
                        i += 2  # Skip to "Over 1.5" button
                        if i < len(lines):
                            match_data.over_1_5 = self.parse_odds_from_text(lines[i])
                        i += 2  # Skip to "Under 1.5" button
                        if i < len(lines):
                            match_data.under_1_5 = self.parse_odds_from_text(lines[i])
                        i += 1
                    
                    # Over/Under 2.5
                    if i < len(lines) and "2.5" in lines[i]:
                        i += 2  # Skip to "Over 2.5" button
                        if i < len(lines):
                            match_data.over_2_5 = self.parse_odds_from_text(lines[i])
                        i += 2  # Skip to "Under 2.5" button
                        if i < len(lines):
                            match_data.under_2_5 = self.parse_odds_from_text(lines[i])
                        i += 1
                    
                    # Over/Under 3.5
                    if i < len(lines) and "3.5" in lines[i]:
                        i += 2  # Skip to "Over 3.5" button
                        if i < len(lines):
                            match_data.over_3_5 = self.parse_odds_from_text(lines[i])
                        i += 2  # Skip to "Under 3.5" button
                        if i < len(lines):
                            match_data.under_3_5 = self.parse_odds_from_text(lines[i])
                        i += 1
                    
                    matches.append(match_data)
                    
                except Exception as e:
                    print(f"Error parsing match at line {i}: {e}")
                    i += 1
            else:
                i += 1
        
        return matches


def parse_snapshot_to_csv(snapshot_text: str, output_path: Path | None = None) -> pd.DataFrame:
    """
    Parse MCP snapshot text and save to CSV.
    
    Args:
        snapshot_text: Raw text from MCP take_snapshot() output
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with match odds
    """
    cfg = get_config()
    parser = SnapshotParser()
    
    # Split into lines for easier parsing
    lines = snapshot_text.split("\n")
    
    # Parse matches
    matches = parser.parse_snapshot_lines(lines)
    
    # Convert to DataFrame
    df = pd.DataFrame([vars(m) for m in matches])
    
    # Save to CSV
    if output_path is None:
        output_path = cfg.data_dir / "upcoming" / "norsk_tipping_odds.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nExtracted {len(matches)} matches")
    print(f"Saved to {output_path}")
    
    return df


def main() -> None:
    """
    Example usage - paste your MCP snapshot text here.
    """
    # This would normally come from the MCP snapshot
    example_snapshot = """
    Lør. 22/11 13:30
    Burnley
    Chelsea
    England - Premier League
    TV3+
    TOTALT ANTALL MÅL - OVER/UNDER 1.5
    Over 1.5
    Over 1.5, odds 1.23
    Under 1.5
    Under 1.5, odds 3.90
    TOTALT ANTALL MÅL - OVER/UNDER 2.5
    Over 2.5
    Over 2.5, odds 1.75
    Under 2.5
    Under 2.5, odds 2.00
    TOTALT ANTALL MÅL - OVER/UNDER 3.5
    Over 3.5
    Over 3.5, odds 2.90
    Under 3.5
    Under 3.5, odds 1.38
    """
    
    df = parse_snapshot_to_csv(example_snapshot)
    print("\nSample data:")
    print(df)


if __name__ == "__main__":
    main()
