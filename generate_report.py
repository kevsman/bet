"""
Unified HTML report generator for betting recommendations.

Handles both domestic and European competition predictions with a consistent
modern dark theme design.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd


def load_and_normalize_data(
    domestic_path: Path = Path("data/processed/recommendations.csv"),
    european_path: Path = Path("data/processed/european/today_predictions.csv"),
    mode: str = "auto"
) -> tuple[pd.DataFrame, str]:
    """
    Load and normalize data from either domestic or European predictions.
    
    Returns a normalized DataFrame with consistent columns and the report type.
    """
    # Determine which file to use
    if mode == "european":
        path = european_path
        report_type = "European Competitions"
    elif mode == "domestic":
        path = domestic_path
        report_type = "Domestic Leagues"
    else:
        # Auto-detect: prefer domestic if it exists and is recent
        if domestic_path.exists():
            path = domestic_path
            report_type = "Domestic Leagues"
        elif european_path.exists():
            path = european_path
            report_type = "European Competitions"
        else:
            return pd.DataFrame(), "Unknown"
    
    if not path.exists():
        print(f"Data file not found: {path}")
        return pd.DataFrame(), report_type
    
    df = pd.read_csv(path)
    
    if df.empty:
        return df, report_type
    
    # Normalize column names based on source
    if "edge" in df.columns:
        # Domestic format - already has most columns
        normalized = pd.DataFrame({
            "date": df.get("date", ""),
            "time": df.get("time", ""),
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "league": df.get("league_code", df.get("league", "Unknown")),
            "bet_type": df["bet"].apply(lambda x: "Over" if "over" in str(x).lower() else "Under"),
            "line": df["line"],
            "odds": df["odds"],
            "probability": df["probability"],
            "poisson_prob": df.get("poisson_prob", df["probability"]),
            "dc_prob": df.get("dc_prob", df["probability"]),
            "ev": df["edge"],  # edge is EV in domestic format
            "stake": df.get("kelly", df.get("stake_fraction", 0)),
            "pred_total": df["model_total"],
            "pred_home": df.get("pred_home", 0),
            "pred_away": df.get("pred_away", 0),
            "dc_home": df.get("dc_home", 0),
            "dc_away": df.get("dc_away", 0),
            "model_used": df.get("model_used", "ensemble"),
            "data_source": df.get("data_source", "Domestic"),
        })
    elif "over_ev" in df.columns:
        # European format - needs transformation
        # Calculate best bet for each match
        df["best_ev"] = df[["over_ev", "under_ev"]].max(axis=1)
        df["bet_type"] = df.apply(
            lambda r: "Over" if r["over_ev"] > r["under_ev"] else "Under", axis=1
        )
        df["best_odds"] = df.apply(
            lambda r: r["over_odds"] if r["over_ev"] > r["under_ev"] else r["under_odds"], axis=1
        )
        df["best_prob"] = df.apply(
            lambda r: r["over_prob"] if r["over_ev"] > r["under_ev"] else r["under_prob"], axis=1
        )
        
        # Calculate Kelly stake
        df["kelly"] = df.apply(
            lambda r: max(0, min(0.05, 0.25 * (r["best_prob"] - (1 - r["best_prob"]) / (r["best_odds"] - 1)) if r["best_odds"] > 1 else 0)),
            axis=1
        )
        
        # Filter for positive EV
        df = df[df["best_ev"] > 0].copy()
        
        normalized = pd.DataFrame({
            "date": "",  # European doesn't have date
            "time": "",
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "league": df.get("league", "European"),
            "bet_type": df["bet_type"],
            "line": 2.5,  # European uses 2.5 line
            "odds": df["best_odds"],
            "probability": df["best_prob"],
            "poisson_prob": df.apply(lambda r: r["over_prob"] if r["bet_type"] == "Over" else r["under_prob"], axis=1),
            "dc_prob": df["best_prob"],  # Same for now
            "ev": df["best_ev"],
            "stake": df["kelly"],
            "pred_total": df["pred_total"],
            "pred_home": df["pred_home"],
            "pred_away": df["pred_away"],
            "dc_home": df.get("dc_home", df["pred_home"]),
            "dc_away": df.get("dc_away", df["pred_away"]),
            "model_used": "ensemble",
            "data_source": df.get("data_source", "European"),
        })
    else:
        print(f"Unknown data format in {path}")
        return pd.DataFrame(), report_type
    
    # Sort by EV descending
    normalized = normalized.sort_values("ev", ascending=False).reset_index(drop=True)
    
    return normalized, report_type


def generate_html_report(
    domestic_path: Path = Path("data/processed/recommendations.csv"),
    european_path: Path = Path("data/processed/european/today_predictions.csv"),
    output_path: Path = Path("betting_report.html"),
    mode: str = "auto"
) -> None:
    """
    Generate unified HTML report for betting recommendations.
    
    Args:
        domestic_path: Path to domestic recommendations CSV
        european_path: Path to European predictions CSV
        output_path: Where to save the HTML report
        mode: "domestic", "european", or "auto" (auto-detect)
    """
    df, report_type = load_and_normalize_data(domestic_path, european_path, mode)
    
    if df.empty:
        print("No recommendations found.")
        return
    
    # Load all predictions for high-confidence section (regardless of edge)
    all_predictions_path = Path("data/processed/all_predictions.csv")
    all_preds_df = None
    if all_predictions_path.exists():
        all_preds_df = pd.read_csv(all_predictions_path)
        all_preds_df = all_preds_df.sort_values("probability", ascending=False)
    
    # Calculate summary stats
    total_stake = df["stake"].sum() * 100
    avg_ev = df["ev"].mean() * 100
    best_ev = df["ev"].max() * 100
    avg_odds = df["odds"].mean()
    avg_prob = df["probability"].mean() * 100
    
    # Count by bet type
    over_count = len(df[df["bet_type"] == "Over"])
    under_count = len(df[df["bet_type"] == "Under"])
    
    # Get highest probability bets (most confident predictions)
    high_prob_df = df.sort_values("probability", ascending=False)
    
    # Filter to optimal calibration range (35-75% where model is most accurate)
    optimal_range_df = df[(df["probability"] >= 0.35) & (df["probability"] <= 0.75)].copy()
    optimal_count = len(optimal_range_df)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Value Bets - {report_type}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #e0e0e0;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #0f0f23;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            overflow: hidden;
            border: 1px solid #2d2d44;
        }}
        
        .header {{
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            color: white;
            padding: 40px;
            text-align: center;
            border-bottom: 2px solid #e94560;
        }}
        
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
            color: #e94560;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            color: #4ecca3;
            margin-bottom: 5px;
        }}
        
        .header .model-info {{
            font-size: 0.95em;
            color: #a0a0b0;
        }}
        
        .header .timestamp {{
            margin-top: 15px;
            font-size: 0.85em;
            color: #666;
        }}
        
        .summary {{
            background: #1a1a2e;
            padding: 25px 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            border-bottom: 1px solid #2d2d44;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            padding: 18px 15px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #2d2d44;
        }}
        
        .summary-card .label {{
            color: #888;
            font-size: 0.8em;
            margin-bottom: 5px;
            text-transform: uppercase;
        }}
        
        .summary-card .value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .summary-card .value.green {{ color: #4ecca3; }}
        .summary-card .value.red {{ color: #e94560; }}
        .summary-card .value.yellow {{ color: #fbbf24; }}
        .summary-card .value.purple {{ color: #a855f7; }}
        .summary-card .value.blue {{ color: #60a5fa; }}
        
        .bet-list {{
            padding: 25px;
        }}
        
        .section-title {{
            color: #e94560;
            font-size: 1.2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #2d2d44;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .section-title .count {{
            background: #2d2d44;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            color: #a0a0b0;
        }}
        
        /* Compact Bet Table */
        .bet-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: #16213e;
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .bet-table th {{
            background: #0f0f23;
            padding: 12px 10px;
            text-align: left;
            font-size: 0.75em;
            text-transform: uppercase;
            color: #666;
            font-weight: 600;
            border-bottom: 2px solid #2d2d44;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        .bet-table th.sortable {{
            cursor: pointer;
            user-select: none;
            transition: all 0.2s;
        }}
        
        .bet-table th.sortable:hover {{
            color: #4ecca3;
            background: #1a1a2e;
        }}
        
        .bet-table th.sortable.active {{
            color: #4ecca3;
        }}
        
        .bet-table th .sort-icon {{
            display: inline-block;
            margin-left: 4px;
            font-size: 0.8em;
            opacity: 0.5;
        }}
        
        .bet-table th.sortable.active .sort-icon {{
            opacity: 1;
        }}
        
        .bet-table th:first-child {{
            padding-left: 15px;
            width: 40px;
        }}
        
        .bet-table td {{
            padding: 12px 10px;
            border-bottom: 1px solid #2d2d44;
            vertical-align: middle;
        }}
        
        .bet-table tr {{
            transition: background 0.2s;
        }}
        
        .bet-table tbody tr:hover {{
            background: #1e2a47;
        }}
        
        .bet-table tbody tr.selected {{
            background: rgba(78, 204, 163, 0.1);
        }}
        
        .bet-table tbody tr.top-pick {{
            border-left: 3px solid #4ecca3;
        }}
        
        .bet-table tbody tr.strong {{
            border-left: 3px solid #fbbf24;
        }}
        
        .bet-row-checkbox input {{
            display: none;
        }}
        
        .bet-row-checkbox label {{
            width: 22px;
            height: 22px;
            background: #2d2d44;
            border: 2px solid #3d3d54;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            color: transparent;
            font-size: 0.85em;
        }}
        
        .bet-row-checkbox input:checked + label {{
            background: #4ecca3;
            border-color: #4ecca3;
            color: #0f0f23;
        }}
        
        .match-cell {{
            min-width: 180px;
        }}
        
        .match-cell .teams {{
            font-weight: 600;
            color: #fff;
            font-size: 0.95em;
            margin-bottom: 2px;
        }}
        
        .match-cell .meta {{
            font-size: 0.75em;
            color: #666;
        }}
        
        .match-cell .league {{
            color: #a0a0b0;
            background: #2d2d44;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 8px;
        }}
        
        .match-cell .date {{
            color: #4ecca3;
        }}
        
        .bet-type-cell {{
            font-weight: 600;
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 0.85em;
            display: inline-block;
            white-space: nowrap;
        }}
        
        .bet-type-cell.over {{
            background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
            color: white;
        }}
        
        .bet-type-cell.under {{
            background: linear-gradient(135deg, #4ecca3 0%, #45b7a0 100%);
            color: white;
        }}
        
        .odds-cell {{
            font-weight: bold;
            color: #fbbf24;
            font-size: 1em;
        }}
        
        .ev-cell {{
            font-weight: bold;
            color: #4ecca3;
            font-size: 1em;
        }}
        
        .edge-cell {{
            color: #60a5fa;
            font-size: 0.95em;
        }}
        
        .prob-cell {{
            color: #a855f7;
            font-size: 0.95em;
        }}
        
        .stake-cell {{
            color: #e94560;
            font-size: 0.95em;
        }}
        
        .goals-cell {{
            color: #fbbf24;
            font-weight: bold;
            font-size: 0.95em;
        }}
        
        .models-cell {{
            font-size: 0.8em;
            min-width: 140px;
        }}
        
        .model-mini {{
            display: flex;
            gap: 3px;
            align-items: center;
            margin-bottom: 2px;
        }}
        
        .model-mini:last-child {{
            margin-bottom: 0;
        }}
        
        .model-label {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            font-size: 0.65em;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .model-label.poisson {{
            background: #60a5fa;
            color: #0f0f23;
        }}
        
        .model-label.dc {{
            background: #a855f7;
            color: #0f0f23;
        }}
        
        .model-mini .model-prob {{
            color: #a0a0b0;
        }}
        
        .model-mini .model-score {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .badge-mini {{
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.65em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .badge-mini.top {{ background: #4ecca3; color: #0f0f23; }}
        .badge-mini.strong {{ background: #fbbf24; color: #0f0f23; }}
        
        /* Legacy styles for Top 5 cards (keeping card format there) */
        .bet-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border: 1px solid #2d2d44;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }}
        
        .bet-card:hover {{
            border-color: #e94560;
            box-shadow: 0 5px 20px rgba(233, 69, 96, 0.15);
            transform: translateY(-2px);
        }}
        
        .bet-card.top-pick {{
            border-left: 4px solid #4ecca3;
        }}
        
        .bet-card.strong {{
            border-left: 4px solid #fbbf24;
        }}
        
        .badge {{
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .badge.top {{ background: #4ecca3; color: #0f0f23; }}
        .badge.strong {{ background: #fbbf24; color: #0f0f23; }}
        
        .bet-type {{
            font-size: 1.1em;
            font-weight: bold;
            padding: 10px 18px;
            border-radius: 8px;
            color: white;
        }}
        
        .bet-type.over {{
            background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
        }}
        
        .bet-type.under {{
            background: linear-gradient(135deg, #4ecca3 0%, #45b7a0 100%);
        }}
        
        .bet-odds {{
            font-size: 1.3em;
            font-weight: bold;
            color: #fff;
        }}
        
        .bet-odds span {{
            color: #888;
            font-size: 0.7em;
            font-weight: normal;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
        }}
        
        .metric {{
            background: #0f0f23;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #2d2d44;
        }}
        
        .metric .label {{
            color: #666;
            font-size: 0.7em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        
        .metric .value {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .metric .value.green {{ color: #4ecca3; }}
        .metric .value.red {{ color: #e94560; }}
        .metric .value.yellow {{ color: #fbbf24; }}
        .metric .value.purple {{ color: #a855f7; }}
        .metric .value.blue {{ color: #60a5fa; }}
        
        .metric .sub {{
            color: #555;
            font-size: 0.7em;
            margin-top: 2px;
        }}
        
        .model-comparison {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #2d2d44;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .model-box {{
            background: #0f0f23;
            border: 1px solid #2d2d44;
            border-radius: 8px;
            padding: 12px 15px;
            min-width: 160px;
        }}
        
        .model-box.poisson {{
            border-left: 3px solid #60a5fa;
        }}
        
        .model-box.dixon-coles {{
            border-left: 3px solid #a855f7;
        }}
        
        .model-title {{
            font-size: 0.75em;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 8px;
            font-weight: bold;
        }}
        
        .model-box.poisson .model-title {{ color: #60a5fa; }}
        .model-box.dixon-coles .model-title {{ color: #a855f7; }}
        
        .model-stats {{
            display: flex;
            gap: 15px;
        }}
        
        .model-stat {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        
        .stat-label {{
            font-size: 0.65em;
            color: #555;
            text-transform: uppercase;
        }}
        
        .stat-value {{
            font-size: 1em;
            font-weight: bold;
            color: #e0e0e0;
        }}
        
        .model-source {{
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.8em;
        }}
        
        .source-label {{
            color: #555;
        }}
        
        .source-value {{
            color: #4ecca3;
            font-weight: 500;
        }}
        
        .footer {{
            background: #0f0f23;
            padding: 20px;
            text-align: center;
            color: #555;
            font-size: 0.85em;
            border-top: 1px solid #2d2d44;
        }}
        
        .footer a {{
            color: #4ecca3;
            text-decoration: none;
        }}
        
        /* Top 5 Section */
        .top-bets {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            padding: 25px;
            border-bottom: 1px solid #2d2d44;
        }}
        
        .top-bets-title {{
            color: #e94560;
            font-size: 1.2em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .top-bets-title .icon {{
            font-size: 1.3em;
        }}
        
        .top-bets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}
        
        .top-bet-card {{
            background: #0f0f23;
            border: 1px solid #2d2d44;
            border-radius: 10px;
            padding: 15px;
            position: relative;
        }}
        
        .top-bet-card .rank {{
            position: absolute;
            top: -8px;
            left: -8px;
            background: #e94560;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.85em;
        }}
        
        .top-bet-card .teams {{
            font-weight: bold;
            color: #fff;
            margin-bottom: 8px;
            font-size: 0.95em;
        }}
        
        .top-bet-card .bet-info {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .top-bet-card .bet-label {{
            font-size: 0.85em;
            padding: 4px 10px;
            border-radius: 5px;
            color: white;
        }}
        
        .top-bet-card .bet-label.over {{ background: #e94560; }}
        .top-bet-card .bet-label.under {{ background: #4ecca3; }}
        
        .top-bet-card .odds-display {{
            color: #fbbf24;
            font-weight: bold;
        }}
        
        .top-bet-card .ev-display {{
            color: #4ecca3;
            font-size: 1.1em;
            font-weight: bold;
        }}
        
        /* Calculator Section */
        .calculator {{
            background: #1a1a2e;
            padding: 25px;
            border-bottom: 1px solid #2d2d44;
        }}
        
        .calculator-title {{
            color: #4ecca3;
            font-size: 1.2em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .calculator-controls {{
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        .bankroll-input {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .bankroll-input label {{
            color: #a0a0b0;
            font-size: 0.9em;
        }}
        
        .bankroll-input input {{
            background: #0f0f23;
            border: 1px solid #2d2d44;
            border-radius: 8px;
            padding: 10px 15px;
            color: #fff;
            font-size: 1.1em;
            width: 150px;
            text-align: right;
        }}
        
        .bankroll-input input:focus {{
            outline: none;
            border-color: #4ecca3;
        }}
        
        .staking-method {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .staking-method label {{
            color: #a0a0b0;
            font-size: 0.9em;
        }}
        
        .staking-method select {{
            background: #0f0f23;
            border: 1px solid #2d2d44;
            border-radius: 8px;
            padding: 10px 15px;
            color: #fff;
            font-size: 0.95em;
            cursor: pointer;
        }}
        
        .staking-method select:focus {{
            outline: none;
            border-color: #4ecca3;
        }}
        
        .prob-filter {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .prob-filter label {{
            color: #a0a0b0;
            font-size: 0.9em;
        }}
        
        .prob-filter select {{
            background: #0f0f23;
            border: 1px solid #2d2d44;
            border-radius: 8px;
            padding: 10px 15px;
            color: #fff;
            font-size: 0.95em;
            cursor: pointer;
        }}
        
        .prob-filter select:focus {{
            outline: none;
            border-color: #4ecca3;
        }}
        
        .calc-buttons {{
            display: flex;
            gap: 10px;
        }}
        
        .calc-btn {{
            background: #2d2d44;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            color: #a0a0b0;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        }}
        
        .calc-btn:hover {{
            background: #3d3d54;
            color: #fff;
        }}
        
        .calc-btn.primary {{
            background: #4ecca3;
            color: #0f0f23;
        }}
        
        .calc-btn.primary:hover {{
            background: #5fd9b0;
        }}
        
        .calc-results {{
            background: #0f0f23;
            border-radius: 10px;
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .calc-result {{
            text-align: center;
        }}
        
        .calc-result .label {{
            color: #666;
            font-size: 0.8em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .calc-result .value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .calc-result .value.green {{ color: #4ecca3; }}
        .calc-result .value.red {{ color: #e94560; }}
        .calc-result .value.yellow {{ color: #fbbf24; }}
        
        /* Bet Breakdown */
        .bet-breakdown {{
            margin-top: 20px;
            background: #0f0f23;
            border-radius: 10px;
            padding: 15px;
            display: none;
        }}
        
        .bet-breakdown.visible {{
            display: block;
        }}
        
        .bet-breakdown-title {{
            color: #a0a0b0;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2d2d44;
        }}
        
        .bet-breakdown-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .bet-breakdown-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: #16213e;
            border-radius: 6px;
            font-size: 0.9em;
        }}
        
        .bet-breakdown-item .bet-info {{
            color: #a0a0b0;
        }}
        
        .bet-breakdown-item .bet-match {{
            color: #fff;
            font-weight: 500;
        }}
        
        .bet-breakdown-item .bet-amount {{
            color: #4ecca3;
            font-weight: bold;
        }}
        
        /* Bet Selection */
        .bet-card {{
            position: relative;
        }}
        
        .bet-checkbox {{
            position: absolute;
            top: 15px;
            right: 15px;
        }}
        
        .bet-checkbox input {{
            display: none;
        }}
        
        .bet-checkbox label {{
            width: 28px;
            height: 28px;
            background: #2d2d44;
            border: 2px solid #3d3d54;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            color: transparent;
            font-size: 1em;
        }}
        
        .bet-checkbox input:checked + label {{
            background: #4ecca3;
            border-color: #4ecca3;
            color: #0f0f23;
        }}
        
        .bet-checkbox label:hover {{
            border-color: #4ecca3;
        }}
        
        .bet-card.selected {{
            border-color: #4ecca3;
            box-shadow: 0 0 15px rgba(78, 204, 163, 0.2);
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.6em; }}
            .match-teams {{ font-size: 1.1em; }}
            .summary {{ grid-template-columns: repeat(2, 1fr); }}
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .top-bets-grid {{ grid-template-columns: 1fr; }}
            .calculator-controls {{ flex-direction: column; align-items: stretch; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Value Bets Report</h1>
            <div class="subtitle">{report_type}</div>
            <div class="model-info">Dixon-Coles + Poisson Ensemble Model</div>
            <div class="timestamp">Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}</div>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <div class="label">Value Bets</div>
                <div class="value green">{len(df)}</div>
            </div>
            <div class="summary-card">
                <div class="label">Total Stake</div>
                <div class="value red">{total_stake:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg EV</div>
                <div class="value green">+{avg_ev:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="label">Best EV</div>
                <div class="value green">+{best_ev:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Odds</div>
                <div class="value yellow">{avg_odds:.2f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Over / Under</div>
                <div class="value blue">{over_count} / {under_count}</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Prob</div>
                <div class="value purple">{avg_prob:.0f}%</div>
            </div>
            <div class="summary-card">
                <div class="label">Optimal Range</div>
                <div class="value green">{optimal_count}</div>
            </div>
        </div>
"""

    # Add Model Insights section based on calibration analysis
    html += """
        <div class="top-bets" style="background: linear-gradient(135deg, #1a2e1a 0%, #1a1a2e 100%); border-left: 3px solid #4ecca3;">
            <div class="top-bets-title" style="color: #4ecca3;">
                <span class="icon">&#128202;</span>
                <span>Model Calibration Insights</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; color: #a0a0b0; font-size: 0.9em;">
                <div style="background: #0f0f23; padding: 15px; border-radius: 8px; border-left: 3px solid #60a5fa;">
                    <div style="color: #60a5fa; font-weight: bold; margin-bottom: 8px;">Poisson Model (Primary)</div>
                    <ul style="margin-left: 18px; line-height: 1.6;">
                        <li>Best calibration: <span style="color: #4ecca3;">1.15% ECE</span> (excellent)</li>
                        <li>Optimal range: <span style="color: #fbbf24;">35-75% probability</span></li>
                        <li>Slight over-prediction bias: +0.08 goals</li>
                    </ul>
                </div>
                <div style="background: #0f0f23; padding: 15px; border-radius: 8px; border-left: 3px solid #a855f7;">
                    <div style="color: #a855f7; font-weight: bold; margin-bottom: 8px;">Dixon-Coles Model</div>
                    <ul style="margin-left: 18px; line-height: 1.6;">
                        <li>Higher ECE: <span style="color: #e94560;">9.84%</span> (needs caution)</li>
                        <li>Under-predicts goals: -0.35 bias</li>
                        <li>Use for secondary confirmation only</li>
                    </ul>
                </div>
                <div style="background: #0f0f23; padding: 15px; border-radius: 8px; border-left: 3px solid #fbbf24;">
                    <div style="color: #fbbf24; font-weight: bold; margin-bottom: 8px;">Edge Analysis (ROI)</div>
                    <ul style="margin-left: 18px; line-height: 1.6;">
                        <li>0-10% edge: <span style="color: #e94560;">Negative ROI</span> (avoid)</li>
                        <li>10-15% edge: <span style="color: #4ecca3;">Break-even to +0.2%</span></li>
                        <li>15%+ edge: <span style="color: #4ecca3;">+47.9% ROI</span> (best)</li>
                    </ul>
                </div>
            </div>
        </div>
"""

    # Add Top 5 bets section
    top_5 = df.head(5)
    html += """
        <div class="top-bets">
            <div class="top-bets-title">
                <span class="icon">&#9733;</span>
                <span>Top 5 Value Bets</span>
            </div>
            <div class="top-bets-grid">
"""
    
    for rank, (idx, row) in enumerate(top_5.iterrows(), 1):
        true_ev = (row["probability"] * row["odds"] - 1) * 100
        bet_class = row["bet_type"].lower()
        html += f"""
                <div class="top-bet-card">
                    <div class="rank">{rank}</div>
                    <div class="teams">{row['home_team']} vs {row['away_team']}</div>
                    <div class="bet-info">
                        <span class="bet-label {bet_class}">{row['bet_type']} {row['line']}</span>
                        <span class="odds-display">@ {row['odds']:.2f}</span>
                    </div>
                    <div class="ev-display">+{true_ev:.1f}% EV</div>
                </div>
"""
    
    html += """
            </div>
        </div>
"""

    # Add Highest Probability section (most confident predictions)
    top_prob = high_prob_df.head(5)
    html += """
        <div class="top-bets" style="border-left: 3px solid #a855f7;">
            <div class="top-bets-title" style="color: #a855f7;">
                <span class="icon">&#127919;</span>
                <span>Highest Probability Bets (Most Confident)</span>
            </div>
            <div class="top-bets-grid">
"""
    
    for rank, (idx, row) in enumerate(top_prob.iterrows(), 1):
        true_ev = (row["probability"] * row["odds"] - 1) * 100
        prob_pct = row["probability"] * 100
        bet_class = row["bet_type"].lower()
        
        # Indicate if in optimal calibration range
        in_optimal = 35 <= prob_pct <= 75
        optimal_badge = '<span style="background: #4ecca3; color: #0f0f23; padding: 2px 6px; border-radius: 4px; font-size: 0.7em; margin-left: 5px;">CALIBRATED</span>' if in_optimal else ''
        
        html += f"""
                <div class="top-bet-card" style="border-left: 3px solid #a855f7;">
                    <div class="rank" style="background: #a855f7;">{rank}</div>
                    <div class="teams">{row['home_team']} vs {row['away_team']}</div>
                    <div class="bet-info">
                        <span class="bet-label {bet_class}">{row['bet_type']} {row['line']}</span>
                        <span class="odds-display">@ {row['odds']:.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="color: #a855f7; font-size: 1.2em; font-weight: bold;">{prob_pct:.0f}% Prob{optimal_badge}</div>
                        <div class="ev-display" style="font-size: 0.9em;">+{true_ev:.1f}% EV</div>
                    </div>
                </div>
"""
    
    html += """
            </div>
        </div>
"""

    # Add High Confidence Bets section (from ALL predictions - no edge requirement)
    if all_preds_df is not None and len(all_preds_df) > 0:
        # Get bets with >65% probability (we'll filter by odds with slider)
        high_conf_df = all_preds_df[
            (all_preds_df['probability'] >= 0.65) & 
            (all_preds_df['odds'] >= 1.10)  # Minimum odds for any display
        ].head(30)  # Get more to allow filtering
        
        if len(high_conf_df) > 0:
            # Build high confidence data for JavaScript
            high_conf_data = []
            for idx, row in high_conf_df.iterrows():
                prob_pct = row["probability"] * 100
                implied = row.get("implied_prob", 1/row["odds"]) * 100
                edge_pct = row.get("edge", row["probability"] - 1/row["odds"]) * 100
                expected_return = (row["probability"] * row["odds"] - 1) * 100
                bet_type = "Over" if "over" in str(row["bet"]).lower() else "Under"
                line = row.get("line", 2.5)
                
                high_conf_data.append({
                    "home": row['home_team'],
                    "away": row['away_team'],
                    "bet_type": bet_type,
                    "line": float(line),
                    "odds": float(row['odds']),
                    "prob": prob_pct,
                    "implied": implied,
                    "edge": edge_pct,
                    "expected_return": expected_return
                })
            
            import json as json_module
            high_conf_json = json_module.dumps(high_conf_data)
            
            html += f"""
        <div class="top-bets" style="border-left: 3px solid #22d3ee;">
            <div class="top-bets-title" style="color: #22d3ee;">
                <span class="icon">&#128176;</span>
                <span>High Confidence Bets (Good Odds + High Probability)</span>
            </div>
            <p style="color: #888; font-size: 0.85em; margin-bottom: 15px; padding: 0 15px;">
                Bets with >65% model probability. Use the slider to set minimum odds threshold.
            </p>
            
            <!-- Odds Slider -->
            <div style="padding: 0 15px 20px 15px;">
                <div style="display: flex; align-items: center; gap: 15px; background: #1a1a2e; padding: 15px; border-radius: 8px;">
                    <label style="color: #22d3ee; font-weight: bold; white-space: nowrap;">Min Odds:</label>
                    <input type="range" id="highConfOddsSlider" min="1.10" max="2.50" step="0.05" value="1.25" 
                           style="flex: 1; accent-color: #22d3ee; cursor: pointer;">
                    <span id="highConfOddsValue" style="color: #fff; font-weight: bold; min-width: 50px; text-align: center; 
                          background: #22d3ee; color: #0f0f23; padding: 5px 10px; border-radius: 5px;">1.25</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 5px 0; color: #666; font-size: 0.8em;">
                    <span>1.10 (safer)</span>
                    <span>Showing <span id="highConfCount" style="color: #22d3ee;">0</span> bets</span>
                    <span>2.50 (higher return)</span>
                </div>
            </div>
            
            <div id="highConfBetsGrid" class="top-bets-grid">
                <!-- Populated by JavaScript -->
            </div>
        </div>
        
        <script>
            const highConfData = {high_conf_json};
            
            function renderHighConfBets(minOdds) {{
                const filtered = highConfData.filter(b => b.odds >= minOdds);
                const container = document.getElementById('highConfBetsGrid');
                const countEl = document.getElementById('highConfCount');
                
                countEl.textContent = filtered.length;
                
                if (filtered.length === 0) {{
                    container.innerHTML = '<div style="color: #888; padding: 20px; text-align: center;">No bets match the current filter. Try lowering the minimum odds.</div>';
                    return;
                }}
                
                let html = '';
                filtered.slice(0, 10).forEach((bet, idx) => {{
                    const edgeColor = bet.edge > 0 ? '#4ecca3' : '#f97316';
                    const edgeStr = bet.edge > 0 ? '+' + bet.edge.toFixed(1) + '%' : bet.edge.toFixed(1) + '%';
                    const returnColor = bet.expected_return > 0 ? '#4ecca3' : '#f97316';
                    const returnStr = (bet.expected_return > 0 ? '+' : '') + bet.expected_return.toFixed(1) + '%';
                    const betClass = bet.bet_type.toLowerCase();
                    
                    html += `
                        <div class="top-bet-card" style="border-left: 3px solid #22d3ee;">
                            <div class="rank" style="background: #22d3ee; color: #0f0f23;">${{idx + 1}}</div>
                            <div class="teams">${{bet.home}} vs ${{bet.away}}</div>
                            <div class="bet-info">
                                <span class="bet-label ${{betClass}}">${{bet.bet_type}} ${{bet.line}}</span>
                                <span class="odds-display">@ ${{bet.odds.toFixed(2)}}</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85em;">
                                <div>
                                    <span style="color: #888;">Model:</span> 
                                    <span style="color: #22d3ee; font-weight: bold;">${{bet.prob.toFixed(0)}}%</span>
                                </div>
                                <div>
                                    <span style="color: #888;">Implied:</span> 
                                    <span style="color: #fff;">${{bet.implied.toFixed(0)}}%</span>
                                </div>
                                <div>
                                    <span style="color: #888;">Edge:</span> 
                                    <span style="color: ${{edgeColor}};">${{edgeStr}}</span>
                                </div>
                                <div>
                                    <span style="color: #888;">E[Return]:</span> 
                                    <span style="color: ${{returnColor}};">${{returnStr}}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }});
                
                container.innerHTML = html;
            }}
            
            // Initialize slider
            const slider = document.getElementById('highConfOddsSlider');
            const valueDisplay = document.getElementById('highConfOddsValue');
            
            slider.addEventListener('input', function() {{
                const val = parseFloat(this.value);
                valueDisplay.textContent = val.toFixed(2);
                renderHighConfBets(val);
            }});
            
            // Initial render
            renderHighConfBets(1.25);
        </script>
"""

    # Add Calculator section
    # Build bet data for JavaScript
    bet_data_js = []
    for idx, row in df.iterrows():
        bet_data_js.append({
            "id": idx,
            "teams": f"{row['home_team']} vs {row['away_team']}",
            "bet": f"{row['bet_type']} {row['line']}",
            "odds": float(row['odds']),
            "stake": float(row['stake']),
            "ev": float(row['ev']),
            "prob": float(row['probability'])
        })
    
    import json
    bet_data_json = json.dumps(bet_data_js)
    
    html += f"""
        <div class="calculator">
            <div class="calculator-title">
                <span>&#128176;</span>
                <span>Bet Calculator</span>
            </div>
            <div class="calculator-controls">
                <div class="bankroll-input">
                    <label for="bankroll">Bankroll (kr):</label>
                    <input type="number" id="bankroll" value="500" min="0" step="100">
                </div>
                <div class="staking-method">
                    <label for="staking">Staking:</label>
                    <select id="staking" onchange="updateCalculator()">
                        <option value="kelly">Kelly (proportional to edge)</option>
                        <option value="fixed">Fixed (equal stakes)</option>
                        <option value="ev-weighted">EV Weighted</option>
                    </select>
                </div>
                <div class="prob-filter">
                    <label for="minProb">Min Prob:</label>
                    <select id="minProb" onchange="applyProbFilter()">
                        <option value="0">All bets</option>
                        <option value="35" selected>≥35% (calibrated range)</option>
                        <option value="50">≥50% (confident)</option>
                        <option value="60">≥60% (high confidence)</option>
                    </select>
                </div>
                <div class="prob-filter">
                    <label for="maxProb">Max Prob:</label>
                    <select id="maxProb" onchange="applyProbFilter()">
                        <option value="100">No limit</option>
                        <option value="75" selected>≤75% (calibrated range)</option>
                        <option value="65">≤65% (moderate)</option>
                    </select>
                </div>
                <div class="calc-buttons">
                    <button class="calc-btn" onclick="selectAll()">Select All</button>
                    <button class="calc-btn" onclick="selectNone()">Clear All</button>
                    <button class="calc-btn" onclick="selectTop5()">Top 5 Only</button>
                </div>
            </div>
            <div class="calc-results">
                <div class="calc-result">
                    <div class="label">Selected Bets</div>
                    <div class="value green" id="selected-count">0</div>
                </div>
                <div class="calc-result">
                    <div class="label">Total to Stake</div>
                    <div class="value red" id="total-stake">0 kr</div>
                </div>
                <div class="calc-result">
                    <div class="label">Expected Return</div>
                    <div class="value yellow" id="expected-return">0 kr</div>
                </div>
                <div class="calc-result">
                    <div class="label">Expected Profit</div>
                    <div class="value green" id="expected-profit">0 kr</div>
                </div>
            </div>
            <div id="bet-breakdown" class="bet-breakdown"></div>
        </div>
        
        <div class="bet-list">
            <div class="section-title">
                <span>All Value Bets</span>
                <span class="count">{len(df)} bets found</span>
            </div>
            
            <table class="bet-table" id="bet-table">
                <thead>
                    <tr>
                        <th></th>
                        <th class="sortable" data-sort="match" onclick="sortTable('match')">Match <span class="sort-icon"></span></th>
                        <th class="sortable" data-sort="bet" onclick="sortTable('bet')">Bet <span class="sort-icon"></span></th>
                        <th class="sortable" data-sort="odds" onclick="sortTable('odds')">Odds <span class="sort-icon"></span></th>
                        <th class="sortable active desc" data-sort="ev" onclick="sortTable('ev')">EV <span class="sort-icon">&#9660;</span></th>
                        <th class="sortable" data-sort="edge" onclick="sortTable('edge')">Edge <span class="sort-icon"></span></th>
                        <th class="sortable" data-sort="prob" onclick="sortTable('prob')">Prob <span class="sort-icon"></span></th>
                        <th class="sortable" data-sort="goals" onclick="sortTable('goals')">Goals <span class="sort-icon"></span></th>
                        <th class="sortable" data-sort="stake" onclick="sortTable('stake')">Stake <span class="sort-icon"></span></th>
                        <th>Models (P / DC)</th>
                    </tr>
                </thead>
                <tbody id="bet-table-body">
"""
    
    # Add each bet as a table row
    for idx, row in df.iterrows():
        ev_pct = row["ev"] * 100  # This is actually edge (prob diff)
        true_ev = (row["probability"] * row["odds"] - 1) * 100  # True EV
        prob_pct = row["probability"] * 100
        stake_pct = row["stake"] * 100
        poisson_prob = row["poisson_prob"] * 100
        dc_prob = row["dc_prob"] * 100
        
        # Determine row class and badge
        if true_ev >= 15:
            row_class = "top-pick"
            badge = '<span class="badge-mini top">Top</span>'
        elif true_ev >= 8:
            row_class = "strong"
            badge = '<span class="badge-mini strong">Strong</span>'
        else:
            row_class = ""
            badge = ""
        
        bet_class = row["bet_type"].lower()
        
        # Format date/time
        date_str = str(row["date"])[:10] if row["date"] else ""
        time_str = str(row["time"])[:5] if row["time"] else ""
        datetime_str = f"{date_str} {time_str}".strip()
        
        # Format predictions
        pred_home = row["pred_home"]
        pred_away = row["pred_away"]
        dc_home = row["dc_home"]
        dc_away = row["dc_away"]
        pred_total = row["pred_total"]
        
        html += f"""
                    <tr class="{row_class}" data-bet-id="{idx}" data-match="{row['home_team']} vs {row['away_team']}" data-bet="{row['bet_type']} {row['line']}" data-odds="{row['odds']}" data-ev="{true_ev}" data-edge="{ev_pct}" data-prob="{prob_pct}" data-goals="{pred_total}" data-stake="{stake_pct}">
                        <td>
                            <div class="bet-row-checkbox">
                                <input type="checkbox" id="bet-{idx}" checked onchange="updateCalculator()">
                                <label for="bet-{idx}">&#10003;</label>
                            </div>
                        </td>
                        <td class="match-cell">
                            <div class="teams">{row['home_team']} vs {row['away_team']}</div>
                            <div class="meta">
                                <span class="league">{row['league']}</span>
                                <span class="date">{datetime_str}</span>
                            </div>
                        </td>
                        <td><span class="bet-type-cell {bet_class}">{row['bet_type']} {row['line']}</span></td>
                        <td class="odds-cell">{row['odds']:.2f}</td>
                        <td class="ev-cell">+{true_ev:.1f}% {badge}</td>
                        <td class="edge-cell">+{ev_pct:.1f}%</td>
                        <td class="prob-cell">{prob_pct:.1f}%</td>
                        <td class="goals-cell">{pred_total:.2f}</td>
                        <td class="stake-cell">{stake_pct:.1f}%</td>
                        <td class="models-cell">
                            <div class="model-mini">
                                <span class="model-label poisson">P</span>
                                <span class="model-prob">{poisson_prob:.0f}%</span>
                                <span class="model-score">({pred_home:.1f}-{pred_away:.1f})</span>
                            </div>
                            <div class="model-mini">
                                <span class="model-label dc">D</span>
                                <span class="model-prob">{dc_prob:.0f}%</span>
                                <span class="model-score">({dc_home:.1f}-{dc_away:.1f})</span>
                            </div>
                        </td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Model: Poisson (primary) + Dixon-Coles (secondary) | Min Edge: 10% | Optimal Prob Range: 35-75%</p>
            <p style="margin-top: 8px; color: #4ecca3;">Based on calibration analysis: Brier Score 0.246, ECE 1.15%</p>
            <p style="margin-top: 8px;">Bet responsibly. Past performance does not guarantee future results.</p>
        </div>
    </div>
    
    <script>
        // Current sort state
        let currentSort = { column: 'ev', direction: 'desc' };
        
        // Initialize calculator on page load
        document.addEventListener('DOMContentLoaded', function() {
            applyProbFilter();
        });
        
        function sortTable(column) {
            const tbody = document.getElementById('bet-table-body');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = document.querySelectorAll('.bet-table th.sortable');
            
            // Determine sort direction
            let direction = 'desc';
            if (currentSort.column === column && currentSort.direction === 'desc') {
                direction = 'asc';
            }
            currentSort = { column, direction };
            
            // Update header styles
            headers.forEach(th => {
                th.classList.remove('active', 'asc', 'desc');
                th.querySelector('.sort-icon').innerHTML = '';
            });
            
            const activeHeader = document.querySelector(`th[data-sort="${column}"]`);
            activeHeader.classList.add('active', direction);
            activeHeader.querySelector('.sort-icon').innerHTML = direction === 'desc' ? '&#9660;' : '&#9650;';
            
            // Sort rows
            rows.sort((a, b) => {
                let aVal = a.dataset[column];
                let bVal = b.dataset[column];
                
                // Check if numeric
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return direction === 'desc' ? bNum - aNum : aNum - bNum;
                } else {
                    // String comparison
                    return direction === 'desc' 
                        ? bVal.localeCompare(aVal) 
                        : aVal.localeCompare(bVal);
                }
            });
            
            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }
        
        function applyProbFilter() {
            const minProb = parseFloat(document.getElementById('minProb').value);
            const maxProb = parseFloat(document.getElementById('maxProb').value);
            const betRows = document.querySelectorAll('.bet-table tbody tr');
            
            betRows.forEach(row => {
                const checkbox = row.querySelector('input[type="checkbox"]');
                const prob = parseFloat(row.dataset.prob);  // Already in percentage
                
                if (prob >= minProb && prob <= maxProb) {
                    checkbox.checked = true;
                } else {
                    checkbox.checked = false;
                }
            });
            
            updateCalculator();
        }
        
        function updateCalculator() {
            const bankroll = parseFloat(document.getElementById('bankroll').value) || 0;
            const stakingMethod = document.getElementById('staking').value;
            const betRows = document.querySelectorAll('.bet-table tbody tr');
            
            // First pass: collect selected bets
            let selectedBets = [];
            betRows.forEach(row => {
                const checkbox = row.querySelector('input[type="checkbox"]');
                if (checkbox && checkbox.checked) {
                    const odds = parseFloat(row.dataset.odds);
                    const stakePercent = parseFloat(row.dataset.stake);  // Already in percentage
                    const prob = parseFloat(row.dataset.prob) / 100;  // Convert to decimal
                    const ev = prob * odds - 1;
                    const teams = row.querySelector('.teams').textContent;
                    const bet = row.querySelector('.bet-type-cell').textContent;
                    selectedBets.push({ row, odds, kelly: stakePercent / 100, prob, ev, teams, bet });
                    row.classList.add('selected');
                } else {
                    row.classList.remove('selected');
                }
            });
            
            // Calculate stakes based on method
            let totalKelly = selectedBets.reduce((sum, b) => sum + b.kelly, 0);
            let totalEV = selectedBets.reduce((sum, b) => sum + Math.max(0, b.ev), 0);
            
            selectedBets.forEach(bet => {
                if (stakingMethod === 'kelly') {
                    // Proportional to Kelly fraction - distribute bankroll proportionally
                    bet.stakePercent = totalKelly > 0 ? bet.kelly / totalKelly : 0;
                } else if (stakingMethod === 'fixed') {
                    // Equal stakes
                    bet.stakePercent = selectedBets.length > 0 ? 1 / selectedBets.length : 0;
                } else if (stakingMethod === 'ev-weighted') {
                    // Weighted by EV
                    bet.stakePercent = totalEV > 0 ? Math.max(0, bet.ev) / totalEV : 0;
                }
                bet.stakeAmount = bankroll * bet.stakePercent;
            });
            
            // Calculate totals
            let totalStake = selectedBets.reduce((sum, b) => sum + b.stakeAmount, 0);
            let expectedReturn = selectedBets.reduce((sum, b) => sum + b.stakeAmount * b.odds * b.prob, 0);
            let expectedProfit = expectedReturn - totalStake;
            
            // Update display
            document.getElementById('selected-count').textContent = selectedBets.length;
            document.getElementById('total-stake').textContent = totalStake.toFixed(0) + ' kr';
            document.getElementById('expected-return').textContent = expectedReturn.toFixed(0) + ' kr';
            document.getElementById('expected-profit').textContent = (expectedProfit >= 0 ? '+' : '') + expectedProfit.toFixed(0) + ' kr';
            
            // Update color based on profit
            const profitEl = document.getElementById('expected-profit');
            profitEl.className = expectedProfit >= 0 ? 'value green' : 'value red';
            
            // Update breakdown
            const breakdownEl = document.getElementById('bet-breakdown');
            if (selectedBets.length > 0) {
                let breakdownHTML = '<div class="bet-breakdown-title">Stake Breakdown</div><div class="bet-breakdown-list">';
                selectedBets.forEach(bet => {
                    breakdownHTML += `
                        <div class="bet-breakdown-item">
                            <div>
                                <span class="bet-match">${bet.teams}</span>
                                <span class="bet-info"> — ${bet.bet} @ ${bet.odds.toFixed(2)}</span>
                            </div>
                            <span class="bet-amount">${bet.stakeAmount.toFixed(0)} kr</span>
                        </div>
                    `;
                });
                breakdownHTML += '</div>';
                breakdownEl.innerHTML = breakdownHTML;
                breakdownEl.classList.add('visible');
            } else {
                breakdownEl.classList.remove('visible');
            }
        }
        
        function selectAll() {
            document.querySelectorAll('.bet-table tbody input[type="checkbox"]').forEach(cb => cb.checked = true);
            updateCalculator();
        }
        
        function selectNone() {
            document.querySelectorAll('.bet-table tbody input[type="checkbox"]').forEach(cb => cb.checked = false);
            updateCalculator();
        }
        
        function selectTop5() {
            const checkboxes = document.querySelectorAll('.bet-table tbody input[type="checkbox"]');
            checkboxes.forEach((cb, index) => {
                cb.checked = index < 5;
            });
            updateCalculator();
        }
        
        // Update when bankroll changes
        document.getElementById('bankroll').addEventListener('input', updateCalculator);
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_path.write_text(html, encoding="utf-8")
    print(f"\n[OK] HTML report generated: {output_path.absolute()}")
    print(f"{len(df)} value bets | Total stake: {total_stake:.1f}% | Avg EV: +{avg_ev:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate betting HTML report")
    parser.add_argument("--european", "-e", action="store_true",
                        help="Generate European competition report")
    parser.add_argument("--domestic", "-d", action="store_true",
                        help="Generate domestic leagues report")
    parser.add_argument("--output", "-o", type=str, default="betting_report.html",
                        help="Output file path")
    args = parser.parse_args()
    
    if args.european:
        mode = "european"
    elif args.domestic:
        mode = "domestic"
    else:
        mode = "auto"
    
    generate_html_report(
        output_path=Path(args.output),
        mode=mode
    )
