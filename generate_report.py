"""
Generate HTML report for Norsk Tipping betting recommendations.

This script reads the recommendations CSV and creates a visual HTML report
with the betting opportunities, sorted by edge percentage.
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd


def generate_european_html_report(
    predictions_path: Path = Path("data/processed/european/today_predictions.csv"),
    output_path: Path = Path("betting_report.html"),
) -> None:
    """Generate HTML report for European competition bets."""
    
    # Read predictions
    df = pd.read_csv(predictions_path)
    
    if df.empty:
        print("No European predictions found.")
        return
    
    # Calculate EV and filter for value bets (EV > 0)
    df["best_ev"] = df[["over_ev", "under_ev"]].max(axis=1)
    df["best_bet"] = df.apply(
        lambda r: "Over 2.5" if r["over_ev"] > r["under_ev"] else "Under 2.5", axis=1
    )
    df["best_odds"] = df.apply(
        lambda r: r["over_odds"] if r["over_ev"] > r["under_ev"] else r["under_odds"], axis=1
    )
    df["best_prob"] = df.apply(
        lambda r: r["over_prob"] if r["over_ev"] > r["under_ev"] else r["under_prob"], axis=1
    )
    
    # Filter for positive EV only
    value_bets = df[df["best_ev"] > 0].copy()
    value_bets = value_bets.sort_values("best_ev", ascending=False)
    
    # Calculate Kelly stake (25% Kelly, max 5%)
    value_bets["kelly_fraction"] = value_bets.apply(
        lambda r: max(0, min(0.05, 0.25 * (r["best_prob"] - (1 - r["best_prob"]) / (r["best_odds"] - 1)) if r["best_odds"] > 1 else 0)),
        axis=1
    )
    
    total_stake = value_bets["kelly_fraction"].sum() * 100
    
    # Get top 5 for featured section
    top_5 = value_bets.head(5)
    
    # Generate HTML with same design as main report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>European Competition Betting Recommendations</title>
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
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .header .timestamp {{
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .summary {{
            background: #f8f9fa;
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            border-bottom: 3px solid #e9ecef;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #0f3460;
        }}
        
        .summary-card .label {{
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .section-title {{
            padding: 30px;
            padding-bottom: 0;
            font-size: 1.5em;
            color: #333;
        }}
        
        .top-picks {{
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        
        .pick-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 25px;
            color: white;
            position: relative;
            overflow: hidden;
        }}
        
        .pick-card.rank-1 {{
            background: linear-gradient(135deg, #f5af19 0%, #f12711 100%);
        }}
        
        .pick-card.rank-2 {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        
        .pick-card.rank-3 {{
            background: linear-gradient(135deg, #4776E6 0%, #8E54E9 100%);
        }}
        
        .pick-card .rank {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        .pick-card .match {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .pick-card .league {{
            opacity: 0.8;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        
        .pick-card .bet-type {{
            background: rgba(255,255,255,0.2);
            display: inline-block;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        
        .pick-card .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }}
        
        .pick-card .stat {{
            text-align: center;
        }}
        
        .pick-card .stat .value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .pick-card .stat .label {{
            font-size: 0.8em;
            opacity: 0.8;
        }}
        
        .all-bets {{
            padding: 30px;
        }}
        
        .bet-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .bet-table th {{
            background: #0f3460;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        
        .bet-table td {{
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .bet-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .bet-table .ev-positive {{
            color: #28a745;
            font-weight: bold;
        }}
        
        .bet-table .bet-type {{
            background: #e3f2fd;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        
        .calculator {{
            background: #f8f9fa;
            padding: 30px;
            margin: 30px;
            border-radius: 15px;
        }}
        
        .calculator h3 {{
            margin-bottom: 20px;
            color: #333;
        }}
        
        .calculator-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .calculator input {{
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 1.1em;
        }}
        
        .calculator input:focus {{
            border-color: #0f3460;
            outline: none;
        }}
        
        .calculator .result {{
            background: #0f3460;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }}
        
        .calculator .result .amount {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .footer {{
            background: #1a1a2e;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .footer p {{
            margin: 5px 0;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚽ European Competition Bets</h1>
            <div class="subtitle">UCL • Europa League • Conference League</div>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <div class="value">{len(value_bets)}</div>
                <div class="label">Value Bets Found</div>
            </div>
            <div class="summary-card">
                <div class="value">{len(df)}</div>
                <div class="label">Matches Analyzed</div>
            </div>
            <div class="summary-card">
                <div class="value">{total_stake:.1f}%</div>
                <div class="label">Total Stake</div>
            </div>
            <div class="summary-card">
                <div class="value">{value_bets['best_ev'].mean()*100:.1f}%</div>
                <div class="label">Avg Expected Value</div>
            </div>
        </div>
        
        <h2 class="section-title">🏆 Top 5 Picks</h2>
        <div class="top-picks">
"""
    
    # Add top 5 picks
    for i, (_, row) in enumerate(top_5.iterrows()):
        rank_class = f"rank-{i+1}" if i < 3 else ""
        ev_pct = row["best_ev"] * 100
        prob_pct = row["best_prob"] * 100
        
        # Calculate expected value (what you expect to win per 1 kr bet)
        expected_return = row['best_prob'] * row['best_odds']
        ev_return_pct = (expected_return - 1) * 100  # Convert to percentage gain
        
        html += f"""
            <div class="pick-card {rank_class}">
                <div class="rank">#{i+1}</div>
                <div class="match">{row['home_team']} vs {row['away_team']}</div>
                <div class="league">{row['league']}</div>
                <div class="bet-type">{row['best_bet']} @ {row['best_odds']:.2f}</div>
                <div class="stats">
                    <div class="stat">
                        <div class="value">+{ev_pct:.1f}%</div>
                        <div class="label">EV</div>
                    </div>
                    <div class="stat">
                        <div class="value">{row['best_odds']:.2f}</div>
                        <div class="label">Odds</div>
                    </div>
                    <div class="stat">
                        <div class="value">{prob_pct:.0f}%</div>
                        <div class="label">Probability</div>
                    </div>
                    <div class="stat">
                        <div class="value">{row['pred_total']:.1f}</div>
                        <div class="label">Pred Goals</div>
                    </div>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="calculator">
            <h3>💰 Bet Calculator</h3>
            <div class="calculator-grid">
                <div>
                    <label>Your Bankroll (kr)</label>
                    <input type="number" id="bankroll" value="1000" onchange="calculateStakes()">
                </div>
                <div>
                    <label>Kelly Fraction</label>
                    <input type="number" id="kellyFraction" value="0.25" step="0.05" onchange="calculateStakes()">
                </div>
            </div>
            <div class="result">
                <div>Recommended Total Stake</div>
                <div class="amount" id="totalStake">-</div>
            </div>
        </div>
        
        <h2 class="section-title">📊 All Value Bets</h2>
        <div class="all-bets">
            <table class="bet-table">
                <thead>
                    <tr>
                        <th>Match</th>
                        <th>Competition</th>
                        <th>Bet</th>
                        <th>Odds</th>
                        <th>Prob</th>
                        <th>EV</th>
                        <th>Goals</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add all value bets to table
    for _, row in value_bets.iterrows():
        ev_pct = row["best_ev"] * 100
        prob_pct = row["best_prob"] * 100
        
        html += f"""
                    <tr>
                        <td><strong>{row['home_team']}</strong> vs {row['away_team']}</td>
                        <td>{row['league'].replace(' - Europe', '')}</td>
                        <td><span class="bet-type">{row['best_bet']}</span></td>
                        <td>{row['best_odds']:.2f}</td>
                        <td>{prob_pct:.0f}%</td>
                        <td class="ev-positive">+{ev_pct:.1f}%</td>
                        <td>{row['pred_total']:.2f}</td>
                    </tr>
"""
    
    html += f"""
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>⚠️ Bet responsibly. These are model predictions and do not guarantee profits.</p>
            <p>Model uses European history + domestic form data • Data from openfootball + Wikipedia</p>
        </div>
    </div>
    
    <script>
        function calculateStakes() {{
            const bankroll = parseFloat(document.getElementById('bankroll').value) || 0;
            const kellyFraction = parseFloat(document.getElementById('kellyFraction').value) || 0.25;
            const totalStakePct = {total_stake:.4f};
            const adjustedStake = bankroll * (totalStakePct / 100) * (kellyFraction / 0.25);
            document.getElementById('totalStake').textContent = adjustedStake.toFixed(0) + ' kr';
        }}
        calculateStakes();
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_path.write_text(html, encoding="utf-8")
    print(f"\n✅ HTML report generated: {output_path.absolute()}")
    print(f"📊 {len(value_bets)} value bets | Total stake: {total_stake:.1f}% of bankroll")


def generate_html_report(
    recommendations_path: Path = Path("data/processed/upcoming_recommendations.csv"),
    output_path: Path = Path("betting_report.html"),
) -> None:
    """Generate HTML report from recommendations."""
    
    # Read recommendations
    df = pd.read_csv(recommendations_path)
    
    if df.empty:
        print("No recommendations found.")
        return
    
    # Sort by edge (descending)
    df = df.sort_values("edge", ascending=False)
    
    # Remove duplicates based on match_id
    df = df.drop_duplicates(subset=["match_id"], keep="first")
    
    # Calculate total stake
    total_stake = df["stake_fraction"].sum() * 100
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Norsk Tipping Betting Recommendations</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .header .timestamp {{
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .summary {{
            background: #f8f9fa;
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            border-bottom: 3px solid #e9ecef;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .summary-card .label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .summary-card .value {{
            color: #1e3c72;
            font-size: 2em;
            font-weight: bold;
        }}
        
        .recommendations {{
            padding: 30px;
        }}
        
        .recommendation {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }}
        
        .recommendation:hover {{
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        
        .recommendation.high-value {{
            border-color: #28a745;
            background: linear-gradient(to right, #f8fff9 0%, white 100%);
        }}
        
        .recommendation.medium-value {{
            border-color: #ffc107;
            background: linear-gradient(to right, #fffef8 0%, white 100%);
        }}
        
        .match-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .match-teams {{
            font-size: 1.5em;
            font-weight: bold;
            color: #1e3c72;
        }}
        
        .match-league {{
            font-size: 0.9em;
            color: #6c757d;
            background: #e9ecef;
            padding: 5px 15px;
            border-radius: 20px;
        }}
        
        .prediction {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        
        .prediction-type {{
            font-size: 1.2em;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            color: white;
        }}
        
        .prediction-type.over {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .prediction-type.under {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        .odds {{
            font-size: 1.1em;
            color: #495057;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }}
        
        .metric .label {{
            color: #6c757d;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        
        .metric .value {{
            font-size: 1.3em;
            font-weight: bold;
        }}
        
        .metric.edge {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        }}
        
        .metric.edge .value {{
            color: #28a745;
            font-size: 1.8em;
        }}
        
        .metric.stake .value {{
            color: #007bff;
        }}
        
        .metric.probability .value {{
            color: #6610f2;
        }}
        
        .metric.total .value {{
            color: #fd7e14;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            border-top: 3px solid #e9ecef;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .badge.top-pick {{
            background: #28a745;
            color: white;
        }}
        
        .badge.strong {{
            background: #ffc107;
            color: #000;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .match-teams {{
                font-size: 1.2em;
            }}
            
            .summary {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Norsk Tipping Betting Recommendations</h1>
            <div class="subtitle">AI-Powered Value Betting Analysis</div>
            <div class="timestamp">Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}</div>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <div class="label">Total Recommendations</div>
                <div class="value">{len(df)}</div>
            </div>
            <div class="summary-card">
                <div class="label">Total Stake</div>
                <div class="value">{total_stake:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Edge</div>
                <div class="value">{df['edge'].mean() * 100:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="label">Best Edge</div>
                <div class="value">{df['edge'].max() * 100:.1f}%</div>
            </div>
        </div>
        
        <div class="recommendations">
"""
    
    # Add each recommendation
    for idx, row in df.iterrows():
        edge_pct = row["edge"] * 100
        prob_pct = row["probability"] * 100
        stake_pct = row["stake_fraction"] * 100
        
        # Determine value class
        if edge_pct >= 30:
            value_class = "high-value"
            badge = '<span class="badge top-pick">⭐ TOP PICK</span>'
        elif edge_pct >= 15:
            value_class = "medium-value"
            badge = '<span class="badge strong">💪 STRONG</span>'
        else:
            value_class = ""
            badge = ""
        
        # Prediction type
        pred_type = row["selection"].title()
        pred_class = pred_type.lower()
        
        html += f"""
            <div class="recommendation {value_class}">
                <div class="match-header">
                    <div>
                        <div class="match-teams">{row['home_team']} vs {row['away_team']}</div>
                        <div class="match-league">{row['league_code']} • {row['date'][:10]}</div>
                    </div>
                    {badge}
                </div>
                
                <div class="prediction">
                    <div class="prediction-type {pred_class}">{pred_type} {row['line']}</div>
                    <div class="odds">@ {row['odds']:.2f}</div>
                </div>
                
                <div class="metrics">
                    <div class="metric edge">
                        <div class="label">Edge</div>
                        <div class="value">{edge_pct:.1f}%</div>
                    </div>
                    <div class="metric probability">
                        <div class="label">Win Probability</div>
                        <div class="value">{prob_pct:.1f}%</div>
                    </div>
                    <div class="metric stake">
                        <div class="label">Recommended Stake</div>
                        <div class="value">{stake_pct:.1f}%</div>
                    </div>
                    <div class="metric total">
                        <div class="label">Model Total Goals</div>
                        <div class="value">{row['model_total']:.2f}</div>
                    </div>
                </div>
            </div>
"""
    
    html += f"""
        </div>
        
        <div class="footer">
            <p>⚠️ Bet responsibly. These are model predictions and do not guarantee profits.</p>
            <p>Stakes are calculated using Kelly Criterion based on model edge.</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    output_path.write_text(html, encoding="utf-8")
    print(f"\n✅ HTML report generated: {output_path.absolute()}")
    print(f"📊 {len(df)} recommendations | Total stake: {total_stake:.1f}% of bankroll")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate betting HTML report")
    parser.add_argument("--european", "-e", action="store_true",
                        help="Generate European competition report")
    args = parser.parse_args()
    
    if args.european:
        generate_european_html_report()
    else:
        generate_html_report()
