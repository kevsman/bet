"""Generate HTML report from fresh Norsk Tipping odds and model predictions."""
import pandas as pd
import numpy as np
from joblib import load
from scipy.stats import poisson
from datetime import datetime
from collections import defaultdict

def main():
    # Load models
    print("Loading models...")
    home_model = load('models/home_poisson.joblib')
    away_model = load('models/away_poisson.joblib')

    # Load feature list
    with open('models/features.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]

    # Load the historical dataset to get team stats
    print("Loading historical dataset...")
    dataset = pd.read_csv('data/processed/match_dataset.csv', parse_dates=['Date'], low_memory=False)
    dataset = dataset.sort_values('Date')

    # Build team feature maps
    home_map = {}
    away_map = {}
    for _, row in dataset.iterrows():
        home_key = (row.get('league_code', 'unknown'), row['HomeTeam'])
        away_key = (row.get('league_code', 'unknown'), row['AwayTeam'])
        home_map[home_key] = row
        away_map[away_key] = row

    # Load fresh Norsk Tipping odds
    print("Loading Norsk Tipping odds...")
    odds_df = pd.read_csv('data/upcoming/norsk_tipping_odds.csv')
    print(f"  Loaded {len(odds_df)} matches")

    # Team name mapping (Norsk Tipping -> our dataset names)
    team_mapping = {
        'Manchester City': 'Man City',
        'Manchester United': 'Man United',
        'Newcastle United': 'Newcastle',
        'Tottenham Hotspur': 'Tottenham',
        'Nottingham Forest': "Nott'm Forest",
        'West Ham United': 'West Ham',
        'Wolverhampton': 'Wolves',
        'Leicester City': 'Leicester',
        'Bayer Leverkusen': 'Leverkusen',
        'Bayern München': 'Bayern Munich',
        'Bayern Munich': 'Bayern Munich',
        'Borussia Dortmund': 'Dortmund',
        "Borussia M'gladbach": "M'gladbach",
        'Eintracht Frankfurt': 'Ein Frankfurt',
        'VfB Stuttgart': 'Stuttgart',
        'VfL Wolfsburg': 'Wolfsburg',
        'TSG 1899 Hoffenheim': 'Hoffenheim',
        '1. FC Union Berlin': 'Union Berlin',
        'FC St. Pauli': 'St Pauli',
        'SC Freiburg': 'Freiburg',
        'FC Augsburg': 'Augsburg',
        '1. FSV Mainz 05': 'Mainz',
        '1. FC Heidenheim 1846': 'Heidenheim',
        'VfL Bochum 1848': 'Bochum',
        'Atletico Madrid': 'Ath Madrid',
        'Athletic Club Bilbao': 'Ath Bilbao',
        'Real Sociedad': 'Sociedad',
        'Real Betis': 'Betis',
        'Celta Vigo': 'Celta',
        'Real Valladolid': 'Valladolid',
        'AC Milan': 'Milan',
        'AS Roma': 'Roma',
        'SSC Napoli': 'Napoli',
        'SS Lazio': 'Lazio',
        'Torino FC': 'Torino',
        'ACF Fiorentina': 'Fiorentina',
        'Bologna FC 1909': 'Bologna',
        'Hellas Verona': 'Verona',
        'Udinese Calcio': 'Udinese',
        'AC Monza': 'Monza',
        'Parma Calcio 1913': 'Parma',
        'Cagliari Calcio': 'Cagliari',
        'Empoli FC': 'Empoli',
        'US Lecce': 'Lecce',
        'Genoa CFC': 'Genoa',
        'Venezia FC': 'Venezia',
        'Como 1907': 'Como',
        'Paris Saint Germain': 'Paris SG',
        'Olympique Lyon': 'Lyon',
        'Olympique Marseille': 'Marseille',
        'AS Monaco': 'Monaco',
        'LOSC Lille': 'Lille',
        'OGC Nice': 'Nice',
        'RC Lens': 'Lens',
        'Stade Rennais': 'Rennes',
        'RC Strasbourg': 'Strasbourg',
        'Stade Brestois 29': 'Brest',
        'Toulouse FC': 'Toulouse',
        'FC Nantes': 'Nantes',
        'AJ Auxerre': 'Auxerre',
        'Angers SCO': 'Angers',
        'Le Havre AC': 'Le Havre',
        'AS Saint-Étienne': 'St Etienne',
        'Montpellier HSC': 'Montpellier',
        'Stade de Reims': 'Reims',
        'PSV Eindhoven': 'PSV',
        'Ajax': 'Ajax',
        'Feyenoord': 'Feyenoord',
        'AZ Alkmaar': 'AZ Alkmaar',
        'FC Twente': 'Twente',
        'FC Utrecht': 'Utrecht',
        'Sporting CP': 'Sporting CP',
        'SL Benfica': 'Benfica',
        'FC Porto': 'Porto',
        'Celtic': 'Celtic',
        'Rangers': 'Rangers',
        'Aberdeen': 'Aberdeen',
        'Hearts': 'Hearts',
        'Hibernian': 'Hibernian',
    }

    def map_team(name):
        return team_mapping.get(name, name)

    # Bivariate Poisson functions for accurate probability calculation
    def bivariate_prob_under(line, home_xg, away_xg):
        """Calculate P(total goals <= line) using bivariate Poisson."""
        prob = 0.0
        for h in range(10):
            for a in range(10):
                if h + a <= line:
                    prob += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
        return prob

    def bivariate_prob_over(line, home_xg, away_xg):
        """Calculate P(total goals > line) using bivariate Poisson."""
        return 1 - bivariate_prob_under(line, home_xg, away_xg)

    # Leagues to search
    LEAGUES = ['E0', 'E1', 'E2', 'E3', 'E4', 'D1', 'D2', 'SP1', 'SP2', 'I1', 'I2', 'F1', 'F2', 'SC0', 'SC1', 'N1', 'B1', 'P1', 'T1', 'G1']

    def get_team_avg_features(team_name, is_home=True, n_matches=5):
        """Get average features from last N home/away matches for a team."""
        for league in LEAGUES:
            if is_home:
                matches = dataset[(dataset['league_code'] == league) & (dataset['HomeTeam'] == team_name)]
            else:
                matches = dataset[(dataset['league_code'] == league) & (dataset['AwayTeam'] == team_name)]
            
            if len(matches) >= 3:
                return matches.tail(n_matches)[features].mean(), league
        return None, None

    # Calculate recommendations
    print("Calculating value bets...")
    recommendations = []

    for _, row in odds_df.iterrows():
        home = map_team(row['home_team'])
        away = map_team(row['away_team'])
        
        # Get averaged features
        home_feats, home_league = get_team_avg_features(home, is_home=True)
        away_feats, away_league = get_team_avg_features(away, is_home=False)
        
        if home_feats is None or away_feats is None:
            continue
        
        if home_feats.isna().any() or away_feats.isna().any():
            continue
        
        # Predict using full feature vectors
        pred_home = float(np.clip(home_model.predict([home_feats])[0], 0.1, 5.0))
        pred_away = float(np.clip(away_model.predict([away_feats])[0], 0.1, 5.0))
        total = pred_home + pred_away
        
        for line in [1.5, 2.5, 3.5]:
            over_col = f'over_{line}'.replace('.', '_')
            under_col = f'under_{line}'.replace('.', '_')
            over_odds = row.get(over_col)
            under_odds = row.get(under_col)
            
            if pd.isna(over_odds) or pd.isna(under_odds):
                continue
            
            # Use bivariate Poisson for accurate probabilities
            p_over = bivariate_prob_over(line, pred_home, pred_away)
            p_under = bivariate_prob_under(line, pred_home, pred_away)
            implied_over = 1 / over_odds
            implied_under = 1 / under_odds
            edge_over = p_over - implied_over
            edge_under = p_under - implied_under
            min_edge = 0.05
            
            if edge_over > min_edge:
                kelly = edge_over / (over_odds - 1) if over_odds > 1 else 0
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
                    'kelly': min(kelly * 0.25, 0.05)
                })
            
            if edge_under > min_edge:
                kelly = edge_under / (under_odds - 1) if under_odds > 1 else 0
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
                    'kelly': min(kelly * 0.25, 0.05)
                })

    # Sort by edge and remove duplicates
    recommendations = sorted(recommendations, key=lambda x: -x['edge'])
    seen = set()
    unique_recs = []
    for r in recommendations:
        key = (r['home_team'], r['away_team'], r['bet'])
        if key not in seen:
            seen.add(key)
            unique_recs.append(r)

    print(f"  Found {len(unique_recs)} value bets")

    # Generate HTML
    print("Generating HTML report...")
    
    # Get top 5 for featured section
    top_5 = unique_recs[:5]
    
    # Generate bet data for JavaScript
    import json
    bets_json = json.dumps([{
        'id': i,
        'home_team': r['home_team'],
        'away_team': r['away_team'],
        'date': r['date'],
        'time': r['time'],
        'country': r['country'],
        'bet': r['bet'],
        'odds': r['odds'],
        'model_total': r['model_total'],
        'probability': r['probability'],
        'edge': r['edge'],
        'kelly': r['kelly']
    } for i, r in enumerate(unique_recs)])
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Value Bets - Norsk Tipping</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background: #0f0f0f;
            color: #e0e0e0; 
            min-height: 100vh; 
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
            padding: 40px 20px;
            text-align: center;
            border-bottom: 1px solid #333;
        }}
        h1 {{ 
            font-size: 2.2em; 
            color: #fff;
            margin-bottom: 8px;
        }}
        .subtitle {{ color: #888; font-size: 0.95em; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 20px; }}
        
        /* Betting Controls */
        .betting-controls {{
            background: linear-gradient(135deg, #1a2e1a 0%, #0f1f0f 100%);
            border: 1px solid #4ade80;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        .betting-controls h2 {{
            color: #4ade80;
            font-size: 1.3em;
            margin-bottom: 20px;
        }}
        .controls-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 25px;
            align-items: center;
            margin-bottom: 20px;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .control-group label {{
            color: #888;
            font-size: 0.85em;
        }}
        .control-group input[type="number"] {{
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 10px 14px;
            color: #fff;
            font-size: 1.1em;
            width: 150px;
        }}
        .control-group input[type="number"]:focus {{
            outline: none;
            border-color: #4ade80;
        }}
        .toggle-group {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .toggle-switch {{
            position: relative;
            width: 50px;
            height: 26px;
        }}
        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        .toggle-slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #333;
            transition: .3s;
            border-radius: 26px;
        }}
        .toggle-slider:before {{
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }}
        .toggle-switch input:checked + .toggle-slider {{
            background-color: #4ade80;
        }}
        .toggle-switch input:checked + .toggle-slider:before {{
            transform: translateX(24px);
        }}
        .toggle-labels {{
            display: flex;
            gap: 8px;
            font-size: 0.9em;
        }}
        .toggle-labels span {{
            color: #666;
        }}
        .toggle-labels span.active {{
            color: #4ade80;
            font-weight: 600;
        }}
        
        /* Summary Box */
        .betting-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 8px;
        }}
        .summary-item {{
            text-align: center;
        }}
        .summary-item .value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #4ade80;
        }}
        .summary-item .value.winnings {{
            color: #fbbf24;
        }}
        .summary-item .label {{
            color: #888;
            font-size: 0.8em;
            margin-top: 4px;
        }}
        
        /* Stats Row */
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-box {{
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .stat-box .value {{ font-size: 2.5em; font-weight: 700; color: #4ade80; }}
        .stat-box .label {{ color: #888; font-size: 0.85em; margin-top: 5px; }}
        
        /* Top 5 Section */
        .top-picks {{
            background: linear-gradient(135deg, #1e3a5f 0%, #1a1a2e 100%);
            border: 1px solid #2563eb;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 40px;
        }}
        .top-picks h2 {{
            color: #60a5fa;
            font-size: 1.3em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .top-list {{
            display: grid;
            gap: 12px;
        }}
        .top-item {{
            display: grid;
            grid-template-columns: 40px 1fr auto auto auto;
            gap: 15px;
            align-items: center;
            background: rgba(0,0,0,0.3);
            padding: 15px 20px;
            border-radius: 8px;
            border-left: 4px solid #4ade80;
        }}
        .top-item:nth-child(1) {{ border-left-color: #fbbf24; }}
        .top-item:nth-child(2) {{ border-left-color: #94a3b8; }}
        .top-item:nth-child(3) {{ border-left-color: #b45309; }}
        .rank {{
            font-size: 1.5em;
            font-weight: 700;
            color: #fff;
        }}
        .top-match {{
            display: flex;
            flex-direction: column;
        }}
        .top-match .teams {{ font-weight: 600; color: #fff; }}
        .top-match .meta {{ font-size: 0.8em; color: #888; }}
        .top-bet {{
            background: #4ade80;
            color: #000;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        .top-bet.under {{ background: #60a5fa; }}
        .top-odds {{
            font-size: 1.3em;
            font-weight: 700;
            color: #fff;
        }}
        .top-edge {{
            background: #4ade80;
            color: #000;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 700;
        }}
        
        /* All Bets Section */
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        .section-header h2 {{
            font-size: 1.3em;
            color: #fff;
        }}
        .select-buttons {{
            display: flex;
            gap: 10px;
        }}
        .select-btn {{
            background: #333;
            border: none;
            color: #fff;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
        }}
        .select-btn:hover {{
            background: #444;
        }}
        .date-group {{
            margin-bottom: 30px;
        }}
        .date-label {{
            background: #2563eb;
            color: #fff;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.9em;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 15px;
        }}
        
        /* Table Layout */
        .bets-table {{
            width: 100%;
            border-collapse: collapse;
            background: #1a1a1a;
            border-radius: 8px;
            overflow: hidden;
        }}
        .bets-table th {{
            background: #252525;
            color: #888;
            font-weight: 600;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 14px 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        .bets-table td {{
            padding: 14px 12px;
            border-bottom: 1px solid #2a2a2a;
        }}
        .bets-table tr:hover {{ background: #222; }}
        .bets-table tr.selected {{ background: rgba(74, 222, 128, 0.1); }}
        .bets-table .match-cell {{
            display: flex;
            flex-direction: column;
        }}
        .bets-table .teams {{ font-weight: 600; color: #fff; }}
        .bets-table .time {{ font-size: 0.8em; color: #666; }}
        .bet-tag {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .bet-tag.over {{ background: rgba(74, 222, 128, 0.2); color: #4ade80; }}
        .bet-tag.under {{ background: rgba(96, 165, 250, 0.2); color: #60a5fa; }}
        .odds-cell {{ font-weight: 700; color: #fff; font-size: 1.1em; }}
        .prob-cell {{ color: #888; }}
        .edge-cell {{ 
            color: #4ade80; 
            font-weight: 700;
        }}
        .stake-cell {{
            color: #4ade80;
            font-weight: 600;
            font-size: 0.9em;
        }}
        .winnings-cell {{
            color: #fbbf24;
            font-weight: 600;
        }}
        
        /* Checkbox styling */
        .bet-checkbox {{
            width: 20px;
            height: 20px;
            cursor: pointer;
            accent-color: #4ade80;
        }}
        
        .footer {{ 
            text-align: center; 
            color: #555; 
            margin-top: 50px; 
            padding: 30px;
            border-top: 1px solid #222;
            font-size: 0.85em;
        }}
        .footer p {{ margin: 5px 0; }}
        
        @media (max-width: 900px) {{
            .stats-row {{ grid-template-columns: 1fr; }}
            .top-item {{ 
                grid-template-columns: 40px 1fr;
                gap: 10px;
            }}
            .top-item > *:nth-child(n+3) {{
                grid-column: 2;
            }}
            .bets-table {{ font-size: 0.9em; }}
            .bets-table th, .bets-table td {{ padding: 10px 8px; }}
            .controls-row {{ flex-direction: column; align-items: flex-start; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Value Bets Report</h1>
        <p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Norsk Tipping Odds</p>
    </div>
    
    <div class="container">
        <!-- Betting Controls -->
        <div class="betting-controls">
            <h2>💰 Bet Calculator</h2>
            <div class="controls-row">
                <div class="control-group">
                    <label>Total Amount to Bet (kr)</label>
                    <input type="number" id="totalBet" value="1000" min="0" step="100" onchange="updateCalculations()">
                </div>
                <div class="control-group">
                    <label>Stake Mode</label>
                    <div class="toggle-group">
                        <span class="toggle-labels">
                            <span id="labelProportional" class="active">Proportional</span>
                        </span>
                        <label class="toggle-switch">
                            <input type="checkbox" id="fixedMode" onchange="updateCalculations()">
                            <span class="toggle-slider"></span>
                        </label>
                        <span class="toggle-labels">
                            <span id="labelFixed">Fixed</span>
                        </span>
                    </div>
                </div>
                <div class="control-group" id="fixedAmountGroup" style="display: none;">
                    <label>Fixed Amount per Bet (kr)</label>
                    <input type="number" id="fixedAmount" value="100" min="0" step="10" onchange="updateCalculations()">
                </div>
            </div>
            <div class="betting-summary">
                <div class="summary-item">
                    <div class="value" id="selectedCount">0</div>
                    <div class="label">Bets Selected</div>
                </div>
                <div class="summary-item">
                    <div class="value" id="totalStake">0 kr</div>
                    <div class="label">Total Stake</div>
                </div>
                <div class="summary-item">
                    <div class="value winnings" id="potentialWinnings">0 kr</div>
                    <div class="label">Potential Winnings (if all win)</div>
                </div>
                <div class="summary-item">
                    <div class="value winnings" id="expectedValue">0 kr</div>
                    <div class="label">Expected Value</div>
                </div>
            </div>
        </div>
        
        <div class="stats-row">
            <div class="stat-box">
                <div class="value">{len(unique_recs)}</div>
                <div class="label">Value Bets Found</div>
            </div>
            <div class="stat-box">
                <div class="value">{len(odds_df)}</div>
                <div class="label">Matches Analyzed</div>
            </div>
            <div class="stat-box">
                <div class="value">{sum(r["kelly"] for r in unique_recs)*100:.1f}%</div>
                <div class="label">Total Recommended Stake</div>
            </div>
        </div>
'''

    # Top 5 Section
    if top_5:
        html += '''
        <div class="top-picks">
            <h2>🏆 Top 5 Recommendations</h2>
            <div class="top-list">
'''
        for i, r in enumerate(top_5, 1):
            bet_class = 'under' if 'Under' in r['bet'] else ''
            html += f'''
                <div class="top-item">
                    <div class="rank">#{i}</div>
                    <div class="top-match">
                        <span class="teams">{r['home_team']} vs {r['away_team']}</span>
                        <span class="meta">{r['date']} {r['time']} • {r['country']}</span>
                    </div>
                    <div class="top-bet {bet_class}">{r['bet']}</div>
                    <div class="top-odds">@{r['odds']:.2f}</div>
                    <div class="top-edge">+{r['edge']*100:.1f}%</div>
                </div>
'''
        html += '''
            </div>
        </div>
'''

    if not unique_recs:
        html += '''
        <div style="text-align: center; padding: 60px 20px; color: #888;">
            <h2 style="color: #fff; margin-bottom: 10px;">No Value Bets Found</h2>
            <p>No bets with >5% edge were identified in the current odds.</p>
        </div>
'''
    else:
        # Group by date
        by_date = defaultdict(list)
        for r in unique_recs:
            by_date[r['date']].append(r)

        html += '''
        <div class="section-header">
            <h2>All Value Bets</h2>
            <div class="select-buttons">
                <button class="select-btn" onclick="selectAll()">Select All</button>
                <button class="select-btn" onclick="selectNone()">Select None</button>
                <button class="select-btn" onclick="selectTop5()">Select Top 5</button>
            </div>
        </div>
'''
        
        bet_idx = 0
        for date in sorted(by_date.keys()):
            recs = sorted(by_date[date], key=lambda x: -x['edge'])
            html += f'''
        <div class="date-group">
            <div class="date-label">📅 {date}</div>
            <table class="bets-table">
                <thead>
                    <tr>
                        <th style="width: 40px;"></th>
                        <th>Match</th>
                        <th>Selection</th>
                        <th>Odds</th>
                        <th>Edge</th>
                        <th>Stake</th>
                        <th>Winnings</th>
                    </tr>
                </thead>
                <tbody>
'''
            for r in recs:
                bet_class = 'over' if 'Over' in r['bet'] else 'under'
                html += f'''
                    <tr id="row-{bet_idx}" data-bet-id="{bet_idx}">
                        <td><input type="checkbox" class="bet-checkbox" data-id="{bet_idx}" onchange="updateCalculations()"></td>
                        <td class="match-cell">
                            <span class="teams">{r['home_team']} vs {r['away_team']}</span>
                            <span class="time">{r['time']} • {r['country']}</span>
                        </td>
                        <td><span class="bet-tag {bet_class}">{r['bet']}</span></td>
                        <td class="odds-cell">{r['odds']:.2f}</td>
                        <td class="edge-cell">+{r['edge']*100:.1f}%</td>
                        <td class="stake-cell" id="stake-{bet_idx}">-</td>
                        <td class="winnings-cell" id="win-{bet_idx}">-</td>
                    </tr>
'''
                bet_idx += 1
            html += '''
                </tbody>
            </table>
        </div>
'''

    html += f'''
        <div class="footer">
            <p>Powered by Poisson regression model trained on historical match data</p>
            <p>Minimum edge threshold: 5% | Kelly criterion: 25% fractional | Max stake: 5%</p>
        </div>
    </div>
    
    <script>
        const betsData = {bets_json};
        
        function updateCalculations() {{
            const totalBet = parseFloat(document.getElementById('totalBet').value) || 0;
            const fixedMode = document.getElementById('fixedMode').checked;
            const fixedAmount = parseFloat(document.getElementById('fixedAmount').value) || 0;
            
            // Update toggle labels
            document.getElementById('labelProportional').className = fixedMode ? '' : 'active';
            document.getElementById('labelFixed').className = fixedMode ? 'active' : '';
            document.getElementById('fixedAmountGroup').style.display = fixedMode ? 'block' : 'none';
            
            // Get selected bets
            const checkboxes = document.querySelectorAll('.bet-checkbox');
            const selectedBets = [];
            let totalKelly = 0;
            
            checkboxes.forEach(cb => {{
                const row = document.getElementById('row-' + cb.dataset.id);
                if (cb.checked) {{
                    selectedBets.push(betsData[parseInt(cb.dataset.id)]);
                    totalKelly += betsData[parseInt(cb.dataset.id)].kelly;
                    row.classList.add('selected');
                }} else {{
                    row.classList.remove('selected');
                }}
            }});
            
            // Calculate stakes and winnings
            let totalStake = 0;
            let totalPotentialWin = 0;
            let totalExpectedValue = 0;
            
            // Reset all cells
            betsData.forEach((bet, i) => {{
                document.getElementById('stake-' + i).textContent = '-';
                document.getElementById('win-' + i).textContent = '-';
            }});
            
            selectedBets.forEach(bet => {{
                let stake;
                if (fixedMode) {{
                    stake = fixedAmount;
                }} else {{
                    // Proportional to Kelly
                    stake = totalKelly > 0 ? (bet.kelly / totalKelly) * totalBet : 0;
                }}
                
                const winnings = stake * bet.odds;
                const profit = winnings - stake;
                const ev = (bet.probability * profit) - ((1 - bet.probability) * stake);
                
                totalStake += stake;
                totalPotentialWin += winnings;
                totalExpectedValue += ev;
                
                document.getElementById('stake-' + bet.id).textContent = stake.toFixed(0) + ' kr';
                document.getElementById('win-' + bet.id).textContent = winnings.toFixed(0) + ' kr';
            }});
            
            // Update summary
            document.getElementById('selectedCount').textContent = selectedBets.length;
            document.getElementById('totalStake').textContent = totalStake.toFixed(0) + ' kr';
            document.getElementById('potentialWinnings').textContent = totalPotentialWin.toFixed(0) + ' kr';
            document.getElementById('expectedValue').textContent = (totalExpectedValue >= 0 ? '+' : '') + totalExpectedValue.toFixed(0) + ' kr';
        }}
        
        function selectAll() {{
            document.querySelectorAll('.bet-checkbox').forEach(cb => cb.checked = true);
            updateCalculations();
        }}
        
        function selectNone() {{
            document.querySelectorAll('.bet-checkbox').forEach(cb => cb.checked = false);
            updateCalculations();
        }}
        
        function selectTop5() {{
            const checkboxes = document.querySelectorAll('.bet-checkbox');
            checkboxes.forEach((cb, i) => cb.checked = i < 5);
            updateCalculations();
        }}
        
        // Initialize
        updateCalculations();
    </script>
</body>
</html>
'''

    output_file = 'value_bets_report.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"Report generated: {output_file}")
    print(f"Value bets: {len(unique_recs)}")
    print(f"Total recommended stake: {sum(r['kelly'] for r in unique_recs)*100:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
