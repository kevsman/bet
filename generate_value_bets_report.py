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

    def prob_under(line, total_lambda):
        return poisson.cdf(int(line), total_lambda)

    def prob_over(line, total_lambda):
        return 1 - prob_under(line, total_lambda)

    # Calculate recommendations
    print("Calculating value bets...")
    recommendations = []

    for _, row in odds_df.iterrows():
        home = map_team(row['home_team'])
        away = map_team(row['away_team'])
        
        home_row = None
        away_row = None
        for league in ['E0', 'E1', 'E2', 'E3', 'E4', 'D1', 'D2', 'SP1', 'SP2', 'I1', 'I2', 'F1', 'F2', 'SC0', 'SC1', 'N1', 'B1', 'P1', 'T1', 'G1']:
            if home_row is None:
                home_row = home_map.get((league, home))
            if away_row is None:
                away_row = away_map.get((league, away))
        
        if home_row is None or away_row is None:
            continue
        
        feature_values = []
        skip = False
        for col in features:
            if col.startswith('home_'):
                val = home_row.get(col, np.nan)
            elif col.startswith('away_'):
                val = away_row.get(col, np.nan)
            else:
                val = home_row.get(col, np.nan)
            if pd.isna(val):
                skip = True
                break
            feature_values.append(val)
        
        if skip:
            continue
        
        X = pd.DataFrame([feature_values], columns=features)
        pred_home = float(np.clip(home_model.predict(X)[0], 0.1, 5.0))
        pred_away = float(np.clip(away_model.predict(X)[0], 0.1, 5.0))
        total = pred_home + pred_away
        
        for line in [1.5, 2.5, 3.5]:
            over_col = f'over_{line}'.replace('.', '_')
            under_col = f'under_{line}'.replace('.', '_')
            over_odds = row.get(over_col)
            under_odds = row.get(under_col)
            
            if pd.isna(over_odds) or pd.isna(under_odds):
                continue
            
            p_over = prob_over(line, total)
            p_under = prob_under(line, total)
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
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            color: #fff; 
            min-height: 100vh; 
            padding: 20px; 
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ 
            text-align: center; 
            margin-bottom: 10px; 
            font-size: 2.5em; 
            background: linear-gradient(90deg, #00d9ff, #00ff88); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .stats-bar {{ 
            display: flex; 
            justify-content: center; 
            gap: 40px; 
            margin-bottom: 30px; 
            flex-wrap: wrap; 
        }}
        .stat {{ 
            text-align: center; 
            background: rgba(255,255,255,0.05); 
            padding: 15px 25px; 
            border-radius: 10px; 
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #00ff88; }}
        .stat-label {{ color: #888; font-size: 0.9em; }}
        .section {{ margin-bottom: 40px; }}
        .section-title {{ 
            font-size: 1.5em; 
            margin-bottom: 15px; 
            padding-left: 10px; 
            border-left: 4px solid #00d9ff; 
        }}
        .card {{ 
            background: rgba(255,255,255,0.05); 
            border-radius: 12px; 
            padding: 20px; 
            margin-bottom: 15px; 
            border: 1px solid rgba(255,255,255,0.1); 
            transition: transform 0.2s, box-shadow 0.2s; 
        }}
        .card:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 10px 30px rgba(0,0,0,0.3); 
        }}
        .card-header {{ 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 15px; 
            flex-wrap: wrap; 
            gap: 10px; 
        }}
        .match-info {{ display: flex; flex-direction: column; }}
        .match-teams {{ font-size: 1.3em; font-weight: bold; }}
        .match-meta {{ color: #888; font-size: 0.9em; }}
        .edge-badge {{ 
            background: linear-gradient(135deg, #00ff88, #00d9ff); 
            color: #000; 
            padding: 8px 16px; 
            border-radius: 20px; 
            font-weight: bold; 
            font-size: 1.1em; 
        }}
        .card-body {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 15px; 
        }}
        .metric {{ 
            background: rgba(0,0,0,0.2); 
            padding: 12px; 
            border-radius: 8px; 
            text-align: center; 
        }}
        .metric-value {{ font-size: 1.4em; font-weight: bold; color: #00d9ff; }}
        .metric-label {{ color: #888; font-size: 0.8em; margin-top: 4px; }}
        .bet-type {{ 
            background: #ff6b6b; 
            color: #fff; 
            padding: 4px 12px; 
            border-radius: 15px; 
            font-size: 0.9em; 
            display: inline-block; 
        }}
        .bet-type.over {{ background: #00ff88; color: #000; }}
        .bet-type.under {{ background: #00d9ff; color: #000; }}
        .footer {{ 
            text-align: center; 
            color: #666; 
            margin-top: 40px; 
            padding: 20px; 
            border-top: 1px solid rgba(255,255,255,0.1); 
        }}
        .no-bets {{
            text-align: center;
            padding: 60px 20px;
            color: #888;
        }}
        .no-bets h2 {{ color: #fff; margin-bottom: 10px; }}
        @media (max-width: 600px) {{ 
            .card-body {{ grid-template-columns: 1fr 1fr; }} 
            h1 {{ font-size: 1.8em; }} 
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Value Bets</h1>
        <p class="subtitle">Generated from Norsk Tipping odds | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="stats-bar">
            <div class="stat">
                <div class="stat-value">{len(unique_recs)}</div>
                <div class="stat-label">Value Bets Found</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(odds_df)}</div>
                <div class="stat-label">Matches Analyzed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{sum(r["kelly"] for r in unique_recs)*100:.1f}%</div>
                <div class="stat-label">Total Stake</div>
            </div>
        </div>
'''

    if not unique_recs:
        html += '''
        <div class="no-bets">
            <h2>No Value Bets Found</h2>
            <p>No bets with >5% edge were identified in the current odds.</p>
        </div>
'''
    else:
        # Group by date
        by_date = defaultdict(list)
        for r in unique_recs:
            by_date[r['date']].append(r)

        for date in sorted(by_date.keys()):
            recs = by_date[date]
            html += f'''
        <div class="section">
            <h2 class="section-title">📅 {date}</h2>
'''
            for r in sorted(recs, key=lambda x: -x['edge']):
                bet_class = 'over' if 'Over' in r['bet'] else 'under'
                html += f'''
            <div class="card">
                <div class="card-header">
                    <div class="match-info">
                        <div class="match-teams">{r['home_team']} vs {r['away_team']}</div>
                        <div class="match-meta">🕐 {r['time']} | 🌍 {r['country']}</div>
                    </div>
                    <div class="edge-badge">+{r['edge']*100:.1f}% Edge</div>
                </div>
                <div class="card-body">
                    <div class="metric">
                        <span class="bet-type {bet_class}">{r['bet']}</span>
                        <div class="metric-label">Selection</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{r['odds']:.2f}</div>
                        <div class="metric-label">Odds</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{r['model_total']:.2f}</div>
                        <div class="metric-label">Model Total</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{r['probability']*100:.0f}%</div>
                        <div class="metric-label">Probability</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{r['kelly']*100:.1f}%</div>
                        <div class="metric-label">Stake</div>
                    </div>
                </div>
            </div>
'''
            html += '''
        </div>
'''

    html += '''
        <div class="footer">
            <p>⚙️ Powered by Poisson regression model trained on historical match data</p>
            <p>📊 Minimum edge threshold: 5% | Kelly criterion: 25% fractional</p>
        </div>
    </div>
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
