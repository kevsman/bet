"""Calculate value bets from fresh Norsk Tipping odds."""
import pandas as pd
import numpy as np
from joblib import load
from scipy.stats import poisson

# Load models
home_model = load('models/home_poisson.joblib')
away_model = load('models/away_poisson.joblib')

# Load feature list
with open('models/features.txt', 'r') as f:
    features = [line.strip() for line in f if line.strip()]

# Load the historical dataset to get team stats
dataset = pd.read_csv('data/processed/match_dataset.csv', parse_dates=['Date'], low_memory=False)
dataset = dataset.sort_values('Date')

# Build team feature maps (most recent stats for each team)
home_map = {}
away_map = {}
for _, row in dataset.iterrows():
    home_key = (row.get('league_code', 'unknown'), row['HomeTeam'])
    away_key = (row.get('league_code', 'unknown'), row['AwayTeam'])
    home_map[home_key] = row
    away_map[away_key] = row

# Load fresh Norsk Tipping odds
odds_df = pd.read_csv('data/upcoming/norsk_tipping_odds.csv')
print(f'Loaded {len(odds_df)} matches from Norsk Tipping')

# Team name mapping (Norsk Tipping -> our dataset names)
team_mapping = {
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle',
    'Tottenham Hotspur': 'Tottenham',
    'Nottingham Forest': "Nott'm Forest",
    'West Ham United': 'West Ham',
    'Brighton': 'Brighton',
    'Wolverhampton': 'Wolves',
    'Leicester City': 'Leicester',
    'Crystal Palace': 'Crystal Palace',
    'Aston Villa': 'Aston Villa',
    'Bayer Leverkusen': 'Leverkusen',
    'Bayern München': 'Bayern Munich',
    'Bayern Munich': 'Bayern Munich',
    'Borussia Dortmund': 'Dortmund',
    "Borussia M'gladbach": "M'gladbach",
    'RB Leipzig': 'RB Leipzig',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'VfB Stuttgart': 'Stuttgart',
    'VfL Wolfsburg': 'Wolfsburg',
    'TSG 1899 Hoffenheim': 'Hoffenheim',
    '1. FC Union Berlin': 'Union Berlin',
    'FC St. Pauli': 'St Pauli',
    'SC Freiburg': 'Freiburg',
    'Werder Bremen': 'Werder Bremen',
    'FC Augsburg': 'Augsburg',
    '1. FSV Mainz 05': 'Mainz',
    '1. FC Heidenheim 1846': 'Heidenheim',
    'VfL Bochum 1848': 'Bochum',
    'Holstein Kiel': 'Holstein Kiel',
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

# Try to find matches in our dataset
recommendations = []

for _, row in odds_df.iterrows():
    home = map_team(row['home_team'])
    away = map_team(row['away_team'])
    
    # Try to find team in dataset (check multiple league codes)
    home_row = None
    away_row = None
    
    for league in ['E0', 'E1', 'E2', 'E3', 'E4', 'D1', 'D2', 'SP1', 'SP2', 'I1', 'I2', 'F1', 'F2', 'SC0', 'SC1', 'N1', 'B1', 'P1', 'T1', 'G1']:
        if home_row is None:
            home_row = home_map.get((league, home))
        if away_row is None:
            away_row = away_map.get((league, away))
    
    if home_row is None or away_row is None:
        continue
    
    # Build features
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
    
    # Predict
    pred_home = float(np.clip(home_model.predict(X)[0], 0.1, 5.0))
    pred_away = float(np.clip(away_model.predict(X)[0], 0.1, 5.0))
    total = pred_home + pred_away
    
    # Check for value bets
    lines = [1.5, 2.5, 3.5]
    for line in lines:
        over_col = f'over_{line}'.replace('.', '_')
        under_col = f'under_{line}'.replace('.', '_')
        
        over_odds = row.get(over_col)
        under_odds = row.get(under_col)
        
        if pd.isna(over_odds) or pd.isna(under_odds):
            continue
        
        p_over = prob_over(line, total)
        p_under = prob_under(line, total)
        
        # Calculate edge
        implied_over = 1 / over_odds
        implied_under = 1 / under_odds
        
        edge_over = p_over - implied_over
        edge_under = p_under - implied_under
        
        min_edge = 0.05  # 5% minimum edge
        
        if edge_over > min_edge:
            kelly = edge_over / (over_odds - 1) if over_odds > 1 else 0
            recommendations.append({
                'date': row['date'],
                'time': row['time'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'bet': f'Over {line}',
                'odds': over_odds,
                'model_total': total,
                'probability': p_over,
                'edge': edge_over,
                'kelly': min(kelly * 0.25, 0.05)  # Quarter Kelly, max 5%
            })
        
        if edge_under > min_edge:
            kelly = edge_under / (under_odds - 1) if under_odds > 1 else 0
            recommendations.append({
                'date': row['date'],
                'time': row['time'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'bet': f'Under {line}',
                'odds': under_odds,
                'model_total': total,
                'probability': p_under,
                'edge': edge_under,
                'kelly': min(kelly * 0.25, 0.05)
            })

# Sort by edge
recommendations = sorted(recommendations, key=lambda x: -x['edge'])

print(f'\n{"="*80}')
print('VALUE BETS FROM FRESH NORSK TIPPING ODDS')
print(f'{"="*80}\n')

if not recommendations:
    print('No value bets found with >5% edge')
else:
    # Remove duplicates (same match, same bet)
    seen = set()
    unique_recs = []
    for r in recommendations:
        key = (r['home_team'], r['away_team'], r['bet'])
        if key not in seen:
            seen.add(key)
            unique_recs.append(r)
    
    for r in unique_recs[:25]:
        print(f"{r['date']} {r['time']} | {r['home_team']} vs {r['away_team']}")
        print(f"  Bet: {r['bet']} @ {r['odds']:.2f}")
        print(f"  Model: {r['model_total']:.2f} goals | Prob: {r['probability']*100:.1f}% | Edge: {r['edge']*100:.1f}%")
        print(f"  Stake: {r['kelly']*100:.1f}% of bankroll")
        print()
    
    print(f'{"="*80}')
    print(f'Total value bets found: {len(unique_recs)}')
    total_stake = sum(r['kelly'] for r in unique_recs[:25])
    print(f'Total recommended stake (top 25): {total_stake*100:.1f}% of bankroll')
