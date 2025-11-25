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

# Build indices for quick team lookups
LEAGUES = ['E0', 'E1', 'E2', 'E3', 'E4', 'D1', 'D2', 'SP1', 'SP2', 'I1', 'I2', 'F1', 'F2', 'SC0', 'SC1', 'N1', 'B1', 'P1', 'T1', 'G1']

def get_team_avg_features(team_name, is_home=True, n_matches=5):
    """Get average features from last N home/away matches for a team.
    
    This is more robust than using a single match's features because
    it smooths out variance from individual match fluctuations.
    """
    for league in LEAGUES:
        if is_home:
            matches = dataset[(dataset['league_code'] == league) & (dataset['HomeTeam'] == team_name)]
        else:
            matches = dataset[(dataset['league_code'] == league) & (dataset['AwayTeam'] == team_name)]
        
        if len(matches) >= 3:  # Need at least 3 matches for reliability
            return matches.tail(n_matches)[features].mean(), league
    
    return None, None

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
    'Atalanta BC': 'Atalanta',
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
    'Olympiakos': 'Olympiakos',
    'Real Madrid': 'Real Madrid',
    'Inter': 'Inter',
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

def bivariate_prob_under(line, home_xg, away_xg):
    """Calculate P(total goals <= line) using bivariate Poisson.
    
    This is more accurate than single-lambda Poisson because it
    properly models home and away goals as independent Poisson processes.
    """
    prob = 0.0
    max_goals = 10  # Sum over reasonable range
    for h in range(max_goals):
        for a in range(max_goals):
            if h + a <= line:
                prob += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
    return prob

def bivariate_prob_over(line, home_xg, away_xg):
    """Calculate P(total goals > line) using bivariate Poisson."""
    return 1 - bivariate_prob_under(line, home_xg, away_xg)

# Try to find matches in our dataset
recommendations = []

for _, row in odds_df.iterrows():
    home = map_team(row['home_team'])
    away = map_team(row['away_team'])
    
    # Get averaged features from last 5 home/away matches (more robust)
    home_feats, home_league = get_team_avg_features(home, is_home=True)
    away_feats, away_league = get_team_avg_features(away, is_home=False)
    
    if home_feats is None or away_feats is None:
        continue
    
    # Check for any missing features
    if home_feats.isna().any() or away_feats.isna().any():
        continue
    
    # Predict using the full feature vectors directly
    # home_feats contains the home team's performance when playing at HOME
    # away_feats contains the away team's performance when playing AWAY
    # Each model expects the full feature vector from that context
    pred_home = float(np.clip(home_model.predict([home_feats])[0], 0.1, 5.0))
    pred_away = float(np.clip(away_model.predict([away_feats])[0], 0.1, 5.0))
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
        
        p_over = bivariate_prob_over(line, pred_home, pred_away)
        p_under = bivariate_prob_under(line, pred_home, pred_away)
        
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
