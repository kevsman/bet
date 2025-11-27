"""
Calculate value bets from fresh Norsk Tipping odds.

Uses both models:
1. Feature-based Poisson model (with xG features if available)
2. Dixon-Coles model (team-specific attack/defense with correlation correction)

The Dixon-Coles model is particularly valuable for:
- Draw predictions
- Low-scoring game probabilities (0-0, 1-0, 0-1, 1-1)
- Under markets
"""
import pandas as pd
import numpy as np
from joblib import load
from scipy.stats import poisson
from pathlib import Path
import sys

# Add src to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.config import get_config

# ============================================================================
# LOAD MODELS
# ============================================================================

# Load feature-based Poisson models
home_model = load('models/home_poisson.joblib')
away_model = load('models/away_poisson.joblib')

# Load feature list
with open('models/features.txt', 'r') as f:
    features = [line.strip() for line in f if line.strip()]

# Separate xG features (may not be available for all teams)
xg_features = [f for f in features if 'xg' in f.lower()]
core_features = [f for f in features if 'xg' not in f.lower()]

# Load Dixon-Coles model if available
dixon_coles_model = None
dc_model_path = Path('models/dixon_coles.joblib')
if dc_model_path.exists():
    try:
        from src.dixon_coles import DixonColesModel
        dixon_coles_model = DixonColesModel.load(dc_model_path)
        print("✓ Dixon-Coles model loaded")
    except Exception as e:
        print(f"Warning: Could not load Dixon-Coles model: {e}")
else:
    print("Note: Dixon-Coles model not found, using feature-based model only")

# Load the historical dataset to get team stats (prefer xG-enhanced if available)
xg_dataset_path = Path('data/processed/match_dataset_with_xg.csv')
std_dataset_path = Path('data/processed/match_dataset.csv')

if xg_dataset_path.exists():
    dataset = pd.read_csv(xg_dataset_path, parse_dates=['Date'], low_memory=False)
else:
    dataset = pd.read_csv(std_dataset_path, parse_dates=['Date'], low_memory=False)
dataset = dataset.sort_values('Date')

# ============================================================================
# TEAM LOOKUPS AND MAPPINGS
# ============================================================================

LEAGUES = ['E0', 'E1', 'E2', 'E3', 'E4', 'D1', 'D2', 'SP1', 'SP2', 'I1', 'I2', 
           'F1', 'F2', 'SC0', 'SC1', 'N1', 'B1', 'P1', 'T1', 'G1']

def get_team_avg_features(team_name, is_home=True, n_matches=5):
    """Get average features from last N home/away matches for a team."""
    for league in LEAGUES:
        if is_home:
            matches = dataset[(dataset['league_code'] == league) & (dataset['HomeTeam'] == team_name)]
        else:
            matches = dataset[(dataset['league_code'] == league) & (dataset['AwayTeam'] == team_name)]
        
        if len(matches) >= 3:  # Need at least 3 matches for reliability
            available_features = [f for f in features if f in matches.columns]
            result = matches.tail(n_matches)[available_features].mean()
            
            # Fill missing xG features with 0
            for f in features:
                if f not in result.index:
                    result[f] = 0.0
            
            return result[features], league
    
    return None, None


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

# ============================================================================
# PROBABILITY CALCULATIONS
# ============================================================================

def bivariate_poisson_prob(line: float, home_xg: float, away_xg: float, 
                           over: bool = True, max_goals: int = 10) -> float:
    """
    Calculate over/under probability using independent bivariate Poisson.
    
    This is more accurate than single-lambda Poisson because it
    properly models home and away goals as independent Poisson processes.
    """
    prob = 0.0
    threshold = int(line)  # Under is ≤ floor(line)
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            total = h + a
            if over and total > threshold:
                prob += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            elif not over and total <= threshold:
                prob += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
    
    return prob


def ensemble_over_under(home_team: str, away_team: str, line: float,
                        poisson_home: float, poisson_away: float,
                        model_weight: float = 0.5) -> dict:
    """
    Ensemble prediction combining Poisson and Dixon-Coles models.
    
    Args:
        home_team, away_team: Team names
        line: Goal line (1.5, 2.5, 3.5)
        poisson_home, poisson_away: Expected goals from feature-based Poisson
        model_weight: Weight for Dixon-Coles (0=pure Poisson, 1=pure DC)
    
    Returns:
        Dict with 'over', 'under' probabilities and model details
    """
    # Feature-based Poisson probabilities
    poisson_over = bivariate_poisson_prob(line, poisson_home, poisson_away, over=True)
    poisson_under = bivariate_poisson_prob(line, poisson_home, poisson_away, over=False)
    
    # Dixon-Coles probabilities (if available)
    if dixon_coles_model is not None and home_team in dixon_coles_model.teams and away_team in dixon_coles_model.teams:
        dc_probs = dixon_coles_model.predict_over_under(home_team, away_team, line)
        dc_over = dc_probs['over']
        dc_under = dc_probs['under']
        
        # Ensemble: weighted average
        final_over = (1 - model_weight) * poisson_over + model_weight * dc_over
        final_under = (1 - model_weight) * poisson_under + model_weight * dc_under
        
        return {
            'over': final_over,
            'under': final_under,
            'poisson_over': poisson_over,
            'poisson_under': poisson_under,
            'dc_over': dc_over,
            'dc_under': dc_under,
            'model_used': 'ensemble'
        }
    else:
        return {
            'over': poisson_over,
            'under': poisson_under,
            'poisson_over': poisson_over,
            'poisson_under': poisson_under,
            'dc_over': None,
            'dc_under': None,
            'model_used': 'poisson_only'
        }


# ============================================================================
# MAIN RECOMMENDATION ENGINE
# ============================================================================

# Load fresh Norsk Tipping odds
odds_path = Path('data/processed/norsk_tipping_odds.csv')
if not odds_path.exists():
    odds_path = Path('data/upcoming/norsk_tipping_odds.csv')

odds_df = pd.read_csv(odds_path)
print(f'Loaded {len(odds_df)} matches from Norsk Tipping')

# Configuration
MIN_EDGE = 0.05  # 5% minimum edge
KELLY_FRACTION = 0.25  # Quarter Kelly
MAX_STAKE = 0.05  # Max 5% of bankroll
DC_WEIGHT = 0.4  # 40% Dixon-Coles, 60% Poisson (if DC available)

recommendations = []
skipped_teams = set()

for _, row in odds_df.iterrows():
    home = map_team(row['home_team'])
    away = map_team(row['away_team'])
    
    # Get averaged features from last 5 home/away matches
    home_feats, home_league = get_team_avg_features(home, is_home=True)
    away_feats, away_league = get_team_avg_features(away, is_home=False)
    
    if home_feats is None or away_feats is None:
        skipped_teams.add((row['home_team'], row['away_team']))
        continue
    
    # Check for any missing features
    if home_feats.isna().any() or away_feats.isna().any():
        continue
    
    # Predict using feature-based Poisson
    pred_home = float(np.clip(home_model.predict([home_feats])[0], 0.1, 5.0))
    pred_away = float(np.clip(away_model.predict([away_feats])[0], 0.1, 5.0))
    total = pred_home + pred_away
    
    # Get Dixon-Coles predictions if available
    dc_home, dc_away = None, None
    if dixon_coles_model is not None and home in dixon_coles_model.teams and away in dixon_coles_model.teams:
        dc_home, dc_away = dixon_coles_model.predict_goals(home, away)
    
    # Check all goal lines
    lines = [1.5, 2.5, 3.5]
    for line in lines:
        over_col = f'over_{line}'.replace('.', '_')
        under_col = f'under_{line}'.replace('.', '_')
        
        over_odds = row.get(over_col)
        under_odds = row.get(under_col)
        
        if pd.isna(over_odds) or pd.isna(under_odds):
            continue
        
        # Get ensemble probabilities
        probs = ensemble_over_under(home, away, line, pred_home, pred_away, DC_WEIGHT)
        p_over = probs['over']
        p_under = probs['under']
        
        # Calculate implied probabilities from odds
        implied_over = 1 / over_odds
        implied_under = 1 / under_odds
        
        # Calculate edge
        edge_over = p_over - implied_over
        edge_under = p_under - implied_under
        
        # Check for value in OVER bet
        if edge_over > MIN_EDGE:
            kelly = edge_over / (over_odds - 1) if over_odds > 1 else 0
            stake = min(kelly * KELLY_FRACTION, MAX_STAKE)
            
            recommendations.append({
                'date': row['date'],
                'time': row.get('time', ''),
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'bet': f'Over {line}',
                'line': line,
                'odds': over_odds,
                'model_total': total,
                'pred_home': pred_home,
                'pred_away': pred_away,
                'dc_home': dc_home,
                'dc_away': dc_away,
                'probability': p_over,
                'poisson_prob': probs['poisson_over'],
                'dc_prob': probs['dc_over'],
                'model_used': probs['model_used'],
                'edge': edge_over,
                'kelly': stake
            })
        
        # Check for value in UNDER bet
        if edge_under > MIN_EDGE:
            kelly = edge_under / (under_odds - 1) if under_odds > 1 else 0
            stake = min(kelly * KELLY_FRACTION, MAX_STAKE)
            
            recommendations.append({
                'date': row['date'],
                'time': row.get('time', ''),
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'bet': f'Under {line}',
                'line': line,
                'odds': under_odds,
                'model_total': total,
                'pred_home': pred_home,
                'pred_away': pred_away,
                'dc_home': dc_home,
                'dc_away': dc_away,
                'probability': p_under,
                'poisson_prob': probs['poisson_under'],
                'dc_prob': probs['dc_under'],
                'model_used': probs['model_used'],
                'edge': edge_under,
                'kelly': stake
            })

# Sort by edge
recommendations = sorted(recommendations, key=lambda x: -x['edge'])

# Remove duplicates (same match, same bet)
seen = set()
unique_recs = []
for r in recommendations:
    key = (r['home_team'], r['away_team'], r['bet'])
    if key not in seen:
        seen.add(key)
        unique_recs.append(r)

# ============================================================================
# OUTPUT
# ============================================================================

print(f'\n{"="*80}')
print('VALUE BETS - DIXON-COLES + POISSON ENSEMBLE')
print(f'{"="*80}\n')

if skipped_teams:
    print(f"Note: {len(skipped_teams)} matches skipped (teams not in dataset)")

if not unique_recs:
    print('No value bets found with >5% edge')
else:
    for r in unique_recs[:30]:
        print(f"{r['date']} {r['time']} | {r['home_team']} vs {r['away_team']}")
        print(f"  Bet: {r['bet']} @ {r['odds']:.2f}")
        
        # Model predictions
        model_str = f"Poisson: {r['pred_home']:.2f}-{r['pred_away']:.2f}"
        if r['dc_home'] is not None:
            model_str += f" | DC: {r['dc_home']:.2f}-{r['dc_away']:.2f}"
        print(f"  {model_str}")
        
        # Probabilities
        prob_str = f"Final: {r['probability']*100:.1f}%"
        if r['dc_prob'] is not None:
            prob_str += f" (Poisson: {r['poisson_prob']*100:.1f}%, DC: {r['dc_prob']*100:.1f}%)"
        print(f"  {prob_str}")
        
        print(f"  Edge: {r['edge']*100:.1f}% | Stake: {r['kelly']*100:.1f}%")
        print()
    
    # Summary statistics
    print(f'{"="*80}')
    print(f'SUMMARY')
    print(f'{"="*80}')
    print(f'Total value bets found: {len(unique_recs)}')
    
    # Breakdown by line
    for line in [1.5, 2.5, 3.5]:
        line_recs = [r for r in unique_recs if r['line'] == line]
        over_count = len([r for r in line_recs if 'Over' in r['bet']])
        under_count = len([r for r in line_recs if 'Under' in r['bet']])
        if line_recs:
            avg_edge = np.mean([r['edge'] for r in line_recs])
            print(f"  {line} line: {len(line_recs)} bets ({over_count} over, {under_count} under), avg edge: {avg_edge*100:.1f}%")
    
    total_stake = sum(r['kelly'] for r in unique_recs[:30])
    print(f'\nTotal recommended stake (top 30): {total_stake*100:.1f}% of bankroll')
    
    # Model usage stats
    ensemble_count = len([r for r in unique_recs if r['model_used'] == 'ensemble'])
    poisson_only = len([r for r in unique_recs if r['model_used'] == 'poisson_only'])
    print(f'Model usage: {ensemble_count} ensemble, {poisson_only} Poisson-only')

# Save recommendations to CSV
if unique_recs:
    recs_df = pd.DataFrame(unique_recs)
    cfg = get_config()
    output_path = cfg.processed_dir / 'recommendations.csv'
    recs_df.to_csv(output_path, index=False)
    print(f'\nRecommendations saved to {output_path}')
