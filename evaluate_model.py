"""Evaluate model performance with new features."""
import pandas as pd
import numpy as np
from pathlib import Path

# Load predictions
preds = pd.read_csv('data/processed/model_predictions.csv', parse_dates=['Date'])

# Filter to test set only
test = preds[preds['dataset_split'] == 'test'].copy()

print('='*60)
print('MODEL PERFORMANCE EVALUATION')
print('='*60)

# Basic metrics
test['home_error'] = test['pred_home_goals'] - test['FTHG']
test['away_error'] = test['pred_away_goals'] - test['FTAG']
test['total_error'] = test['pred_total_goals'] - test['total_goals']

print(f'\nTest Set Size: {len(test)} matches')
print(f'Seasons: {test["season_code"].unique()}')

print('\n--- Goal Prediction Accuracy ---')
print(f'Home Goals MAE:  {test["home_error"].abs().mean():.3f}')
print(f'Away Goals MAE:  {test["away_error"].abs().mean():.3f}')
print(f'Total Goals MAE: {test["total_error"].abs().mean():.3f}')

print(f'\nHome Goals RMSE: {np.sqrt((test["home_error"]**2).mean()):.3f}')
print(f'Away Goals RMSE: {np.sqrt((test["away_error"]**2).mean()):.3f}')
print(f'Total Goals RMSE: {np.sqrt((test["total_error"]**2).mean()):.3f}')

# Over/Under accuracy
test['actual_over'] = test['total_goals'] > 2.5
test['pred_over'] = test['cal_over_prob'] > 0.5

accuracy = (test['actual_over'] == test['pred_over']).mean()
print(f'\n--- Over/Under 2.5 Classification ---')
print(f'Accuracy: {accuracy*100:.1f}%')

# Calibration check
print('\n--- Probability Calibration ---')
for thresh in [0.4, 0.5, 0.6, 0.7]:
    high_conf = test[test['cal_over_prob'] >= thresh]
    if len(high_conf) > 0:
        actual_rate = high_conf['actual_over'].mean()
        print(f'When P(over) >= {thresh}: {len(high_conf)} bets, actual over rate = {actual_rate*100:.1f}%')

# Check feature usage
print('\n--- Features Used ---')
features_file = Path('models/features.txt')
if features_file.exists():
    features = features_file.read_text().strip().split('\n')
    print(f'Total features: {len(features)}')
    
    shot_features = [f for f in features if 'shot' in f.lower()]
    xg_features = [f for f in features if 'xg' in f.lower()]
    conversion_features = [f for f in features if 'conversion' in f.lower() or 'accuracy' in f.lower()]
    
    print(f'Shot-related features: {len(shot_features)}')
    print(f'xG features: {len(xg_features)}')
    print(f'Conversion/Accuracy features: {len(conversion_features)}')
    
    if shot_features:
        print(f'\nShot features: {shot_features[:6]}...')
    if conversion_features:
        print(f'Quality features: {conversion_features[:6]}...')

# League breakdown
print('\n--- Performance by League ---')
league_perf = test.groupby('league_code').agg({
    'total_error': lambda x: np.sqrt((x**2).mean()),
    'actual_over': 'mean',
    'match_id': 'count'
}).rename(columns={'total_error': 'RMSE', 'actual_over': 'Over_Rate', 'match_id': 'Matches'})
league_perf = league_perf.sort_values('Matches', ascending=False).head(10)
print(league_perf.round(3).to_string())
