import pandas as pd

df = pd.read_csv('data/processed/match_dataset.csv', low_memory=False)
leagues = df['league_code'].unique()

print('Available leagues and sample teams:\n')
for league in sorted(leagues):
    teams = sorted(set(list(df[df['league_code']==league]['home_team'].unique()) + 
                      list(df[df['league_code']==league]['away_team'].unique())))
    print(f'\n{league}: {len(teams)} teams')
    print(', '.join(teams[:15]))
    if len(teams) > 15:
        print('...')
