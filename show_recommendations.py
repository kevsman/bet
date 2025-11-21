import pandas as pd

df = pd.read_csv('data/processed/upcoming_recommendations.csv')

print("\n" + "="*80)
print("BETTING RECOMMENDATIONS FROM NORSK TIPPING")
print("="*80 + "\n")

for _, row in df.iterrows():
    print(f"{row['home_team']} vs {row['away_team']}")
    print(f"  Prediction: {row['selection']} {row['line']} @ {row['odds']:.2f}")
    print(f"  Model Total: {row['model_total']:.2f} goals")
    print(f"  Edge: {row['edge']*100:.1f}%")
    print(f"  Probability: {row['probability']*100:.1f}%")
    print(f"  Stake: {row['stake_fraction']*100:.1f}% of bankroll")
    print()

print("="*80)
print(f"Total recommendations: {len(df)}")
print(f"Total stake: {df['stake_fraction'].sum()*100:.1f}% of bankroll")
