
import pandas as pd

def recommend_portfolio(budget=500, num_bets=5):
    # Load recommendations
    df = pd.read_csv("data/processed/upcoming_recommendations.csv")
    
    # Sort by edge (highest value first)
    df = df.sort_values("edge", ascending=False)
    
    # Remove duplicates (keep best line for each match)
    df = df.drop_duplicates(subset=["match_id"], keep="first")
    
    # Select top N bets
    top_picks = df.head(num_bets).copy()
    
    # Calculate stake per bet (equal stake strategy as requested)
    stake_per_bet = budget / num_bets
    
    print(f"\n💰 RECOMMENDED PORTFOLIO ({budget} NOK Budget)")
    print(f"Strategy: {num_bets} bets of {stake_per_bet:.0f} NOK each\n")
    print(f"{'MATCH':<35} {'SELECTION':<15} {'ODDS':<8} {'EDGE':<8} {'POTENTIAL RETURN'}")
    print("-" * 90)
    
    total_potential_return = 0
    
    for _, row in top_picks.iterrows():
        match_name = f"{row['home_team']} vs {row['away_team']}"
        selection = f"{row['selection']} {row['line']}"
        potential_return = stake_per_bet * row['odds']
        total_potential_return += potential_return
        
        print(f"{match_name:<35} {selection:<15} {row['odds']:<8.2f} {row['edge']*100:<7.1f}% {potential_return:.0f} NOK")
        
    print("-" * 90)
    print(f"TOTAL INVESTMENT: {budget} NOK")
    print(f"MAX POTENTIAL RETURN: {total_potential_return:.0f} NOK")
    print(f"AVG ODDS: {top_picks['odds'].mean():.2f}")

if __name__ == "__main__":
    recommend_portfolio(500, 5)
