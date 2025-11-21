
import math

def poisson_probability(k, lam):
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def explain_bet(home_team, away_team, model_total, odds_under):
    print(f"--- Analysis for {home_team} vs {away_team} ---")
    print(f"Model Predicted Total Goals (Lambda): {model_total}")
    print(f"Market Odds for Under 2.5: {odds_under}")
    
    # Calculate probabilities for 0, 1, and 2 goals
    p0 = poisson_probability(0, model_total)
    p1 = poisson_probability(1, model_total)
    p2 = poisson_probability(2, model_total)
    
    prob_under_2_5 = p0 + p1 + p2
    
    print(f"\nProbability breakdown:")
    print(f"  Chance of exactly 0 goals: {p0*100:.1f}%")
    print(f"  Chance of exactly 1 goal:  {p1*100:.1f}%")
    print(f"  Chance of exactly 2 goals: {p2*100:.1f}%")
    print(f"  --------------------------------")
    print(f"  Total Probability (Under 2.5): {prob_under_2_5*100:.1f}%")
    
    # Value Calculation
    implied_prob = 1 / odds_under
    edge = prob_under_2_5 - implied_prob
    
    print(f"\nValue Calculation:")
    print(f"  Implied Probability (1/{odds_under}): {implied_prob*100:.1f}%")
    print(f"  Model Probability:            {prob_under_2_5*100:.1f}%")
    print(f"  Edge (Model - Implied):       {edge*100:.1f}%")
    
    if edge > 0:
        print(f"\nCONCLUSION: Even though the average is {model_total} (which is > 2.5),")
        print(f"the {prob_under_2_5*100:.1f}% chance of it staying under is HIGHER than the {implied_prob*100:.1f}% chance the odds imply.")
        print("Therefore, it is a profitable bet in the long run.")
    else:
        print("\nCONCLUSION: No value found.")

# Example from your report
explain_bet("Liverpool", "Nott'm Forest", 2.79, 2.50)
