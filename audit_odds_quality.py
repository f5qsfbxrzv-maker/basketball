"""
Audit 2023-24 Odds Data Quality
- Check for outliers
- Verify moneyline odds (not point spreads)
- Validate payout calculations
- Compare to 2024-25 odds distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*90)
print("ODDS DATA QUALITY AUDIT - 2023-24 vs 2024-25")
print("="*90)

# Load both seasons
odds_2023 = pd.read_csv('data/closing_odds_2023_24.csv')
odds_2024 = pd.read_csv('data/live/closing_odds_2024_25.csv')

print(f"\n[1/5] BASIC STATISTICS")
print(f"="*90)

for name, df in [('2023-24', odds_2023), ('2024-25', odds_2024)]:
    print(f"\n{name} Season:")
    print(f"  Total games: {len(df)}")
    print(f"  Home odds range: [{df['home_ml_odds'].min():.0f}, {df['home_ml_odds'].max():.0f}]")
    print(f"  Away odds range: [{df['away_ml_odds'].min():.0f}, {df['away_ml_odds'].max():.0f}]")
    print(f"  Home odds mean: {df['home_ml_odds'].mean():.1f}")
    print(f"  Away odds mean: {df['away_ml_odds'].mean():.1f}")

# Check for extreme outliers
print(f"\n[2/5] OUTLIER DETECTION")
print(f"="*90)

def check_outliers(df, season_name):
    print(f"\n{season_name}:")
    
    # Extreme values (likely errors)
    extreme = df[
        (df['home_ml_odds'] < -2000) | (df['home_ml_odds'] > 2000) |
        (df['away_ml_odds'] < -2000) | (df['away_ml_odds'] > 2000)
    ]
    print(f"  Extreme outliers (|odds| > 2000): {len(extreme)} games")
    if len(extreme) > 0:
        print(f"    Home odds range: [{extreme['home_ml_odds'].min():.0f}, {extreme['home_ml_odds'].max():.0f}]")
        print(f"    Away odds range: [{extreme['away_ml_odds'].min():.0f}, {extreme['away_ml_odds'].max():.0f}]")
        print(f"\n    Sample outliers:")
        for _, row in extreme.head(3).iterrows():
            print(f"      {row['game_date']} - {row['home_team']} vs {row['away_team']}")
            print(f"        Home: {row['home_ml_odds']:.0f}, Away: {row['away_ml_odds']:.0f}")
    
    # Heavy favorites/underdogs (plausible but rare)
    heavy = df[
        ((df['home_ml_odds'] < -1000) | (df['home_ml_odds'] > 1000) |
         (df['away_ml_odds'] < -1000) | (df['away_ml_odds'] > 1000)) &
        ((df['home_ml_odds'] >= -2000) & (df['home_ml_odds'] <= 2000) &
         (df['away_ml_odds'] >= -2000) & (df['away_ml_odds'] <= 2000))
    ]
    print(f"  Heavy favorites/dogs (1000 < |odds| < 2000): {len(heavy)} games")
    
    # Normal range
    normal = df[
        (df['home_ml_odds'] >= -1000) & (df['home_ml_odds'] <= 1000) &
        (df['away_ml_odds'] >= -1000) & (df['away_ml_odds'] <= 1000)
    ]
    print(f"  Normal range (|odds| <= 1000): {len(normal)} games ({len(normal)/len(df)*100:.1f}%)")

check_outliers(odds_2023, '2023-24')
check_outliers(odds_2024, '2024-25')

# Check for point spread contamination
print(f"\n[3/5] MONEYLINE vs SPREAD CHECK")
print(f"="*90)

def check_spread_contamination(df, season_name):
    """
    Moneyline odds are typically:
    - Negative for favorites (-110 to -500 common)
    - Positive for underdogs (+100 to +500 common)
    
    Point spreads are small numbers like:
    - -7.5, +7.5, -3.5, +3.5
    
    If we see lots of small numbers (< 100 absolute value), might be spreads
    """
    print(f"\n{season_name}:")
    
    # Count very small odds (suspicious for moneyline)
    small_home = df[(df['home_ml_odds'] > -100) & (df['home_ml_odds'] < 100)]
    small_away = df[(df['away_ml_odds'] > -100) & (df['away_ml_odds'] < 100)]
    
    print(f"  Games with |home odds| < 100: {len(small_home)} ({len(small_home)/len(df)*100:.1f}%)")
    print(f"  Games with |away odds| < 100: {len(small_away)} ({len(small_away)/len(df)*100:.1f}%)")
    
    if len(small_home) > 0:
        print(f"    Sample small home odds: {small_home['home_ml_odds'].head(10).tolist()}")
    
    # Check if both teams have similar odds (common in spreads, rare in moneyline)
    df['odds_diff'] = abs(df['home_ml_odds'] - df['away_ml_odds'])
    similar = df[df['odds_diff'] < 50]
    print(f"  Games with very similar odds (diff < 50): {len(similar)} ({len(similar)/len(df)*100:.1f}%)")
    
    # Typical moneyline pattern: one negative, one positive
    typical = df[
        ((df['home_ml_odds'] < 0) & (df['away_ml_odds'] > 0)) |
        ((df['home_ml_odds'] > 0) & (df['away_ml_odds'] < 0))
    ]
    print(f"  Typical moneyline pattern (one neg, one pos): {len(typical)} ({len(typical)/len(df)*100:.1f}%)")

check_spread_contamination(odds_2023, '2023-24')
check_spread_contamination(odds_2024, '2024-25')

# Validate implied probability
print(f"\n[4/5] IMPLIED PROBABILITY CHECK")
print(f"="*90)

def american_to_implied_prob(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def check_probabilities(df, season_name):
    print(f"\n{season_name}:")
    
    df['home_prob'] = df['home_ml_odds'].apply(american_to_implied_prob)
    df['away_prob'] = df['away_ml_odds'].apply(american_to_implied_prob)
    df['total_prob'] = df['home_prob'] + df['away_prob']
    
    print(f"  Home win probability: {df['home_prob'].mean():.1%} ± {df['home_prob'].std():.1%}")
    print(f"  Away win probability: {df['away_prob'].mean():.1%} ± {df['away_prob'].std():.1%}")
    print(f"  Total probability (should be >100% due to vig):")
    print(f"    Mean: {df['total_prob'].mean():.1%}")
    print(f"    Range: [{df['total_prob'].min():.1%}, {df['total_prob'].max():.1%}]")
    
    # Flag suspicious probabilities
    suspicious = df[(df['total_prob'] < 1.0) | (df['total_prob'] > 1.3)]
    print(f"  Suspicious total prob (< 100% or > 130%): {len(suspicious)} games")
    if len(suspicious) > 0:
        print(f"    Sample:")
        for _, row in suspicious.head(3).iterrows():
            print(f"      {row['home_team']} vs {row['away_team']}")
            print(f"        Home: {row['home_ml_odds']:.0f} ({row['home_prob']:.1%})")
            print(f"        Away: {row['away_ml_odds']:.0f} ({row['away_prob']:.1%})")
            print(f"        Total: {row['total_prob']:.1%}")

check_probabilities(odds_2023, '2023-24')
check_probabilities(odds_2024, '2024-25')

# Sample payout calculations
print(f"\n[5/5] SAMPLE PAYOUT VALIDATION")
print(f"="*90)

def validate_payouts(df, season_name):
    print(f"\n{season_name} - Sample Payout Examples:")
    
    # Get a few typical games
    samples = df[
        (df['home_ml_odds'] >= -1000) & (df['home_ml_odds'] <= 1000) &
        (df['away_ml_odds'] >= -1000) & (df['away_ml_odds'] <= 1000)
    ].sample(min(5, len(df)))
    
    for _, row in samples.iterrows():
        home_odds = row['home_ml_odds']
        away_odds = row['away_ml_odds']
        
        # Calculate payouts for $100 bet
        if home_odds > 0:
            home_payout = (home_odds / 100) * 100
        else:
            home_payout = (100 / abs(home_odds)) * 100
        
        if away_odds > 0:
            away_payout = (away_odds / 100) * 100
        else:
            away_payout = (100 / abs(away_odds)) * 100
        
        print(f"\n  {row['home_team']} vs {row['away_team']}")
        print(f"    Bet $100 on home ({home_odds:+.0f}): Win ${home_payout:.2f}")
        print(f"    Bet $100 on away ({away_odds:+.0f}): Win ${away_payout:.2f}")

validate_payouts(odds_2023, '2023-24')
validate_payouts(odds_2024, '2024-25')

print(f"\n{'='*90}")
print(f"AUDIT COMPLETE")
print(f"{'='*90}\n")
