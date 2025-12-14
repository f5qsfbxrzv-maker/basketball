"""
Check Moneyline Data Integrity: Compare ML odds vs Spread implied probabilities
Identify corrupted odds that don't match expected values from spreads
"""

import sqlite3
import pandas as pd
import numpy as np

# Load both odds and training data to cross-check
conn = sqlite3.connect('data/live/historical_closing_odds.db')
odds_df = pd.read_sql('SELECT * FROM moneyline_odds ORDER BY game_date', conn)
conn.close()

# Load training data with spreads
train_df = pd.read_csv('data/training_data_with_temporal_features.csv')
train_df['date'] = pd.to_datetime(train_df['date'])

# Convert game_date to datetime for proper merge
odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

# Merge to get spreads alongside ML odds
merged = train_df.merge(
    odds_df,
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='inner'
)

def american_to_prob(odds):
    if pd.isna(odds) or odds == 0: return 0.5
    if odds < 0:
        return (-odds) / (-odds + 100)
    else:
        return 100 / (odds + 100)

def spread_to_ml_prob(spread):
    # Approximation: 50% baseline, adjust by spread
    return 0.50 - (spread * 0.0335)

# Calculate probabilities
merged['home_ml_prob'] = merged['home_ml_odds'].apply(american_to_prob)
merged['spread_implied_prob'] = merged['target_spread'].apply(spread_to_ml_prob)
merged['prob_diff'] = abs(merged['home_ml_prob'] - merged['spread_implied_prob'])

# Find suspicious odds
suspicious = merged[merged['prob_diff'] > 0.20].copy()

print('='*100)
print('MONEYLINE DATA INTEGRITY CHECK')
print('='*100)
print(f'Total games with odds: {len(merged)}')
print(f'Games with >20% spread/ML mismatch: {len(suspicious)} ({len(suspicious)/len(merged)*100:.1f}%)')
print()

if len(suspicious) > 0:
    print('TOP 20 SUSPICIOUS ODDS (Spread vs ML Mismatch):')
    print('='*100)
    suspicious_sorted = suspicious.sort_values('prob_diff', ascending=False).head(20)
    
    for _, row in suspicious_sorted.iterrows():
        game_date_str = row['game_date'].strftime('%Y-%m-%d') if pd.notna(row['game_date']) else 'Unknown'
        game_str = f"{game_date_str} | {row['home_team']} vs {row['away_team']}"
        spread_str = f"  Spread: {row['target_spread']:+.1f} -> Implied {row['spread_implied_prob']:.1%} home win prob"
        ml_str = f"  ML Odds: {row['home_ml_odds']:+.0f} -> Implied {row['home_ml_prob']:.1%} home win prob"
        diff_str = f"  DIFFERENCE: {row['prob_diff']:.1%} (RED FLAG if >20%)"
        
        print(game_str)
        print(spread_str)
        print(ml_str)
        print(diff_str)
        print()

# Statistics on extreme odds
extreme_favorites = merged[merged['home_ml_odds'] < -500]
extreme_underdogs = merged[merged['home_ml_odds'] > 500]

print('='*100)
print('EXTREME ODDS ANALYSIS')
print('='*100)
print(f'Home teams at <-500 (>83% implied): {len(extreme_favorites)}')
print(f'Home teams at >+500 (<17% implied): {len(extreme_underdogs)}')
print()

if len(extreme_favorites) > 0:
    print('EXTREME FAVORITES (check if realistic):')
    for _, row in extreme_favorites.head(10).iterrows():
        game_date_str = row['game_date'].strftime('%Y-%m-%d') if pd.notna(row['game_date']) else 'Unknown'
        game_str = f"  {game_date_str} | {row['home_team']} vs {row['away_team']} | ML: {row['home_ml_odds']:+.0f} | Spread: {row['target_spread']:+.1f}"
        print(game_str)

print()
print('='*100)
print('RECOMMENDATION')
print('='*100)
if len(suspicious) > 50:
    print('❌ HIGH CORRUPTION: >50 games with bad odds. Use HYBRID approach (Real + Spread-derived).')
elif len(suspicious) > 10:
    print('⚠️  MODERATE CORRUPTION: 10-50 games with bad odds. Filter or use hybrid approach.')
else:
    print('✅ LOW CORRUPTION: <10 games with bad odds. Data mostly clean.')
