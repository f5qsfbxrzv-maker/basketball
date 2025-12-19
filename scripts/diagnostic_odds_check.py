"""
Diagnostic Check - Verify Odds and Bet Logic
"""

import pandas as pd

# Load a sample of the odds
odds_df = pd.read_csv('data/live/closing_odds_2024_25.csv')
odds_df = odds_df.rename(columns={
    'game_date': 'date',
    'home_ml_odds': 'home_ml',
    'away_ml_odds': 'away_ml'
})

print("Sample odds data:")
print(odds_df[['date', 'home_team', 'away_team', 'home_ml', 'away_ml']].head(20))

# Check odds distribution
print("\n" + "="*70)
print("ODDS DISTRIBUTION")
print("="*70)

home_fav = (odds_df['home_ml'] < 0).sum()
away_fav = (odds_df['away_ml'] < 0).sum()
total = len(odds_df)

print(f"Home favorites: {home_fav} ({home_fav/total*100:.1f}%)")
print(f"Away favorites: {away_fav} ({away_fav/total*100:.1f}%)")
print(f"Total games: {total}")

# Check for any weird odds
print("\n" + "="*70)
print("ODDS SANITY CHECK")
print("="*70)

print(f"Home odds range: {odds_df['home_ml'].min()} to {odds_df['home_ml'].max()}")
print(f"Away odds range: {odds_df['away_ml'].min()} to {odds_df['away_ml'].max()}")

# Sample games with different scenarios
print("\n" + "="*70)
print("SAMPLE SCENARIOS")
print("="*70)

print("\nHome favorite games:")
home_fav_games = odds_df[odds_df['home_ml'] < 0].head(3)
print(home_fav_games[['date', 'home_team', 'away_team', 'home_ml', 'away_ml']])

print("\nAway favorite games:")
away_fav_games = odds_df[odds_df['away_ml'] < 0].head(3)
print(away_fav_games[['date', 'home_team', 'away_team', 'home_ml', 'away_ml']])

print("\nPick'em games (close odds):")
pickem = odds_df[(odds_df['home_ml'] > -150) & (odds_df['home_ml'] < 150) & 
                 (odds_df['away_ml'] > -150) & (odds_df['away_ml'] < 150)].head(3)
print(pickem[['date', 'home_team', 'away_team', 'home_ml', 'away_ml']])

# Test odds conversion
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

print("\n" + "="*70)
print("ODDS CONVERSION CHECK")
print("="*70)

test_odds = [-200, -150, -110, 100, 150, 200, 500]
for odds in test_odds:
    prob = american_to_prob(odds)
    print(f"  {odds:>5} → {prob*100:.1f}%")
