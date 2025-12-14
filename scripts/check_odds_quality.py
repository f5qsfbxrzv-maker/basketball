"""
Deep dive into the odds data to find remaining issues
"""

import pandas as pd
import sqlite3

# Load the odds database
conn = sqlite3.connect('data/backups/nba_ODDS_history.db')

# Get all odds with details
odds_df = pd.read_sql("""
    SELECT game_date, home_team, away_team, home_ml_odds, away_ml_odds, spread_line
    FROM odds_history 
    WHERE home_ml_odds IS NOT NULL
    ORDER BY game_date, home_team
""", conn)
conn.close()

print("="*80)
print("ODDS DATA QUALITY CHECK")
print("="*80)

print(f"\nTotal rows: {len(odds_df)}")

# Check for extreme odds (likely data errors)
print(f"\n{'='*80}")
print("EXTREME ODDS DETECTION")
print("="*80)

extreme_favorites = odds_df[(odds_df['home_ml_odds'] < -500) | (odds_df['away_ml_odds'] < -500)]
extreme_underdogs = odds_df[(odds_df['home_ml_odds'] > 500) | (odds_df['away_ml_odds'] > 500)]

print(f"\nExtreme favorites (< -500):")
print(f"  Count: {len(extreme_favorites)}")
if len(extreme_favorites) > 0:
    print(extreme_favorites[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']].head(10))

print(f"\nExtreme underdogs (> +500):")
print(f"  Count: {len(extreme_underdogs)}")
if len(extreme_underdogs) > 0:
    print(extreme_underdogs[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']].head(10))

# Check for duplicates
print(f"\n{'='*80}")
print("DUPLICATE GAMES CHECK")
print("="*80)

odds_df['matchup'] = odds_df['home_team'] + '_' + odds_df['away_team']
duplicates = odds_df.groupby('matchup').size()
dupe_matchups = duplicates[duplicates > 1].sort_values(ascending=False)

print(f"\nMatchups with multiple odds entries: {len(dupe_matchups)}")
print(f"\nTop 10 most duplicated:")
print(dupe_matchups.head(10))

# Show examples of duplicates for a specific game
if len(dupe_matchups) > 0:
    top_dupe = dupe_matchups.index[0]
    print(f"\nExample: {top_dupe}")
    example = odds_df[odds_df['matchup'] == top_dupe][['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds', 'spread_line']]
    print(example.to_string())

# Check odds distribution
print(f"\n{'='*80}")
print("ODDS DISTRIBUTION")
print("="*80)

print("\nHome ML odds distribution:")
print(odds_df['home_ml_odds'].describe())

print("\nAway ML odds distribution:")
print(odds_df['away_ml_odds'].describe())

# Check for unrealistic odds patterns
print(f"\n{'='*80}")
print("UNREALISTIC PATTERNS")
print("="*80)

# Both teams favorites (impossible)
both_favorites = odds_df[(odds_df['home_ml_odds'] < 0) & (odds_df['away_ml_odds'] < 0)]
print(f"\nGames where both teams are favorites: {len(both_favorites)}")
if len(both_favorites) > 0:
    print(both_favorites.head())

# Both teams underdogs (impossible)
both_underdogs = odds_df[(odds_df['home_ml_odds'] > 0) & (odds_df['away_ml_odds'] > 0)]
print(f"\nGames where both teams are underdogs: {len(both_underdogs)}")
if len(both_underdogs) > 0:
    print(both_underdogs.head())

# Check implied probability sum (should be > 100% due to vig)
def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

odds_df['home_implied'] = odds_df['home_ml_odds'].apply(american_to_implied)
odds_df['away_implied'] = odds_df['away_ml_odds'].apply(american_to_implied)
odds_df['total_implied'] = odds_df['home_implied'] + odds_df['away_implied']

print(f"\n{'='*80}")
print("VIG CHECK (Total Implied Probability)")
print("="*80)
print("\nShould be 1.04-1.06 (4-6% vig)")
print(odds_df['total_implied'].describe())

suspicious_vig = odds_df[(odds_df['total_implied'] < 1.00) | (odds_df['total_implied'] > 1.15)]
print(f"\nGames with suspicious vig (< 100% or > 115%): {len(suspicious_vig)}")
if len(suspicious_vig) > 0:
    print(suspicious_vig[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds', 'total_implied']].head(10))

print("\n" + "="*80)
