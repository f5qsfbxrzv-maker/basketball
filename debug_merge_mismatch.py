"""Debug why cleaned odds still show 50% coverage"""

import pandas as pd

# Load data
training = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
training['date'] = pd.to_datetime(training['date'])

odds_2023 = pd.read_csv('data/closing_odds_2023_24_CLEANED.csv')
odds_2024 = pd.read_csv('data/closing_odds_2024_25_CLEANED.csv')

all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
all_odds['game_date'] = pd.to_datetime(all_odds['game_date'])

# Check what dates/teams exist in each
print("="*80)
print("🔍 DEBUGGING MERGE MISMATCH")
print("="*80)

# Sample from Oct 24, 2023 (first unmatched)
print("\n📅 October 24, 2023 - Sample Day Analysis")
print("-"*80)

training_sample = training[training['date'] == '2023-10-24']
odds_sample = all_odds[all_odds['game_date'] == '2023-10-24']

print(f"\nTraining data games ({len(training_sample)}):")
for _, row in training_sample.iterrows():
    print(f"  {row['away_team']} @ {row['home_team']}")

print(f"\nOdds data games ({len(odds_sample)}):")
for _, row in odds_sample.iterrows():
    print(f"  {row['away_team']} @ {row['home_team']}")

# Check team columns
print("\n\n🏀 COLUMN NAMES CHECK")
print("-"*80)
print("Training columns:", [c for c in training.columns if 'team' in c])
print("Odds columns:", [c for c in all_odds.columns if 'team' in c])

# Look at raw values
print("\n\n📝 RAW VALUES COMPARISON")
print("-"*80)
print("\nTraining - first game:")
print(training_sample.iloc[0][['date', 'home_team', 'away_team']])

print("\nOdds - first game:")
print(odds_sample.iloc[0][['game_date', 'home_team', 'away_team']])

# Check for any data type issues
print("\n\n🔤 DATA TYPES")
print("-"*80)
print("Training date type:", training['date'].dtype)
print("Training home_team type:", training['home_team'].dtype)
print("Odds game_date type:", all_odds['game_date'].dtype)
print("Odds home_team type:", all_odds['home_team'].dtype)

# Check for whitespace or case issues
print("\n\n🎯 EXACT STRING CHECK (first 5 teams from each)")
print("-"*80)
training_teams = sorted(training['home_team'].unique())[:5]
odds_teams = sorted(all_odds['home_team'].unique())[:5]

print("Training teams:", training_teams)
print("Odds teams:", odds_teams)

for t in training_teams:
    print(f"  Training: '{t}' | len={len(t)} | repr={repr(t)}")
    
for t in odds_teams:
    print(f"  Odds: '{t}' | len={len(t)} | repr={repr(t)}")

# Test simple merge
print("\n\n🧪 TEST MERGE (single game)")
print("-"*80)

test_game = training_sample.iloc[0]
print(f"Looking for: {test_game['date'].date()} | {test_game['away_team']} @ {test_game['home_team']}")

match = all_odds[
    (all_odds['game_date'] == test_game['date']) &
    (all_odds['home_team'] == test_game['home_team']) &
    (all_odds['away_team'] == test_game['away_team'])
]

if len(match) > 0:
    print("✅ FOUND MATCH")
    print(match[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']])
else:
    print("❌ NO MATCH")
    print("\nChecking if game exists with ANY team combo...")
    date_matches = all_odds[all_odds['game_date'] == test_game['date']]
    print(f"Games on {test_game['date'].date()}: {len(date_matches)}")
    if len(date_matches) > 0:
        print(date_matches[['game_date', 'home_team', 'away_team']])

print("\n" + "="*80)
