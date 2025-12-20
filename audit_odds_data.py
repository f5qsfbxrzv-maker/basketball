"""
ODDS DATA QUALITY AUDIT
========================
Checks historical odds files for:
1. Coverage (% of games with odds)
2. Data quality (outliers, corrupted values)
3. Merge issues (why games aren't matching)
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Load training data
print("="*80)
print("📊 ODDS DATA QUALITY AUDIT")
print("="*80)

# 1. Load training data
print("\n1️⃣ Loading training data...")
training_df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
training_df['date'] = pd.to_datetime(training_df['date'])
training_df = training_df.sort_values('date')

print(f"✓ Total games in training data: {len(training_df):,}")
print(f"  Date range: {training_df['date'].min().date()} to {training_df['date'].max().date()}")

# 2. Load odds files
print("\n2️⃣ Loading odds files...")
odds_2023 = pd.read_csv('data/closing_odds_2023_24.csv')
odds_2024 = pd.read_csv('data/live/closing_odds_2024_25.csv')

odds_2023['game_date'] = pd.to_datetime(odds_2023['game_date'])
odds_2024['game_date'] = pd.to_datetime(odds_2024['game_date'])

print(f"✓ 2023-24 odds: {len(odds_2023):,} games")
print(f"  Date range: {odds_2023['game_date'].min().date()} to {odds_2023['game_date'].max().date()}")
print(f"\n✓ 2024-25 odds: {len(odds_2024):,} games")
print(f"  Date range: {odds_2024['game_date'].min().date()} to {odds_2024['game_date'].max().date()}")

# 3. Check odds quality
print("\n3️⃣ Checking odds quality...")

def check_odds_quality(df, season_name):
    print(f"\n{season_name}:")
    
    # Check for nulls
    null_home = df['home_ml_odds'].isna().sum()
    null_away = df['away_ml_odds'].isna().sum()
    print(f"  Null values: Home={null_home}, Away={null_away}")
    
    # Check for extreme outliers
    home_odds = df['home_ml_odds'].dropna()
    away_odds = df['away_ml_odds'].dropna()
    
    print(f"  Home odds range: {home_odds.min():.0f} to {home_odds.max():.0f}")
    print(f"  Away odds range: {away_odds.min():.0f} to {away_odds.max():.0f}")
    
    # Flag suspicious odds (absolute value > 2000 or between -100 and +100 but wrong sign)
    suspicious_home = ((home_odds.abs() > 2000) | 
                       ((home_odds > -100) & (home_odds < 100))).sum()
    suspicious_away = ((away_odds.abs() > 2000) | 
                       ((away_odds > -100) & (away_odds < 100))).sum()
    
    print(f"  Suspicious odds: Home={suspicious_home}, Away={suspicious_away}")
    
    # Check for duplicates
    dupes = df.duplicated(subset=['game_date', 'home_team', 'away_team']).sum()
    print(f"  Duplicate games: {dupes}")
    
    return df

odds_2023_clean = check_odds_quality(odds_2023, "2023-24 Season")
odds_2024_clean = check_odds_quality(odds_2024, "2024-25 Season")

# 4. Check team name consistency
print("\n4️⃣ Checking team name matching...")

# Get unique team names from each dataset
training_teams = set(training_df['home_team'].unique()) | set(training_df['away_team'].unique())
odds_2023_teams = set(odds_2023['home_team'].unique()) | set(odds_2023['away_team'].unique())
odds_2024_teams = set(odds_2024['home_team'].unique()) | set(odds_2024['away_team'].unique())

print(f"\nTeams in training data: {len(training_teams)}")
print(f"Teams in 2023-24 odds: {len(odds_2023_teams)}")
print(f"Teams in 2024-25 odds: {len(odds_2024_teams)}")

# Check for mismatches
training_not_in_odds23 = training_teams - odds_2023_teams
odds23_not_in_training = odds_2023_teams - training_teams

if training_not_in_odds23:
    print(f"\n⚠️  Teams in training but NOT in 2023-24 odds:")
    for team in sorted(training_not_in_odds23):
        print(f"  - {team}")

if odds23_not_in_training:
    print(f"\n⚠️  Teams in 2023-24 odds but NOT in training:")
    for team in sorted(odds23_not_in_training):
        print(f"  - {team}")

training_not_in_odds24 = training_teams - odds_2024_teams
odds24_not_in_training = odds_2024_teams - training_teams

if training_not_in_odds24:
    print(f"\n⚠️  Teams in training but NOT in 2024-25 odds:")
    for team in sorted(training_not_in_odds24):
        print(f"  - {team}")

if odds24_not_in_training:
    print(f"\n⚠️  Teams in 2024-25 odds but NOT in training:")
    for team in sorted(odds24_not_in_training):
        print(f"  - {team}")

# 5. Sample comparison
print("\n5️⃣ Sample date/team matching...")

# Check a specific date
test_date = '2023-10-24'
train_games = training_df[training_df['date'] == test_date][['date', 'home_team', 'away_team']].head()
odds_games = odds_2023[odds_2023['game_date'] == test_date][['game_date', 'home_team', 'away_team']].head()

print(f"\nGames on {test_date}:")
print("\nTraining data:")
print(train_games)
print("\nOdds data:")
print(odds_games)

# 6. Expected vs actual coverage
print("\n6️⃣ Coverage analysis...")

# Filter training data to test period
test_period = training_df[training_df['date'] >= '2023-10-01']
season_2023 = test_period[(test_period['date'] >= '2023-10-01') & (test_period['date'] <= '2024-04-30')]
season_2024 = test_period[test_period['date'] >= '2024-10-01']

print(f"\nExpected games in test period:")
print(f"  2023-24 season: {len(season_2023):,} games")
print(f"  2024-25 season: {len(season_2024):,} games")
print(f"  Total: {len(test_period):,} games")

print(f"\nActual odds available:")
print(f"  2023-24 season: {len(odds_2023):,} games")
print(f"  2024-25 season: {len(odds_2024):,} games")
print(f"  Total: {len(odds_2023) + len(odds_2024):,} games")

coverage_2023 = (len(odds_2023) / len(season_2023)) * 100 if len(season_2023) > 0 else 0
coverage_2024 = (len(odds_2024) / len(season_2024)) * 100 if len(season_2024) > 0 else 0

print(f"\nCoverage rates:")
print(f"  2023-24: {coverage_2023:.1f}%")
print(f"  2024-25: {coverage_2024:.1f}%")

# 7. Recommendations
print("\n" + "="*80)
print("💡 RECOMMENDATIONS")
print("="*80)

if coverage_2023 < 95 or coverage_2024 < 95:
    print("⚠️  LOW COVERAGE: Missing odds for many games")
    print("   Likely cause: Team name mismatch between data sources")
    print("   Action: Need to standardize team names")

if len(training_not_in_odds23) > 0 or len(training_not_in_odds24) > 0:
    print("\n⚠️  TEAM NAME MISMATCH DETECTED")
    print("   Action: Create team name mapping dictionary")

suspicious_total = ((odds_2023['home_ml_odds'].abs() > 2000).sum() + 
                   (odds_2024['home_ml_odds'].abs() > 2000).sum())
if suspicious_total > 0:
    print(f"\n⚠️  SUSPICIOUS ODDS: {suspicious_total} odds with absolute value > 2000")
    print("   Action: Filter out extreme outliers")

print("\n✅ Next steps:")
print("1. Standardize team names if mismatches found")
print("2. Filter out odds outside reasonable range (-1000 to +1000)")
print("3. Remove duplicates if any")
print("4. Re-run merge and verify >95% coverage")
