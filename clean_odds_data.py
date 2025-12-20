"""
CLEAN AND STANDARDIZE ODDS DATA
=================================
Fixes:
1. Maps full team names to abbreviations
2. Filters extreme outliers
3. Removes duplicates
4. Validates coverage
"""

import pandas as pd
import numpy as np

# Team name mapping: Full Name → Abbreviation
TEAM_NAME_MAP = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

print("="*80)
print("🧹 CLEANING ODDS DATA")
print("="*80)

# 1. Load odds
print("\n1️⃣ Loading raw odds files...")
odds_2023 = pd.read_csv('data/closing_odds_2023_24.csv')
odds_2024 = pd.read_csv('data/live/closing_odds_2024_25.csv')

print(f"✓ 2023-24: {len(odds_2023):,} rows")
print(f"✓ 2024-25: {len(odds_2024):,} rows")

# 2. Standardize team names
print("\n2️⃣ Standardizing team names...")

def standardize_teams(df):
    df['home_team'] = df['home_team'].map(TEAM_NAME_MAP).fillna(df['home_team'])
    df['away_team'] = df['away_team'].map(TEAM_NAME_MAP).fillna(df['away_team'])
    return df

odds_2023 = standardize_teams(odds_2023)
odds_2024 = standardize_teams(odds_2024)

print("✓ Team names converted to abbreviations")

# 3. Filter extreme outliers
print("\n3️⃣ Filtering extreme outliers...")

def filter_outliers(df, season_name):
    before = len(df)
    
    # Remove odds outside reasonable range (-1500 to +1500)
    # NBA favorites rarely exceed -1000, underdogs rarely exceed +1000
    df = df[
        (df['home_ml_odds'] >= -1500) & (df['home_ml_odds'] <= 1500) &
        (df['away_ml_odds'] >= -1500) & (df['away_ml_odds'] <= 1500)
    ].copy()
    
    removed = before - len(df)
    print(f"  {season_name}: Removed {removed} games with extreme odds (kept {len(df):,})")
    
    return df

odds_2023 = filter_outliers(odds_2023, "2023-24")
odds_2024 = filter_outliers(odds_2024, "2024-25")

# 4. Remove duplicates
print("\n4️⃣ Removing duplicates...")

def remove_duplicates(df, season_name):
    before = len(df)
    
    # Keep first occurrence (usually most recent snapshot)
    df = df.drop_duplicates(subset=['game_date', 'home_team', 'away_team'], keep='first')
    
    removed = before - len(df)
    print(f"  {season_name}: Removed {removed} duplicates (kept {len(df):,})")
    
    return df

odds_2023 = remove_duplicates(odds_2023, "2023-24")
odds_2024 = remove_duplicates(odds_2024, "2024-25")

# 5. Validate cleaned data
print("\n5️⃣ Validating cleaned data...")

for df, name in [(odds_2023, "2023-24"), (odds_2024, "2024-25")]:
    print(f"\n{name}:")
    print(f"  Games: {len(df):,}")
    print(f"  Home odds range: {df['home_ml_odds'].min():.0f} to {df['home_ml_odds'].max():.0f}")
    print(f"  Away odds range: {df['away_ml_odds'].min():.0f} to {df['away_ml_odds'].max():.0f}")
    print(f"  Null values: {df[['home_ml_odds', 'away_ml_odds']].isna().sum().sum()}")
    print(f"  Date range: {pd.to_datetime(df['game_date']).min().date()} to {pd.to_datetime(df['game_date']).max().date()}")

# 6. Save cleaned files
print("\n6️⃣ Saving cleaned files...")

odds_2023.to_csv('data/closing_odds_2023_24_CLEANED.csv', index=False)
odds_2024.to_csv('data/closing_odds_2024_25_CLEANED.csv', index=False)

print("✓ Saved: data/closing_odds_2023_24_CLEANED.csv")
print("✓ Saved: data/closing_odds_2024_25_CLEANED.csv")

# 7. Test merge with training data
print("\n7️⃣ Testing merge with training data...")

training = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
training['date'] = pd.to_datetime(training['date'])

all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
all_odds['game_date'] = pd.to_datetime(all_odds['game_date'])

# Test merge
test_period = training[training['date'] >= '2023-10-01'].copy()
merged = test_period.merge(
    all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='left'
)

matched = merged['home_ml_odds'].notna().sum()
coverage = (matched / len(test_period)) * 100

print(f"\nMerge test results:")
print(f"  Test period games: {len(test_period):,}")
print(f"  Matched with odds: {matched:,}")
print(f"  Coverage: {coverage:.1f}%")

if coverage > 95:
    print("\n✅ SUCCESS: Coverage > 95%")
elif coverage > 90:
    print("\n⚠️  GOOD: Coverage > 90% (acceptable)")
else:
    print("\n🔴 ISSUE: Coverage < 90% (investigate remaining mismatches)")
    
    # Show sample of unmatched games
    unmatched = merged[merged['home_ml_odds'].isna()].head(10)
    print("\nSample unmatched games:")
    print(unmatched[['date', 'home_team', 'away_team']])

print("\n" + "="*80)
print("✅ ODDS CLEANING COMPLETE")
print("="*80)
print("Next: Re-run optimization with CLEANED odds files")
