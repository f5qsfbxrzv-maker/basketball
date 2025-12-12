"""
Test PIE weighting with player name normalization.
"""

import sys
sys.path.append('.')

from src.features.feature_calculator_v5 import FeatureCalculatorV5

calc = FeatureCalculatorV5()

print("\n" + "="*80)
print("🧪 TESTING PIE-WEIGHTED INJURY IMPACT")
print("="*80)

# Test case: BOS @ NYK on 2024-10-22
# Expected: 
# - BOS: Porzingis out (PIE ~0.12) 
# - NYK: Robinson, Achiuwa, McCullar out

test_date = '2024-10-22'
test_home = 'BOS'
test_away = 'NYK'

print(f"\n📅 Test Game: {test_away} @ {test_home} on {test_date}")
print("="*80)

result = calc._calculate_historical_injury_impact(test_home, test_away, test_date)

print(f"\n📊 RESULTS:")
print(f"   Home ({test_home}) Injury Impact: {result['home_injury_impact']:.2f}")
print(f"   Away ({test_away}) Injury Impact: {result['away_injury_impact']:.2f}")
print(f"   Net Differential: {result['home_injury_impact'] - result['away_injury_impact']:.2f}")

# Check if values are non-zero
if result['home_injury_impact'] > 0 or result['away_injury_impact'] > 0:
    print("\n✅ SUCCESS - Non-zero injury impacts detected")
else:
    print("\n❌ FAILED - All zeros (should have injuries)")

# Test superstar case: Giannis out
print("\n" + "="*80)
print("🌟 TESTING SUPERSTAR IMPACT")
print("="*80)

# Find a game where Giannis was out
import sqlite3
import pandas as pd

db_path = 'data/live/nba_betting_data.db'
with sqlite3.connect(db_path) as conn:
    giannis_out = pd.read_sql_query(
        """
        SELECT DISTINCT game_date, team_abbreviation
        FROM historical_inactives
        WHERE player_name LIKE '%Antetokounmpo%'
        AND game_date >= '2024-01-01'
        ORDER BY game_date DESC
        LIMIT 5
        """,
        conn
    )

if not giannis_out.empty:
    print(f"\n📋 Found {len(giannis_out)} games with Giannis out:")
    for idx, row in giannis_out.iterrows():
        print(f"   {row['game_date']}: {row['team_abbreviation']}")
    
    # Test first occurrence
    test_date = giannis_out.iloc[0]['game_date']
    test_team = giannis_out.iloc[0]['team_abbreviation']
    
    print(f"\n🧪 Testing {test_team} on {test_date} (Giannis out)")
    
    # Calculate impact (assume home game for simplicity)
    result = calc._calculate_historical_injury_impact(test_team, 'OPP', test_date)
    
    print(f"\n📊 GIANNIS IMPACT:")
    print(f"   {test_team} Injury Impact: {result['home_injury_impact']:.2f}")
    print(f"   Expected: ~6-9 points (PIE=0.20 × 20 × 3.0 multiplier)")
    
    if result['home_injury_impact'] > 3.0:
        print("   ✅ Superstar multiplier appears to be working")
    else:
        print("   ⚠️ Impact lower than expected for superstar")

print("\n" + "="*80)
