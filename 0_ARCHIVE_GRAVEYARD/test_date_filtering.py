"""Test the new date-filtered team stats method"""
import sys
sys.path.insert(0, '.')

from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.settings import DB_PATH

# Initialize feature calculator
calc = FeatureCalculatorV5(db_path=str(DB_PATH))

# Test: Get Lakers stats as of Nov 1, 2024 (early season)
print("\n" + "="*60)
print("TEST: Get LAL stats as of 2024-11-01 (L10 games)")
print("="*60)
lal_stats_nov = calc.get_team_stats_as_of_date('LAL', '2024-11-01', lookback_games=10)
print(f"\n✓ LAL Stats (as of Nov 1, 2024):")
for key, val in lal_stats_nov.items():
    if key != 'games_used':
        print(f"  {key:15s}: {val:.2f}")
    else:
        print(f"  {key:15s}: {val}")

# Test: Get Lakers stats as of Dec 1, 2024 (mid season)
print("\n" + "="*60)
print("TEST: Get LAL stats as of 2024-12-01 (L10 games)")
print("="*60)
lal_stats_dec = calc.get_team_stats_as_of_date('LAL', '2024-12-01', lookback_games=10)
print(f"\n✓ LAL Stats (as of Dec 1, 2024):")
for key, val in lal_stats_dec.items():
    if key != 'games_used':
        print(f"  {key:15s}: {val:.2f}")
    else:
        print(f"  {key:15s}: {val}")

# Verify they're different (proving date filtering works!)
print("\n" + "="*60)
print("VERIFICATION: Stats should be DIFFERENT (date filtering working)")
print("="*60)
if lal_stats_nov and lal_stats_dec:
    diff_off = abs(lal_stats_nov['off_rating'] - lal_stats_dec['off_rating'])
    diff_def = abs(lal_stats_nov['def_rating'] - lal_stats_dec['def_rating'])
    print(f"Off Rating difference: {diff_off:.2f}")
    print(f"Def Rating difference: {diff_def:.2f}")
    
    if diff_off > 1.0 or diff_def > 1.0:
        print("\n✅ SUCCESS: Date filtering is working!")
        print("   Stats change between Nov and Dec (as expected)")
    else:
        print("\n⚠️  WARNING: Stats are very similar - check if data is correct")

# Test: Try to get stats for future date (should return empty or limited)
print("\n" + "="*60)
print("TEST: Try to get stats for BEFORE season starts")
print("="*60)
early_stats = calc.get_team_stats_as_of_date('LAL', '2024-10-15', lookback_games=10)
print(f"Stats before season: {early_stats}")
if not early_stats or early_stats.get('games_used', 0) == 0:
    print("✅ CORRECT: No data leakage - can't see future games!")
else:
    print(f"⚠️  Found {early_stats.get('games_used')} games (should be 0)")
