"""
Trace exactly where away_composite_elo disappears
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.features.feature_calculator_live import FeatureCalculatorV5
import pandas as pd

# Test the exact flow
calc = FeatureCalculatorV5()

home_team = "OKC"
away_team = "SAS"
game_date = "2025-12-16"

print(f"\n{'='*60}")
print(f"TESTING: {away_team} @ {home_team} on {game_date}")
print(f"{'='*60}\n")

# STEP 1: Test _calc_baseline_features directly
print("[STEP 1] Testing _calc_baseline_features() directly...")
baseline_features = calc._calc_baseline_features(home_team, away_team, game_date)
print(f"  Returned features: {list(baseline_features.keys())}")
print(f"  home_composite_elo: {baseline_features.get('home_composite_elo', 'MISSING')}")
print(f"  away_composite_elo: {baseline_features.get('away_composite_elo', 'MISSING')}")

# STEP 2: Test full calculate_game_features
print(f"\n[STEP 2] Testing calculate_game_features() full flow...")
full_features = calc.calculate_game_features(
    home_team=home_team,
    away_team=away_team,
    game_date=game_date
)
print(f"  Total features returned: {len(full_features)}")
print(f"  home_composite_elo: {full_features.get('home_composite_elo', 'MISSING')}")
print(f"  away_composite_elo: {full_features.get('away_composite_elo', 'MISSING')}")

# STEP 3: Check if FEATURE_WHITELIST is filtering it out
print(f"\n[STEP 3] Checking FEATURE_WHITELIST...")
from src.features.feature_calculator_live import FEATURE_WHITELIST
if FEATURE_WHITELIST is not None:
    print(f"  FEATURE_WHITELIST exists with {len(FEATURE_WHITELIST)} features")
    if 'home_composite_elo' in FEATURE_WHITELIST:
        print(f"  ✓ home_composite_elo IS in whitelist")
    else:
        print(f"  ✗ home_composite_elo NOT in whitelist")
    
    if 'away_composite_elo' in FEATURE_WHITELIST:
        print(f"  ✓ away_composite_elo IS in whitelist")
    else:
        print(f"  ✗ away_composite_elo NOT in whitelist [BUG FOUND!]")
else:
    print(f"  FEATURE_WHITELIST is None (no filtering)")

# STEP 4: Show all features
print(f"\n[STEP 4] All features in final dict:")
for i, (key, value) in enumerate(sorted(full_features.items()), 1):
    print(f"  {i:2d}. {key:30s} = {value}")

print(f"\n{'='*60}")
if full_features.get('away_composite_elo') == 'MISSING' or 'away_composite_elo' not in full_features:
    print("❌ BUG CONFIRMED: away_composite_elo is missing from final features")
else:
    print("✓ away_composite_elo is present in final features")
print(f"{'='*60}\n")
