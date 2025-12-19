#!/usr/bin/env python3
"""Quick test for away_composite_elo feature"""
import sys
sys.path.insert(0, 'src')

from features.feature_calculator_live import FeatureCalculatorV5

calc = FeatureCalculatorV5()
features = calc.calculate_game_features('OKC', 'SAS', game_date='2025-12-14')

print(f"\nTotal features: {len(features)}")
print(f"\nhome_composite_elo: {features.get('home_composite_elo', 'MISSING')}")
print(f"away_composite_elo: {features.get('away_composite_elo', 'MISSING')}")
print(f"\noff_elo_diff: {features.get('off_elo_diff', 'MISSING')}")
print(f"def_elo_diff: {features.get('def_elo_diff', 'MISSING')}")

if 'away_composite_elo' in features:
    print("\n[OK] away_composite_elo is now included!")
    if features['away_composite_elo'] != 0:
        print(f"[OK] away_composite_elo = {features['away_composite_elo']} (not zero)")
    else:
        print("[ERROR] away_composite_elo is zero")
else:
    print("\n[ERROR] away_composite_elo still missing")
