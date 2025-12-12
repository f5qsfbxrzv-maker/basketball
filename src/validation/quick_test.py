"""Quick feature generation test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

calc = FeatureCalculatorV5()

# Test game: GSW @ CLE (Nov 15, 2024)
features = calc.calculate_game_features(
    home_team='CLE',
    away_team='GSW',
    game_date='2024-11-15'
)

print(f"\nFeatures generated: {len(features)}")
print(f"Whitelist target: {len(FEATURE_WHITELIST)}")

missing = [f for f in FEATURE_WHITELIST if f not in features]
print(f"Missing: {len(missing)}")

if missing:
    print("\nMissing features:")
    for f in missing:
        print(f"  - {f}")

print("\nKey features:")
print(f"  home_composite_elo: {features.get('home_composite_elo', 'MISSING')}")
print(f"  off_elo_diff: {features.get('off_elo_diff', 'MISSING')}")
print(f"  def_elo_diff: {features.get('def_elo_diff', 'MISSING')}")
print(f"  ewma_efg_diff: {features.get('ewma_efg_diff', 'MISSING')}")
print(f"  rest_advantage: {features.get('rest_advantage', 'MISSING')}")

print(f"\nSUCCESS!" if len(missing) == 0 else f"\nINCOMPLETE")
