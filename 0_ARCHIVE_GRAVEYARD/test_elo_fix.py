"""Test ELO fix"""
from src.features.feature_calculator_live import FeatureCalculatorV5

calc = FeatureCalculatorV5()
feats = calc.calculate_game_features('OKC', 'SAS', game_date='2025-12-13')

print("=" * 80)
print("ELO FEATURES AFTER FIX")
print("=" * 80)

print(f"\nActual values:")
print(f"  home_composite_elo: {feats.get('home_composite_elo'):.2f}")
print(f"  off_elo_diff: {feats.get('off_elo_diff'):.2f}")
print(f"  def_elo_diff: {feats.get('def_elo_diff'):.2f}")

print(f"\nExpected values from database:")
print(f"  OKC composite: 1651.43")
print(f"  SAS composite: 1537.43")
print(f"  off_diff: 1581.93 - 1555.93 = 26.00")
print(f"  def_diff: 1720.93 - 1518.93 = 202.00")

print(f"\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

match_composite = abs(feats.get('home_composite_elo') - 1651.43) < 1.0
match_off = abs(feats.get('off_elo_diff') - 26.0) < 1.0
match_def = abs(feats.get('def_elo_diff') - 202.0) < 1.0

print(f"{'✓' if match_composite else '❌'} Composite ELO: {match_composite}")
print(f"{'✓' if match_off else '❌'} Offensive diff: {match_off}")
print(f"{'✓' if match_def else '❌'} Defensive diff: {match_def}")

if all([match_composite, match_off, match_def]):
    print("\n✓✓✓ ELO CALIBRATION FIXED ✓✓✓")
    print(f"OKC now correctly shows as elite team (composite: {feats.get('home_composite_elo'):.1f})")
    print(f"OKC defensive advantage over SAS: +{feats.get('def_elo_diff'):.1f} points")
else:
    print("\n❌ ELO still has issues")
