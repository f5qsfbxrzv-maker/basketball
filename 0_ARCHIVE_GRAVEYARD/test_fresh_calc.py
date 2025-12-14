"""Test with fresh instance of FeatureCalculatorV5"""
# Force reimport to get latest data
import sys
if 'src.features.feature_calculator_live' in sys.modules:
    del sys.modules['src.features.feature_calculator_live']

from src.features.feature_calculator_live import FeatureCalculatorV5

print("="*80)
print("Creating FRESH FeatureCalculatorV5 instance")
print("="*80)

calc = FeatureCalculatorV5()

print(f"\ngame_logs_df rows: {len(calc.game_logs_df)}")
if not calc.game_logs_df.empty:
    print(f"Date range: {calc.game_logs_df['GAME_DATE'].min()} to {calc.game_logs_df['GAME_DATE'].max()}")

# Test rest days
print("\n" + "="*80)
print("Testing rest days calculation:")
print("="*80)

feats = calc.calculate_game_features('ORL', 'NYK', game_date='2025-12-13')
print(f"\nORL vs NYK:")
print(f"  home_rest_days: {feats['home_rest_days']} (expected: 3)")
print(f"  away_rest_days: {feats['away_rest_days']} (expected: 3)")

feats2 = calc.calculate_game_features('OKC', 'SAS', game_date='2025-12-13')
print(f"\nOKC vs SAS:")
print(f"  home_rest_days: {feats2['home_rest_days']} (expected: 2)")
print(f"  away_rest_days: {feats2['away_rest_days']} (expected: 2)")
