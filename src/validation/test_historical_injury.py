"""
Test the historical injury calculator directly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features.feature_calculator_v5 import FeatureCalculatorV5

calc = FeatureCalculatorV5()

# Test BOS vs NYK on 2024-10-22
result = calc._calculate_historical_injury_impact('BOS', 'NYK', '2024-10-22')

print(f"Direct method call result:")
print(f"  home (BOS) injury_impact: {result.get('home_injury_impact', 'MISSING')}")
print(f"  away (NYK) injury_impact: {result.get('away_injury_impact', 'MISSING')}")

# Now test through calculate_game_features
features = calc.calculate_game_features('BOS', 'NYK', game_date='2024-10-22')

print(f"\nThrough calculate_game_features:")
print(f"  injury_impact_diff: {features.get('injury_impact_diff', 'MISSING')}")
print(f"  injury_impact_abs: {features.get('injury_impact_abs', 'MISSING')}")
print(f"  injury_elo_interaction: {features.get('injury_elo_interaction', 'MISSING')}")
