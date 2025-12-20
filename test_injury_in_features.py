import logging
logging.basicConfig(level=logging.WARNING)

from src.features.feature_calculator_v5 import FeatureCalculatorV5

calc = FeatureCalculatorV5(db_path=r"data\live\nba_betting_data.db")

print("=== TEST INJURY CALCULATION ===\n")

# Call the method that should use the scraper
injury_data = calc._calculate_injury_impact('LAL', 'LAC')

print("Result:")
for key, value in injury_data.items():
    print(f"  {key}: {value}")

# Now test with feature calculation
print("\n=== FULL FEATURE CALCULATION ===\n")
from datetime import datetime
features = calc.calculate_game_features(
    home_team='LAC',
    away_team='LAL', 
    season='2025-26',
    game_date=None
)

print(f"home_injury_impact: {features.get('home_injury_impact', 'NOT FOUND')}")
print(f"away_injury_impact: {features.get('away_injury_impact', 'NOT FOUND')}")
print(f"injury_matchup_advantage: {features.get('injury_matchup_advantage', 'NOT FOUND')}")
print(f"injury_shock_diff: {features.get('injury_shock_diff', 'NOT FOUND')}")
print(f"star_mismatch: {features.get('star_mismatch', 'NOT FOUND')}")
print(f"star_power_leverage: {features.get('star_power_leverage', 'NOT FOUND')}")
