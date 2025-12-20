from datetime import datetime
from src.features.feature_calculator_v5 import FeatureCalculatorV5

calc = FeatureCalculatorV5(r"data\live\nba_betting_data.db")

# Test rest calculation for Dec 20 game
game_date = datetime(2025, 12, 20)
rest_features = calc._calc_rest_features('LAC', 'LAL', game_date)

print("=== REST FEATURES FOR LAC (HOME) VS LAL (AWAY) ON DEC 20 ===")
print(f"rest_days_diff: {rest_features['rest_days_diff']}")
print(f"is_b2b_diff: {rest_features['is_b2b_diff']}")

print("\n=== EXPECTED ===")
print("Both teams played Dec 18")
print("Game is Dec 20")
print("Both should have 1 day rest")
print("rest_days_diff should be 0 (same rest)")
