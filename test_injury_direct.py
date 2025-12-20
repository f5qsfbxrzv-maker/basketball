from src.features.feature_calculator_v5 import FeatureCalculatorV5
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize calculator
calc = FeatureCalculatorV5(db_path=r"data\live\nba_betting_data.db")

# Test injury calculation directly
print("=== TESTING INJURY CALCULATION DIRECTLY ===\n")
try:
    injury_data = calc._calculate_injury_impact('LAL', 'LAC')
    print(f"✅ Success! Injury data:")
    for key, value in injury_data.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
