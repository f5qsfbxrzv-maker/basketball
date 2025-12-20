import logging
logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

print("=== DEBUG INJURY SCRAPER CALL ===\n")

# Test directly
from src.services.injury_scraper import InjuryScraper

scraper = InjuryScraper()
print("1. Testing scraper directly:")
result = scraper.get_team_injury_impact('LAL')
print(f"   Result: {result['total_penalty']}")

# Now test through feature calculator
print("\n2. Testing through feature calculator:")
try:
    from src.features.feature_calculator_v5 import FeatureCalculatorV5
    calc = FeatureCalculatorV5(db_path=r"data\live\nba_betting_data.db")
    injury_data = calc._calculate_injury_impact('LAL', 'LAC')
    print(f"   Result: {injury_data}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
