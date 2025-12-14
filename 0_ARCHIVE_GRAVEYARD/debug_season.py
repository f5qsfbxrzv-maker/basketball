"""Debug ELO differential calculation"""
from src.features.feature_calculator_live import FeatureCalculatorV5
from src.features.off_def_elo_system import OffDefEloSystem

calc = FeatureCalculatorV5()

# Manually test the get_differentials method
elo_system = OffDefEloSystem(db_path='data/live/nba_betting_data.db')

print("=" * 80)
print("DEBUGGING ELO DIFFERENTIALS")
print("=" * 80)

# Test with different season strings
seasons_to_try = ['2025-26', '2024-25', '2025']

for season in seasons_to_try:
    print(f"\nTrying season='{season}':")
    try:
        diffs = elo_system.get_differentials(season, 'OKC', 'SAS', game_date='2025-12-13')
        print(f"  off_elo_diff: {diffs['off_elo_diff']:.2f}")
        print(f"  def_elo_diff: {diffs['def_elo_diff']:.2f}")
        print(f"  composite_diff: {diffs['composite_elo_diff']:.2f}")
        
        # Check if these are the raw ratings
        okc = elo_system.get_latest('OKC', season, before_date='2025-12-13')
        sas = elo_system.get_latest('SAS', season, before_date='2025-12-13')
        if okc and sas:
            print(f"  OKC: off={okc.off_elo:.2f}, def={okc.def_elo:.2f}")
            print(f"  SAS: off={sas.off_elo:.2f}, def={sas.def_elo:.2f}")
        else:
            print(f"  No ratings found for this season")
    except Exception as e:
        print(f"  Error: {e}")

# Check what season the feature calculator is using
print("\n" + "=" * 80)
print("FEATURE CALCULATOR SEASON LOGIC")
print("=" * 80)

import pandas as pd
game_dt = pd.to_datetime('2025-12-13')
year = game_dt.year
month = game_dt.month
season = f"{year - 1}-{str(year)[-2:]}" if month < 10 else f"{year}-{str(year + 1)[-2:]}"
print(f"\nFor game_date='2025-12-13':")
print(f"  year: {year}")
print(f"  month: {month}")
print(f"  Calculated season: '{season}'")
