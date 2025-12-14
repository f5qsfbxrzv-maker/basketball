"""Debug why stats are returning zeros"""
from src.features.feature_calculator_live import FeatureCalculatorV5

calc = FeatureCalculatorV5()

print("=" * 80)
print("DEBUGGING STATS LOOKUP")
print("=" * 80)

# Test get_team_stats_as_of_date
print("\n1. Testing get_team_stats_as_of_date for LAL:")
stats = calc.get_team_stats_as_of_date('LAL', '2025-12-13', lookback_games=10)
print(f"   Keys returned: {list(stats.keys())}")
print(f"   efg_pct: {stats.get('efg_pct', 'NOT FOUND')}")
print(f"   off_rating: {stats.get('off_rating', 'NOT FOUND')}")
print(f"   games_used: {stats.get('games_used', 'NOT FOUND')}")

# Test full feature calculation
print("\n2. Testing full feature calculation for LAL vs MEM:")
feats = calc.calculate_game_features('LAL', 'MEM', game_date='2025-12-13')
print(f"   Feature count: {len(feats)}")
print(f"   home_efg_pct: {feats.get('home_efg_pct', 'NOT FOUND')}")
print(f"   home_off_elo: {feats.get('home_off_elo', 'NOT FOUND')}")
print(f"   away_efg_pct: {feats.get('away_efg_pct', 'NOT FOUND')}")

# Test with correct games
print("\n3. Testing CORRECT Dec 13 games:")
print("\n   OKC vs SAS:")
feats_okc = calc.calculate_game_features('OKC', 'SAS', game_date='2025-12-13')
print(f"   home_efg_pct: {feats_okc.get('home_efg_pct', 'NOT FOUND')}")
print(f"   home_off_elo: {feats_okc.get('home_off_elo', 'NOT FOUND')}")

print("\n   ORL vs NYK:")
feats_orl = calc.calculate_game_features('ORL', 'NYK', game_date='2025-12-13')
print(f"   home_efg_pct: {feats_orl.get('home_efg_pct', 'NOT FOUND')}")
print(f"   home_off_elo: {feats_orl.get('home_off_elo', 'NOT FOUND')}")
