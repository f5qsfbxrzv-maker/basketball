"""
Comprehensive Season Data Check
Validates that ALL features are using 2025-26 season data
"""
import sqlite3
import pandas as pd
from datetime import datetime
from src.features.feature_calculator_live import FeatureCalculatorV5

print("=" * 80)
print("SEASON DATA VALIDATION - December 13, 2025")
print("=" * 80)

# 1. Check database has current season data
conn = sqlite3.connect('data/live/nba_betting_data.db')

print("\n[1/5] Checking game_logs table...")
game_logs_check = pd.read_sql("""
    SELECT 
        season,
        COUNT(*) as games,
        MIN(game_date) as first_game,
        MAX(game_date) as last_game
    FROM game_logs
    WHERE season IN ('2024-25', '2025-26')
    GROUP BY season
    ORDER BY season DESC
""", conn)
print(game_logs_check.to_string())

print("\n[2/5] Checking game_advanced_stats table...")
adv_stats_check = pd.read_sql("""
    SELECT 
        season,
        COUNT(*) as games,
        MIN(game_date) as first_game,
        MAX(game_date) as last_game
    FROM game_advanced_stats
    WHERE season IN ('2024-25', '2025-26')
    GROUP BY season
    ORDER BY season DESC
""", conn)
print(adv_stats_check.to_string())

print("\n[3/5] Checking ELO ratings table...")
elo_check = pd.read_sql("""
    SELECT 
        season,
        COUNT(DISTINCT team) as teams,
        MIN(game_date) as first_game,
        MAX(game_date) as last_game
    FROM elo_ratings
    WHERE season IN ('2024-25', '2025-26')
    GROUP BY season
    ORDER BY season DESC
""", conn)
print(elo_check.to_string())

conn.close()

print("\n[4/5] Testing feature calculation with NO season specified (should auto-detect)...")
calc = FeatureCalculatorV5()

# Test without specifying season - should auto-calculate to 2025-26
feats = calc.calculate_game_features('OKC', 'SAS', game_date='2025-12-13')

# Check if we're getting current season data
print(f"\n  season_year: {feats.get('season_year')} (should be 2025)")
print(f"  games_into_season: {feats.get('games_into_season'):.1f} (should be ~25)")
print(f"  home_composite_elo: {feats.get('home_composite_elo'):.1f} (should be 1651, NOT default 1500)")

# Check ELO values to ensure we're using 2025-26 data
if abs(feats.get('home_composite_elo') - 1651.43) < 5:
    print("  ✓ ELO using 2025-26 season data")
else:
    print(f"  ❌ ELO NOT using correct season (got {feats.get('home_composite_elo'):.1f})")

print("\n[5/5] Checking EWMA features are using recent games...")
# EWMA should be calculated from recent game_advanced_stats
ewma_features = {
    'ewma_efg_diff': feats.get('ewma_efg_diff'),
    'home_ewma_3p_pct': feats.get('home_ewma_3p_pct'),
    'ewma_pace_diff': feats.get('ewma_pace_diff')
}

print("  EWMA feature samples:")
for k, v in ewma_features.items():
    if v is not None and abs(v) > 0.0001:
        print(f"    ✓ {k}: {v:.4f} (populated)")
    else:
        print(f"    ❌ {k}: {v} (zero or missing)")

# Final validation
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

issues = []

# Check 1: 2025-26 season has data
games_2025 = game_logs_check[game_logs_check['season'] == '2025-26']['games'].values
if len(games_2025) == 0 or games_2025[0] < 100:
    issues.append("❌ Insufficient 2025-26 game logs")
else:
    print(f"✓ 2025-26 game logs: {games_2025[0]} games")

# Check 2: Data is recent (last game within 3 days)
last_game = game_logs_check[game_logs_check['season'] == '2025-26']['last_game'].values[0]
days_old = (datetime.strptime('2025-12-13', '%Y-%m-%d') - datetime.strptime(last_game, '%Y-%m-%d')).days
if days_old > 3:
    issues.append(f"❌ Game logs {days_old} days old (last: {last_game})")
else:
    print(f"✓ Game logs current (last: {last_game}, {days_old} days ago)")

# Check 3: Features using correct season
if feats.get('season_year') != 2025:
    issues.append(f"❌ season_year = {feats.get('season_year')} (should be 2025)")
else:
    print(f"✓ season_year: 2025")

# Check 4: ELO using current season
if abs(feats.get('home_composite_elo') - 1651.43) > 10:
    issues.append(f"❌ ELO not using 2025-26 data")
else:
    print(f"✓ ELO ratings: 2025-26 season data")

# Check 5: EWMA features populated
ewma_populated = sum(1 for v in ewma_features.values() if v is not None and abs(v) > 0.0001)
if ewma_populated < 2:
    issues.append(f"❌ EWMA features not populated ({ewma_populated}/3)")
else:
    print(f"✓ EWMA features: populated ({ewma_populated}/3)")

print("\n" + "=" * 80)
if len(issues) == 0:
    print("✓✓✓ ALL SEASON DATA CHECKS PASSED ✓✓✓")
    print("System is using 2025-26 season data correctly")
else:
    print("❌❌❌ SEASON DATA ISSUES FOUND ❌❌❌")
    for issue in issues:
        print(f"  {issue}")
print("=" * 80)
