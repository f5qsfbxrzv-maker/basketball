from src.features.feature_calculator_v5 import FeatureCalculatorV5
from datetime import datetime

# Initialize calculator
calc = FeatureCalculatorV5(db_path=r"data\live\nba_betting_data.db")

# Calculate features for LAL @ LAC
print("=" * 80)
print("FEATURE CALCULATION TEST - LAL @ LAC")
print("=" * 80)

features = calc.calculate_game_features(
    home_team='LAC',
    away_team='LAL',
    season='2025-26',
    game_date=None
)

print(f"\nTotal features calculated: {len(features)}")

# Check MDP model's 19 required features
mdp_features = [
    'off_elo_diff', 'def_elo_diff', 'home_composite_elo',
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
    'injury_matchup_advantage', 'injury_shock_diff', 'star_power_leverage',
    'season_progress', 'league_offensive_context', 'total_foul_environment',
    'net_free_throw_advantage', 'offense_vs_defense_matchup',
    'pace_efficiency_interaction', 'star_mismatch'
]

print("\nMDP MODEL'S 19 FEATURES:")
good = []
zero = []
missing = []

for f in mdp_features:
    value = features.get(f, 'MISSING')
    if value == 'MISSING':
        missing.append(f)
        print(f"  ❌ MISSING {f}: {value}")
    elif value == 0 or value == 0.0:
        zero.append(f)
        print(f"  ⚠️  ZERO {f}: {value}")
    else:
        good.append(f)
        print(f"  ✅ {f}: {value}")

print(f"\nSUMMARY:")
print(f"  ✅ Good ({len(good)}): {good[:5]}...")
print(f"  ⚠️  Zero ({len(zero)}): {zero}")
print(f"  ❌ Missing ({len(missing)}): {missing}")

# Show injury features specifically
print(f"\n" + "=" * 80)
print("INJURY FEATURES DETAIL:")
print("=" * 80)
print(f"home_injury_impact: {features.get('home_injury_impact', 'MISSING')}")
print(f"away_injury_impact: {features.get('away_injury_impact', 'MISSING')}")
print(f"injury_matchup_advantage: {features.get('injury_matchup_advantage', 'MISSING')}")
print(f"injury_shock_diff: {features.get('injury_shock_diff', 'MISSING')}")
print(f"star_mismatch: {features.get('star_mismatch', 'MISSING')}")
print(f"star_power_leverage: {features.get('star_power_leverage', 'MISSING')}")

# Show rest features
print(f"\n" + "=" * 80)
print("REST/FATIGUE FEATURES DETAIL:")
print("=" * 80)
print(f"home_rest_days: {features.get('home_rest_days', 'MISSING')}")
print(f"away_rest_days: {features.get('away_rest_days', 'MISSING')}")
print(f"rest_days_diff: {features.get('rest_days_diff', 'MISSING')}")
print(f"net_fatigue_score: {features.get('net_fatigue_score', 'MISSING')}")
