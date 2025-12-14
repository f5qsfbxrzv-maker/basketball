"""Verify actual model features are calculated correctly"""
from src.features.feature_calculator_live import FeatureCalculatorV5

calc = FeatureCalculatorV5()

print("=" * 80)
print("CORRECT GAMES - December 13, 2025 (NBA Cup Semifinals)")
print("=" * 80)

# Correct games from user
games = [
    ('OKC', 'SAS'),
    ('ORL', 'NYK')
]

model_features = [
    'ewma_efg_diff', 'ewma_tov_diff', 'ewma_orb_diff', 'home_ewma_3p_pct',
    'home_rest_days', 'away_rest_days', 'rest_advantage',
    'injury_impact_diff', 'home_star_missing', 'away_star_missing',
    'off_elo_diff', 'def_elo_diff', 'home_composite_elo',
    'season_year', 'games_into_season', 'season_progress'
]

for home, away in games:
    print(f"\n{home} vs {away}:")
    print("-" * 80)
    feats = calc.calculate_game_features(home, away, game_date='2025-12-13')
    
    print(f"Total features: {len(feats)} (Expected: 43)")
    print("\nKey feature values:")
    for feat in model_features:
        val = feats.get(feat, "MISSING")
        if val == "MISSING":
            print(f"  ❌ {feat:25} = MISSING")
        elif isinstance(val, (int, float)) and val == 0.0:
            print(f"  ⚠️  {feat:25} = {val:.4f} (ZERO)")
        else:
            print(f"  ✓  {feat:25} = {val:.4f}" if isinstance(val, (int, float)) else f"  ✓  {feat:25} = {val}")

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print(f"All 43 features present: {len(feats) == 43}")
print(f"No missing features: {all(feats.get(f) != 'MISSING' for f in model_features)}")
print(f"Temporal features correct: season_year={feats.get('season_year')} (expect 2025)")
print(f"Rest features populated: home_rest={feats.get('home_rest_days')} (expect 2-3)")
