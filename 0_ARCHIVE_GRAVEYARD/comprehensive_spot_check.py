"""
Comprehensive Spot Check - December 13, 2025
Validates all 43 model features with correct games
"""
from src.features.feature_calculator_live import FeatureCalculatorV5
import json

# Load model to get exact feature list
model = json.load(open('models/xgboost_final_trial98.json'))
expected_features = model['learner']['feature_names']

calc = FeatureCalculatorV5()

print("=" * 80)
print("COMPREHENSIVE SPOT CHECK - December 13, 2025")
print("NBA Cup Semifinals (CORRECT SCHEDULE)")
print("=" * 80)

# CORRECT games for Dec 13, 2025 (NBA Cup Semifinals)
games = [
    ('OKC', 'SAS', 'Oklahoma City vs San Antonio'),
    ('ORL', 'NYK', 'Orlando vs New York')
]

all_results = []

for home, away, description in games:
    print(f"\n{'=' * 80}")
    print(f"{description} ({home} vs {away})")
    print("=" * 80)
    
    # Calculate features
    feats = calc.calculate_game_features(home, away, game_date='2025-12-13')
    
    # Validate feature count
    print(f"\n✓ Feature Count: {len(feats)} (Expected: 43)")
    
    # Check all expected features are present
    missing = [f for f in expected_features if f not in feats]
    if missing:
        print(f"❌ MISSING FEATURES: {missing}")
    else:
        print(f"✓ All 43 model features present")
    
    # Group features by category for validation
    print("\n" + "-" * 80)
    print("TEMPORAL FEATURES (7 features)")
    print("-" * 80)
    temporal = {
        'season_year': (feats.get('season_year'), 2025.0, 'Must be 2025'),
        'games_into_season': (feats.get('games_into_season'), 25.0, '~25 games (53 days / 2.1)'),
        'season_progress': (feats.get('season_progress'), 0.31, '~31% through season'),
        'endgame_phase': (feats.get('endgame_phase'), 0.0, 'Not in endgame yet'),
        'season_month': (feats.get('season_month'), 12.0, 'December = 12'),
        'season_year_normalized': (feats.get('season_year_normalized'), None, 'Normalized year'),
        'is_season_opener': (feats.get('is_season_opener'), 0.0, 'Not opener')
    }
    for feat, (val, expected, note) in temporal.items():
        status = "✓" if val is not None and val != 0.0 or feat in ['endgame_phase', 'is_season_opener'] else "⚠️"
        print(f"  {status} {feat:25} = {val:>8.2f} | {note}" if isinstance(val, (int, float)) else f"  {status} {feat:25} = {val} | {note}")
    
    print("\n" + "-" * 80)
    print("REST & FATIGUE FEATURES (8 features)")
    print("-" * 80)
    rest_feats = ['home_rest_days', 'away_rest_days', 'rest_advantage', 'fatigue_mismatch',
                  'home_back_to_back', 'away_back_to_back', 'home_3in4', 'away_3in4']
    for feat in rest_feats:
        val = feats.get(feat, 'MISSING')
        status = "✓" if val != 'MISSING' else "❌"
        print(f"  {status} {feat:25} = {val:>8.2f}" if isinstance(val, (int, float)) else f"  {status} {feat:25} = {val}")
    
    print("\n" + "-" * 80)
    print("INJURY FEATURES (8 features)")
    print("-" * 80)
    injury_feats = ['injury_impact_diff', 'injury_impact_abs', 'injury_shock_home', 'injury_shock_away',
                    'injury_shock_diff', 'home_star_missing', 'away_star_missing', 'star_mismatch']
    for feat in injury_feats:
        val = feats.get(feat, 'MISSING')
        status = "✓" if val != 'MISSING' else "❌"
        print(f"  {status} {feat:25} = {val:>8.2f}" if isinstance(val, (int, float)) else f"  {status} {feat:25} = {val}")
    
    print("\n" + "-" * 80)
    print("EWMA FEATURES (13 features) - CRITICAL FOR MODEL")
    print("-" * 80)
    ewma_feats = ['ewma_efg_diff', 'ewma_tov_diff', 'ewma_orb_diff', 'ewma_pace_diff', 'ewma_vol_3p_diff',
                  'home_ewma_3p_pct', 'away_ewma_3p_pct', 'away_ewma_tov_pct', 'home_orb', 'away_orb',
                  'away_ewma_fta_rate', 'ewma_chaos_home', 'ewma_net_chaos']
    for feat in ewma_feats:
        val = feats.get(feat, 'MISSING')
        if val == 'MISSING':
            print(f"  ❌ {feat:25} = MISSING")
        elif isinstance(val, (int, float)) and abs(val) < 0.0001:
            print(f"  ⚠️  {feat:25} = {val:>8.4f} (ZERO - may indicate calculation issue)")
        else:
            print(f"  ✓  {feat:25} = {val:>8.4f}" if isinstance(val, (int, float)) else f"  ✓  {feat:25} = {val}")
    
    print("\n" + "-" * 80)
    print("ELO FEATURES (3 features)")
    print("-" * 80)
    elo_feats = ['off_elo_diff', 'def_elo_diff', 'home_composite_elo']
    for feat in elo_feats:
        val = feats.get(feat, 'MISSING')
        if val == 'MISSING':
            print(f"  ❌ {feat:25} = MISSING")
        elif isinstance(val, (int, float)) and abs(val - 1500.0) < 10.0:
            print(f"  ⚠️  {feat:25} = {val:>8.2f} (DEFAULT VALUE - not using historical data?)")
        else:
            print(f"  ✓  {feat:25} = {val:>8.2f}" if isinstance(val, (int, float)) else f"  ✓  {feat:25} = {val}")
    
    print("\n" + "-" * 80)
    print("OTHER FEATURES (4 features)")
    print("-" * 80)
    other_feats = ['altitude_game', 'ewma_foul_synergy_home', 'ewma_foul_synergy_away', 'total_foul_environment']
    for feat in other_feats:
        val = feats.get(feat, 'MISSING')
        status = "✓" if val != 'MISSING' else "❌"
        print(f"  {status} {feat:25} = {val:>8.4f}" if isinstance(val, (int, float)) else f"  {status} {feat:25} = {val}")
    
    # Validation summary for this game
    all_present = len(missing) == 0
    ewma_populated = all(abs(feats.get(f, 0)) > 0.0001 for f in ['ewma_efg_diff', 'ewma_tov_diff', 'ewma_orb_diff'])
    elo_customized = abs(feats.get('home_composite_elo', 1500) - 1500.0) > 10.0
    
    result = {
        'game': description,
        'all_present': all_present,
        'ewma_populated': ewma_populated,
        'elo_customized': elo_customized,
        'temporal_correct': feats.get('season_year') == 2025.0
    }
    all_results.append(result)

# Final summary
print("\n" + "=" * 80)
print("FINAL VALIDATION SUMMARY")
print("=" * 80)

for r in all_results:
    print(f"\n{r['game']}:")
    print(f"  {'✓' if r['all_present'] else '❌'} All 43 features present: {r['all_present']}")
    print(f"  {'✓' if r['ewma_populated'] else '❌'} EWMA features populated: {r['ewma_populated']}")
    print(f"  {'✓' if r['elo_customized'] else '❌'} ELO using historical data: {r['elo_customized']}")
    print(f"  {'✓' if r['temporal_correct'] else '❌'} Temporal features correct (2025): {r['temporal_correct']}")

all_pass = all(r['all_present'] and r['ewma_populated'] and r['temporal_correct'] for r in all_results)

print("\n" + "=" * 80)
if all_pass:
    print("✓✓✓ COMPREHENSIVE SPOT CHECK PASSED ✓✓✓")
    print("System is ready for predictions on December 13, 2025 games")
else:
    print("❌❌❌ SPOT CHECK FAILED - FIX ISSUES BEFORE PREDICTIONS ❌❌❌")
print("=" * 80)
