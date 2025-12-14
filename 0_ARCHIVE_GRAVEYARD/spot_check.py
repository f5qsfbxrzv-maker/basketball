"""
Spot Check Validation - Quick verification of feature generation
"""
from src.features.feature_calculator_live import FeatureCalculatorV5

print("=" * 80)
print("SPOT CHECK VALIDATION - December 13, 2025")
print("=" * 80)

calc = FeatureCalculatorV5()

# Test different games
games = [
    ('LAL', 'MEM', '2025-12-13'),
    ('BOS', 'DET', '2025-12-13'),
    ('CHI', 'TOR', '2025-12-13')
]

results = []
for home, away, date in games:
    print(f"\n{home} vs {away} ({date}):")
    feats = calc.calculate_game_features(home, away, game_date=date)
    
    print(f"  Features: {len(feats)} total")
    print(f"  Temporal: season_year={feats.get('season_year', '?'):.0f}, " +
          f"games_into_season={feats.get('games_into_season', 0):.1f}, " +
          f"season_progress={feats.get('season_progress', 0):.3f}")
    print(f"  Rest: home={feats.get('home_rest_days', 0):.0f}, " +
          f"away={feats.get('away_rest_days', 0):.0f}, " +
          f"home_b2b={feats.get('home_back_to_back', 0):.0f}, " +
          f"away_b2b={feats.get('away_back_to_back', 0):.0f}")
    print(f"  Injury: diff={feats.get('injury_impact_diff', 0):.2f}, " +
          f"home_star={feats.get('home_star_missing', 0):.0f}, " +
          f"away_star={feats.get('away_star_missing', 0):.0f}")
    print(f"  ELO: home_off={feats.get('home_off_elo', 1500):.1f}, " +
          f"home_def={feats.get('home_def_elo', 1500):.1f}, " +
          f"away_off={feats.get('away_off_elo', 1500):.1f}")
    print(f"  Four Factors: home_efg={feats.get('home_efg_pct', 0):.3f}, " +
          f"home_tov={feats.get('home_tov_pct', 0):.3f}")
    
    results.append(len(feats))

print(f"\n{'=' * 80}")
print("VERIFICATION SUMMARY")
print("=" * 80)
print(f"Feature counts: {results}")
print(f"All games returning 43 features: {all(r == 43 for r in results)}")
print(f"Season year correct (2025): {feats.get('season_year', 0) == 2025}")
print(f"Season progress reasonable (0.30-0.35): {0.30 <= feats.get('season_progress', 0) <= 0.35}")

if all(r == 43 for r in results):
    print("\n✓✓✓ SPOT CHECK PASSED ✓✓✓")
else:
    print("\n✗✗✗ SPOT CHECK FAILED ✗✗✗")
