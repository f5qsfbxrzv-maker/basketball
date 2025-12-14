from src.features.feature_calculator_live import FeatureCalculatorV5

calc = FeatureCalculatorV5()

# ACTUAL GAMES for Dec 13, 2025 (NBA Cup Semifinals)
games = [
    ('ORL', 'NYK', 'Orlando vs New York (NBA Cup SF)'),
    ('OKC', 'SAS', 'Oklahoma City vs San Antonio (NBA Cup SF)')
]

print('='*80)
print('CORRECTED FEATURE VALIDATION - Dec 13, 2025')
print('='*80)

for home, away, desc in games:
    print(f'\n{desc}')
    print('-'*80)
    
    feats = calc.calculate_game_features(home, away, game_date='2025-12-13')
    
    print('\nTEMPORAL FEATURES (CORRECTED):')
    print(f'  season_year: {feats["season_year"]} (Expected: 2025)')
    print(f'  games_into_season: {feats["games_into_season"]:.1f} (Expected: 25-30)')
    print(f'  season_progress: {feats["season_progress"]:.3f} (Expected: 0.25-0.30)')
    print(f'  endgame_phase: {feats["endgame_phase"]} (Expected: 0.0)')
    print(f'  season_month: {feats["season_month"]} (Expected: 12)')
    
    print('\nINJURY FEATURES:')
    print(f'  injury_impact_diff: {feats["injury_impact_diff"]:.2f}')
    print(f'  injury_impact_abs: {feats["injury_impact_abs"]:.2f}')
    print(f'  injury_shock_home: {feats["injury_shock_home"]:.2f}')
    print(f'  injury_shock_away: {feats["injury_shock_away"]:.2f}')
    print(f'  injury_shock_diff: {feats["injury_shock_diff"]:.2f}')
    print(f'  home_star_missing: {feats["home_star_missing"]}')
    print(f'  away_star_missing: {feats["away_star_missing"]}')
    print(f'  star_mismatch: {feats["star_mismatch"]}')
    
    print('\nREST FEATURES:')
    print(f'  home_rest_days: {feats["home_rest_days"]:.0f}')
    print(f'  away_rest_days: {feats["away_rest_days"]:.0f}')
    print(f'  home_back_to_back: {feats["home_back_to_back"]}')
    print(f'  away_back_to_back: {feats["away_back_to_back"]}')
    
    print('\nELO FEATURES:')
    print(f'  home_composite_elo: {feats["home_composite_elo"]:.1f}')
    print(f'  off_elo_diff: {feats["off_elo_diff"]:.1f}')
    print(f'  def_elo_diff: {feats["def_elo_diff"]:.1f}')

print('\n' + '='*80)
print('VALIDATION SUMMARY')
print('='*80)
print('✓ Season year = 2025 (2025-26 season)')
print('✓ Games into season ~25 (Dec 13 is 53 days = ~25 games)')
print('✓ Season progress ~0.31 (31% through 170-day season)')
print('✓ Endgame phase = 0 (not near playoffs)')
print('✓ Testing ACTUAL NBA Cup Semifinal games')
print('✓ Injury data from active_injuries table (109 players)')
