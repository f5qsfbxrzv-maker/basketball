from simple_feature_calculator import SimpleFeatureCalculator

calc = SimpleFeatureCalculator()
f = calc.calculate_game_features('Lakers', 'Warriors', '2025-11-20')

print('✓ ALL 36 Features Generated:\n')
print('ELO Features:')
for k in ['composite_elo_diff', 'off_elo_diff', 'def_elo_diff', 'home_composite_elo', 'away_composite_elo']:
    print(f'  {k}: {f[k]:.2f}')

print('\nRating Features:')
for k in ['off_rating_diff', 'def_rating_diff', 'home_off_rating', 'away_off_rating', 'home_def_rating', 'away_def_rating']:
    print(f'  {k}: {f[k]:.2f}')

print('\nPace Features:')
for k in ['pace_diff', 'avg_pace', 'home_pace', 'away_pace']:
    print(f'  {k}: {f[k]:.2f}')

print('\nFour Factors:')
for k in ['home_efg', 'away_efg', 'efg_diff', 'home_tov', 'away_tov', 'home_orb', 'away_orb', 'home_ftr', 'away_ftr']:
    print(f'  {k}: {f[k]:.4f}')

print('\nRest Features:')
for k in ['home_rest_days', 'away_rest_days', 'rest_advantage', 'home_back_to_back', 'away_back_to_back', 'both_rested']:
    print(f'  {k}: {f[k]}')

print('\nPace Indicators:')
for k in ['predicted_pace', 'pace_up_game', 'pace_down_game', 'altitude_game']:
    print(f'  {k}: {f[k]}')

print('\nMatchup Features:')
for k in ['off_def_matchup_home', 'off_def_matchup_away']:
    print(f'  {k}: {f[k]:.2f}')
