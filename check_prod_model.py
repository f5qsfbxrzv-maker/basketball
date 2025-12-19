import joblib

bundle = joblib.load('models/production/basket_ats_model_20251121_210643.joblib')
features = bundle['features']

print(f'\n{"="*70}')
print('PRODUCTION MODEL TRAINED FEATURES')
print(f'{"="*70}\n')

print(f'Total: {len(features)} features\n')

print('ELO Features Check:')
has_home = 'home_composite_elo' in features
has_away = 'away_composite_elo' in features

print(f'  home_composite_elo: {"PRESENT" if has_home else "MISSING"}')
print(f'  away_composite_elo: {"PRESENT" if has_away else "MISSING"}')

if has_home and not has_away:
    print('\n!!! CRITICAL: Model trained WITHOUT away_composite_elo')
    print('    The model never learned away team raw ELO signal')
    print('    Must retrain with corrected 44-feature set\n')

print('\nAll features:')
for i, f in enumerate(features, 1):
    marker = ''
    if f == 'home_composite_elo':
        marker = ' <- HOME ELO ONLY'
    print(f'{i:2d}. {f}{marker}')

print(f'\n{"="*70}\n')
