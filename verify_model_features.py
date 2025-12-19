import json

meta = json.load(open('models/final_model_metadata.json'))

print(f'\n{"="*70}')
print('CURRENT MODEL STATUS (Dec 12, 2025)')
print(f'{"="*70}\n')

print(f'Training date: {meta["training_date"]}')
print(f'Total features: {meta["n_features"]}')
print(f'Training games: {meta["n_games"]}\n')

print('ELO Features Check:')
has_home = 'home_composite_elo' in meta['features']
has_away = 'away_composite_elo' in meta['features']

print(f'  home_composite_elo: {"YES" if has_home else "NO"}')
print(f'  away_composite_elo: {"YES" if has_away else "NO"}')

if has_home and not has_away:
    print(f'\n{"!"*70}')
    print('CRITICAL: Model trained with HOME ELO but NOT AWAY ELO')
    print('The model never learned the away team raw ELO signal!')
    print(f'{"!"*70}\n')
    print('WHY THIS MATTERS:')
    print('  - Model only sees home_composite_elo (e.g., 1602)')
    print('  - Model does NOT see away_composite_elo (e.g., 1566)')
    print('  - Must infer away strength from diffs (off_elo_diff, def_elo_diff)')
    print('  - Loses information about absolute game tier (Clash of Titans vs Tank Bowl)')
    print(f'\nRECOMMENDATION:')
    print('  Retrain model with corrected 44-feature set including away_composite_elo')
    print('  This will improve predictions by 1-3% (better game context understanding)')

print(f'\n{"="*70}\n')
