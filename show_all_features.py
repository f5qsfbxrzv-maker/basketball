import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.features.feature_calculator_live import FeatureCalculatorV5

calc = FeatureCalculatorV5()

print('Generating features for OKC vs SAS...')
features = calc.calculate_game_features('OKC', 'SAS', game_date='2025-12-14')

print(f'\nTotal features: {len(features)}')

# Group features
four_factors = {k: v for k, v in features.items() if 'efg' in k or 'tov' in k or 'orb' in k or 'ftr' in k or 'ts_pct' in k}
ewma = {k: v for k, v in features.items() if 'ewma' in k}
elo = {k: v for k, v in features.items() if 'elo' in k}
injury = {k: v for k, v in features.items() if 'injury' in k or 'star' in k}

print(f'\nFOUR FACTORS features: {len(four_factors)}')
for k, v in sorted(four_factors.items()):
    print(f'  {k:40s}: {v:8.4f}')

print(f'\nEWMA features: {len(ewma)}')
for k, v in sorted(ewma.items()):
    print(f'  {k:40s}: {v:8.4f}')

print(f'\nELO features: {len(elo)}')
for k, v in sorted(elo.items()):
    print(f'  {k:40s}: {v:8.4f}')

print(f'\nINJURY features: {len(injury)}')
for k, v in sorted(injury.items()):
    print(f'  {k:40s}: {v:8.4f}')

print('\nALL FEATURES:')
for k, v in sorted(features.items()):
    print(f'  {k:40s}: {v:8.4f}')
