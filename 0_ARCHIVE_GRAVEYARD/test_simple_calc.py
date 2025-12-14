from simple_feature_calculator import SimpleFeatureCalculator
import joblib

# Test calculator
calc = SimpleFeatureCalculator()
features = calc.calculate_game_features('Lakers', 'Warriors', '2025-11-20')

print(f'✓ Returned {len(features)} features')

# Load model and check
model = joblib.load('models/production/moneyline_model_enhanced.pkl')
model_features = set(model.feature_names_in_)
calc_features = set(features.keys())

missing = model_features - calc_features
extra = calc_features - model_features

if len(missing) == 0 and len(extra) == 0:
    print('✓ ALL 36 features match perfectly!')
else:
    print(f'✗ Missing {len(missing)} features: {list(missing)}')
    print(f'✗ Extra {len(extra)} features: {list(extra)}')

# Print first 10 features
print('\nFirst 10 features:')
for i, (k, v) in enumerate(list(features.items())[:10]):
    print(f'  {k}: {v}')
