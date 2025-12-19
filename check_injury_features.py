"""Compare injury features between models and feature calculator"""
import xgboost as xgb

# Check 43-feature model
print('=' * 80)
print('43-FEATURE MODEL (xgboost_final_trial98.json) - INJURY FEATURES')
print('=' * 80)

model = xgb.Booster()
model.load_model('models/xgboost_final_trial98.json')
features = model.feature_names
scores = model.get_score(importance_type='gain')

injury_features = [f for f in features if 'injury' in f.lower() or 'star' in f.lower()]
print(f"\nInjury-related features: {len(injury_features)}")
for feat in injury_features:
    importance = scores.get(feat, 0.0)
    status = "✅" if importance > 0 else "❌"
    print(f"{status} {feat:35s} {importance:8.1f}")

# Check what feature calculator generates
print('\n' + '=' * 80)
print('FEATURE CALCULATOR OUTPUT - INJURY FEATURES')
print('=' * 80)

try:
    from src.features.feature_calculator_v5 import FeatureCalculatorV5
    fc = FeatureCalculatorV5('data/nba_betting_data.db')
    features_calc = fc.calculate_game_features('PHX', 'LAL', '2025-12-15')
    
    injury_calc = [k for k in features_calc.keys() if 'injury' in k.lower() or 'star' in k.lower()]
    print(f"\nInjury features generated: {len(injury_calc)}")
    for feat in injury_calc:
        print(f"✅ {feat:35s} {features_calc[feat]:8.2f}")
    
    if not injury_calc:
        print("❌ NO INJURY FEATURES GENERATED!")
        
except Exception as e:
    print(f"Error: {e}")

print('\n' + '=' * 80)
print('ANALYSIS')
print('=' * 80)
print("\nModel needs these injury features:")
for feat in injury_features:
    print(f"  - {feat}")
