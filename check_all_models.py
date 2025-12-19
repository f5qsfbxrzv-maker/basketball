"""Check all available models and their feature counts"""
import xgboost as xgb
import os

models_to_check = [
    'models/xgboost_final_trial98.json',
    'models/model_v6_ml.xgb',
    'models/model_v6_ats.xgb',
    'models/model_v6_total.xgb'
]

print('=' * 80)
print('AVAILABLE MODELS AND THEIR FEATURES')
print('=' * 80)
print()

for model_path in models_to_check:
    if not os.path.exists(model_path):
        print(f"❌ {model_path} - NOT FOUND")
        continue
    
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        features = model.feature_names
        
        print(f"✅ {model_path}")
        print(f"   Features: {len(features)}")
        print(f"   First 5: {features[:5]}")
        print()
    except Exception as e:
        print(f"❌ {model_path} - ERROR: {e}")
        print()
