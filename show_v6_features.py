"""Show the complete feature list for model_v6_ml.xgb (19 features)"""
import xgboost as xgb
import pandas as pd

model_path = 'models/model_v6_ml.xgb'
print(f"Loading: {model_path}")

model = xgb.Booster()
model.load_model(model_path)

features = model.feature_names
print(f"\nTotal features: {len(features)}")

# Get importance
scores = model.get_score(importance_type='gain')

# Create ranked list
data = []
for feat in features:
    importance = scores.get(feat, 0.0)
    data.append({'feature': feat, 'importance': importance})

df = pd.DataFrame(data).sort_values('importance', ascending=False)

print('\n' + '=' * 80)
print('MODEL V6 ML - 19 FEATURES RANKED BY IMPORTANCE')
print('=' * 80)
print()

for i, row in df.iterrows():
    rank = i + 1
    if row['importance'] > 0:
        print(f"{rank:2d}. {row['feature']:35s} {row['importance']:10.1f}")
    else:
        print(f"--. {row['feature']:35s} {'NOT USED':>10s}")

print()
print('=' * 80)
