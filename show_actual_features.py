"""Show the ACTUAL features being used by the dashboard model"""
import xgboost as xgb
import pandas as pd

# Load the exact same model the dashboard uses
model_path = 'models/xgboost_final_trial98.json'
print(f"Loading model: {model_path}")

model = xgb.Booster()
model.load_model(model_path)

# Get feature names from the model
feature_names = model.feature_names
print(f"\nTotal features in model: {len(feature_names)}")

# Get importance scores
scores = model.get_score(importance_type='gain')
print(f"Features with importance scores: {len(scores)}")

# Create DataFrame with all features
feature_data = []
for i, feat in enumerate(feature_names):
    importance = scores.get(feat, 0.0)  # 0 if feature not used
    feature_data.append({
        'rank': i + 1,
        'feature': feat,
        'importance': importance
    })

df = pd.DataFrame(feature_data)
df = df.sort_values('importance', ascending=False)

print('\n' + '=' * 80)
print('ACTUAL FEATURES IN DASHBOARD MODEL (RANKED BY IMPORTANCE)')
print('=' * 80)
print()

rank = 1
for _, row in df.iterrows():
    if row['importance'] > 0:
        print(f"{rank:2d}. {row['feature']:40s} {row['importance']:10.1f}")
        rank += 1
    else:
        print(f"--. {row['feature']:40s} {'NOT USED':>10s}")

print()
print('=' * 80)
print(f"Features with importance > 0: {(df['importance'] > 0).sum()}")
print(f"Features with importance = 0: {(df['importance'] == 0).sum()}")
print(f"Total features: {len(df)}")
