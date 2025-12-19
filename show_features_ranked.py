"""Show features in ranked order by importance"""
import xgboost as xgb
import pandas as pd

# Load model
model = xgb.Booster()
model.load_model('models/xgboost_final_trial98.json')

# Get feature importance scores
scores = model.get_score(importance_type='gain')

# Create DataFrame and sort
df = pd.DataFrame(list(scores.items()), columns=['feature', 'importance'])
df = df.sort_values('importance', ascending=False)

print('=' * 80)
print('FEATURES RANKED BY IMPORTANCE (Gain)')
print('=' * 80)
print()

for i, row in df.iterrows():
    print(f"{i+1:2d}. {row['feature']:40s} {row['importance']:10.1f}")

print()
print('=' * 80)
print(f'Total features: {len(df)}')
