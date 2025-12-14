"""Check what features the model expects"""
import json

model = json.load(open('models/xgboost_final_trial98.json'))
feats = model['learner']['feature_names']

print('Model expects these 43 features:')
print('=' * 80)
for i, f in enumerate(feats, 1):
    print(f'{i:2}. {f}')
