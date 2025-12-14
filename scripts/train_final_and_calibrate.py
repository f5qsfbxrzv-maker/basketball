"""
Train final model with Trial 98 params + Isotonic calibration for Kelly optimization
No CV needed - we already validated. Just train on all data and calibrate.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from datetime import datetime
import pickle

print("="*70)
print("FINAL MODEL: TRIAL 98 + ISOTONIC CALIBRATION")
print("="*70)

# Load best params
with open('models/single_fold_best_params.json', 'r') as f:
    config = json.load(f)

params = config['best_params']
print(f"\nTrial: #{config['best_trial']}")
print(f"Validated AUC: {config['best_auc']:.5f}")

# Load data
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df['target_spread_cover']

print(f"\nDataset: {len(X):,} games, {X.shape[1]} features")

# Train XGBoost on ALL data
xgb_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'random_state': 42,
    'verbosity': 0,
    **params
}

print("\n" + "="*70)
print("TRAINING XGBOOST ON ALL DATA")
print("="*70)

model = xgb.XGBClassifier(**xgb_params)
model.fit(X, y, verbose=False)

# Get raw predictions
raw_probs = model.predict_proba(X)[:, 1]
raw_auc = roc_auc_score(y, raw_probs)
raw_brier = brier_score_loss(y, raw_probs)
raw_logloss = log_loss(y, raw_probs)

print(f"\nRaw XGBoost performance (in-sample):")
print(f"  AUC:      {raw_auc:.5f}")
print(f"  Brier:    {raw_brier:.5f}")
print(f"  Log Loss: {raw_logloss:.5f}")

# Isotonic calibration
print("\n" + "="*70)
print("FITTING ISOTONIC CALIBRATION")
print("="*70)

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_probs, y)

cal_probs = iso.transform(raw_probs)
cal_auc = roc_auc_score(y, cal_probs)
cal_brier = brier_score_loss(y, cal_probs)
cal_logloss = log_loss(y, cal_probs)

print(f"\nCalibrated performance (in-sample):")
print(f"  AUC:      {cal_auc:.5f}")
print(f"  Brier:    {cal_brier:.5f} ({(raw_brier - cal_brier)*100:+.2f}%)")
print(f"  Log Loss: {cal_logloss:.5f} ({(raw_logloss - cal_logloss)*100:+.2f}%)")

# Analyze calibration curve
print("\n" + "="*70)
print("CALIBRATION ANALYSIS")
print("="*70)

bins = np.linspace(0, 1, 11)
bin_centers = []
observed_freq = []
calibrated_freq = []

for i in range(len(bins)-1):
    mask = (raw_probs >= bins[i]) & (raw_probs < bins[i+1])
    if mask.sum() > 0:
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        observed_freq.append(y[mask].mean())
        calibrated_freq.append(cal_probs[mask].mean())

print("\nBin   Raw Pred  Observed  Calibrated  Count")
print("-" * 50)
for i in range(len(bins)-1):
    mask = (raw_probs >= bins[i]) & (raw_probs < bins[i+1])
    if mask.sum() > 0:
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}   {raw_probs[mask].mean():.3f}     {y[mask].mean():.3f}      {cal_probs[mask].mean():.3f}      {mask.sum():,}")

# Save models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

model.save_model('models/xgboost_final_trial98.json')
with open('models/isotonic_calibrator_final.pkl', 'wb') as f:
    pickle.dump(iso, f)

# Feature importance
importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Save metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'dataset': 'training_data_with_temporal_features.csv',
    'n_games': len(X),
    'n_features': X.shape[1],
    'features': features,
    'trial_number': config['best_trial'],
    'hyperparameters': params,
    'performance': {
        'raw_auc': float(raw_auc),
        'raw_brier': float(raw_brier),
        'raw_logloss': float(raw_logloss),
        'calibrated_auc': float(cal_auc),
        'calibrated_brier': float(cal_brier),
        'calibrated_logloss': float(cal_logloss),
        'brier_improvement': float(raw_brier - cal_brier),
        'logloss_improvement': float(raw_logloss - cal_logloss)
    },
    'feature_importance': importance_df.to_dict('records')[:20]
}

with open('models/final_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ XGBoost model: models/xgboost_final_trial98.json")
print(f"✓ Isotonic calibrator: models/isotonic_calibrator_final.pkl")
print(f"✓ Metadata: models/final_model_metadata.json")

# Top features
print("\n" + "="*70)
print("TOP 15 FEATURES")
print("="*70)

for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']*100:5.2f}%")

print("\n" + "="*70)
print("✓ PRODUCTION MODEL READY FOR KELLY OPTIMIZATION")
print("="*70)
print(f"\nKey Stats:")
print(f"  - Calibrated Brier: {cal_brier:.5f} (Kelly input)")
print(f"  - Brier improvement: {(raw_brier - cal_brier)*100:+.2f}%")
print(f"  - Training games: {len(X):,}")
print(f"  - Model complexity: {params['n_estimators']} trees, depth {params['max_depth']}")
