"""
Train final model with Trial 98 params on 5-fold CV + Isotonic calibration
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from datetime import datetime
import pickle

print("="*70)
print("FINAL MODEL TRAINING - TRIAL 98 PARAMS + ISOTONIC CALIBRATION")
print("="*70)

# Load best params from Trial 98
with open('models/single_fold_best_params.json', 'r') as f:
    best_config = json.load(f)
    
params = best_config['best_params']
print(f"\nBest Trial: #{best_config['best_trial']}")
print(f"Single Fold AUC: {best_config['best_auc']:.5f}")

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

# Configure XGBoost with Trial 98 params
xgb_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'random_state': 42,
    'verbosity': 0,
    **params
}

print("\n" + "="*70)
print("5-FOLD CROSS-VALIDATION")
print("="*70)

tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"\nFold {fold}:")
    print(f"  Train: {len(X_train):,} games")
    print(f"  Val:   {len(X_val):,} games")
    
    # Train XGBoost
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, verbose=False)
    
    # Raw predictions
    raw_probs = model.predict_proba(X_val)[:, 1]
    raw_auc = roc_auc_score(y_val, raw_probs)
    raw_brier = brier_score_loss(y_val, raw_probs)
    
    # Isotonic calibration on training fold
    iso = IsotonicRegression(out_of_bounds='clip')
    train_probs = model.predict_proba(X_train)[:, 1]
    iso.fit(train_probs, y_train)
    
    # Apply calibration to validation
    cal_probs = iso.transform(raw_probs)
    cal_auc = roc_auc_score(y_val, cal_probs)
    cal_brier = brier_score_loss(y_val, cal_probs)
    
    print(f"  Raw:        AUC={raw_auc:.5f}, Brier={raw_brier:.5f}")
    print(f"  Calibrated: AUC={cal_auc:.5f}, Brier={cal_brier:.5f}")
    print(f"  Δ Brier:    {(raw_brier - cal_brier)*100:+.2f}%")
    
    fold_results.append({
        'fold': fold,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'raw_auc': raw_auc,
        'raw_brier': raw_brier,
        'cal_auc': cal_auc,
        'cal_brier': cal_brier,
        'brier_improvement': raw_brier - cal_brier
    })

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

avg_raw_auc = np.mean([r['raw_auc'] for r in fold_results])
avg_raw_brier = np.mean([r['raw_brier'] for r in fold_results])
avg_cal_auc = np.mean([r['cal_auc'] for r in fold_results])
avg_cal_brier = np.mean([r['cal_brier'] for r in fold_results])
avg_brier_improvement = np.mean([r['brier_improvement'] for r in fold_results])

print(f"\n5-Fold Averages:")
print(f"  Raw XGBoost:        AUC={avg_raw_auc:.5f}, Brier={avg_raw_brier:.5f}")
print(f"  Isotonic Calibrated: AUC={avg_cal_auc:.5f}, Brier={avg_cal_brier:.5f}")
print(f"  Brier Improvement:   {avg_brier_improvement*100:+.2f}%")

# Train final model on ALL data
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*70)

final_model = xgb.XGBClassifier(**xgb_params)
print(f"\nTraining on {len(X):,} games...")
final_model.fit(X, y, verbose=False)

# Fit isotonic calibration on all data
final_iso = IsotonicRegression(out_of_bounds='clip')
all_train_probs = final_model.predict_proba(X)[:, 1]
final_iso.fit(all_train_probs, y)

# Save models
final_model.save_model('models/xgboost_final_trial98.json')
with open('models/isotonic_calibrator_final.pkl', 'wb') as f:
    pickle.dump(final_iso, f)

# Save metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'dataset': 'training_data_with_temporal_features.csv',
    'n_games': len(X),
    'n_features': X.shape[1],
    'features': features,
    'trial_number': best_config['best_trial'],
    'hyperparameters': params,
    'cv_results': {
        'raw_auc_mean': float(avg_raw_auc),
        'raw_brier_mean': float(avg_raw_brier),
        'calibrated_auc_mean': float(avg_cal_auc),
        'calibrated_brier_mean': float(avg_cal_brier),
        'brier_improvement_mean': float(avg_brier_improvement),
        'fold_details': fold_results
    }
}

with open('models/final_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Saved XGBoost model: models/xgboost_final_trial98.json")
print(f"✓ Saved Isotonic calibrator: models/isotonic_calibrator_final.pkl")
print(f"✓ Saved metadata: models/final_model_metadata.json")

# Feature importance
print("\n" + "="*70)
print("TOP 15 FEATURES")
print("="*70)

importance_df = pd.DataFrame({
    'feature': features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']*100:5.2f}%")

print("\n" + "="*70)
print("✓ COMPLETE - Model ready for production")
print("="*70)
