"""
Single Fold Hyperparameter Tuning - Fold 5 Only
Fast tuning (100 trials) on most recent data to compare to baseline.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

print("="*70)
print("SINGLE FOLD HYPERTUNING - FOLD 5 ONLY")
print("="*70)

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

# Get Fold 5 split
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    pass

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

print(f"\n1. Single fold split:")
print(f"   Train: {len(X_train):,} games")
print(f"   Val:   {len(X_val):,} games")
print(f"   Features: {X.shape[1]}")

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.12, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 800, 3000, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 35),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 5.0),
        'random_state': 42,
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    return auc

print("\n2. Running Optuna (100 trials on single fold)...")
print("   Baseline (fixed params): 0.63888")
print("   Target: 0.650+")
print("")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Best trial: #{study.best_trial.number}")
print(f"Best AUC: {study.best_value:.5f}")
print(f"Baseline: 0.63888")
print(f"Improvement: {(study.best_value - 0.63888)*100:+.2f}%")

print("\nBest parameters:")
for key, value in study.best_trial.params.items():
    if isinstance(value, float):
        print(f"  {key:20s}: {value:.4f}")
    else:
        print(f"  {key:20s}: {value}")

# Save
with open('models/single_fold_best_params.json', 'w') as f:
    json.dump({
        'best_trial': study.best_trial.number,
        'best_auc': study.best_value,
        'best_params': study.best_trial.params,
        'baseline_auc': 0.63888,
        'improvement_pct': (study.best_value - 0.63888) * 100,
        'tuning_date': datetime.now().isoformat()
    }, f, indent=2)

print(f"\n✓ Saved: models/single_fold_best_params.json")
print("="*70)
