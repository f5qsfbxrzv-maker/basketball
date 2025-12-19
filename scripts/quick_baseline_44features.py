"""
Quick baseline hyperparameter search with adjusted ranges:
- Higher learning rate (0.01-0.15)
- Deeper trees (max_depth 6-10)
- Lower gamma (0-1.5)
- 100 trials (~30 minutes)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import optuna
from datetime import datetime

print("\n" + "="*70)
print("QUICK BASELINE - 44 FEATURES")
print("="*70)

# Load data
print("\n[1/3] Loading training data...")
df = pd.read_csv('data/training_data_with_temporal_features.csv')

# Filter to actual features (exclude IDs, dates, targets)
exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win', 
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"  Loaded: {len(df):,} games x {len(feature_cols)} features")

# Use target_spread_cover as target
X = df[feature_cols]
y = df['target_spread_cover']

print(f"  Covers: {y.sum():,} / Non-covers: {(~y).sum():,}")

# Define objective with aggressive parameters
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 1.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5, log=True),
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0
    }
    
    # 5-fold TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    fold_aucs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        fold_logloss = log_loss(y_val, preds)
        fold_auc = roc_auc_score(y_val, preds)
        
        fold_scores.append(fold_logloss)
        fold_aucs.append(fold_auc)
        
        # Pruning
        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    avg_logloss = np.mean(fold_scores)
    avg_auc = np.mean(fold_aucs)
    
    trial.set_user_attr('avg_auc', avg_auc)
    
    return avg_logloss

# Run optimization
print("\n[2/3] Running 100-trial baseline...")
print("  Learning rate: 0.01-0.15 (higher)")
print("  Max depth: 6-10 (deeper)")
print("  Gamma: 0-1.5 (lower)")
print("  Expected runtime: ~30 minutes\n")

storage_path = "sqlite:///models/nba_baseline_44features.db"
study_name = "baseline_44features_v1"

study = optuna.create_study(
    direction="minimize",
    storage=storage_path,
    study_name=study_name,
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
)

def print_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        auc = trial.user_attrs.get('avg_auc', None)
        auc_str = f"{auc:.5f}" if auc else "N/A"
        print(f"  Trial {trial.number:3d}: LogLoss={trial.value:.6f}, AUC={auc_str}")
        if trial.number == study.best_trial.number:
            print(f"    NEW BEST!")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"  Trial {trial.number:3d}: PRUNED")

try:
    study.optimize(
        objective,
        n_trials=100,
        callbacks=[print_callback],
        show_progress_bar=False
    )
except KeyboardInterrupt:
    print("\n\nOptimization stopped early (Ctrl+C)")

# Results
print("\n" + "="*70)
print("BASELINE RESULTS")
print("="*70)

best_trial = study.best_trial
best_auc = best_trial.user_attrs.get('avg_auc', 'N/A')

print(f"\nBest Trial: #{best_trial.number}")
print(f"  LogLoss: {best_trial.value:.6f}")
print(f"  AUC: {best_auc:.5f}" if isinstance(best_auc, float) else f"  AUC: {best_auc}")
print(f"\nBest Hyperparameters:")
for key, value in best_trial.params.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'best_trial': best_trial.number,
    'logloss': best_trial.value,
    'auc': best_auc,
    'params': best_trial.params
}

import json
with open('models/baseline_44features_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[3/3] Results saved: models/baseline_44features_results.json")
print("\nNext: If AUC > 0.75, train final model with these params")
print("="*70)
