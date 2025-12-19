"""
2000-trial hyperparameter optimization for 37-feature model
Same deep search strategy as 8-hour optimization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import optuna
from datetime import datetime
import json

print("\n" + "="*70)
print("2000-TRIAL OPTIMIZATION - 37 FEATURES")
print("="*70)

# Load data
print("\n[1/4] Loading training data...")
df = pd.read_csv('data/training_data_36features.csv')

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win', 
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]
y = df['target_spread_cover']

print(f"  Loaded: {len(df):,} games x {len(feature_cols)} features")
print(f"  Covers: {y.sum():,} ({y.mean()*100:.1f}%)")

# Objective function
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0.1, 5),
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
print("\n[2/4] Initializing Optuna study...")

storage_path = "sqlite:///models/nba_optuna_37features.db"
study_name = "nba_37features_2000trials"

study = optuna.create_study(
    direction="minimize",
    storage=storage_path,
    study_name=study_name,
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=2
    )
)

print(f"  Study: {study_name}")
print(f"  Storage: {storage_path}")
print(f"  Direction: minimize log_loss")
print(f"  Pruning: MedianPruner")

# Check if resuming
n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
if n_completed > 0:
    print(f"\n  Resuming from {n_completed} completed trials")
    print(f"  Best so far: {study.best_value:.6f}")

print("\n[3/4] Starting 2000-trial optimization...")
print("  Expected runtime: ~8-10 hours")
print("  Press Ctrl+C to stop early (results will be saved)\n")

# Callback to print progress
def print_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        auc = trial.user_attrs.get('avg_auc', None)
        auc_str = f"{auc:.5f}" if auc else "N/A"
        print(f"  Trial {trial.number:4d}: LogLoss={trial.value:.6f}, AUC={auc_str}")
        if trial.number == study.best_trial.number:
            print(f"    ** NEW BEST **")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"  Trial {trial.number:4d}: PRUNED")

try:
    study.optimize(
        objective,
        n_trials=2000,
        callbacks=[print_callback],
        show_progress_bar=False
    )
except KeyboardInterrupt:
    print("\n\nOptimization stopped early (Ctrl+C)")

# Results
print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
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
    'n_features': len(feature_cols),
    'n_trials': len(study.trials),
    'best_trial': best_trial.number,
    'logloss': best_trial.value,
    'auc': best_auc,
    'params': best_trial.params
}

with open('models/optuna_37features_2000trials_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[4/4] Results saved: models/optuna_37features_2000trials_results.json")
print("="*70)
