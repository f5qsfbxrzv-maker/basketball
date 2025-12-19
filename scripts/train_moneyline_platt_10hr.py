"""
Optuna Hyperparameter Optimization for MONEYLINE Prediction
- Target: target_moneyline_win (not spread covers)
- Key changes: Low gamma (allow confident predictions), depth 3-5, class balance
- Goal: LogLoss < 0.650, ideally < 0.620
- Duration: 10 hours with 5-fold TimeSeriesSplit
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import optuna
from optuna.samplers import TPESampler
import json
from datetime import datetime

print("\n" + "="*90)
print("MONEYLINE-FOCUSED HYPERPARAMETER OPTIMIZATION")
print("="*90)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
print("\n[1/3] Loading training data...")
df = pd.read_csv('data/training_data_36features.csv')
df['date'] = pd.to_datetime(df['date'])

# Critical change: Use MONEYLINE target, not spread cover
exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df['target_moneyline_win']  # MONEYLINE target

print(f"  Samples: {len(df):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Home win rate: {y.mean()*100:.1f}%")
print(f"  Favorite win rate: ~67% (class imbalance)")

# Calculate class imbalance for scale_pos_weight
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
class_ratio = neg_count / pos_count
print(f"  Class ratio (away/home): {class_ratio:.3f}")

print("\n[2/3] Setting up Optuna study...")

def objective(trial):
    # Hyperparameters optimized for moneyline prediction
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 5),  # Opened up for complex favorites
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),  # Low gamma = confident predictions
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.5, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.1, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.5),  # Balance classes
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 50
    }
    
    # 5-fold TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, y_pred)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

# Create study
study_name = 'nba_moneyline_platt_10hr'
storage = f'sqlite:///models/{study_name}.db'

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction='minimize',
    sampler=TPESampler(seed=42),
    load_if_exists=True
)

print(f"  Study: {study_name}")
print(f"  Storage: {storage}")
print(f"  Objective: Minimize LogLoss on moneyline prediction")
print(f"  Duration: 10 hours (~600 trials estimated)")
print(f"\n  Key parameter changes:")
print(f"    - gamma: 0.0-3.0 (allow confident predictions, not 5.0+)")
print(f"    - max_depth: 3-5 (capture favorite strength nuances)")
print(f"    - scale_pos_weight: 0.8-1.5 (handle class imbalance)")
print(f"\n  Target: LogLoss < 0.650 (monster: < 0.620)")

# Run optimization
print(f"\n[3/3] Running optimization for 10 hours...")
print(f"{'='*90}\n")

try:
    study.optimize(
        objective,
        n_trials=1000,
        timeout=36000,  # 10 hours
        show_progress_bar=True
    )
except KeyboardInterrupt:
    print("\n\nOptimization interrupted by user.")

# Results
print(f"\n{'='*90}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*90}")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total trials: {len(study.trials)}")
print(f"Best trial: #{study.best_trial.number}")
print(f"Best LogLoss: {study.best_value:.6f}")

best_params = study.best_params
print(f"\nBest parameters:")
for key, value in best_params.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Key insights
print(f"\n{'='*90}")
print("KEY INSIGHTS")
print(f"{'='*90}")

gamma_val = best_params['gamma']
if gamma_val < 1.0:
    print(f"\n✓ Gamma = {gamma_val:.3f}: Model is CONFIDENT (gloves off!)")
elif gamma_val < 2.0:
    print(f"\n→ Gamma = {gamma_val:.3f}: Moderate confidence")
else:
    print(f"\n⚠ Gamma = {gamma_val:.3f}: Still conservative (may need more tuning)")

depth_val = best_params['max_depth']
if depth_val > 3:
    print(f"✓ Max depth = {depth_val}: Using extra complexity for favorites")
else:
    print(f"→ Max depth = {depth_val}: Simple tree structure sufficient")

weight_val = best_params['scale_pos_weight']
if 0.9 < weight_val < 1.1:
    print(f"→ Scale pos weight = {weight_val:.3f}: Balanced")
elif weight_val < 0.9:
    print(f"⚠ Scale pos weight = {weight_val:.3f}: Favoring majority class")
else:
    print(f"✓ Scale pos weight = {weight_val:.3f}: Helping underdog detection")

if study.best_value < 0.620:
    print(f"\n🔥 MONSTER MODEL: LogLoss {study.best_value:.6f} < 0.620!")
elif study.best_value < 0.650:
    print(f"\n✓ Strong performance: LogLoss {study.best_value:.6f} < 0.650")
else:
    print(f"\n→ Decent model: LogLoss {study.best_value:.6f} (target: < 0.650)")

# Save results
results = {
    'study_name': study_name,
    'completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_trials': len(study.trials),
    'best_trial': study.best_trial.number,
    'best_logloss': study.best_value,
    'best_params': best_params,
    'target': 'target_moneyline_win',
    'features': len(feature_cols),
    'samples': len(df),
    'cv_folds': 5
}

output_file = f'models/{study_name}_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {output_file}")
print(f"\n{'='*90}")
