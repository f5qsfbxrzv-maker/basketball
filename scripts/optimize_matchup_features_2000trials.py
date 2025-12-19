"""
Hyperparameter Optimization - Matchup-Optimized Features
- 24 features (consolidated from 37)
- Expanded parameter ranges for cleaner signal
- 2000 trials targeting LogLoss < 0.620
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
import os

print("\n" + "="*90)
print("HYPERPARAMETER OPTIMIZATION - MATCHUP-OPTIMIZED (24 FEATURES)")
print("="*90)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load matchup-optimized data
print("\n[1/3] Loading matchup-optimized dataset...")
df = pd.read_csv('data/training_data_matchup_optimized.csv')
df['date'] = pd.to_datetime(df['date'])

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df['target_moneyline_win']

print(f"  Samples: {len(df):,}")
print(f"  Features: {len(feature_cols)} (optimized from 37)")
print(f"  Home win rate: {y.mean()*100:.1f}%")

# Calculate class imbalance
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
class_ratio = neg_count / pos_count
print(f"  Class ratio (away/home): {class_ratio:.3f}")

print("\n[2/3] Setting up Optuna study...")

def objective(trial):
    # Expanded hyperparameter ranges for clean signal
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # Log scale!
        'max_depth': trial.suggest_int('max_depth', 3, 7),  # Expanded from 3-5
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 80),  # Much higher floor
        'gamma': trial.suggest_float('gamma', 0.5, 5.0),  # Wider range for split control
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.5, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.1, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.5),
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

# Create study (use absolute path for SQLite)
study_name = 'nba_matchup_optimized_2000trials'
db_path = os.path.abspath(f'models/{study_name}.db')
storage = f'sqlite:///{db_path}'

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction='minimize',
    sampler=TPESampler(seed=42),
    load_if_exists=True
)

print(f"  Study: {study_name}")
print(f"  Storage: {storage}")
print(f"  Objective: Minimize LogLoss")
print(f"  Trials: 2000 (~10 hours)")
print(f"\n  Parameter ranges (expanded for clean signal):")
print(f"    - learning_rate: 0.01-0.2 (log scale)")
print(f"    - max_depth: 3-7 (was 3-5)")
print(f"    - min_child_weight: 10-80 (was 5-20, prevents small sample overfitting)")
print(f"    - gamma: 0.5-5.0 (was 0.0-3.0, requires substantial gain to split)")
print(f"    - scale_pos_weight: 0.8-1.5")
print(f"\n  Baseline: LogLoss 0.627 (Trial #873 params on 37 features)")
print(f"  Target: LogLoss < 0.620 (monster model)")

# Run optimization
print(f"\n[3/3] Running 2000-trial optimization...")
print(f"{'='*90}\n")

try:
    study.optimize(
        objective,
        n_trials=2000,
        timeout=None,  # No timeout, let it finish 2000 trials
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

baseline_logloss = 0.627432  # From Trial #873 on 24 features
improvement = ((baseline_logloss - study.best_value) / baseline_logloss) * 100

print(f"\nComparison:")
print(f"  Baseline (old params): {baseline_logloss:.6f}")
print(f"  Optimized (new params): {study.best_value:.6f}")
print(f"  Improvement: {improvement:+.2f}%")

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

depth_val = best_params['max_depth']
if depth_val <= 4:
    print(f"\n→ Max depth = {depth_val}: Clean signal, simple structure sufficient")
elif depth_val <= 6:
    print(f"\n✓ Max depth = {depth_val}: Using extra complexity for matchup nuances")
else:
    print(f"\n⚠ Max depth = {depth_val}: High complexity (watch for overfitting)")

min_child = best_params['min_child_weight']
if min_child >= 40:
    print(f"✓ Min child weight = {min_child}: Conservative (strong generalization)")
elif min_child >= 20:
    print(f"→ Min child weight = {min_child}: Moderate (balanced)")
else:
    print(f"⚠ Min child weight = {min_child}: Low (may overfit small samples)")

gamma_val = best_params['gamma']
if gamma_val >= 3.0:
    print(f"✓ Gamma = {gamma_val:.2f}: High bar for splits (prevents noise)")
elif gamma_val >= 1.5:
    print(f"→ Gamma = {gamma_val:.2f}: Moderate split threshold")
else:
    print(f"⚠ Gamma = {gamma_val:.2f}: Low threshold (may split on marginal gains)")

lr_val = best_params['learning_rate']
if lr_val < 0.03:
    print(f"→ Learning rate = {lr_val:.4f}: Conservative (slow learning)")
elif lr_val < 0.08:
    print(f"✓ Learning rate = {lr_val:.4f}: Moderate (balanced)")
else:
    print(f"→ Learning rate = {lr_val:.4f}: Aggressive (fast learning)")

if study.best_value < 0.620:
    print(f"\n🔥 MONSTER MODEL: LogLoss {study.best_value:.6f} < 0.620!")
elif study.best_value < 0.625:
    print(f"\n🎉 Excellent: LogLoss {study.best_value:.6f} < 0.625")
elif study.best_value < 0.627:
    print(f"\n✓ Improved: LogLoss {study.best_value:.6f} (beat baseline)")
else:
    print(f"\n→ Similar: LogLoss {study.best_value:.6f} (close to baseline)")

# Save results
results = {
    'study_name': study_name,
    'completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_trials': len(study.trials),
    'best_trial': study.best_trial.number,
    'best_logloss': study.best_value,
    'best_params': best_params,
    'baseline_logloss': baseline_logloss,
    'improvement_pct': improvement,
    'target': 'target_moneyline_win',
    'features': len(feature_cols),
    'samples': len(df),
    'cv_folds': 5,
    'dataset': 'matchup_optimized'
}

output_file = f'models/{study_name}_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {output_file}")
print(f"\n{'='*90}")
print("NEXT STEPS")
print(f"{'='*90}")
print(f"\n1. Train final model with optimized parameters")
print(f"2. Apply Platt calibration")
print(f"3. Backtest with real moneyline odds")
print(f"4. Compare ROI to baseline -2.29%")
print(f"\n{'='*90}")
