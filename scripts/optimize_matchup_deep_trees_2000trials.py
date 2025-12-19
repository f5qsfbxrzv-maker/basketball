"""
Hyperparameter Optimization - Deep Trees Strategy
- Lower learning rate: 0.005-0.03 (precise minimum)
- Deeper trees: 8-12 (complex interactions)
- Relaxed gamma: 0-0.5 (allow growth, control with L1/L2)
- Pruning enabled for computational efficiency
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
from datetime import datetime
import os

print("\n" + "="*90)
print("HYPERPARAMETER OPTIMIZATION - DEEP TREES + LOW LEARNING RATE")
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
print(f"  Features: {len(feature_cols)} (matchup-optimized)")
print(f"  Home win rate: {y.mean()*100:.1f}%")

# Calculate class imbalance
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
class_ratio = neg_count / pos_count
print(f"  Class ratio (away/home): {class_ratio:.3f}")

print("\n[2/3] Setting up Optuna study with pruning...")

def objective(trial):
    # Deep trees strategy with lower learning rate
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),  # Lower + precise
        'max_depth': trial.suggest_int('max_depth', 8, 12),  # Deep for interactions
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 80),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),  # Relaxed - allow growth
        'subsample': trial.suggest_float('subsample', 0.4, 0.95),  # Wider range
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),  # Control with L2
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),  # Control with L1
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.5),
        'n_estimators': 2000,  # More rounds for low learning rate
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 100
    }
    
    # 5-fold TimeSeriesSplit with manual pruning
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
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
        
        # Manual pruning: Report intermediate value and check if should prune
        trial.report(np.mean(cv_scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(cv_scores)

# Create study with pruning
study_name = 'nba_matchup_deep_trees_2000trials'
db_path = os.path.abspath(f'models/{study_name}.db')
storage = f'sqlite:///{db_path}'

# MedianPruner: Stops trials performing worse than median
pruner = MedianPruner(
    n_startup_trials=50,  # Don't prune first 50 trials
    n_warmup_steps=2,     # Evaluate at least 2 folds before pruning
    interval_steps=1      # Check every fold
)

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction='minimize',
    sampler=TPESampler(seed=42),
    pruner=pruner,
    load_if_exists=False  # Fresh start
)

print(f"  Study: {study_name}")
print(f"  Storage: {storage}")
print(f"  Objective: Minimize LogLoss")
print(f"  Trials: 2000")
print(f"  Pruning: MedianPruner (startup=50, warmup=2)")
print(f"\n  STRATEGY: Deep Trees + Low Learning Rate")
print(f"  ─────────────────────────────────────────")
print(f"    - learning_rate: 0.005-0.03 (log, precise minimum)")
print(f"    - max_depth: 8-12 (was 3-7, capture complex interactions)")
print(f"    - gamma: 0.0-0.5 (was 0.5-5.0, allow growth)")
print(f"    - L1/L2 regularization: Higher ranges (control overfitting)")
print(f"    - n_estimators: 2000 (was 1000, for low learning rate)")
print(f"    - subsample: 0.4-0.95 (wider variance)")
print(f"\n  Baseline: LogLoss 0.6274 (Trial #873 params, 24 features)")
print(f"  Target: LogLoss < 0.620 (monster model)")

# Run optimization
print(f"\n[3/3] Running 2000-trial optimization...")
print(f"{'='*90}\n")

try:
    study.optimize(
        objective,
        n_trials=2000,
        timeout=None,
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
print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Best trial: #{study.best_trial.number}")
print(f"Best LogLoss: {study.best_value:.6f}")

baseline_logloss = 0.627432  # From Trial #873 on 24 features
improvement = ((baseline_logloss - study.best_value) / baseline_logloss) * 100

print(f"\nComparison:")
print(f"  Baseline (Trial #873 params): {baseline_logloss:.6f}")
print(f"  Optimized (deep trees): {study.best_value:.6f}")
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
print("KEY INSIGHTS - DEEP TREES STRATEGY")
print(f"{'='*90}")

depth_val = best_params['max_depth']
print(f"\n→ Max depth = {depth_val}: ", end="")
if depth_val >= 11:
    print(f"Maximum complexity (capturing subtle interactions)")
elif depth_val >= 9:
    print(f"Deep interactions (balanced complexity)")
else:
    print(f"Moderate depth (simpler structure preferred)")

min_child = best_params['min_child_weight']
print(f"→ Min child weight = {min_child}: ", end="")
if min_child >= 40:
    print(f"Conservative (strong generalization)")
else:
    print(f"Moderate (balanced fit)")

gamma_val = best_params['gamma']
print(f"→ Gamma = {gamma_val:.3f}: ", end="")
if gamma_val < 0.1:
    print(f"Very relaxed (allows easy splitting)")
elif gamma_val < 0.3:
    print(f"Relaxed (balanced split threshold)")
else:
    print(f"Moderate control (some split resistance)")

lr_val = best_params['learning_rate']
print(f"→ Learning rate = {lr_val:.4f}: ", end="")
if lr_val < 0.01:
    print(f"Very low (precise, slow convergence)")
elif lr_val < 0.02:
    print(f"Low (careful optimization)")
else:
    print(f"Moderate-low (balanced)")

l2_val = best_params['reg_lambda']
l1_val = best_params['reg_alpha']
print(f"→ Regularization: L2={l2_val:.3f}, L1={l1_val:.3f}")
if l2_val > 5.0 or l1_val > 2.0:
    print(f"  Strong regularization (controlling deep tree overfitting)")
elif l2_val > 1.0 or l1_val > 0.5:
    print(f"  Moderate regularization (balanced)")
else:
    print(f"  Light regularization (confident in structure)")

if study.best_value < 0.620:
    print(f"\n🔥 MONSTER MODEL: LogLoss {study.best_value:.6f} < 0.620!")
elif study.best_value < 0.625:
    print(f"\n🎉 Excellent: LogLoss {study.best_value:.6f} < 0.625")
elif study.best_value < baseline_logloss:
    print(f"\n✓ Improved: LogLoss {study.best_value:.6f} (beat baseline)")
else:
    print(f"\n→ Similar: LogLoss {study.best_value:.6f} (comparable to baseline)")

# Save results
results = {
    'study_name': study_name,
    'completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_trials': len(study.trials),
    'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    'best_trial': study.best_trial.number,
    'best_logloss': study.best_value,
    'best_params': best_params,
    'baseline_logloss': baseline_logloss,
    'improvement_pct': improvement,
    'strategy': 'deep_trees_low_lr',
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
print(f"\n1. Train final model with optimized deep tree parameters")
print(f"2. Compare feature importance (depth 8-12 vs depth 3)")
print(f"3. Apply Platt calibration")
print(f"4. Backtest with real moneyline odds")
print(f"\n{'='*90}")
