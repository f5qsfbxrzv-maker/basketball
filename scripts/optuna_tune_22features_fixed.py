"""
Optuna Hyperparameter Tuning - 22 Features with FIXED ELO Dataset
- Uses repaired home_composite_elo (no more wild oscillations)
- Removed redundant injury components (injury_impact_diff, injury_shock_diff, star_mismatch)
- Keeps injury_matchup_advantage (optimized composite)
- Conservative strategy: 3000 trials
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import optuna
from datetime import datetime
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_PATH = 'data/training_data_matchup_with_injury_advantage_FIXED.csv'
OUTPUT_MODEL_PATH = 'models/xgboost_22features_fixed_optuna_{timestamp}.json'

# 22 Features (removed 3 redundant injury components)
FEATURES = [
    # ELO Ratings (4) - NOW WITH FIXED HOME ELO!
    'home_composite_elo',
    'away_composite_elo',
    'off_elo_diff',
    'def_elo_diff',
    
    # EWMA Stats (6)
    'ewma_efg_diff',
    'ewma_pace_diff',
    'ewma_tov_diff',
    'ewma_orb_diff',
    'ewma_vol_3p_diff',
    'ewma_chaos_home',
    
    # Injuries (1) - COMPOSITE ONLY, removed components
    'injury_matchup_advantage',
    
    # Advanced Matchup Features (11)
    'net_fatigue_score',
    'ewma_foul_synergy_home',
    'total_foul_environment',
    'league_offensive_context',
    'season_progress',
    'pace_efficiency_interaction',
    'projected_possession_margin',
    'three_point_matchup',
    'net_free_throw_advantage',
    'star_power_leverage',
    'offense_vs_defense_matchup'
]

TARGET = 'target_moneyline_win'

# Conservative hyperparameter ranges (same as before)
PARAM_RANGES = {
    'max_depth': (3, 5),
    'min_child_weight': (25, 75),
    'gamma': (2.0, 10.0),
    'learning_rate': (0.001, 0.02),
    'n_estimators': (5000, 12000),
    'subsample': (0.5, 0.7),
    'colsample_bytree': (0.5, 0.7),
    'reg_alpha': (5.0, 20.0)
}

N_TRIALS = 3000
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 100

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("="*90)
print("OPTUNA HYPERPARAMETER TUNING - 22 FEATURES WITH FIXED ELO")
print("="*90)

print(f"\n[1/5] Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values('date').reset_index(drop=True)

print(f"  Total samples: {len(df):,}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# Verify all features exist
missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    print(f"\n❌ ERROR: Missing features: {missing_features}")
    exit(1)

X = df[FEATURES].values
y = df[TARGET].values

print(f"\n  Features: {len(FEATURES)}")
print(f"  Target balance: {y.mean():.1%} wins")

# Check for NaNs
if np.isnan(X).any():
    print(f"\n❌ ERROR: NaN values detected in features!")
    nan_cols = [FEATURES[i] for i in range(len(FEATURES)) if np.isnan(X[:, i]).any()]
    print(f"  Columns with NaNs: {nan_cols}")
    exit(1)

print(f"\n  ✓ Data validation passed")

# ==============================================================================
# OPTUNA OBJECTIVE
# ==============================================================================
def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Sample hyperparameters
    n_estimators = trial.suggest_int('n_estimators', *PARAM_RANGES['n_estimators'])
    
    params = {
        'max_depth': trial.suggest_int('max_depth', *PARAM_RANGES['max_depth']),
        'min_child_weight': trial.suggest_int('min_child_weight', *PARAM_RANGES['min_child_weight']),
        'gamma': trial.suggest_float('gamma', *PARAM_RANGES['gamma']),
        'learning_rate': trial.suggest_float('learning_rate', *PARAM_RANGES['learning_rate'], log=True),
        'subsample': trial.suggest_float('subsample', *PARAM_RANGES['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *PARAM_RANGES['colsample_bytree']),
        'reg_alpha': trial.suggest_float('reg_alpha', *PARAM_RANGES['reg_alpha']),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42
    }
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )
        
        # Predict and score
        y_pred = model.predict(dval)
        loss = log_loss(y_val, y_pred)
        fold_scores.append(loss)
    
    # Return mean validation loss
    return np.mean(fold_scores)

# ==============================================================================
# RUN OPTIMIZATION
# ==============================================================================
print(f"\n[2/5] Starting Optuna optimization...")
print(f"  Trials: {N_TRIALS}")
print(f"  Cross-validation folds: {N_SPLITS}")
print(f"  Strategy: Conservative (deep regularization)")
print(f"\n  Hyperparameter ranges:")
for param, (low, high) in PARAM_RANGES.items():
    print(f"    {param:<20} [{low}, {high}]")

print(f"\n  Starting optimization at {datetime.now().strftime('%H:%M:%S')}...")
print(f"  This will take approximately 3-4 hours.\n")

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n[3/5] Optimization complete!")
print(f"  Best trial: {study.best_trial.number}. Best value: {study.best_value:.6f}")
print(f"  Winning hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key:<20} {value}")

# ==============================================================================
# TRAIN FINAL MODEL
# ==============================================================================
print(f"\n[4/5] Training final model on full dataset...")

best_params = study.best_params.copy()
n_estimators_final = best_params.pop('n_estimators')  # Remove n_estimators from params
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42
})

dtrain_full = xgb.DMatrix(X, label=y)
final_model = xgb.train(
    best_params,
    dtrain_full,
    num_boost_round=n_estimators_final
)

# Evaluate on training set
y_pred_train = final_model.predict(dtrain_full)
train_loss = log_loss(y, y_pred_train)
train_auc = roc_auc_score(y, y_pred_train)
train_acc = np.mean((y_pred_train > 0.5) == y)
train_brier = np.mean((y_pred_train - y) ** 2)

print(f"\n  Training Set Performance:")
print(f"    Log Loss:    {train_loss:.6f}")
print(f"    AUC:         {train_auc:.4f}")
print(f"    Brier Score: {train_brier:.6f}")
print(f"    Accuracy:    {train_acc:.4f}")

# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================
print(f"\n[5/5] Feature importance analysis...")

importance_gain = final_model.get_score(importance_type='gain')
importance_weight = final_model.get_score(importance_type='weight')

# Map feature indices to names
feature_importance = []
for i, feat_name in enumerate(FEATURES):
    gain = importance_gain.get(f'f{i}', 0)
    weight = importance_weight.get(f'f{i}', 0)
    feature_importance.append({
        'feature': feat_name,
        'gain': gain,
        'weight': weight
    })

# Sort by gain
feature_importance.sort(key=lambda x: x['gain'], reverse=True)

print(f"\n  Top Features by Gain:")
print(f"  {'Rank':<6} {'Feature':<40} {'Gain':<12} {'Weight':<12}")
print(f"  {'-'*70}")
for rank, fi in enumerate(feature_importance[:15], 1):
    print(f"  {rank:<6} {fi['feature']:<40} {fi['gain']:<12.1f} {fi['weight']:<12.0f}")

# Check home_composite_elo rank (should be much higher now!)
home_elo_rank = next((i+1 for i, x in enumerate(feature_importance) if x['feature'] == 'home_composite_elo'), None)
injury_rank = next((i+1 for i, x in enumerate(feature_importance) if x['feature'] == 'injury_matchup_advantage'), None)

print(f"\n  Key Feature Ranks:")
print(f"    home_composite_elo: Rank #{home_elo_rank}/22 (was #24/25 with broken ELO)")
print(f"    injury_matchup_advantage: Rank #{injury_rank}/22")

# ==============================================================================
# SAVE MODEL
# ==============================================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = OUTPUT_MODEL_PATH.format(timestamp=timestamp)
final_model.save_model(output_path)

print(f"\n{'='*90}")
print(f"✅ SUCCESS!")
print(f"{'='*90}")
print(f"  Model saved: {output_path}")
print(f"  Validation log loss: {study.best_value:.6f}")
print(f"  Training log loss: {train_loss:.6f}")
print(f"  Features: {len(FEATURES)} (removed 3 redundant injury components)")
print(f"  Dataset: FIXED (home_composite_elo repaired)")
print(f"\n  Next steps:")
print(f"    1. Compare to previous best: 0.6584 (25 features, broken home ELO)")
print(f"    2. home_composite_elo should now contribute meaningfully")
print(f"    3. Deploy if validation loss < 0.655")
print(f"{'='*90}\n")
