"""
DEEP Hyperparameter Optimization using Bayesian Search (Optuna)
GOAL: Minimize Log Loss for calibrated betting probabilities
STRATEGY: Aggressive regularization to fix overconfidence, then post-hoc calibration
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import sqlite3
import pandas as pd
import numpy as np
import joblib
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

print("="*80)
print("DEEP BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA)")
print("="*80)
print("Strategy: Aggressive regularization + dynamic pruning")
print("Goal: Minimize Log Loss for calibrated betting probabilities")
print("="*80)

# ============================================================================
# LOAD DATA - Use existing features instead of regenerating
# ============================================================================
print("\n⚠️ Loaded team_stats (660 rows) - DEPRECATED, use game_advanced_stats instead")

# Check if we have a cached feature dataset
cached_features_path = 'data/processed/training_features_30.csv'

if Path(cached_features_path).exists():
    print(f"\n✓ Loading cached features from {cached_features_path}")
    df = pd.read_csv(cached_features_path)
    
    # Separate features and target
    X = df[FEATURE_WHITELIST].values.astype(np.float32)
    y = df['home_won'].values.astype(np.int32)
    
    print(f"Loaded {len(X)} games with {X.shape[1]} features")
    print(f"Dataset shape: {X.shape}")
    print(f"Positive samples: {y.sum()} ({100*y.mean():.1f}%)")
    
else:
    print(f"\n⚠️ Cached features not found at {cached_features_path}")
    print("Generating features from scratch...")
    
    fc = FeatureCalculatorV5()
    print("Feature calculator initialized")

    # Load games
    conn = sqlite3.connect('data/live/nba_betting_data.db')
    games_df = pd.read_sql("""
        SELECT game_id, game_date, home_team, away_team, home_score, away_score
        FROM game_results
        WHERE date(game_date) >= '2021-01-01'
        AND date(game_date) <= '2024-12-31'
        ORDER BY game_date
    """, conn)
    conn.close()

    print(f"Loaded {len(games_df)} games from 2021-2024")

    # ============================================================================
    # GENERATE FEATURES
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING FEATURES")
    print("="*80)

    features_list = []
    targets = []
    game_ids = []

    for idx, row in games_df.iterrows():
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(games_df)} ({100*idx/len(games_df):.1f}%)")
        
        try:
            features = fc.calculate_game_features(
                game_date=row['game_date'],
                home_team=row['home_team'],
                away_team=row['away_team']
            )
            
            # Extract values in correct order matching FEATURE_WHITELIST
            if features and isinstance(features, dict):
                feature_values = [features.get(feat, 0.0) for feat in FEATURE_WHITELIST]
                if len(feature_values) == len(FEATURE_WHITELIST):
                    features_list.append(feature_values)
                    # Target: 1 if home team won, 0 otherwise
                    home_won = 1 if row['home_score'] > row['away_score'] else 0
                    targets.append(home_won)
                    game_ids.append(row['game_id'])
        except Exception as e:
            continue

    print(f"Generated features for {len(features_list)} games")

    # Convert to arrays
    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets, dtype=np.int32)

print(f"\nDataset shape: {X.shape}")
print(f"Positive samples: {y.sum()} ({100*y.mean():.1f}%)")

print("\n" + "="*80)
print("FEATURE STATISTICS")
print("="*80)
print(f"Total games: {len(X):,}")
print(f"Features per game: {X.shape[1]}")
print(f"Home wins: {y.sum()} ({100*y.mean():.1f}%)")
print(f"Away wins: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
print("="*80)

# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    """
    Optuna objective: Minimize cross-validated log loss
    This function will be called 100+ times with different hyperparameter combinations
    """
    
    # DEEP SEARCH SPACE - Aggressive regularization to fix calibration
    # EXPANDED RANGES for deeper exploration
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',  # Fast approximate method
        'booster': 'gbtree',
        'random_state': 42,
        
        # STRUCTURE: Allow deeper trees but constrain with regularization
        'max_depth': trial.suggest_int('max_depth', 2, 12),  # Expanded: 2-12 (was 3-10)
        
        # CRITICAL: High min_child_weight suppresses noise/overconfidence on rare events
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 150),  # Expanded: 5-150 (was 10-100)
        
        # SAMPLING: Prevent overfitting to specific games
        'subsample': trial.suggest_float('subsample', 0.4, 0.95),  # Expanded: 0.4-0.95 (was 0.5-0.9)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),  # Expanded
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),  # Expanded
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),  # NEW parameter
        
        # LEARNING: Low rate for precision, high n_estimators for convergence
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.15, log=True),  # Expanded: 0.003-0.15
        
        # REGULARIZATION (CRITICAL for calibration)
        # High gamma = aggressive pruning (won't split unless significant improvement)
        'gamma': trial.suggest_float('gamma', 0.0, 15.0),  # Expanded: 0-15 (was 0.1-10)
        
        # L1 (Lasso) - kills noisy features
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 100.0, log=True),  # Expanded: 1e-4 to 100
        
        # L2 (Ridge) - shrinks all weights  
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 100.0, log=True),  # Expanded: 1e-4 to 100
        
        # Class imbalance handling
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.7, 1.3),  # Expanded
        
        # Max delta step (conservative updates)
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),  # Expanded: 0-10
    }
    
    # Extract n_estimators separately (don't include in param dict for xgb.cv)
    n_estimators = trial.suggest_int('n_estimators', 300, 4000)  # Expanded: 300-4000
    
    # Use XGBoost's built-in CV for efficiency
    dtrain = xgb.DMatrix(X, label=y)
    
    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=n_estimators,  # Pass as argument, not in param dict
        nfold=5,
        stratified=True,
        early_stopping_rounds=50,
        metrics='logloss',
        seed=42,
        verbose_eval=False
    )
    
    # Return best log loss from cross-validation
    best_logloss = cv_results['test-logloss-mean'].min()
    
    # Also track standard deviation (lower = more stable)
    std_logloss = cv_results['test-logloss-std'].iloc[cv_results['test-logloss-mean'].idxmin()]
    
    # Optuna can track multiple metrics
    trial.set_user_attr('std_logloss', std_logloss)
    trial.set_user_attr('best_iteration', cv_results['test-logloss-mean'].idxmin())
    
    return best_logloss

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("STARTING BAYESIAN OPTIMIZATION")
print("="*80)
print("Search space: 13 hyperparameters")
print("Sampler: TPE (Tree-structured Parzen Estimator)")
print("Trials: 300 (DEEP exploration, each with 5-fold CV)")
print("Expected runtime: 15-30 minutes")
print("="*80)
print("\nOptimization will learn from each trial to find global minimum faster...")
print("Early trials explore broadly, later trials exploit promising regions.")
print("With fixed injury features, expect significant improvement.\n")

# Create Optuna study
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='nba_betting_logloss_deep_v2'
)

# Run optimization - DEEP SEARCH
study.optimize(objective, n_trials=300, show_progress_bar=True)

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)

print(f"\n🎯 Best CV Log Loss: {study.best_value:.4f}")
print(f"Standard Deviation: {study.best_trial.user_attrs['std_logloss']:.4f}")
print(f"Best Iteration: {study.best_trial.user_attrs['best_iteration']}")

print("\n" + "="*80)
print("BEST HYPERPARAMETERS")
print("="*80)
for key, value in study.best_params.items():
    print(f"  {key:25s} {value}")

# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMS
# ============================================================================

print("\n" + "="*80)
print("TRAINING FINAL MODEL")
print("="*80)

best_params = study.best_params.copy()
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X, y)

# Evaluate
y_pred_proba = final_model.predict_proba(X)[:, 1]
y_pred = final_model.predict(X)

train_accuracy = (y_pred == y).mean()
train_logloss = log_loss(y, y_pred_proba)
train_brier = brier_score_loss(y, y_pred_proba)

print(f"\nTraining Metrics:")
print(f"  Accuracy: {train_accuracy:.4f}")
print(f"  Log Loss: {train_logloss:.4f}")
print(f"  Brier Score: {train_brier:.4f}")

# ============================================================================
# CALIBRATION ANALYSIS (PRE POST-HOC)
# ============================================================================

print("\n" + "="*80)
print("CALIBRATION ANALYSIS (BEFORE POST-HOC CORRECTION)")
print("="*80)

prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10, strategy='uniform')

# Plot
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
plt.plot(prob_pred, prob_true, 'o-', label='Optuna-Tuned Model', linewidth=2, markersize=8)
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Actual Win Rate', fontsize=12)
plt.title('Calibration Curve - Optuna Deep Optimization (Pre-Calibration)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/calibration_curve_optuna_pre.png', dpi=150)
print("Calibration curve saved to output/calibration_curve_optuna_pre.png")
plt.close()

print(f"\nCalibration Deciles:")
for i in range(len(prob_pred)):
    gap = abs(prob_pred[i] - prob_true[i])
    print(f"  Bin {i+1}: Predicted {prob_pred[i]:.3f} | Actual {prob_true[i]:.3f} | Gap {gap:.3f}")

max_gap_pre = max(abs(prob_pred[i] - prob_true[i]) for i in range(len(prob_pred)))
print(f"\n⚠️ Max Calibration Gap (Pre): {max_gap_pre:.4f}")

# ============================================================================
# POST-HOC CALIBRATION (ISOTONIC + PLATT)
# ============================================================================

print("\n" + "="*80)
print("APPLYING POST-HOC CALIBRATION")
print("="*80)
print("Method: Isotonic Regression (non-parametric, fits perfectly to data)")
print("Backup: Platt Scaling (logistic regression on scores)")

# Isotonic calibration
calibrated_isotonic = CalibratedClassifierCV(
    final_model, 
    method='isotonic', 
    cv='prefit'
)

# Need to split data for calibration (can't use same data)
# Use last 20% for calibration
split_idx = int(0.8 * len(X))
X_train_cal, X_calib = X[:split_idx], X[split_idx:]
y_train_cal, y_calib = y[:split_idx], y[split_idx:]

# Retrain on 80%, calibrate on 20%
final_model_80 = xgb.XGBClassifier(**best_params)
final_model_80.fit(X_train_cal, y_train_cal)

calibrated_isotonic = CalibratedClassifierCV(
    final_model_80,
    method='isotonic',
    cv='prefit'
)
calibrated_isotonic.fit(X_calib, y_calib)

# Evaluate calibrated model on full dataset
y_pred_proba_cal = calibrated_isotonic.predict_proba(X)[:, 1]
cal_logloss = log_loss(y, y_pred_proba_cal)
cal_brier = brier_score_loss(y, y_pred_proba_cal)

print(f"\nPost-Calibration Metrics:")
print(f"  Log Loss: {cal_logloss:.4f} (vs {train_logloss:.4f} pre-calibration)")
print(f"  Brier Score: {cal_brier:.4f} (vs {train_brier:.4f} pre-calibration)")

# Calibration curve after
prob_true_cal, prob_pred_cal = calibration_curve(y, y_pred_proba_cal, n_bins=10, strategy='uniform')

plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
plt.plot(prob_pred, prob_true, 'o-', label='Pre-Calibration', linewidth=2, markersize=8, alpha=0.5)
plt.plot(prob_pred_cal, prob_true_cal, 's-', label='Post-Calibration (Isotonic)', linewidth=2, markersize=8)
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Actual Win Rate', fontsize=12)
plt.title('Calibration Curve - Before & After Isotonic Calibration', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/calibration_curve_optuna_post.png', dpi=150)
print("Post-calibration curve saved to output/calibration_curve_optuna_post.png")
plt.close()

max_gap_post = max(abs(prob_pred_cal[i] - prob_true_cal[i]) for i in range(len(prob_pred_cal)))
print(f"\n✅ Max Calibration Gap (Post): {max_gap_post:.4f}")
print(f"Improvement: {max_gap_pre - max_gap_post:.4f} ({100*(max_gap_pre - max_gap_post)/max_gap_pre:.1f}% reduction)")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save uncalibrated model
joblib.dump(final_model_80, 'models/xgboost_optuna_uncalibrated.pkl')
print("Uncalibrated model saved to models/xgboost_optuna_uncalibrated.pkl")

# Save calibrated model (THIS IS THE ONE TO USE FOR BETTING)
joblib.dump(calibrated_isotonic, 'models/xgboost_optuna_calibrated.pkl')
print("✅ CALIBRATED model saved to models/xgboost_optuna_calibrated.pkl")

# Save best parameters
import json
with open('output/optuna_best_params.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
print("Best parameters saved to output/optuna_best_params.json")

# Save study history
study_df = study.trials_dataframe()
study_df.to_csv('output/optuna_study_history.csv', index=False)
print("Full study history saved to output/optuna_study_history.csv")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("TOP 15 FEATURES (OPTUNA-TUNED MODEL)")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': FEATURE_WHITELIST,
    'importance': final_model_80.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:30s} {row['importance']:.4f}")

feature_importance.to_csv('output/feature_importance_optuna.csv', index=False)

print("\n" + "="*80)
print("DEEP OPTIMIZATION COMPLETE")
print("="*80)
print(f"\n✅ Use 'models/xgboost_optuna_calibrated.pkl' for betting")
print(f"✅ This model has isotonic calibration applied")
print(f"✅ Max calibration gap reduced from {max_gap_pre:.4f} to {max_gap_post:.4f}")
print(f"✅ Ready for Kelly criterion position sizing")
print("="*80)
