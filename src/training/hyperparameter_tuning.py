"""
Hyperparameter tuning for XGBoost model using historical data
BETTING OPTIMIZED: Log loss scoring for calibrated probabilities
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent  # Go up to project root
sys.path.insert(0, str(root_dir))

import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBClassifier
from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

print("="*80)
print("HYPERPARAMETER TUNING - XGBOOST")
print("="*80)

# Initialize feature calculator
calc = FeatureCalculatorV5()
print("\nFeature calculator initialized")

# Load game results
conn = sqlite3.connect('data/live/nba_betting_data.db')

query = """
SELECT 
    game_id,
    game_date,
    home_team,
    away_team,
    home_score,
    away_score,
    home_won
FROM game_results
WHERE game_date >= '2021-01-01' AND game_date < '2024-11-01'
ORDER BY game_date
"""

games_df = pd.read_sql(query, conn)
conn.close()

print(f"Loaded {len(games_df)} games from 2021-2024")

# Generate features
print("\n" + "="*80)
print("GENERATING FEATURES")
print("="*80)

features_list = []
labels = []

for idx, row in games_df.iterrows():
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{len(games_df)} ({100*idx/len(games_df):.1f}%)", end='\r')
    
    try:
        features = calc.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            season='2023-24' if row['game_date'] < '2024-07-01' else '2024-25',
            game_date=str(row['game_date'])[:10]
        )
        
        if all(f in features for f in FEATURE_WHITELIST):
            features_list.append([features[f] for f in FEATURE_WHITELIST])
            labels.append(row['home_won'])
    except:
        pass

print(f"\nGenerated features for {len(features_list)} games")

# Create dataset
X = pd.DataFrame(features_list, columns=FEATURE_WHITELIST).astype(float)
y = np.array(labels)

print(f"\nDataset shape: {X.shape}")
print(f"Positive samples: {y.sum()} ({100*y.sum()/len(y):.1f}%)")

# Define hyperparameter search space - DEEP EXPLORATION
# OPTIMIZED FOR BETTING: Conservative ranges to prevent overfitting
# Focus: Calibrated probabilities for Kelly criterion, not raw accuracy
param_distributions = {
    'max_depth': [2, 3, 4, 5, 6],  # Shallow to moderate depth
    'learning_rate': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07],  # Fine-grained learning rates
    'n_estimators': [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000],  # Wide ensemble range
    'min_child_weight': [3, 5, 7, 10, 12, 15, 18, 20, 25, 30],  # Fine-grained weight control
    'subsample': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],  # Extensive row sampling
    'colsample_bytree': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],  # Extensive feature sampling
    'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Per-level feature sampling
    'colsample_bynode': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Per-node feature sampling
    'gamma': [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],  # Loss reduction thresholds
    'reg_alpha': [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],  # L1 regularization
    'reg_lambda': [0.5, 1, 1.5, 2, 3, 5, 7, 10, 15],  # L2 regularization
    'scale_pos_weight': [0.8, 0.9, 1.0, 1.1, 1.2],  # Class imbalance handling
    'max_delta_step': [0, 0.5, 1, 2, 5]  # Conservative updates for imbalanced classes
}

print("\n" + "="*80)
print("HYPERPARAMETER SEARCH")
print("="*80)
print(f"\nDEEP HYPERPARAMETER SEARCH")
print(f"Search space: {sum(len(v) for v in param_distributions.values())} total parameter values")
print(f"Possible combinations: {np.prod([len(v) for v in param_distributions.values()]):,.0f}")
print(f"Random search iterations: 300 (deep exploration)")
print(f"Cross-validation: 5-fold time series split")
print(f"OPTIMIZATION METRIC: LOG LOSS (probability calibration for betting)")
print(f"  -> Log loss punishes confident wrong predictions")
print(f"  -> Essential for calibrated probabilities in Kelly criterion")
print(f"  -> Accuracy metric inappropriate for betting models")

# Base model
base_model = XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    tree_method='hist'  # Faster training
)

# Time series cross-validation (respects temporal order)
tscv = TimeSeriesSplit(n_splits=5)

# Randomized search with LOG LOSS scoring (CRITICAL for betting)
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=300,  # DEEP SEARCH: 300 random combinations
    scoring='neg_log_loss',  # CHANGED: neg_log_loss instead of accuracy
    cv=tscv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nStarting DEEP hyperparameter search (this may take 2-3 hours)...")
print("Expected runtime: ~120-180 minutes with 300 iterations")
print("Progress will be shown every 50 fits...\n")
random_search.fit(X, y)

print("\n" + "="*80)
print("SEARCH RESULTS")
print("="*80)

print(f"\nBest log loss: {-random_search.best_score_:.4f}")  # Negate to show positive
print(f"\nBest parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

# Train final model with best parameters
print("\n" + "="*80)
print("TRAINING FINAL MODEL")
print("="*80)

best_model = XGBClassifier(
    **random_search.best_params_,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

best_model.fit(X, y)

# Evaluate on full dataset
y_pred = best_model.predict(X)
y_pred_proba = best_model.predict_proba(X)[:, 1]
train_accuracy = (y_pred == y).mean()

print(f"\nTraining accuracy: {train_accuracy:.4f}")

# === CALIBRATION CURVE (CRITICAL FOR BETTING) ===
print("\n" + "="*80)
print("CALIBRATION ANALYSIS")
print("="*80)

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Generate calibration curve
prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10, strategy='uniform')

# Plot calibration curve
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.plot(prob_pred, prob_true, 'o-', label='Model Calibration', linewidth=2, markersize=8)
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Actual Win Rate', fontsize=12)
plt.title('Calibration Curve - Tuned Model (LOG LOSS Optimized)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/calibration_curve_tuned.png', dpi=150)
print("Calibration curve saved to output/calibration_curve_tuned.png")
plt.close()

# Calculate calibration metrics
from sklearn.metrics import brier_score_loss, log_loss

brier_score = brier_score_loss(y, y_pred_proba)
logloss_score = log_loss(y, y_pred_proba)

print(f"\nCalibration Metrics:")
print(f"  Brier Score: {brier_score:.4f} (lower is better, perfect = 0)")
print(f"  Log Loss: {logloss_score:.4f} (lower is better)")

# Show calibration deciles
print(f"\nCalibration Deciles (Predicted vs Actual):")
for i in range(len(prob_pred)):
    print(f"  Bin {i+1}: Predicted {prob_pred[i]:.3f} | Actual {prob_true[i]:.3f} | Gap {abs(prob_pred[i] - prob_true[i]):.3f}")

max_calibration_gap = max(abs(prob_pred[i] - prob_true[i]) for i in range(len(prob_pred)))
print(f"\nMax Calibration Gap: {max_calibration_gap:.4f} (should be < 0.05 for betting)")

# Save model
model_path = 'models/xgboost_tuned.pkl'
joblib.dump(best_model, model_path)
print(f"\nModel saved to {model_path}")

# Save tuning results - TOP 20 for deep analysis
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')

# Save full results
results_df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']].head(20).to_csv(
    'output/hyperparameter_tuning_results_deep.csv', index=False
)
print("Top 20 tuning results saved to output/hyperparameter_tuning_results_deep.csv")

# Also save all results for later analysis
results_df.to_csv('output/hyperparameter_tuning_all_results.csv', index=False)
print("All 300 results saved to output/hyperparameter_tuning_all_results.csv")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': FEATURE_WHITELIST,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*80)
print("TOP 10 FEATURES (TUNED MODEL)")
print("="*80)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:30s} {row['importance']:.4f}")

feature_importance.to_csv('output/feature_importance_tuned.csv', index=False)

print("\n" + "="*80)
print("HYPERPARAMETER TUNING COMPLETE")
print("="*80)
print(f"\nBest CV score: {random_search.best_score_:.4f}")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Improvement over baseline: {(train_accuracy - 0.6791)*100:+.2f}%")
