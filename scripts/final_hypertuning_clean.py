"""
Final Hyperparameter Tuning on CLEAN 36-Feature Dataset
Uses Syndicate constrained search (1500 trials) optimized for clean data.

Why this approach:
1. Data bugs fixed (rest days, ELO inflation, outliers)
2. Clean data trains 42% faster
3. Current hyperparameters optimized for exploiting bugs
4. 36 features proven baseline (59.44% accuracy)
5. Matchup features didn't help (-0.46%)

Target: Break 0.56 AUC with proper tuning on clean data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import optuna
from datetime import datetime
import json

print("="*70)
print("FINAL HYPERPARAMETER TUNING - CLEAN 36 FEATURES")
print("="*70)

# Load clean dataset
print("\n1. Loading clean 36-feature dataset...")
df = pd.read_csv("data/training_data_with_features_cleaned.csv")
print(f"   Games: {len(df):,}")

# Prepare features
exclude_cols = ['date','game_id','home_team','away_team','season','target_spread',
                'target_spread_cover','target_moneyline_win','target_game_total',
                'target_over_under','target_home_cover','target_over']

X = df[[c for c in df.columns if c not in exclude_cols]]
y = df['target_spread_cover']

print(f"   Features: {X.shape[1]}")
print(f"   Target: {y.value_counts().to_dict()}")

# Define objective
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000, step=100),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'gamma': trial.suggest_float('gamma', 0, 3.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 7),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        scores.append(auc)
    
    return np.mean(scores)

# Run optimization
print("\n2. Running Optuna optimization (1500 trials)...")
print("   Constrained search:")
print("     - depth: 6-12 (avoid shallow overfitting)")
print("     - gamma: 0-3.0 (allow splits)")
print("     - colsample: 0.5-0.8 (feature diversity)")
print("     - learning_rate: 0.01-0.1 (faster convergence)")
print("     - n_estimators: 1000-3000 (match learning rate)")
print("")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1500, show_progress_bar=True)

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Best trial: #{study.best_trial.number}")
print(f"Best AUC: {study.best_value:.4f}")
print("\nBest parameters:")
for key, value in study.best_params.items():
    print(f"  {key:20s}: {value}")

# Save best params
output_path = "models/final_tuned_params.json"
with open(output_path, 'w') as f:
    json.dump({
        'best_trial': study.best_trial.number,
        'best_auc': study.best_value,
        'best_params': study.best_params,
        'tuning_date': datetime.now().isoformat(),
        'dataset': 'training_data_with_features_cleaned.csv',
        'n_trials': 1500,
        'strategy': 'Syndicate constrained search on clean data'
    }, f, indent=2)

print(f"\n✓ Saved best parameters: {output_path}")

# Train final model
print("\n3. Training final model with best parameters...")
model_final = xgb.XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss', verbosity=0)
model_final.fit(X, y)

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 features:")
for idx, row in importances.head(10).iterrows():
    print(f"     {row['feature']:30s}: {row['importance']*100:5.2f}%")

# Save model
model_path = "models/final_tuned_model.json"
model_final.save_model(model_path)
print(f"\n✓ Saved model: {model_path}")

print("\n" + "="*70)
print("TUNING COMPLETE")
print("="*70)
print(f"AUC: {study.best_value:.4f}")
print(f"Improvement target: Break 0.56 (need +{(0.56 - study.best_value)*100:.2f}%)")
print("\nNext steps:")
print("  1. Walk-forward backtest on validation set")
print("  2. Calibrate probabilities (isotonic/Platt)")
print("  3. Test Kelly sizing on historical bets")
print("="*70)
