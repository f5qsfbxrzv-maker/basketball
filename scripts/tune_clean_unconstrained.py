"""
Unconstrained Hyperparameter Tuning on CLEAN 36-Feature Dataset
No monotonic constraints - let the model learn relationships from clean data.

Strategy: Trust the data cleaning. The bugs are fixed, so let XGBoost find the patterns.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import json
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

print("="*70)
print("UNCONSTRAINED TUNING - CLEAN 36 FEATURES (NO CONSTRAINTS)")
print("="*70)

# Load clean data
print("\n1. Loading clean dataset...")
df = pd.read_csv('data/training_data_with_features_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"   Games: {len(df):,}")

# Prepare features
target = 'target_spread_cover'
drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df[target]

print(f"   Features: {len(features)}")

# Define objective
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        
        # Hyperparameters - wider ranges for clean data
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 3000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'random_state': 42,
        'verbosity': 0
    }
    
    # Time Series CV
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

# Run optimization
print("\n2. Running Optuna optimization (300 trials, unconstrained)...")
print("   Strategy: Trust clean data, no artificial constraints")
print("")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300, show_progress_bar=True)

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Best trial: #{study.best_trial.number}")
print(f"Best AUC: {study.best_value:.4f}")

print("\nBest parameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key:20s}: {value}")

# Compare to constrained
print("\nComparison:")
print(f"  Unconstrained: {study.best_value:.4f}")
print(f"  Constrained:   0.5507")
print(f"  Difference:    {(study.best_value - 0.5507):.4f} ({((study.best_value - 0.5507)/0.5507)*100:+.2f}%)")

# Save
output_path = "models/clean_unconstrained_best_params.json"
with open(output_path, 'w') as f:
    json.dump({
        'best_trial': study.best_trial.number,
        'best_auc': study.best_value,
        'best_params': study.best_trial.params,
        'tuning_date': datetime.now().isoformat(),
        'dataset': 'training_data_with_features_cleaned.csv',
        'n_trials': 300,
        'strategy': 'Unconstrained - trust clean data'
    }, f, indent=2)

print(f"\n✓ Saved: {output_path}")

# Train final model
print("\n3. Training final model...")
model_final = xgb.XGBClassifier(**study.best_trial.params, objective='binary:logistic', 
                                 tree_method='hist', eval_metric='auc', random_state=42, verbosity=0)
model_final.fit(X, y)

preds_full = model_final.predict_proba(X)[:, 1]
auc_full = roc_auc_score(y, preds_full)
acc_full = accuracy_score(y, (preds_full > 0.5).astype(int))

print(f"   Full dataset AUC: {auc_full:.4f}")
print(f"   Full dataset Accuracy: {acc_full:.4f}")

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 15 features:")
for idx, row in importances.head(15).iterrows():
    print(f"     {row['feature']:30s}: {row['importance']*100:5.2f}%")

# Save
model_final.save_model("models/clean_unconstrained_model.json")
importances.to_csv("output/clean_unconstrained_feature_importance.csv", index=False)

print("\n" + "="*70)
print("VERDICT")
print("="*70)
if study.best_value >= 0.56:
    print(f"✓ SUCCESS: {study.best_value:.4f} AUC (target: 0.56)")
elif study.best_value > 0.5507:
    print(f"✓ IMPROVEMENT: {study.best_value:.4f} vs 0.5507 constrained (+{((study.best_value-0.5507)/0.5507)*100:.2f}%)")
else:
    print(f"✗ NO IMPROVEMENT: {study.best_value:.4f} (constraints weren't the issue)")

print("\nConclusion:")
if study.best_value > 0.5507:
    print("  → Constraints were hurting performance. Use unconstrained model.")
else:
    print("  → Performance ceiling reached. Need new features or different approach.")
print("="*70)
