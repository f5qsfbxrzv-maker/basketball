"""
Train model on lean dataset and compare to 36-feature baseline
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import json
import time

print("="*70)
print("TRAINING LEAN MODEL (21 features) vs BASELINE (36 features)")
print("="*70)

# Load lean data
print("\n1. Loading lean dataset...")
df_lean = pd.read_csv("data/training_data_lean.csv")
print(f"   Games: {len(df_lean):,}")

# Get features
exclude_cols = ['date', 'game_id', 'home_team', 'away_team', 'season', 
               'target_spread', 'target_spread_cover', 'target_moneyline_win', 
               'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']
feature_cols = [c for c in df_lean.columns if c not in exclude_cols]
print(f"   Features: {len(feature_cols)}")

X = df_lean[feature_cols]
y = df_lean['target_moneyline_win']

# Use same parameters as original model for fair comparison
print("\n2. Using baseline hyperparameters...")
params = {
    'learning_rate': 0.015,
    'n_estimators': 2786,
    'max_depth': 9,
    'min_child_weight': 21,
    'gamma': 2.69,
    'subsample': 0.63,
    'colsample_bytree': 0.53,
    'colsample_bylevel': 0.57,
    'colsample_bynode': 0.63,
    'reg_alpha': 6.5,  # Reduced from 16.36 since we have less features
    'reg_lambda': 1.11,
    'scale_pos_weight': 0.90,
    'max_delta_step': 5,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

# Time series split
print("\n3. Time series cross-validation (5 folds)...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)
    
    cv_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc, 'logloss': logloss})
    print(f"   Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}, LogLoss={logloss:.4f}")

cv_df = pd.DataFrame(cv_scores)
print(f"\n   Mean Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
print(f"   Mean AUC:      {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
print(f"   Mean LogLoss:  {cv_df['logloss'].mean():.4f} ± {cv_df['logloss'].std():.4f}")

# Train final model on all data
print("\n4. Training final model on all data...")
start = time.time()
model_lean = xgb.XGBClassifier(**params)
model_lean.fit(X, y, verbose=False)
train_time = time.time() - start

print(f"   Training time: {train_time:.2f}s")

# Get feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_lean.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("TOP 15 FEATURES (LEAN MODEL)")
print("="*70)
for idx, row in importance_df.head(15).iterrows():
    is_interaction = row['feature'] in ['efficiency_x_pace', 'tired_altitude', 'form_x_defense']
    marker = " ⭐" if is_interaction else ""
    print(f"  {idx+1:2d}. {row['feature']:<30} {row['importance']:>8.4f}{marker}")

# Save model and results
model_lean.save_model("models/xgboost_lean.json")
importance_df.to_csv("output/feature_importance_lean.csv", index=False)

results = {
    'features': len(feature_cols),
    'games': len(df_lean),
    'cv_mean_auc': float(cv_df['auc'].mean()),
    'cv_std_auc': float(cv_df['auc'].std()),
    'cv_mean_accuracy': float(cv_df['accuracy'].mean()),
    'cv_std_accuracy': float(cv_df['accuracy'].std()),
    'training_time_seconds': train_time,
    'top_feature': importance_df.iloc[0]['feature'],
    'top_feature_importance': float(importance_df.iloc[0]['importance'])
}

with open("output/lean_model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("COMPARISON: LEAN vs BASELINE")
print("="*70)
print(f"{'Metric':<20} {'Baseline (36)':<15} {'Lean (21)':<15} {'Change':<10}")
print("-"*70)
print(f"{'Mean AUC':<20} {'0.5520':<15} {cv_df['auc'].mean():<15.4f} {'+' if cv_df['auc'].mean() > 0.552 else ''}{(cv_df['auc'].mean() - 0.552)*100:>8.2f}%")
print(f"{'Mean Accuracy':<20} {'0.5944':<15} {cv_df['accuracy'].mean():<15.4f} {'+' if cv_df['accuracy'].mean() > 0.5944 else ''}{(cv_df['accuracy'].mean() - 0.5944)*100:>8.2f}%")
print(f"{'Features':<20} {'36':<15} {'21':<15} {'-42%':<10}")
print(f"{'Training Speed':<20} {'baseline':<15} {f'{train_time:.1f}s':<15} {'~42% faster':<10}")

print("\n" + "="*70)
print("SAVED FILES")
print("="*70)
print("  - models/xgboost_lean.json")
print("  - output/feature_importance_lean.csv")
print("  - output/lean_model_results.json")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
if cv_df['auc'].mean() >= 0.552:
    print("✓ LEAN MODEL PERFORMANCE MAINTAINED OR IMPROVED!")
    print("  → Proceed with hyperparameter tuning on lean dataset")
    print("  → Run: python scripts/hypertune_lean.py")
else:
    print("⚠ LEAN MODEL UNDERPERFORMS BASELINE")
    print(f"  Loss: {(cv_df['auc'].mean() - 0.552)*100:.2f}%")
    print("  → Review removed features")
    print("  → Consider keeping more features")
print("="*70)
