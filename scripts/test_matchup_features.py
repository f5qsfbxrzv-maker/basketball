"""
Quick Training: 41 Features (36 Original + 5 Matchups) vs 36 Baseline
Tests if schematic matchup features improve AUC.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

print("="*70)
print("TESTING MATCHUP FEATURES: 41 vs 36 Features")
print("="*70)

# Load datasets
print("\n1. Loading datasets...")
df_baseline = pd.read_csv("data/training_data_with_features_cleaned.csv")
df_matchups = pd.read_csv("data/training_data_with_matchups.csv")

print(f"   Baseline: {len(df_baseline):,} games")
print(f"   Matchups: {len(df_matchups):,} games")

# Prepare features
exclude_cols = ['date','game_id','home_team','away_team','season','target_spread',
                'target_spread_cover','target_moneyline_win','target_game_total',
                'target_over_under','target_home_cover','target_over']

X_baseline = df_baseline[[c for c in df_baseline.columns if c not in exclude_cols]]
X_matchups = df_matchups[[c for c in df_matchups.columns if c not in exclude_cols]]

y = df_baseline['target_spread_cover']

print(f"   Baseline features: {X_baseline.shape[1]}")
print(f"   Matchups features: {X_matchups.shape[1]}")

# Time series split
print("\n2. Training with TimeSeriesSplit (5 folds)...")
tscv = TimeSeriesSplit(n_splits=5)

# Train baseline (36 features)
print("\n   BASELINE (36 features):")
baseline_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_baseline), 1):
    X_train, X_val = X_baseline.iloc[train_idx], X_baseline.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    
    baseline_scores.append(auc)
    print(f"     Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}")

baseline_mean = np.mean(baseline_scores)
baseline_std = np.std(baseline_scores)
print(f"   → Mean AUC: {baseline_mean:.4f} ± {baseline_std:.4f}")

# Train with matchups (41 features)
print("\n   WITH MATCHUPS (41 features):")
matchup_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_matchups), 1):
    X_train, X_val = X_matchups.iloc[train_idx], X_matchups.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    
    matchup_scores.append(auc)
    print(f"     Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}")

matchup_mean = np.mean(matchup_scores)
matchup_std = np.std(matchup_scores)
print(f"   → Mean AUC: {matchup_mean:.4f} ± {matchup_std:.4f}")

# Calculate improvement
improvement = matchup_mean - baseline_mean
improvement_pct = (improvement / baseline_mean) * 100

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Baseline (36 features):  {baseline_mean:.4f} ± {baseline_std:.4f}")
print(f"Matchups (41 features):  {matchup_mean:.4f} ± {matchup_std:.4f}")
print(f"Improvement:             {improvement:+.4f} ({improvement_pct:+.2f}%)")

if improvement > 0:
    print("\n✓ MATCHUP FEATURES IMPROVED PERFORMANCE")
else:
    print("\n✗ MATCHUP FEATURES DID NOT IMPROVE PERFORMANCE")

# Train final model to get feature importance
print("\n3. Feature importance (top matchup features)...")
model_final = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
model_final.fit(X_matchups, y)

# Get importance
importances = pd.DataFrame({
    'feature': X_matchups.columns,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 15 features:")
for idx, row in importances.head(15).iterrows():
    is_matchup = '***' if row['feature'] in ['off_vs_def_mismatch','pace_control_advantage','three_point_exploit','turnover_pressure','rebounding_clash'] else ''
    print(f"     {row['feature']:30s}: {row['importance']*100:5.2f}% {is_matchup}")

matchup_features = importances[importances['feature'].isin(['off_vs_def_mismatch','pace_control_advantage','three_point_exploit','turnover_pressure','rebounding_clash'])]
print(f"\n   Matchup features combined importance: {matchup_features['importance'].sum()*100:.2f}%")

print("\n" + "="*70)
print("DECISION:")
if improvement > 0.001:  # More than 0.1% improvement
    print("✓ KEEP MATCHUP FEATURES - They improve AUC")
    print("  Next: Hyperparameter tune on 41-feature dataset")
else:
    print("✗ SKIP MATCHUP FEATURES - No meaningful improvement")
    print("  Next: Focus on PIE roster share or other features")
print("="*70)
