"""
UNCONSTRAINED SANITY TEST - Clean Data Performance Check
Removes all constraints and uses aggressive parameters to verify clean data quality.

If AUC jumps to ~0.560: Data is good, constraints were choking
If AUC stays ~0.550: Clean data removed some "accidentally helpful" dirty signal
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

print("="*70)
print("UNCONSTRAINED SANITY TEST - CLEAN DATA")
print("="*70)

# 1. Load Clean Data
print("\n1. Loading clean dataset...")
df = pd.read_csv('data/training_data_with_features_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"   Games: {len(df):,}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# 2. Setup (NO CONSTRAINTS)
target = 'target_spread_cover'
drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df[target]

print(f"   Features: {len(features)}")
print(f"   Target: {y.value_counts().to_dict()}")

# 3. Aggressive Params (Force deeper learning)
print("\n2. Testing UNCONSTRAINED model with aggressive parameters...")
print("   Strategy: NO constraints, forced depth 6, learning rate 0.02")

params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'max_depth': 6,           # Forced deeper than constrained model's 3
    'learning_rate': 0.02,    # Forced faster than constrained model's 0.005
    'n_estimators': 1500,     # Forced longer than constrained model's 500
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'min_child_weight': 10,
    'gamma': 1.0,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'verbosity': 0
}

# 4. Rigorous Validation (5-Fold Time Series Split)
print("\n3. Running 5-Fold Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=5)
scores = []
accuracies = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    
    print(f"   Fold {fold} AUC: {auc:.5f}  Accuracy: {acc:.4f}")
    scores.append(auc)
    accuracies.append(acc)

mean_auc = np.mean(scores)
std_auc = np.std(scores)
mean_acc = np.mean(accuracies)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Average AUC:      {mean_auc:.5f} ± {std_auc:.5f}")
print(f"Average Accuracy: {mean_acc:.4f}")
print(f"Target (0.560):   {'✓ ACHIEVED' if mean_auc >= 0.560 else f'Miss by {(0.560 - mean_auc)*100:.2f}%'}")

# 5. Feature Importance (Unconstrained Model)
print("\n4. Feature importance (unconstrained model)...")
model_full = xgb.XGBClassifier(**params)
model_full.fit(X, y)

imp = pd.Series(model_full.feature_importances_, index=features).sort_values(ascending=False)

print("\n   Top 10 Features (Unshackled):")
for feat, importance in imp.head(10).items():
    print(f"     {feat:30s}: {importance*100:5.2f}%")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if mean_auc >= 0.560:
    print("✓ CLEAN DATA IS GOOD - Constraints were choking the model")
    print("  → Proceed with UNCONSTRAINED hyperparameter tuning")
    print("  → Use wider search space (depth 4-10, lr 0.005-0.1)")
elif mean_auc >= 0.550:
    print("⚠ CLEAN DATA COMPARABLE - Marginal improvement")
    print(f"  → Clean: {mean_auc:.4f} vs Dirty baseline: ~0.5508")
    print("  → May need better features, not just tuning")
else:
    print("✗ CLEAN DATA WORSE - Something went wrong")
    print("  → Check data cleaning script for errors")
    print("  → Consider reverting some fixes (e.g., ELO normalization)")

print("\nNext steps:")
if mean_auc >= 0.555:
    print("  1. Run unconstrained Optuna (200+ trials)")
    print("  2. Search space: depth 4-10, lr 0.005-0.1, lower reg")
    print("  3. Target: 0.565+ AUC")
else:
    print("  1. Investigate feature engineering")
    print("  2. Try lean 21-feature dataset")
    print("  3. Consider ensemble models")
print("="*70)
