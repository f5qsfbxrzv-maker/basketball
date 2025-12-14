"""
5-Fold Constrained vs Unconstrained Comparison
Fair comparison with same parameters, same CV splits.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

print("="*70)
print("5-FOLD COMPARISON: CONSTRAINED VS UNCONSTRAINED")
print("="*70)

# Load data
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df['target_spread_cover']

print(f"\n1. Dataset: {len(df):,} games, {X.shape[1]} features")

# Define constraints
constraint_map = {
    'off_elo_diff': 1, 'def_elo_diff': 1, 'home_composite_elo': 1,
    'ewma_efg_diff': 1, 'home_ewma_3p_pct': 1,
    'away_back_to_back': -1, 'home_back_to_back': -1, 'away_3in4': -1,
    'rest_advantage': 1,
    'injury_shock_diff': 1, 'injury_shock_home': -1,
    'injury_impact_abs': -1, 'injury_impact_diff': 1,
    'away_star_missing': 1, 'home_star_missing': -1, 'star_mismatch': 1,
    'ewma_tov_diff': -1, 'ewma_orb_diff': 1,
    'home_orb': 1, 'home_drb': 1,
    'home_ewma_fta_rate': 1, 'ewma_foul_synergy_home': 1,
    'fatigue_mismatch': 1,
}

constraints = tuple(constraint_map.get(col, 0) for col in features)

print(f"\n2. Testing with same parameters on both models...")

# Same params for fair comparison
params_base = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.02,
    'n_estimators': 1500,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'min_child_weight': 10,
    'gamma': 1.0,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'verbosity': 0
}

tscv = TimeSeriesSplit(n_splits=5)

# Test UNCONSTRAINED
print("\n3. UNCONSTRAINED (baseline):")
unconstrained_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**params_base)
    model.fit(X_train, y_train, verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    
    print(f"   Fold {fold}: AUC={auc:.5f}, Acc={acc:.4f}")
    unconstrained_scores.append(auc)

unconstrained_mean = np.mean(unconstrained_scores)
unconstrained_std = np.std(unconstrained_scores)

# Test CONSTRAINED
print("\n4. CONSTRAINED (with 21 constraints):")
constrained_scores = []
params_constrained = params_base.copy()
params_constrained['monotone_constraints'] = constraints

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**params_constrained)
    model.fit(X_train, y_train, verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    
    print(f"   Fold {fold}: AUC={auc:.5f}, Acc={acc:.4f}")
    constrained_scores.append(auc)

constrained_mean = np.mean(constrained_scores)
constrained_std = np.std(constrained_scores)

# Comparison
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"UNCONSTRAINED:  {unconstrained_mean:.5f} ± {unconstrained_std:.5f}")
print(f"CONSTRAINED:    {constrained_mean:.5f} ± {constrained_std:.5f}")
print(f"Difference:     {(constrained_mean - unconstrained_mean)*100:+.2f}%")
print("")

if constrained_mean >= unconstrained_mean:
    print("✓ CONSTRAINTS HELPED - More robust, interpretable model")
    print("  → Proceed with constrained hyperparameter tuning")
elif constrained_mean >= unconstrained_mean - 0.005:
    print("≈ CONSTRAINTS NEUTRAL - Small tradeoff (<0.5%)")
    print("  → Consider constrained tuning for interpretability")
else:
    print("✗ CONSTRAINTS HURT - Significant performance loss")
    print(f"  → Loss: {abs((constrained_mean - unconstrained_mean)*100):.2f}%")
    print("  → Proceed with UNCONSTRAINED tuning")

# Fold-by-fold comparison
print("\n5. Fold-by-fold impact:")
for fold in range(5):
    diff = (constrained_scores[fold] - unconstrained_scores[fold]) * 100
    symbol = "✓" if diff >= 0 else "✗"
    print(f"   Fold {fold+1}: {diff:+.2f}% {symbol}")

print("\n" + "="*70)
print("DECISION")
print("="*70)

if constrained_mean >= unconstrained_mean - 0.003:
    print("→ Use CONSTRAINED tuning (interpretable + robust)")
    print("  Expected: 0.560-0.570 AUC after tuning")
else:
    print("→ Use UNCONSTRAINED tuning (maximize performance)")
    print("  Expected: 0.565-0.575 AUC after tuning")
print("="*70)
