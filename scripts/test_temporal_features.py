"""
Test 43-Feature Dataset (36 + 7 Temporal)
Verify if explicit temporal features recover lost 1.01% AUC.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

print("="*70)
print("TESTING 43 FEATURES (36 + 7 TEMPORAL)")
print("="*70)

# Load data
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

drop_cols = ['date','game_id','home_team','away_team','season'] + [c for c in df.columns if c.startswith('target_')]
X = df[[c for c in df.columns if c not in drop_cols]]
y = df['target_spread_cover']

print(f"\n1. Dataset: {len(df):,} games, {X.shape[1]} features")

# Same params as sanity test for fair comparison
params = {
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

print("\n2. Running 5-Fold Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=5)
scores = []
accuracies = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    
    print(f"   Fold {fold}: AUC={auc:.5f}, Acc={acc:.4f}")
    scores.append(auc)
    accuracies.append(acc)

mean_auc = np.mean(scores)
std_auc = np.std(scores)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"43 Features (36 + 7 temporal): {mean_auc:.5f} ± {std_auc:.5f}")
print(f"Clean 36 Features (baseline):  0.54070")
print(f"Dirty 36 Features (target):    0.55080")
print("")
print(f"Recovery:     {(mean_auc - 0.54070)*100:+.2f}%")
print(f"vs Dirty:     {(mean_auc - 0.55080)*100:+.2f}%")

if mean_auc >= 0.55080:
    print("\n✓ SUCCESS - Recovered lost performance + improved!")
elif mean_auc >= 0.54070:
    print(f"\n⚠ PARTIAL - Recovered {((mean_auc - 0.54070) / 0.01010)*100:.1f}% of lost signal")
else:
    print("\n✗ FAILURE - Temporal features didn't help")

# Feature importance
print("\n3. Top 10 Features (with temporal):")
model_full = xgb.XGBClassifier(**params)
model_full.fit(X, y)

imp = pd.Series(model_full.feature_importances_, index=X.columns).sort_values(ascending=False)
for idx, (feat, importance) in enumerate(imp.head(10).items(), 1):
    is_temporal = '***' if feat in ['is_season_opener','season_year','season_year_normalized','games_into_season','season_progress','endgame_phase','season_month'] else ''
    print(f"   {idx:2d}. {feat:30s}: {importance*100:5.2f}% {is_temporal}")

temporal_importance = imp[['is_season_opener','season_year','season_year_normalized','games_into_season','season_progress','endgame_phase','season_month']].sum()
print(f"\n   Temporal features combined: {temporal_importance*100:.2f}%")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
if mean_auc >= 0.560:
    print("✓ BREAKTHROUGH - Ready for production hyperparameter tuning")
elif mean_auc >= 0.555:
    print("✓ PROGRESS - Hyperparameter tuning should break 0.560")
else:
    print("⚠ PLATEAU - May need different approach (ensemble, more features)")
print("="*70)
