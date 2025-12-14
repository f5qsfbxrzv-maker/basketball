"""
Quick Single-Fold Test - Preview Final Performance
Tests best parameters from unconstrained run with constraints applied.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

print("="*70)
print("SINGLE-FOLD PREVIEW - CONSTRAINED MODEL")
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

print(f"\n2. Applied {len([c for c in constraints if c != 0])}/43 constraints")

# Test params (from unconstrained baseline)
params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'monotone_constraints': constraints,
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

print("\n3. Testing on most recent fold (Fold 5)...")
tscv = TimeSeriesSplit(n_splits=5)

# Get last fold
for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    if fold_num == 5:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"   Train: {len(X_train):,} games")
        print(f"   Val:   {len(X_val):,} games")
        print(f"   Val date range: {df.iloc[val_idx]['date'].min()} to {df.iloc[val_idx]['date'].max()}")
        
        # Train
        print("\n4. Training...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        # Predict
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        acc = accuracy_score(y_val, (preds > 0.5).astype(int))
        
        print("\n" + "="*70)
        print("RESULTS - FOLD 5 (Most Recent Data)")
        print("="*70)
        print(f"AUC:      {auc:.5f}")
        print(f"Accuracy: {acc:.4f}")
        print("")
        print("COMPARISON:")
        print(f"  Unconstrained Fold 5: 0.63888")
        print(f"  Constrained Fold 5:   {auc:.5f}")
        print(f"  Difference:           {(auc - 0.63888)*100:+.2f}%")
        
        if auc >= 0.63888:
            print("\n✓ Constraints IMPROVED performance (more robust)")
        elif auc >= 0.630:
            print("\n≈ Constraints maintained performance (~1% loss acceptable)")
        else:
            print("\n⚠ Constraints hurt significantly (>1.5% loss)")
        
        # Feature importance
        print("\n5. Top 10 Features (Constrained):")
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        for idx, (feat, importance) in enumerate(imp.head(10).items(), 1):
            constr = constraint_map.get(feat, 0)
            constr_str = '(↑)' if constr == 1 else '(↓)' if constr == -1 else '   '
            print(f"   {idx:2d}. {feat:30s}: {importance*100:5.2f}% {constr_str}")
        
        print("\n" + "="*70)
        print("PROJECTION FOR FULL 300-TRIAL TUNING:")
        print("="*70)
        
        # Estimate full 5-fold average (assuming similar performance on other folds)
        # Fold 5 typically best (most data), so average will be lower
        estimated_avg = auc * 0.88  # Typical Fold 5 to average ratio
        
        print(f"Current Fold 5:       {auc:.5f}")
        print(f"Estimated 5-fold avg: {estimated_avg:.5f}")
        print(f"Target (0.565):       {'✓ ON TRACK' if estimated_avg >= 0.560 else '⚠ BELOW TARGET'}")
        
        if estimated_avg >= 0.565:
            print("\n✓ Projected to BEAT target with hyperparameter tuning")
        elif estimated_avg >= 0.560:
            print("\n≈ Projected to MEET target with hyperparameter tuning")
        else:
            print(f"\n⚠ Projected to MISS target by {(0.565 - estimated_avg)*100:.2f}%")
            print("   May need: Ensemble models, more features, or different approach")
        
        print("="*70)
