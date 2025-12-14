"""
Analyze feature importance and correlations from Syndicate hypertuning results
- Load best model parameters
- Train model and extract feature importance
- Calculate feature correlations
- Identify features to drop (bottom 50%)
"""

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

print("="*70)
print("FEATURE IMPORTANCE & CORRELATION ANALYSIS")
print("="*70)

# Load data
print("\n1. Loading training data...")
df = pd.read_csv("data/training_data_with_features.csv")

# Get features
exclude_cols = ['date', 'game_id', 'home_team', 'away_team', 'season', 
               'target_spread', 'target_spread_cover', 'target_moneyline_win', 
               'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df['target_moneyline_win']

print(f"   Total games: {len(df):,}")
print(f"   Total features: {len(feature_cols)}")

# Load best Syndicate parameters
print("\n2. Loading best Syndicate parameters...")
try:
    with open("output/syndicate_best_params.json", "r") as f:
        best_params = json.load(f)
    print(f"   Best AUC from Syndicate: {best_params.get('best_auc', 'N/A')}")
except FileNotFoundError:
    print("   WARNING: syndicate_best_params.json not found, using defaults")
    best_params = {
        'learning_rate': 0.015,
        'n_estimators': 2786,
        'max_depth': 9,
        'min_child_weight': 21,
        'gamma': 2.69,
        'subsample': 0.63,
        'colsample_bytree': 0.53,
        'colsample_bylevel': 0.57,
        'colsample_bynode': 0.63,
        'reg_alpha': 16.36,
        'reg_lambda': 1.11,
        'scale_pos_weight': 0.90,
        'max_delta_step': 5
    }

# Add fixed params
best_params['random_state'] = 42
best_params['tree_method'] = 'hist'
best_params['eval_metric'] = 'logloss'

# Train final model with best params
print("\n3. Training model with best parameters...")
model = xgb.XGBClassifier(**best_params)
model.fit(X, y, verbose=False)

# Get feature importance
print("\n4. Extracting feature importance...")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
importance_df['rank'] = importance_df.index + 1

# Print top 20 features
print("\n" + "="*70)
print("TOP 20 FEATURES BY IMPORTANCE")
print("="*70)
print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Cumulative':<12}")
print("-"*70)
for idx, row in importance_df.head(20).iterrows():
    print(f"{row['rank']:<6} {row['feature']:<30} {row['importance']:>10.4f}  {row['cumulative_importance']:>10.4f}")

# Find features that make up 90% of importance
cutoff_90 = importance_df[importance_df['cumulative_importance'] <= 0.90]
print(f"\nFeatures accounting for 90% importance: {len(cutoff_90)}")

# Bottom 50% features
bottom_half = importance_df.tail(len(importance_df) // 2)
print(f"\nBOTTOM 50% FEATURES (candidates for removal):")
print(f"Total: {len(bottom_half)} features")
print(f"Total importance: {bottom_half['importance'].sum():.4f} ({bottom_half['importance'].sum()*100:.1f}%)")
print("\nBottom 18 features:")
for idx, row in bottom_half.iterrows():
    print(f"  {row['rank']:2d}. {row['feature']:<30} {row['importance']:>8.4f}")

# Calculate feature correlations
print("\n" + "="*70)
print("FEATURE CORRELATIONS")
print("="*70)
print("Calculating correlation matrix...")
corr_matrix = X.corr().abs()

# Find highly correlated features (>0.8)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.8:
            high_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print(f"\nHIGHLY CORRELATED FEATURES (>0.8): {len(high_corr_pairs)} pairs")
    print("-"*70)
    for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True):
        print(f"  {pair['feature_1']:<30} <-> {pair['feature_2']:<30} r={pair['correlation']:.3f}")
else:
    print("\nNo highly correlated features (>0.8) found")

# Calculate correlation with target
print("\n" + "="*70)
print("FEATURE-TARGET CORRELATIONS")
print("="*70)
target_corr = X.corrwith(y).abs().sort_values(ascending=False)
print("\nTop 15 features by target correlation:")
for idx, (feat, corr) in enumerate(target_corr.head(15).items(), 1):
    print(f"  {idx:2d}. {feat:<30} r={corr:.4f}")

print("\nBottom 15 features by target correlation:")
for idx, (feat, corr) in enumerate(target_corr.tail(15).items(), 1):
    rank = len(target_corr) - 15 + idx
    print(f"  {rank:2d}. {feat:<30} r={corr:.4f}")

# Save results
importance_df.to_csv("output/feature_importance_analysis.csv", index=False)
target_corr.to_csv("output/feature_target_correlations.csv", header=['correlation'])
print("\n" + "="*70)
print(f"Results saved:")
print(f"  - output/feature_importance_analysis.csv")
print(f"  - output/feature_target_correlations.csv")
print("="*70)

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("\n1. FEATURE REDUCTION:")
print(f"   - Consider removing bottom {len(bottom_half)} features")
print(f"   - These account for only {bottom_half['importance'].sum()*100:.1f}% of total importance")
print(f"   - Could speed up training by ~{len(bottom_half)/len(feature_cols)*100:.0f}%")

if high_corr_pairs:
    print("\n2. MULTICOLLINEARITY:")
    print(f"   - Found {len(high_corr_pairs)} highly correlated pairs")
    print(f"   - Consider removing one feature from each pair")

print("\n3. FEATURE ENGINEERING:")
print("   - AUC ~0.55 suggests model has plateaued")
print("   - Hyperparameter tuning unlikely to break 0.56")
print("   - New features needed (interaction terms, non-linear transforms, etc.)")

print("\n4. ENSEMBLING:")
print("   - Blend Syndicate + Unconstrained models")
print("   - May stabilize variance and improve AUC by 0.5-1%")

# Generate test code snippet
print("\n" + "="*70)
print("TEST CODE SNIPPET")
print("="*70)
print("""
# Load best Syndicate parameters
import json
import xgboost as xgb

with open('output/syndicate_best_params.json', 'r') as f:
    params = json.load(f)

params.update({
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
})

# Train model
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Predict
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")
""")
