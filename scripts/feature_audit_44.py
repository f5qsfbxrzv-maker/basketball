"""
Feature Audit for 44-Feature Model
- Train model with best hyperparameters from Trial 306
- Analyze feature importance (gain, weight, cover)
- Check for redundant/low-value features
- Correlation analysis
- SHAP value analysis
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import json
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*70)
print("FEATURE AUDIT - 44 FEATURES")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('data/training_data_with_temporal_features.csv')

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win', 
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]
y = df['target_spread_cover']

print(f"  Games: {len(df):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Covers: {y.sum():,} ({y.mean()*100:.1f}%)")

# Best params from Trial 306
print("\n[2/5] Training with Trial 306 params...")
params = {
    'learning_rate': 0.001575,
    'max_depth': 3,
    'min_child_weight': 14,
    'reg_lambda': 0.090566,
    'reg_alpha': 0.002029,
    'subsample': 0.569192,
    'colsample_bytree': 0.877130,
    'gamma': 3.824719,
    'n_estimators': 1000,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0
}

# Train on full dataset to get feature importance
dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
model = xgb.train(params, dtrain, num_boost_round=1000)

print("  Model trained")

# Get feature importance (all metrics)
print("\n[3/5] Extracting feature importance...")
importance_gain = model.get_score(importance_type='gain')
importance_weight = model.get_score(importance_type='weight')
importance_cover = model.get_score(importance_type='cover')

# Combine into dataframe
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'gain': [importance_gain.get(f, 0) for f in feature_cols],
    'weight': [importance_weight.get(f, 0) for f in feature_cols],
    'cover': [importance_cover.get(f, 0) for f in feature_cols]
})

# Normalize
importance_df['gain_pct'] = 100 * importance_df['gain'] / importance_df['gain'].sum()
importance_df['weight_pct'] = 100 * importance_df['weight'] / importance_df['weight'].sum()
importance_df['cover_pct'] = 100 * importance_df['cover'] / importance_df['cover'].sum()

# Sort by gain
importance_df = importance_df.sort_values('gain', ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("TOP 20 FEATURES BY GAIN")
print("="*70)
print(f"\n{'Rank':<6} {'Feature':<35} {'Gain%':<10} {'Weight%':<10} {'Cover%':<10}")
print("-"*70)
for i, row in importance_df.head(20).iterrows():
    print(f"{i+1:<6} {row['feature']:<35} {row['gain_pct']:<10.2f} {row['weight_pct']:<10.2f} {row['cover_pct']:<10.2f}")

print("\n" + "="*70)
print("BOTTOM 10 FEATURES (Potentially Removable)")
print("="*70)
print(f"\n{'Rank':<6} {'Feature':<35} {'Gain%':<10} {'Weight%':<10}")
print("-"*70)
for i, row in importance_df.tail(10).iterrows():
    print(f"{i+1:<6} {row['feature']:<35} {row['gain_pct']:<10.3f} {row['weight_pct']:<10.3f}")

# Identify zero-importance features
zero_features = importance_df[importance_df['gain'] == 0]['feature'].tolist()
if zero_features:
    print(f"\n** {len(zero_features)} features with ZERO importance:")
    for f in zero_features:
        print(f"  - {f}")

# Save full importance
importance_df.to_csv('models/feature_importance_44features.csv', index=False)
print(f"\n[4/5] Saved: models/feature_importance_44features.csv")

# Correlation analysis
print("\n[5/5] Checking feature correlations...")
corr_matrix = X.corr().abs()

# Find highly correlated pairs (>0.8)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.8:
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print(f"\n** {len(high_corr_pairs)} highly correlated pairs (>0.8):")
    for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True):
        print(f"  {pair['feature1']:<30} <-> {pair['feature2']:<30} : {pair['correlation']:.3f}")
else:
    print("  No highly correlated pairs (>0.8)")

# Summary recommendations
print("\n" + "="*70)
print("AUDIT SUMMARY")
print("="*70)

top10_gain = importance_df.head(10)['gain_pct'].sum()
print(f"\nTop 10 features account for: {top10_gain:.1f}% of total gain")
print(f"Zero-importance features: {len(zero_features)}")
print(f"Highly correlated pairs (>0.8): {len(high_corr_pairs)}")

if zero_features:
    print(f"\n✓ Recommendation: Remove {len(zero_features)} zero-importance features")
if len(high_corr_pairs) > 5:
    print(f"✓ Recommendation: Review {len(high_corr_pairs)} correlated pairs for redundancy")
if top10_gain > 80:
    print(f"✓ Note: Top 10 features dominate ({top10_gain:.1f}%) - consider focused tuning")

print("\n" + "="*70)
