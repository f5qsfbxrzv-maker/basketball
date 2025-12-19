"""
Compare feature importance between old model (Trial 1306) and new model (Trial 340).
Identify which features perform better/worse with Gold Standard ELO.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Trial 1306 hyperparameters (OLD system, K=32 ELO)
TRIAL_1306_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'min_child_weight': 25,
    'learning_rate': 0.0105,
    'n_estimators': 9947,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0
}

# Trial 340 hyperparameters (GOLD system, K=15 ELO) - best from recent optimization
TRIAL_340_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.178,
    'n_estimators': 281,
    'min_child_weight': 10,
    'subsample': 0.84,
    'colsample_bytree': 0.946,
    'colsample_bylevel': 0.978,
    'gamma': 9.537,
    'reg_alpha': 1.887,
    'reg_lambda': 1.327,
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0
}

# 25 features used by Trial 1306
FEATURES = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'net_fatigue_score', 'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff', 
    'ewma_orb_diff', 'ewma_vol_3p_diff', 'injury_impact_diff', 'injury_shock_diff',
    'star_mismatch', 'ewma_chaos_home', 'ewma_foul_synergy_home', 
    'total_foul_environment', 'league_offensive_context', 'season_progress',
    'pace_efficiency_interaction', 'projected_possession_margin',
    'three_point_matchup', 'net_free_throw_advantage', 'star_power_leverage',
    'offense_vs_defense_matchup', 'injury_matchup_advantage'
]

print("=" * 80)
print("FEATURE IMPORTANCE COMPARISON")
print("=" * 80)
print()
print("Comparing:")
print("  OLD MODEL (Trial 1306): K=32 ELO, Log-Loss 0.6222")
print("  NEW MODEL (Trial 340):  K=15 GOLD ELO, Log-Loss 0.6297")
print()

# Load Gold Standard training data
print("Loading training data with Gold Standard ELO...")
df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')

# Ensure we only have games with valid ELO values
df = df.dropna(subset=['home_composite_elo', 'away_composite_elo'])

# Split features and target
X = df[FEATURES]
y = df['target_moneyline_win']

print(f"Total games: {len(df)}")
print(f"Features: {len(FEATURES)}")
print()

# Train both models using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

print("Training OLD MODEL (Trial 1306 hyperparameters on GOLD ELO data)...")
model_old = xgb.XGBClassifier(**TRIAL_1306_PARAMS)
fold_idx = 0
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    fold_idx += 1
    if fold_idx == 5:  # Use last fold for final model
        model_old.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        break

print("Training NEW MODEL (Trial 340 hyperparameters on GOLD ELO data)...")
model_new = xgb.XGBClassifier(**TRIAL_340_PARAMS)
fold_idx = 0
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    fold_idx += 1
    if fold_idx == 5:  # Use last fold for final model
        model_new.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        break

print()
print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)
print()

# Get feature importance from both models
importance_old = model_old.feature_importances_
importance_new = model_new.feature_importances_

# Create comparison DataFrame
comparison = pd.DataFrame({
    'Feature': FEATURES,
    'Old_Importance': importance_old,
    'New_Importance': importance_new
})

# Calculate change
comparison['Change'] = comparison['New_Importance'] - comparison['Old_Importance']
comparison['Pct_Change'] = ((comparison['New_Importance'] - comparison['Old_Importance']) / 
                            comparison['Old_Importance'] * 100)

# Sort by absolute change
comparison['Abs_Change'] = comparison['Change'].abs()
comparison = comparison.sort_values('Abs_Change', ascending=False)

print("BIGGEST CHANGES (by absolute change):")
print("-" * 80)
print(f"{'Feature':<30} {'Old':<8} {'New':<8} {'Change':<10} {'Pct':<10}")
print("-" * 80)

for _, row in comparison.head(10).iterrows():
    feature = row['Feature']
    old = row['Old_Importance']
    new = row['New_Importance']
    change = row['Change']
    pct = row['Pct_Change']
    
    arrow = "↑" if change > 0 else "↓"
    print(f"{feature:<30} {old:>6.4f}  {new:>6.4f}  {arrow} {abs(change):>6.4f}  {pct:>6.1f}%")

print()
print()
print("OVER-PERFORMERS (features better with Gold ELO):")
print("-" * 80)
over_performers = comparison[comparison['Pct_Change'] > 10].sort_values('Pct_Change', ascending=False)
if len(over_performers) > 0:
    for _, row in over_performers.iterrows():
        print(f"  {row['Feature']:<30} {row['Old_Importance']:.4f} → {row['New_Importance']:.4f} (+{row['Pct_Change']:.1f}%)")
else:
    print("  None (no features improved by >10%)")

print()
print("UNDER-PERFORMERS (features worse with Gold ELO):")
print("-" * 80)
under_performers = comparison[comparison['Pct_Change'] < -10].sort_values('Pct_Change')
if len(under_performers) > 0:
    for _, row in under_performers.iterrows():
        print(f"  {row['Feature']:<30} {row['Old_Importance']:.4f} → {row['New_Importance']:.4f} ({row['Pct_Change']:.1f}%)")
else:
    print("  None (no features declined by >10%)")

print()
print()
print("ELO FEATURES BREAKDOWN:")
print("-" * 80)
elo_features = comparison[comparison['Feature'].str.contains('elo', case=False)]
elo_features = elo_features.sort_values('New_Importance', ascending=False)
for _, row in elo_features.iterrows():
    print(f"  {row['Feature']:<30} Old: {row['Old_Importance']:.4f}  New: {row['New_Importance']:.4f}  Change: {row['Change']:+.4f}")

print()
print()
print("FOUR FACTORS BREAKDOWN:")
print("-" * 80)
four_factors = comparison[
    (comparison['Feature'].str.contains('efg|tov|oreb|ft_fga', case=False))
]
four_factors = four_factors.sort_values('New_Importance', ascending=False)
for _, row in four_factors.iterrows():
    print(f"  {row['Feature']:<30} Old: {row['Old_Importance']:.4f}  New: {row['New_Importance']:.4f}  Change: {row['Change']:+.4f}")

print()
print()
print("TOP 5 MOST IMPORTANT FEATURES:")
print("-" * 80)
print("OLD MODEL (Trial 1306 on OLD ELO, 0.6222):")
top_old = comparison.sort_values('Old_Importance', ascending=False).head(5)
for i, row in enumerate(top_old.itertuples(), 1):
    print(f"  {i}. {row.Feature:<30} {row.Old_Importance:.4f}")

print()
print("NEW MODEL (Trial 340 on GOLD ELO, 0.6297):")
top_new = comparison.sort_values('New_Importance', ascending=False).head(5)
for i, row in enumerate(top_new.itertuples(), 1):
    print(f"  {i}. {row.Feature:<30} {row.New_Importance:.4f}")

print()
print()
print("KEY INSIGHTS:")
print("-" * 80)

# Calculate total ELO importance
old_elo_total = comparison[comparison['Feature'].str.contains('elo', case=False)]['Old_Importance'].sum()
new_elo_total = comparison[comparison['Feature'].str.contains('elo', case=False)]['New_Importance'].sum()

print(f"Total ELO importance:")
print(f"  OLD: {old_elo_total:.4f} ({old_elo_total*100:.1f}%)")
print(f"  NEW: {new_elo_total:.4f} ({new_elo_total*100:.1f}%)")
print(f"  Change: {(new_elo_total - old_elo_total):+.4f} ({(new_elo_total - old_elo_total)*100:+.1f} percentage points)")

print()

# Check if composite ELO features became more/less important
home_elo_change = comparison[comparison['Feature'] == 'home_composite_elo']['Change'].values[0]
away_elo_change = comparison[comparison['Feature'] == 'away_composite_elo']['Change'].values[0]
off_diff_change = comparison[comparison['Feature'] == 'off_elo_diff']['Change'].values[0]
def_diff_change = comparison[comparison['Feature'] == 'def_elo_diff']['Change'].values[0]

print(f"Composite ELO changes:")
print(f"  home_composite_elo: {home_elo_change:+.4f}")
print(f"  away_composite_elo: {away_elo_change:+.4f}")
print(f"  off_elo_diff: {off_diff_change:+.4f}")
print(f"  def_elo_diff: {def_diff_change:+.4f}")

print()
print()
print("HYPOTHESIS:")
print("-" * 80)
if new_elo_total < old_elo_total:
    print("✗ Gold Standard ELO features have LESS total importance")
    print("  This suggests the new ELO is less predictive, explaining worse performance.")
elif new_elo_total > old_elo_total:
    print("✓ Gold Standard ELO features have MORE total importance")
    print("  This suggests the new ELO is MORE predictive.")
    print("  Old hyperparameters (max_depth=3, min_child_weight=25) were too conservative")
    print("  for clean ELO data. Trial 340 (max_depth=5, min_child_weight=10) is better.")

# Save comparison to CSV
comparison_output = comparison.drop('Abs_Change', axis=1).sort_values('New_Importance', ascending=False)
comparison_output.to_csv('feature_importance_comparison.csv', index=False)
print()
print("✓ Saved: feature_importance_comparison.csv")
print()
print("=" * 80)
