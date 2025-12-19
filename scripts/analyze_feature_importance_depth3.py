"""
Feature Importance Analysis for Moneyline Model
- Load best trial parameters from Optuna
- Train model and extract feature importance (Gain, Weight, Cover)
- Identify zero-importance features
- Visualize which features the depth-3 tree actually uses
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("\n" + "="*90)
print("FEATURE IMPORTANCE ANALYSIS - MONEYLINE MODEL")
print("="*90)

# Load data
print("\n[1/5] Loading training data...")
df = pd.read_csv('data/training_data_36features.csv')
df['date'] = pd.to_datetime(df['date'])

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df['target_moneyline_win']

print(f"  Samples: {len(df):,}")
print(f"  Features: {len(feature_cols)}")

# Load best parameters from Optuna
print("\n[2/5] Loading best parameters from Optuna...")
study = optuna.load_study(
    study_name='nba_moneyline_platt_10hr',
    storage='sqlite:///models/nba_moneyline_platt_10hr.db'
)

best_params = study.best_params
best_params['n_estimators'] = 1000
best_params['objective'] = 'binary:logistic'
best_params['tree_method'] = 'hist'
best_params['random_state'] = 42
best_params['verbosity'] = 0
best_params['early_stopping_rounds'] = 50

print(f"  Best trial: #{study.best_trial.number}")
print(f"  Best LogLoss: {study.best_value:.6f}")
print(f"  Max depth: {best_params['max_depth']}")
print(f"  Gamma: {best_params['gamma']:.3f}")

# Train model
print("\n[3/5] Training model with best parameters...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

model = xgb.XGBClassifier(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"  Model trained with {model.n_estimators} trees")

# Extract feature importance
print("\n[4/5] Extracting feature importance...")

# Get importance by different metrics
importance_gain = model.get_booster().get_score(importance_type='gain')
importance_weight = model.get_booster().get_score(importance_type='weight')
importance_cover = model.get_booster().get_score(importance_type='cover')

# Create comprehensive importance dataframe
importance_data = []
for feature in feature_cols:
    # XGBoost uses f0, f1, etc internally
    feature_idx = feature_cols.index(feature)
    xgb_feature_name = f'f{feature_idx}'
    
    importance_data.append({
        'feature': feature,
        'gain': importance_gain.get(xgb_feature_name, 0),
        'weight': importance_weight.get(xgb_feature_name, 0),
        'cover': importance_cover.get(xgb_feature_name, 0)
    })

importance_df = pd.DataFrame(importance_data)

# Calculate percentages
total_gain = importance_df['gain'].sum()
total_weight = importance_df['weight'].sum()
importance_df['gain_pct'] = (importance_df['gain'] / total_gain * 100).round(2)
importance_df['weight_pct'] = (importance_df['weight'] / total_weight * 100).round(2)

# Sort by gain
importance_df = importance_df.sort_values('gain', ascending=False).reset_index(drop=True)

# Identify feature categories
def categorize_feature(feat):
    if 'elo' in feat:
        return 'ELO'
    elif any(x in feat for x in ['rest', 'back_to_back', '3in4']):
        return 'Fatigue'
    elif any(x in feat for x in ['injury', 'star']):
        return 'Injury'
    elif any(x in feat for x in ['ewma', 'orb', '3p', 'fta', 'tov', 'efg']):
        return 'Shooting/Offense'
    elif 'chaos' in feat or 'foul' in feat:
        return 'Game Flow'
    elif 'season' in feat or 'altitude' in feat:
        return 'Context'
    else:
        return 'Other'

importance_df['category'] = importance_df['feature'].apply(categorize_feature)

# Print results
print(f"\n{'='*90}")
print("FEATURE IMPORTANCE RANKINGS (by Gain)")
print(f"{'='*90}")
print(f"\n{'Rank':<6} {'Feature':<35} {'Gain%':<10} {'Weight':<10} {'Category':<15}")
print("-"*90)

for idx, row in importance_df.head(37).iterrows():
    print(f"{idx+1:<6} {row['feature']:<35} {row['gain_pct']:<10.2f} {row['weight']:<10.0f} {row['category']:<15}")

# Zero importance features
zero_importance = importance_df[importance_df['gain'] == 0]
print(f"\n{'='*90}")
print("ZERO-IMPORTANCE FEATURES (Ignored by Model)")
print(f"{'='*90}")

if len(zero_importance) > 0:
    print(f"\n❌ The model IGNORED {len(zero_importance)} features at depth-3:")
    for idx, row in zero_importance.iterrows():
        print(f"  • {row['feature']} ({row['category']})")
else:
    print("\n✓ All features used (none ignored)")

# Category analysis
print(f"\n{'='*90}")
print("CATEGORY ANALYSIS")
print(f"{'='*90}")

category_summary = importance_df.groupby('category').agg({
    'gain_pct': 'sum',
    'feature': 'count'
}).rename(columns={'feature': 'count'}).sort_values('gain_pct', ascending=False)

print(f"\n{'Category':<20} {'Features':<12} {'Total Gain%':<15}")
print("-"*90)
for category, row in category_summary.iterrows():
    print(f"{category:<20} {row['count']:<12.0f} {row['gain_pct']:<15.2f}")

# Top 10 vs Bottom 10 analysis
top_10_gain = importance_df.head(10)['gain_pct'].sum()
print(f"\n{'='*90}")
print("CONCENTRATION ANALYSIS")
print(f"{'='*90}")
print(f"\nTop 10 features account for: {top_10_gain:.1f}% of total gain")
print(f"Bottom 27 features account for: {100 - top_10_gain:.1f}% of total gain")

# Identify "Raw" vs "Differential" features
raw_features = []
diff_features = []

for feat in feature_cols:
    if 'diff' in feat or 'advantage' in feat or 'mismatch' in feat:
        diff_features.append(feat)
    elif feat.startswith('away_') or feat.startswith('home_'):
        # Check if it's isolated (no differential pair exists)
        if feat.startswith('away_'):
            base = feat.replace('away_', '')
            if f'home_{base}' in feature_cols and f'{base}_diff' not in feature_cols:
                raw_features.append(feat)
        elif feat.startswith('home_'):
            base = feat.replace('home_', '')
            if f'away_{base}' in feature_cols and f'{base}_diff' not in feature_cols:
                raw_features.append(feat)

# Get importance for raw vs differential
raw_importance = importance_df[importance_df['feature'].isin(raw_features)]
diff_importance = importance_df[importance_df['feature'].isin(diff_features)]

print(f"\n{'='*90}")
print("RAW vs DIFFERENTIAL FEATURES")
print(f"{'='*90}")

print(f"\nDifferential features ({len(diff_features)}):")
print(f"  Total gain: {diff_importance['gain_pct'].sum():.1f}%")
for feat in diff_features:
    gain = importance_df[importance_df['feature'] == feat]['gain_pct'].values
    if len(gain) > 0:
        print(f"    {feat}: {gain[0]:.2f}%")

print(f"\nRaw isolated features ({len(raw_features)}):")
print(f"  Total gain: {raw_importance['gain_pct'].sum():.1f}%")
for feat in raw_features:
    gain = importance_df[importance_df['feature'] == feat]['gain_pct'].values
    if len(gain) > 0:
        print(f"    {feat}: {gain[0]:.2f}%")

# Visualizations
print("\n[5/5] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Importance Analysis - Moneyline Model (Depth 3)', 
             fontsize=16, fontweight='bold')

# Top 20 features by gain
ax = axes[0, 0]
top_20 = importance_df.head(20)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_20)))
ax.barh(range(len(top_20)), top_20['gain_pct'], color=colors)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('Gain %')
ax.set_title('Top 20 Features by Gain')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Category distribution
ax = axes[0, 1]
category_summary_sorted = category_summary.sort_values('gain_pct', ascending=True)
ax.barh(range(len(category_summary_sorted)), category_summary_sorted['gain_pct'], 
        color='steelblue')
ax.set_yticks(range(len(category_summary_sorted)))
ax.set_yticklabels(category_summary_sorted.index)
ax.set_xlabel('Total Gain %')
ax.set_title('Feature Importance by Category')
ax.grid(axis='x', alpha=0.3)

# Gain distribution
ax = axes[1, 0]
ax.hist(importance_df['gain_pct'], bins=30, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(importance_df['gain_pct'].median(), color='red', linestyle='--', 
           label=f'Median: {importance_df["gain_pct"].median():.2f}%')
ax.axvline(importance_df['gain_pct'].mean(), color='blue', linestyle='--', 
           label=f'Mean: {importance_df["gain_pct"].mean():.2f}%')
ax.set_xlabel('Gain %')
ax.set_ylabel('Number of Features')
ax.set_title('Distribution of Feature Importance')
ax.legend()
ax.grid(alpha=0.3)

# Cumulative importance
ax = axes[1, 1]
cumsum = importance_df['gain_pct'].cumsum()
ax.plot(range(1, len(cumsum)+1), cumsum, linewidth=2, color='darkgreen')
ax.axhline(80, color='red', linestyle='--', label='80% threshold')
ax.axhline(90, color='orange', linestyle='--', label='90% threshold')
ax.set_xlabel('Number of Features')
ax.set_ylabel('Cumulative Gain %')
ax.set_title('Cumulative Feature Importance')
ax.legend()
ax.grid(alpha=0.3)

# Find 80% and 90% thresholds
features_for_80 = (cumsum >= 80).idxmax() + 1
features_for_90 = (cumsum >= 90).idxmax() + 1
ax.text(features_for_80, 82, f'{features_for_80} features', ha='center')
ax.text(features_for_90, 92, f'{features_for_90} features', ha='center')

plt.tight_layout()
plt.savefig('models/feature_importance_moneyline_depth3.png', dpi=300, bbox_inches='tight')
print(f"  Saved: models/feature_importance_moneyline_depth3.png")

# Save CSV
importance_df.to_csv('models/feature_importance_moneyline_detailed.csv', index=False)
print(f"  Saved: models/feature_importance_moneyline_detailed.csv")

print(f"\n{'='*90}")
print("DIAGNOSIS SUMMARY")
print(f"{'='*90}")

print(f"\n🎯 Model Configuration:")
print(f"  Max depth: {best_params['max_depth']} (shallow tree)")
print(f"  Gamma: {best_params['gamma']:.3f}")
print(f"  LogLoss: {study.best_value:.6f}")

print(f"\n📊 Feature Usage:")
print(f"  Total features: {len(feature_cols)}")
print(f"  Used features: {len(importance_df[importance_df['gain'] > 0])}")
print(f"  Ignored features: {len(zero_importance)}")
print(f"  Top 10 account for: {top_10_gain:.1f}% of gain")
print(f"  Features for 80% gain: {features_for_80}")

print(f"\n💡 Key Findings:")
if len(zero_importance) > 0:
    print(f"  ⚠️  {len(zero_importance)} features completely ignored at depth-3")
if raw_importance['gain_pct'].sum() < 5:
    print(f"  ⚠️  Raw isolated features contribute only {raw_importance['gain_pct'].sum():.1f}%")
if diff_importance['gain_pct'].sum() > 30:
    print(f"  ✓ Differential features dominate ({diff_importance['gain_pct'].sum():.1f}% gain)")

print(f"\n🔧 Recommended Actions:")
print(f"  1. Convert raw features to matchup differentials")
print(f"  2. Consolidate 7 fatigue features into net_fatigue_score")
print(f"  3. Replace season_year with league_avg_offensive_rating")
print(f"  4. Add efficiency_pace_interaction")
print(f"  5. Drop zero-importance features")

print(f"\n{'='*90}")
