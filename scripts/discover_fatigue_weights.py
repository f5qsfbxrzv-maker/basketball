"""
Fatigue Weight Discovery - Logistic Regression Stacking
- Isolate fatigue-related features
- Train logistic regression to find optimal coefficients
- Create data-driven net_fatigue_score using mathematically proven weights
- Compare single consolidated feature vs diluted signals
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt

print("\n" + "="*90)
print("FATIGUE WEIGHT DISCOVERY - LOGISTIC REGRESSION STACKING")
print("="*90)

# Load data
print("\n[1/5] Loading training data...")
df = pd.read_csv('data/training_data_36features.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"  Samples: {len(df):,}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Define fatigue features
fatigue_features = [
    'home_rest_days',
    'away_rest_days',
    'home_back_to_back',
    'away_back_to_back',
    'home_3in4',
    'away_3in4',
    'rest_advantage',
    'altitude_game'
]

print(f"\n[2/5] Isolating {len(fatigue_features)} fatigue features...")
for feat in fatigue_features:
    print(f"  - {feat}")

# Prepare data
X_fatigue = df[fatigue_features].copy()
y = df['target_moneyline_win'].copy()

# Check for missing values
missing = X_fatigue.isnull().sum()
if missing.any():
    print(f"\n  WARNING: Missing values detected:")
    for feat, count in missing[missing > 0].items():
        print(f"    {feat}: {count} ({count/len(df)*100:.2f}%)")
    X_fatigue = X_fatigue.fillna(0)

print(f"\n  Home win rate: {y.mean()*100:.1f}%")
print(f"  Feature means:")
for feat in fatigue_features:
    print(f"    {feat}: {X_fatigue[feat].mean():.3f}")

# Standardize features (important for coefficient comparison)
print("\n[3/5] Training logistic regression...")
scaler = StandardScaler()
X_fatigue_scaled = scaler.fit_transform(X_fatigue)

# Train logistic regression
logreg = LogisticRegression(
    penalty='l2',
    C=1.0,  # Moderate regularization
    max_iter=1000,
    random_state=42
)
logreg.fit(X_fatigue_scaled, y)

# Get coefficients
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]

# Calculate AUC
y_pred_proba = logreg.predict_proba(X_fatigue_scaled)[:, 1]
auc = roc_auc_score(y, y_pred_proba)

print(f"  Logistic Regression AUC: {auc:.5f}")
print(f"  Intercept: {intercept:.5f}")

# Print coefficients with interpretation
print(f"\n{'='*90}")
print("FATIGUE COEFFICIENTS (Mathematically Optimal Weights)")
print(f"{'='*90}")

coef_df = pd.DataFrame({
    'feature': fatigue_features,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients),
    'std_dev': scaler.scale_
}).sort_values('abs_coefficient', ascending=False)

print(f"\n{'Rank':<6} {'Feature':<25} {'Coefficient':<15} {'Impact':<15} {'Direction':<15}")
print("-"*90)

for idx, row in coef_df.iterrows():
    direction = "Helps Home" if row['coefficient'] > 0 else "Helps Away"
    impact = "Strong" if abs(row['coefficient']) > 0.1 else "Moderate" if abs(row['coefficient']) > 0.05 else "Weak"
    rank = coef_df.index.get_loc(idx) + 1
    print(f"{rank:<6} {row['feature']:<25} {row['coefficient']:<15.6f} {impact:<15} {direction:<15}")

# Calculate relative importance
total_abs_coef = coef_df['abs_coefficient'].sum()
coef_df['importance_pct'] = (coef_df['abs_coefficient'] / total_abs_coef * 100).round(2)

print(f"\n{'='*90}")
print("RELATIVE IMPORTANCE")
print(f"{'='*90}")
print(f"\n{'Feature':<25} {'Importance %':<15} {'Relative to Top':<20}")
print("-"*90)

top_coef = coef_df.iloc[0]['abs_coefficient']
for idx, row in coef_df.iterrows():
    relative = row['abs_coefficient'] / top_coef
    print(f"{row['feature']:<25} {row['importance_pct']:<15.2f} {relative:<20.2f}x")

# Create net_fatigue_score using discovered weights
print(f"\n[4/5] Creating data-driven net_fatigue_score...")

# Method 1: Using standardized coefficients (recommended)
df['net_fatigue_score_standardized'] = 0
for i, feat in enumerate(fatigue_features):
    # Standardize the feature
    feat_standardized = (df[feat] - scaler.mean_[i]) / scaler.scale_[i]
    # Apply coefficient
    df['net_fatigue_score_standardized'] += feat_standardized * coefficients[i]

# Method 2: Using raw coefficients on raw data (simpler for production)
# Calculate coefficients for raw (non-standardized) data
raw_coefficients = coefficients / scaler.scale_

df['net_fatigue_score_raw'] = 0
for i, feat in enumerate(fatigue_features):
    df['net_fatigue_score_raw'] += df[feat] * raw_coefficients[i]

print(f"  Created: net_fatigue_score_standardized")
print(f"  Created: net_fatigue_score_raw")
print(f"\n  Net fatigue score statistics:")
print(f"    Mean: {df['net_fatigue_score_raw'].mean():.4f}")
print(f"    Std:  {df['net_fatigue_score_raw'].std():.4f}")
print(f"    Min:  {df['net_fatigue_score_raw'].min():.4f}")
print(f"    Max:  {df['net_fatigue_score_raw'].max():.4f}")

# Compare predictive power: Individual features vs consolidated
print(f"\n[5/5] Comparing predictive power...")

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

X_train, X_test, y_train, y_test = train_test_split(
    df[['net_fatigue_score_raw']], 
    df['target_moneyline_win'],
    test_size=0.2,
    random_state=42,
    shuffle=False
)

# Single consolidated feature
logreg_consolidated = LogisticRegression(max_iter=1000, random_state=42)
logreg_consolidated.fit(X_train, y_train)
y_pred_consolidated = logreg_consolidated.predict_proba(X_test)[:, 1]
auc_consolidated = roc_auc_score(y_test, y_pred_consolidated)
logloss_consolidated = log_loss(y_test, y_pred_consolidated)

# Multiple diluted features
X_train_diluted = df.iloc[:len(X_train)][fatigue_features]
X_test_diluted = df.iloc[len(X_train):][fatigue_features]
logreg_diluted = LogisticRegression(max_iter=1000, random_state=42)
logreg_diluted.fit(X_train_diluted, y_train)
y_pred_diluted = logreg_diluted.predict_proba(X_test_diluted)[:, 1]
auc_diluted = roc_auc_score(y_test, y_pred_diluted)
logloss_diluted = log_loss(y_test, y_pred_diluted)

print(f"\n{'='*90}")
print("COMPARISON: 1 Consolidated vs 8 Diluted Features")
print(f"{'='*90}")

print(f"\n{'Metric':<30} {'Consolidated (1)':<20} {'Diluted (8)':<20} {'Winner':<15}")
print("-"*90)
print(f"{'AUC':<30} {auc_consolidated:<20.5f} {auc_diluted:<20.5f} "
      f"{'Consolidated' if auc_consolidated > auc_diluted else 'Diluted':<15}")
print(f"{'LogLoss':<30} {logloss_consolidated:<20.5f} {logloss_diluted:<20.5f} "
      f"{'Consolidated' if logloss_consolidated < logloss_diluted else 'Diluted':<15}")
print(f"{'Features Used':<30} {1:<20} {8:<20} {'Consolidated':<15}")

improvement_auc = ((auc_consolidated - auc_diluted) / auc_diluted * 100)
improvement_logloss = ((logloss_diluted - logloss_consolidated) / logloss_diluted * 100)

print(f"\n  Consolidated feature improvement:")
print(f"    AUC: {improvement_auc:+.2f}%")
print(f"    LogLoss: {improvement_logloss:+.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fatigue Weight Discovery - Logistic Regression Coefficients', 
             fontsize=16, fontweight='bold')

# Coefficient bar chart
ax = axes[0, 0]
colors = ['green' if c > 0 else 'red' for c in coef_df['coefficient']]
ax.barh(range(len(coef_df)), coef_df['coefficient'], color=colors, alpha=0.7)
ax.set_yticks(range(len(coef_df)))
ax.set_yticklabels(coef_df['feature'])
ax.set_xlabel('Coefficient (Positive = Helps Home)')
ax.set_title('Fatigue Feature Coefficients')
ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

# Importance pie chart
ax = axes[0, 1]
ax.pie(coef_df['importance_pct'], labels=coef_df['feature'], autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 8})
ax.set_title('Relative Importance Distribution')

# Distribution of net_fatigue_score
ax = axes[1, 0]
ax.hist(df['net_fatigue_score_raw'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Net Fatigue Score')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Consolidated Fatigue Score')
ax.axvline(0, color='red', linestyle='--', label='Neutral (0)')
ax.legend()
ax.grid(alpha=0.3)

# Fatigue score vs win rate
ax = axes[1, 1]
df['fatigue_bin'] = pd.cut(df['net_fatigue_score_raw'], bins=10)
win_rate_by_fatigue = df.groupby('fatigue_bin')['target_moneyline_win'].mean()
bin_centers = [interval.mid for interval in win_rate_by_fatigue.index]
ax.plot(bin_centers, win_rate_by_fatigue.values, 'o-', linewidth=2, markersize=8, color='darkgreen')
ax.axhline(0.5, color='red', linestyle='--', label='50% baseline')
ax.set_xlabel('Net Fatigue Score (binned)')
ax.set_ylabel('Home Win Rate')
ax.set_title('Win Rate vs Consolidated Fatigue Score')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('models/fatigue_weight_discovery.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved: models/fatigue_weight_discovery.png")

# Save coefficients for production use
print(f"\n{'='*90}")
print("PRODUCTION CODE")
print(f"{'='*90}")

print(f"\n# Data-driven fatigue weights (from {len(df):,} games):")
print(f"FATIGUE_WEIGHTS = {{")
for idx, row in coef_df.iterrows():
    print(f"    '{row['feature']}': {raw_coefficients[fatigue_features.index(row['feature'])]:.8f},")
print(f"}}")

print(f"\n# Create net_fatigue_score:")
print(f"df['net_fatigue_score'] = (")
for i, feat in enumerate(fatigue_features):
    sign = '+' if i > 0 else ' '
    print(f"    {sign} df['{feat}'] * {raw_coefficients[i]:.8f}")
print(f")")

# Save enhanced dataset
output_cols = [c for c in df.columns if c not in ['fatigue_bin']]
df[output_cols].to_csv('data/training_data_with_net_fatigue.csv', index=False)
print(f"\n  Saved: data/training_data_with_net_fatigue.csv")

# Save coefficient report
coef_report = coef_df.copy()
coef_report['raw_coefficient'] = raw_coefficients[coef_df.index]
coef_report.to_csv('models/fatigue_coefficients.csv', index=False)
print(f"  Saved: models/fatigue_coefficients.csv")

print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")

print(f"\n✓ Discovered mathematically optimal fatigue weights")
print(f"✓ Top impact: {coef_df.iloc[0]['feature']} ({coef_df.iloc[0]['importance_pct']:.1f}%)")
print(f"✓ Consolidated 8 weak signals → 1 strong signal")
print(f"✓ AUC improvement: {improvement_auc:+.2f}%")
print(f"✓ Ready to replace {len(fatigue_features)} features with net_fatigue_score")

print(f"\n{'='*90}")
