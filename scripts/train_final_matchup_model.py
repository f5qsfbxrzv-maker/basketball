"""
Final Model Training - Matchup-Optimized Features
- Uses 24-feature matchup_optimized dataset
- Trains with Trial #873 best parameters
- Compares to baseline LogLoss 0.6564
- Expected: Break the 0.650 barrier
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

print("\n" + "="*90)
print("FINAL MODEL TRAINING - MATCHUP-OPTIMIZED FEATURES")
print("="*90)

# Load optimized dataset
print("\n[1/5] Loading matchup-optimized dataset...")
df = pd.read_csv('data/training_data_matchup_optimized.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"  Samples: {len(df):,}")
print(f"  Features: 24 (optimized from 37)")

# Prepare features and target
exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df['target_moneyline_win']

print(f"\n  Features in model:")
for i, feat in enumerate(feature_cols, 1):
    print(f"    {i:2}. {feat}")

print(f"\n  Home win rate: {y.mean()*100:.1f}%")

# Best parameters from Trial #873
print("\n[2/5] Loading Trial #873 best parameters...")
params = {
    'learning_rate': 0.04990986414755364,
    'max_depth': 3,
    'min_child_weight': 14,
    'gamma': 0.9318079110160647,
    'subsample': 0.6349355890275402,
    'colsample_bytree': 0.7368685047037173,
    'reg_lambda': 0.0238006824217908,
    'reg_alpha': 0.01681714016473699,
    'scale_pos_weight': 0.9591928649383534,
    'n_estimators': 1000,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds': 50
}

print(f"  Max depth: {params['max_depth']}")
print(f"  Learning rate: {params['learning_rate']:.6f}")
print(f"  Gamma: {params['gamma']:.4f}")
print(f"  Scale pos weight: {params['scale_pos_weight']:.4f}")

# Train/test split (time-based)
print("\n[3/5] Training model...")
cutoff = pd.to_datetime('2024-10-01')
train_mask = df['date'] < cutoff
test_mask = df['date'] >= cutoff

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"  Train: {len(X_train):,} games (before Oct 2024)")
print(f"  Test:  {len(X_test):,} games (2024-25 & 2025-26)")

# Train model
model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Predictions
train_probs = model.predict_proba(X_train)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]

# Metrics
train_logloss = log_loss(y_train, train_probs)
test_logloss = log_loss(y_test, test_probs)
train_auc = roc_auc_score(y_train, train_probs)
test_auc = roc_auc_score(y_test, test_probs)
train_brier = brier_score_loss(y_train, train_probs)
test_brier = brier_score_loss(y_test, test_probs)

print(f"\n[4/5] Evaluating performance...")
print(f"\n{'='*90}")
print("BASELINE vs OPTIMIZED COMPARISON")
print(f"{'='*90}")

baseline_logloss = 0.656442  # Trial #873 on 37 features
optimized_logloss = test_logloss
improvement = ((baseline_logloss - optimized_logloss) / baseline_logloss) * 100

print(f"\n{'Metric':<25} {'Baseline (37 feat)':<20} {'Optimized (24 feat)':<20} {'Change':<15}")
print("-"*90)
print(f"{'Test LogLoss':<25} {baseline_logloss:<20.6f} {optimized_logloss:<20.6f} {improvement:+.2f}%")
print(f"{'Test AUC':<25} {0.686:<20.3f} {test_auc:<20.5f} {(test_auc-0.686)/0.686*100:+.2f}%")
print(f"{'Test Brier':<25} {0.228:<20.3f} {test_brier:<20.5f} {(0.228-test_brier)/0.228*100:+.2f}%")

if optimized_logloss < 0.650:
    print(f"\n🎉 SUCCESS: Broke the 0.650 barrier!")
elif optimized_logloss < 0.655:
    print(f"\n✓ Strong performance: {optimized_logloss:.6f}")
else:
    print(f"\n→ Decent: {optimized_logloss:.6f}")

print(f"\n{'='*90}")
print("FEATURE IMPACT ANALYSIS")
print(f"{'='*90}")

# Get feature importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance,
    'pct': importance / importance.sum() * 100
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Features:")
print(f"{'Rank':<6} {'Feature':<35} {'Importance %':<15}")
print("-"*90)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{idx+1:<6} {row['feature']:<35} {row['pct']:<15.2f}")

top_10_pct = feature_importance.head(10)['pct'].sum()
print(f"\nTop 10 account for: {top_10_pct:.1f}% of total importance")

# Check if consolidation worked
fatigue_importance = feature_importance[feature_importance['feature'] == 'net_fatigue_score']
if not fatigue_importance.empty:
    fatigue_pct = fatigue_importance.iloc[0]['pct']
    fatigue_rank = feature_importance.index.get_loc(fatigue_importance.index[0]) + 1
    print(f"\nFatigue consolidation:")
    print(f"  net_fatigue_score: #{fatigue_rank} ({fatigue_pct:.2f}%)")
    print(f"  Expected: ~18% (was 8 diluted features)")

# Calibration with Platt scaling
print(f"\n[5/5] Applying Platt calibration...")
calibrated_model = CalibratedClassifierCV(
    model,
    method='sigmoid',  # Platt scaling
    cv='prefit'
)
calibrated_model.fit(X_train, y_train)

test_probs_cal = calibrated_model.predict_proba(X_test)[:, 1]
test_logloss_cal = log_loss(y_test, test_probs_cal)
test_brier_cal = brier_score_loss(y_test, test_probs_cal)

cal_improvement = ((test_logloss - test_logloss_cal) / test_logloss) * 100

print(f"\n  Raw LogLoss:        {test_logloss:.6f}")
print(f"  Calibrated LogLoss: {test_logloss_cal:.6f} ({cal_improvement:+.2f}%)")
print(f"  Raw Brier:          {test_brier:.6f}")
print(f"  Calibrated Brier:   {test_brier_cal:.6f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Final Model Performance - Matchup-Optimized Features', 
             fontsize=16, fontweight='bold')

# Feature importance
ax = axes[0, 0]
top_15 = feature_importance.head(15)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_15)))
ax.barh(range(len(top_15)), top_15['pct'], color=colors)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['feature'])
ax.set_xlabel('Importance %')
ax.set_title('Top 15 Features')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# LogLoss comparison
ax = axes[0, 1]
metrics_comparison = pd.DataFrame({
    'Model': ['Baseline\n(37 feat)', 'Optimized\n(24 feat)', 'Calibrated\n(Platt)'],
    'LogLoss': [baseline_logloss, test_logloss, test_logloss_cal]
})
colors_bars = ['orange', 'steelblue', 'green']
ax.bar(metrics_comparison['Model'], metrics_comparison['LogLoss'], color=colors_bars, alpha=0.7)
ax.axhline(0.650, color='red', linestyle='--', label='Target: 0.650')
ax.axhline(0.620, color='darkred', linestyle='--', label='Monster: 0.620')
ax.set_ylabel('LogLoss')
ax.set_title('Model Progression')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Probability distribution
ax = axes[1, 0]
ax.hist(test_probs, bins=50, alpha=0.5, label='Raw', edgecolor='black')
ax.hist(test_probs_cal, bins=50, alpha=0.5, label='Calibrated', edgecolor='black')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Frequency')
ax.set_title('Probability Distribution')
ax.legend()
ax.grid(alpha=0.3)

# Predicted vs Actual win rate by decile
ax = axes[1, 1]
df_test = df[test_mask].copy()
df_test['pred_prob_cal'] = test_probs_cal
df_test['pred_decile'] = pd.qcut(test_probs_cal, 10, labels=False, duplicates='drop')
decile_stats = df_test.groupby('pred_decile').agg({
    'pred_prob_cal': 'mean',
    'target_moneyline_win': 'mean'
}).reset_index()

ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(decile_stats['pred_prob_cal'], decile_stats['target_moneyline_win'], 
        'o-', linewidth=2, markersize=8, label='Actual', color='green')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Win Rate')
ax.set_title('Calibration Curve (Deciles)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('models/final_model_performance.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved: models/final_model_performance.png")

# Save model
model.save_model('models/nba_moneyline_matchup_optimized.json')
print(f"  Saved: models/nba_moneyline_matchup_optimized.json")

# Save importance
feature_importance.to_csv('models/final_feature_importance.csv', index=False)
print(f"  Saved: models/final_feature_importance.csv")

print(f"\n{'='*90}")
print("FINAL SUMMARY")
print(f"{'='*90}")

print(f"\n✓ Feature Engineering:")
print(f"  • 37 → 24 features ({(37-24)/37*100:.0f}% reduction)")
print(f"  • Consolidated 8 fatigue → 1 net_fatigue_score")
print(f"  • Added 6 matchup interactions")
print(f"  • Fixed season_year temporal trap")

print(f"\n✓ Model Performance:")
print(f"  • Baseline LogLoss: {baseline_logloss:.6f}")
print(f"  • Optimized LogLoss: {optimized_logloss:.6f} ({improvement:+.2f}%)")
print(f"  • Calibrated LogLoss: {test_logloss_cal:.6f}")

if optimized_logloss < 0.650:
    print(f"  • 🎉 Broke 0.650 barrier!")
if test_logloss_cal < 0.645:
    print(f"  • 🔥 Approaching monster territory (<0.620)!")

print(f"\n✓ Next Steps:")
print(f"  1. Backtest with real moneyline odds")
print(f"  2. Compare ROI to previous -2.29%")
print(f"  3. Expected: Better calibration → reduced underdog losses")

print(f"\n{'='*90}")
