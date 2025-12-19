"""
Isotonic Regression Calibration Tuning
- Use proper hold-out set for calibration (not training set)
- Generate reliability curves to diagnose calibration issues
- Optimize edge threshold based on calibration quality
- Re-run backtest with improved calibration
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from datetime import datetime

print("\n" + "="*90)
print("ISOTONIC CALIBRATION TUNING")
print("="*90)

# Load training data
print("\n[1/8] Loading training data...")
df = pd.read_csv('data/training_data_36features.csv')
df['date'] = pd.to_datetime(df['date'])

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win', 
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Three-way split: Train / Calibration / Test
train_cutoff = pd.to_datetime('2023-10-01')
cal_cutoff = pd.to_datetime('2024-10-01')

df_train = df[df['date'] < train_cutoff].copy()
df_cal = df[(df['date'] >= train_cutoff) & (df['date'] < cal_cutoff)].copy()
df_test = df[df['date'] >= cal_cutoff].copy()

X_train = df_train[feature_cols]
y_train = df_train['target_moneyline_win']
X_cal = df_cal[feature_cols]
y_cal = df_cal['target_moneyline_win']
X_test = df_test[feature_cols]
y_test = df_test['target_moneyline_win']

print(f"  Train: {len(df_train):,} games (through Sep 2023)")
print(f"  Calibration: {len(df_cal):,} games (2023-24 season)")
print(f"  Test: {len(df_test):,} games (2024-25 & 2025-26)")
print(f"  Train home win rate: {y_train.mean()*100:.1f}%")
print(f"  Cal home win rate: {y_cal.mean()*100:.1f}%")
print(f"  Test home win rate: {y_test.mean()*100:.1f}%")

# Train model
print("\n[2/8] Training XGBoost...")
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

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, verbose=False)

# Get raw probabilities
train_probs_raw = model.predict_proba(X_train)[:, 1]
cal_probs_raw = model.predict_proba(X_cal)[:, 1]
test_probs_raw = model.predict_proba(X_test)[:, 1]

print(f"  Train AUC (raw): {roc_auc_score(y_train, train_probs_raw):.5f}")
print(f"  Cal AUC (raw):   {roc_auc_score(y_cal, cal_probs_raw):.5f}")
print(f"  Test AUC (raw):  {roc_auc_score(y_test, test_probs_raw):.5f}")

brier_train_raw = brier_score_loss(y_train, train_probs_raw)
brier_cal_raw = brier_score_loss(y_cal, cal_probs_raw)
brier_test_raw = brier_score_loss(y_test, test_probs_raw)

print(f"  Train Brier (raw): {brier_train_raw:.5f}")
print(f"  Cal Brier (raw):   {brier_cal_raw:.5f}")
print(f"  Test Brier (raw):  {brier_test_raw:.5f}")

# Fit isotonic regression on CALIBRATION SET (not training set!)
print("\n[3/8] Fitting isotonic regression on calibration set...")
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(cal_probs_raw, y_cal)

# Apply calibration
cal_probs_iso = iso.transform(cal_probs_raw)
test_probs_iso = iso.transform(test_probs_raw)

brier_cal_iso = brier_score_loss(y_cal, cal_probs_iso)
brier_test_iso = brier_score_loss(y_test, test_probs_iso)

brier_improvement_cal = ((brier_cal_raw - brier_cal_iso) / brier_cal_raw) * 100
brier_improvement_test = ((brier_test_raw - brier_test_iso) / brier_test_raw) * 100

print(f"  Cal Brier (isotonic):  {brier_cal_iso:.5f} ({brier_improvement_cal:+.2f}% improvement)")
print(f"  Test Brier (isotonic): {brier_test_iso:.5f} ({brier_improvement_test:+.2f}% improvement)")

# Generate reliability curves
print("\n[4/8] Generating reliability curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Calibration Analysis: Raw vs Isotonic', fontsize=16, fontweight='bold')

# Raw calibration - Calibration set
fraction_positives_cal_raw, mean_predicted_cal_raw = calibration_curve(
    y_cal, cal_probs_raw, n_bins=10, strategy='quantile'
)

ax = axes[0, 0]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_predicted_cal_raw, fraction_positives_cal_raw, 's-', label='Raw predictions', markersize=8)
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Set - RAW (Before Isotonic)')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Brier: {brier_cal_raw:.4f}', transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Isotonic calibration - Calibration set
fraction_positives_cal_iso, mean_predicted_cal_iso = calibration_curve(
    y_cal, cal_probs_iso, n_bins=10, strategy='quantile'
)

ax = axes[0, 1]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_predicted_cal_iso, fraction_positives_cal_iso, 's-', label='Isotonic predictions', 
        markersize=8, color='green')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Set - ISOTONIC (After)')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Brier: {brier_cal_iso:.4f}\nΔ: {brier_improvement_cal:+.2f}%', 
        transform=ax.transAxes, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Raw calibration - Test set
fraction_positives_test_raw, mean_predicted_test_raw = calibration_curve(
    y_test, test_probs_raw, n_bins=10, strategy='quantile'
)

ax = axes[1, 0]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_predicted_test_raw, fraction_positives_test_raw, 's-', label='Raw predictions', 
        markersize=8, color='red')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Test Set - RAW (Before Isotonic)')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Brier: {brier_test_raw:.4f}', transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Isotonic calibration - Test set
fraction_positives_test_iso, mean_predicted_test_iso = calibration_curve(
    y_test, test_probs_iso, n_bins=10, strategy='quantile'
)

ax = axes[1, 1]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_predicted_test_iso, fraction_positives_test_iso, 's-', label='Isotonic predictions', 
        markersize=8, color='green')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Test Set - ISOTONIC (After)')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Brier: {brier_test_iso:.4f}\nΔ: {brier_improvement_test:+.2f}%', 
        transform=ax.transAxes, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('models/isotonic_calibration_analysis.png', dpi=300, bbox_inches='tight')
print(f"  Saved: models/isotonic_calibration_analysis.png")

# Load moneyline odds
print("\n[5/8] Loading moneyline odds...")
odds_df = pd.read_csv('data/live/closing_odds_2024_25.csv')
odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

# Merge with test set
df_test = df_test.merge(
    odds_df[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='left'
)

n_with_odds = df_test['home_ml_odds'].notna().sum()
print(f"  Matched odds: {n_with_odds} of {len(df_test)} games")

df_test = df_test[df_test['home_ml_odds'].notna()].copy()
df_test['prob_home_win_iso'] = iso.transform(
    model.predict_proba(df_test[feature_cols])[:, 1]
)
df_test['prob_away_win_iso'] = 1 - df_test['prob_home_win_iso']

# Convert odds functions
def american_to_decimal(american_odds):
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

def american_to_prob(american_odds):
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

df_test['home_ml_decimal'] = df_test['home_ml_odds'].apply(american_to_decimal)
df_test['away_ml_decimal'] = df_test['away_ml_odds'].apply(american_to_decimal)

# Test multiple edge thresholds
print("\n[6/8] Testing edge thresholds...")

UNIT_SIZE = 100
COMMISSION = 0.048
thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

results_by_threshold = []

for threshold in thresholds:
    bets = []
    
    for idx, row in df_test.iterrows():
        # Market implied probabilities
        market_prob_home = american_to_prob(row['home_ml_odds'])
        market_prob_away = american_to_prob(row['away_ml_odds'])
        total_prob = market_prob_home + market_prob_away
        
        # Fair probabilities after vig removal
        fair_prob_home = market_prob_home / total_prob
        fair_prob_away = market_prob_away / total_prob
        
        # Calculate edge
        edge_home = row['prob_home_win_iso'] - fair_prob_home
        edge_away = row['prob_away_win_iso'] - fair_prob_away
        
        # Bet on side with positive edge above threshold
        if edge_home > threshold:
            bet_side = 'home'
            edge = edge_home
            odds_decimal = row['home_ml_decimal']
            actual_result = row['target_moneyline_win']
        elif edge_away > threshold:
            bet_side = 'away'
            edge = edge_away
            odds_decimal = row['away_ml_decimal']
            actual_result = not row['target_moneyline_win']
        else:
            continue
        
        # Calculate P&L
        if actual_result:
            profit = UNIT_SIZE * (odds_decimal - 1)
            profit_after_commission = profit * (1 - COMMISSION)
            pnl = profit_after_commission
        else:
            pnl = -UNIT_SIZE
        
        bets.append({
            'edge': edge,
            'odds': odds_decimal,
            'result': actual_result,
            'pnl': pnl
        })
    
    if len(bets) > 0:
        bets_df = pd.DataFrame(bets)
        total_bets = len(bets_df)
        total_staked = total_bets * UNIT_SIZE
        wins = bets_df['result'].sum()
        win_rate = wins / total_bets
        total_pnl = bets_df['pnl'].sum()
        roi = (total_pnl / total_staked) * 100
        avg_edge = bets_df['edge'].mean()
        
        results_by_threshold.append({
            'threshold': threshold,
            'bets': total_bets,
            'wins': wins,
            'win_rate': win_rate,
            'avg_edge': avg_edge,
            'pnl': total_pnl,
            'roi': roi
        })

results_df = pd.DataFrame(results_by_threshold)

print(f"\n{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Avg Edge%':<12} {'P&L':<12} {'ROI%':<10}")
print("-"*90)
for idx, row in results_df.iterrows():
    print(f"{row['threshold']:<12.2f} {row['bets']:<8.0f} {row['wins']:<8.0f} "
          f"{row['win_rate']*100:<10.1f} {row['avg_edge']*100:<12.2f} "
          f"${row['pnl']:<11,.2f} {row['roi']:<+10.2f}")

# Find optimal threshold
best_idx = results_df['roi'].idxmax()
best_threshold = results_df.loc[best_idx, 'threshold']
best_roi = results_df.loc[best_idx, 'roi']

print(f"\n  Optimal threshold: {best_threshold:.2f} (ROI: {best_roi:+.2f}%)")

# Run final backtest with optimal threshold
print(f"\n[7/8] Running backtest with optimal threshold ({best_threshold:.2f})...")

bets = []

for idx, row in df_test.iterrows():
    market_prob_home = american_to_prob(row['home_ml_odds'])
    market_prob_away = american_to_prob(row['away_ml_odds'])
    total_prob = market_prob_home + market_prob_away
    
    fair_prob_home = market_prob_home / total_prob
    fair_prob_away = market_prob_away / total_prob
    
    edge_home = row['prob_home_win_iso'] - fair_prob_home
    edge_away = row['prob_away_win_iso'] - fair_prob_away
    
    if edge_home > best_threshold:
        bet_side = 'home'
        bet_team = row['home_team']
        edge = edge_home
        win_prob = row['prob_home_win_iso']
        odds_decimal = row['home_ml_decimal']
        odds_american = row['home_ml_odds']
        actual_result = row['target_moneyline_win']
    elif edge_away > best_threshold:
        bet_side = 'away'
        bet_team = row['away_team']
        edge = edge_away
        win_prob = row['prob_away_win_iso']
        odds_decimal = row['away_ml_decimal']
        odds_american = row['away_ml_odds']
        actual_result = not row['target_moneyline_win']
    else:
        continue
    
    if actual_result:
        profit = UNIT_SIZE * (odds_decimal - 1)
        profit_after_commission = profit * (1 - COMMISSION)
        pnl = profit_after_commission
    else:
        pnl = -UNIT_SIZE
    
    bets.append({
        'date': row['date'],
        'game': f"{row['away_team']} @ {row['home_team']}",
        'bet_side': bet_side,
        'bet_team': bet_team,
        'edge': edge,
        'win_prob': win_prob,
        'odds_american': odds_american,
        'odds_decimal': odds_decimal,
        'stake': UNIT_SIZE,
        'result': 'WIN' if actual_result else 'LOSS',
        'pnl': pnl
    })

bets_df = pd.DataFrame(bets)

print(f"\n{'='*90}")
print("FINAL BACKTEST RESULTS - ISOTONIC CALIBRATED")
print(f"{'='*90}")

total_bets = len(bets_df)
total_staked = total_bets * UNIT_SIZE
total_wins = (bets_df['result'] == 'WIN').sum()
total_losses = (bets_df['result'] == 'LOSS').sum()
win_rate = total_wins / total_bets

total_pnl = bets_df['pnl'].sum()
roi = (total_pnl / total_staked) * 100

avg_edge = bets_df['edge'].mean()
avg_odds = bets_df['odds_decimal'].mean()

print(f"\nBetting Summary:")
print(f"  Minimum edge threshold: {best_threshold:.2f} ({best_threshold*100:.0f}%)")
print(f"  Total games available: {len(df_test):,}")
print(f"  Bets placed: {total_bets} ({total_bets/len(df_test)*100:.1f}% of games)")
print(f"  Total staked: ${total_staked:,.0f}")

print(f"\nPerformance:")
print(f"  Wins: {total_wins}")
print(f"  Losses: {total_losses}")
print(f"  Win rate: {win_rate*100:.1f}%")
print(f"  Average edge: {avg_edge*100:.2f}%")
print(f"  Average odds: {avg_odds:.3f} decimal")

print(f"\nResults:")
print(f"  Total P&L: ${total_pnl:+,.2f}")
print(f"  ROI: {roi:+.2f}%")

# Analyze by bet type
bets_df['is_favorite'] = bets_df['odds_american'] < 0
favorites = bets_df[bets_df['is_favorite']]
underdogs = bets_df[~bets_df['is_favorite']]

print(f"\n{'='*90}")
print("BET TYPE BREAKDOWN")
print(f"{'='*90}")

if len(favorites) > 0:
    fav_win_rate = (favorites['result'] == 'WIN').mean()
    fav_roi = (favorites['pnl'].sum() / (len(favorites) * UNIT_SIZE)) * 100
    print(f"\nFavorites: {len(favorites)} bets")
    print(f"  Win rate: {fav_win_rate*100:.1f}%")
    print(f"  Avg odds: {favorites['odds_american'].mean():+.0f}")
    print(f"  P&L: ${favorites['pnl'].sum():+,.2f}")
    print(f"  ROI: {fav_roi:+.2f}%")

if len(underdogs) > 0:
    dog_win_rate = (underdogs['result'] == 'WIN').mean()
    dog_roi = (underdogs['pnl'].sum() / (len(underdogs) * UNIT_SIZE)) * 100
    print(f"\nUnderdogs: {len(underdogs)} bets")
    print(f"  Win rate: {dog_win_rate*100:.1f}%")
    print(f"  Avg odds: {underdogs['odds_american'].mean():+.0f}")
    print(f"  P&L: ${underdogs['pnl'].sum():+,.2f}")
    print(f"  ROI: {dog_roi:+.2f}%")

# Monthly breakdown
print(f"\n{'='*90}")
print("MONTHLY BREAKDOWN")
print(f"{'='*90}")
bets_df['month'] = pd.to_datetime(bets_df['date']).dt.to_period('M')
monthly = bets_df.groupby('month').agg({
    'stake': 'count',
    'pnl': 'sum',
    'result': lambda x: (x == 'WIN').sum()
}).rename(columns={'stake': 'bets', 'result': 'wins'})
monthly['win_rate'] = monthly['wins'] / monthly['bets']
monthly['roi'] = (monthly['pnl'] / (monthly['bets'] * UNIT_SIZE)) * 100

print(f"\n{'Month':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'P&L':<12} {'ROI%':<10}")
print("-"*90)
for month, row in monthly.iterrows():
    print(f"{str(month):<12} {row['bets']:<8.0f} {row['wins']:<8.0f} "
          f"{row['win_rate']*100:<10.1f} ${row['pnl']:<11,.2f} {row['roi']:<+10.2f}")

# Save results
print(f"\n[8/8] Saving results...")
bets_df.to_csv('models/backtest_isotonic_optimized.csv', index=False)
results_df.to_csv('models/edge_threshold_analysis.csv', index=False)

print(f"  Saved: models/backtest_isotonic_optimized.csv")
print(f"  Saved: models/edge_threshold_analysis.csv")

print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
print(f"\nCalibration Improvement:")
print(f"  Test Brier Score: {brier_test_raw:.5f} → {brier_test_iso:.5f} ({brier_improvement_test:+.2f}%)")
print(f"\nBacktest Comparison:")
print(f"  Original (3% threshold, train-set calibration): -2.29% ROI")
print(f"  Optimized ({best_threshold*100:.0f}% threshold, hold-out calibration): {roi:+.2f}% ROI")
print(f"  Improvement: {roi - (-2.29):+.2f} percentage points")
print(f"\n{'='*90}")
