"""
MONEYLINE MODEL TRAINING & OPTIMIZATION
- Target: target_moneyline_win (not spread covers)
- Features: 37-feature set (same features, different target)
- Calibration: Platt scaling (sigmoid) for better probability bounds
- Optimization: 500 trials for moneyline-specific hyperparameters
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
import optuna
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

print("\n" + "="*90)
print("MONEYLINE MODEL TRAINING WITH PLATT SCALING")
print("="*90)

# Load training data
print("\n[1/7] Loading training data...")
df = pd.read_csv('data/training_data_36features.csv')
df['date'] = pd.to_datetime(df['date'])

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win', 
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"  Total games: {len(df):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Three-way split: Train / Calibration / Test
train_cutoff = pd.to_datetime('2023-10-01')
cal_cutoff = pd.to_datetime('2024-10-01')

df_train = df[df['date'] < train_cutoff].copy()
df_cal = df[(df['date'] >= train_cutoff) & (df['date'] < cal_cutoff)].copy()
df_test = df[df['date'] >= cal_cutoff].copy()

X_train = df_train[feature_cols]
y_train = df_train['target_moneyline_win']  # KEY CHANGE: Moneyline target
X_cal = df_cal[feature_cols]
y_cal = df_cal['target_moneyline_win']
X_test = df_test[feature_cols]
y_test = df_test['target_moneyline_win']

print(f"\n  Train: {len(df_train):,} games (through Sep 2023)")
print(f"  Calibration: {len(df_cal):,} games (2023-24 season)")
print(f"  Test: {len(df_test):,} games (2024-25 & 2025-26)")
print(f"  Train home win rate: {y_train.mean()*100:.1f}%")
print(f"  Cal home win rate: {y_cal.mean()*100:.1f}%")
print(f"  Test home win rate: {y_test.mean()*100:.1f}%")

# Hyperparameter optimization with Optuna
print("\n[2/7] Running Optuna optimization (500 trials for moneyline)...")

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0
    }
    
    # Time series cross-validation on training set
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)
        
        y_pred = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, y_pred)
        scores.append(score)
    
    return np.mean(scores)

study = optuna.create_study(
    direction='minimize',
    study_name='moneyline_500trials',
    storage='sqlite:///models/moneyline_optuna.db',
    load_if_exists=True
)

study.optimize(objective, n_trials=500, show_progress_bar=True)

print(f"\n  Best trial: {study.best_trial.number}")
print(f"  Best LogLoss: {study.best_value:.5f}")
print(f"  Best params:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Train final model with best params
print("\n[3/7] Training final model with best hyperparameters...")

best_params = study.best_params.copy()
best_params.update({
    'n_estimators': 1000,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0
})

model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train, verbose=False)

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

# Apply Platt scaling on calibration set
print("\n[4/7] Applying Platt scaling (sigmoid calibration)...")

# Fit calibrator on calibration set
from sklearn.linear_model import LogisticRegression

# Platt scaling: fit logistic regression on raw probabilities
platt = LogisticRegression()
platt.fit(cal_probs_raw.reshape(-1, 1), y_cal)

# Apply to all sets
cal_probs_platt = platt.predict_proba(cal_probs_raw.reshape(-1, 1))[:, 1]
test_probs_platt = platt.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]

brier_cal_platt = brier_score_loss(y_cal, cal_probs_platt)
brier_test_platt = brier_score_loss(y_test, test_probs_platt)

brier_improvement_cal = ((brier_cal_raw - brier_cal_platt) / brier_cal_raw) * 100
brier_improvement_test = ((brier_test_raw - brier_test_platt) / brier_test_raw) * 100

print(f"  Cal Brier (Platt):  {brier_cal_platt:.5f} ({brier_improvement_cal:+.2f}% improvement)")
print(f"  Test Brier (Platt): {brier_test_platt:.5f} ({brier_improvement_test:+.2f}% improvement)")

# Generate reliability curves
print("\n[5/7] Generating reliability curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Moneyline Model: Raw vs Platt Scaling', fontsize=16, fontweight='bold')

# Raw - Cal set
fraction_positives_cal_raw, mean_predicted_cal_raw = calibration_curve(
    y_cal, cal_probs_raw, n_bins=10, strategy='quantile'
)

ax = axes[0, 0]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_predicted_cal_raw, fraction_positives_cal_raw, 's-', label='Raw predictions', markersize=8)
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Set - RAW (Before Platt)')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Brier: {brier_cal_raw:.4f}', transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Platt - Cal set
fraction_positives_cal_platt, mean_predicted_cal_platt = calibration_curve(
    y_cal, cal_probs_platt, n_bins=10, strategy='quantile'
)

ax = axes[0, 1]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_predicted_cal_platt, fraction_positives_cal_platt, 's-', label='Platt predictions', 
        markersize=8, color='green')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Set - PLATT (After)')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Brier: {brier_cal_platt:.4f}\nΔ: {brier_improvement_cal:+.2f}%', 
        transform=ax.transAxes, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Raw - Test set
fraction_positives_test_raw, mean_predicted_test_raw = calibration_curve(
    y_test, test_probs_raw, n_bins=10, strategy='quantile'
)

ax = axes[1, 0]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_predicted_test_raw, fraction_positives_test_raw, 's-', label='Raw predictions', 
        markersize=8, color='red')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Test Set - RAW (Before Platt)')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Brier: {brier_test_raw:.4f}', transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Platt - Test set
fraction_positives_test_platt, mean_predicted_test_platt = calibration_curve(
    y_test, test_probs_platt, n_bins=10, strategy='quantile'
)

ax = axes[1, 1]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_predicted_test_platt, fraction_positives_test_platt, 's-', label='Platt predictions', 
        markersize=8, color='green')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Test Set - PLATT (After)')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Brier: {brier_test_platt:.4f}\nΔ: {brier_improvement_test:+.2f}%', 
        transform=ax.transAxes, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('models/moneyline_platt_calibration.png', dpi=300, bbox_inches='tight')
print(f"  Saved: models/moneyline_platt_calibration.png")

# Save model and calibrator
print("\n[6/7] Saving model artifacts...")
with open('models/moneyline_xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/moneyline_platt_calibrator.pkl', 'wb') as f:
    pickle.dump(platt, f)

# Save metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'train_games': len(df_train),
    'cal_games': len(df_cal),
    'test_games': len(df_test),
    'features': feature_cols,
    'target': 'target_moneyline_win',
    'calibration_method': 'platt_scaling',
    'best_params': best_params,
    'train_auc': float(roc_auc_score(y_train, train_probs_raw)),
    'cal_auc': float(roc_auc_score(y_cal, cal_probs_raw)),
    'test_auc': float(roc_auc_score(y_test, test_probs_raw)),
    'train_brier_raw': float(brier_train_raw),
    'cal_brier_raw': float(brier_cal_raw),
    'test_brier_raw': float(brier_test_raw),
    'cal_brier_platt': float(brier_cal_platt),
    'test_brier_platt': float(brier_test_platt),
    'brier_improvement_cal_pct': float(brier_improvement_cal),
    'brier_improvement_test_pct': float(brier_improvement_test)
}

import json
with open('models/moneyline_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  Saved: models/moneyline_xgboost_model.pkl")
print(f"  Saved: models/moneyline_platt_calibrator.pkl")
print(f"  Saved: models/moneyline_model_metadata.json")

# Backtest with real moneyline odds
print("\n[7/7] Running backtest with real moneyline odds...")

odds_df = pd.read_csv('data/live/closing_odds_2024_25.csv')
odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

df_test = df_test.merge(
    odds_df[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='left'
)

n_with_odds = df_test['home_ml_odds'].notna().sum()
print(f"  Matched odds: {n_with_odds} of {len(df_test)} games")

df_test = df_test[df_test['home_ml_odds'].notna()].copy()
df_test['prob_home_win'] = platt.predict_proba(
    model.predict_proba(df_test[feature_cols])[:, 1].reshape(-1, 1)
)[:, 1]
df_test['prob_away_win'] = 1 - df_test['prob_home_win']

# Odds conversion functions
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

# Test multiple thresholds
UNIT_SIZE = 100
COMMISSION = 0.048
thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

print(f"\n{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'P&L':<12} {'ROI%':<10}")
print("-"*90)

best_roi = -999
best_threshold = 0.03

for threshold in thresholds:
    bets = []
    
    for idx, row in df_test.iterrows():
        market_prob_home = american_to_prob(row['home_ml_odds'])
        market_prob_away = american_to_prob(row['away_ml_odds'])
        total_prob = market_prob_home + market_prob_away
        
        fair_prob_home = market_prob_home / total_prob
        fair_prob_away = market_prob_away / total_prob
        
        edge_home = row['prob_home_win'] - fair_prob_home
        edge_away = row['prob_away_win'] - fair_prob_away
        
        if edge_home > threshold:
            odds_decimal = row['home_ml_decimal']
            actual_result = row['target_moneyline_win']
        elif edge_away > threshold:
            odds_decimal = row['away_ml_decimal']
            actual_result = not row['target_moneyline_win']
        else:
            continue
        
        if actual_result:
            profit = UNIT_SIZE * (odds_decimal - 1)
            profit_after_commission = profit * (1 - COMMISSION)
            pnl = profit_after_commission
        else:
            pnl = -UNIT_SIZE
        
        bets.append({'result': actual_result, 'pnl': pnl})
    
    if len(bets) > 0:
        bets_df = pd.DataFrame(bets)
        total_bets = len(bets_df)
        wins = bets_df['result'].sum()
        win_rate = wins / total_bets
        total_pnl = bets_df['pnl'].sum()
        roi = (total_pnl / (total_bets * UNIT_SIZE)) * 100
        
        print(f"{threshold:<12.2f} {total_bets:<8.0f} {wins:<8.0f} "
              f"{win_rate*100:<10.1f} ${total_pnl:<11,.2f} {roi:<+10.2f}")
        
        if roi > best_roi:
            best_roi = roi
            best_threshold = threshold

print(f"\n{'='*90}")
print(f"BEST THRESHOLD: {best_threshold:.2f} ({best_threshold*100:.0f}%) with ROI: {best_roi:+.2f}%")
print(f"{'='*90}")

print(f"\nCalibration Summary:")
print(f"  Method: Platt Scaling (Sigmoid)")
print(f"  Test Brier: {brier_test_raw:.5f} → {brier_test_platt:.5f} ({brier_improvement_test:+.2f}%)")
print(f"  Test AUC: {roc_auc_score(y_test, test_probs_raw):.5f}")
print(f"\nBacktest Summary:")
print(f"  Target: Moneyline wins (not spread covers)")
print(f"  Best ROI: {best_roi:+.2f}% at {best_threshold*100:.0f}% edge threshold")
print(f"\n{'='*90}")
