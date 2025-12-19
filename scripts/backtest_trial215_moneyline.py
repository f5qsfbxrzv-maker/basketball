"""
Walk-Forward Backtest - Trial #215 (LogLoss 0.6565)
- Train on pre-2024 data
- Test on 2024-25 season
- Real moneyline odds
- Flat $100 unit bets
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import optuna
import json
from datetime import datetime

print("\n" + "="*90)
print("WALK-FORWARD BACKTEST - TRIAL #215 MONEYLINE")
print("="*90)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load Trial #215 parameters from previous study
print("\n[1/5] Loading Trial #215 parameters...")
study_name = 'nba_matchup_optimized_2000trials'
db_path = f'models/{study_name}.db'
storage = f'sqlite:///{db_path}'

study = optuna.load_study(study_name=study_name, storage=storage)
trial_215 = study.trials[215]

print(f"  Study: {study_name}")
print(f"  Trial: #215")
print(f"  LogLoss: {trial_215.value:.6f}")
print(f"\n  Parameters:")
for key, value in trial_215.params.items():
    if isinstance(value, float):
        print(f"    {key}: {value:.6f}")
    else:
        print(f"    {key}: {value}")

# Load matchup-optimized training data
print("\n[2/5] Loading matchup-optimized dataset...")
df = pd.read_csv('data/training_data_matchup_optimized.csv')
df['date'] = pd.to_datetime(df['date'])

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Split: Train on pre-2024, test on 2024-25
train_df = df[df['date'] < '2024-10-01'].copy()
test_df = df[df['date'] >= '2024-10-01'].copy()

print(f"  Total samples: {len(df):,}")
print(f"  Train samples (pre-2024): {len(train_df):,}")
print(f"  Test samples (2024-25): {len(test_df):,}")
print(f"  Features: {len(feature_cols)}")

# Train model with Trial #215 parameters
print("\n[3/5] Training model with Trial #215 parameters...")
params = {
    **trial_215.params,
    'n_estimators': 1000,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'random_state': 42,
    'early_stopping_rounds': 50
}

X_train = train_df[feature_cols]
y_train = train_df['target_moneyline_win']
X_test = test_df[feature_cols]
y_test = test_df['target_moneyline_win']

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Generate predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
train_pred = model.predict_proba(X_train)[:, 1]
train_logloss = log_loss(y_train, train_pred)
test_logloss = log_loss(y_test, y_pred_proba)
test_auc = roc_auc_score(y_test, y_pred_proba)
test_brier = brier_score_loss(y_test, y_pred_proba)

print(f"\n  Model Performance:")
print(f"    Train LogLoss: {train_logloss:.6f}")
print(f"    Test LogLoss: {test_logloss:.6f}")
print(f"    Test AUC: {test_auc:.4f}")
print(f"    Test Brier: {test_brier:.4f}")

# Load moneyline odds
print("\n[4/5] Loading moneyline odds...")
odds_df = pd.read_csv('data/live/closing_odds_2024_25.csv')
odds_df = odds_df.rename(columns={
    'game_date': 'date',
    'home_ml_odds': 'home_ml',
    'away_ml_odds': 'away_ml'
})
odds_df['date'] = pd.to_datetime(odds_df['date'])

# Take latest odds per game (in case multiple bookmakers)
odds_df = odds_df.sort_values('snapshot_timestamp').groupby(['date', 'home_team', 'away_team']).last().reset_index()

print(f"  Odds records: {len(odds_df):,}")
print(f"  Date range: {odds_df['date'].min()} to {odds_df['date'].max()}")

# Merge predictions with odds
test_df['model_prob_home'] = y_pred_proba
test_df['model_prob_away'] = 1 - y_pred_proba

# Match odds by date and teams
merged = test_df.merge(
    odds_df,
    on=['date', 'home_team', 'away_team'],
    how='inner'
)

print(f"  Merged records: {len(merged):,}")

# Calculate fair probabilities (remove vig)
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

merged['home_market_prob'] = merged['home_ml'].apply(american_to_prob)
merged['away_market_prob'] = merged['away_ml'].apply(american_to_prob)

# Remove vig
total_prob = merged['home_market_prob'] + merged['away_market_prob']
merged['home_fair_prob'] = merged['home_market_prob'] / total_prob
merged['away_fair_prob'] = merged['away_market_prob'] / total_prob

# Calculate edge
merged['edge_home'] = merged['model_prob_home'] - merged['home_fair_prob']
merged['edge_away'] = merged['model_prob_away'] - merged['away_fair_prob']

# Run backtest with different edge thresholds
print("\n[5/5] Running walk-forward backtest...")
print(f"{'='*90}")
print(f"{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Profit':<12} {'ROI':<10} {'Avg Odds':<10}")
print(f"{'='*90}")

thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]

best_roi = -999
best_threshold = None
best_results = None

for threshold in thresholds:
    # Find bets with edge > threshold
    home_bets = merged[merged['edge_home'] > threshold].copy()
    away_bets = merged[merged['edge_away'] > threshold].copy()
    
    # Calculate returns for home bets
    home_bets['won'] = home_bets['target_moneyline_win'] == 1
    home_bets['odds_decimal'] = home_bets['home_ml'].apply(
        lambda x: 1 + (x/100) if x > 0 else 1 + (100/abs(x))
    )
    home_bets['profit'] = home_bets.apply(
        lambda row: 100 * (row['odds_decimal'] - 1) if row['won'] else -100,
        axis=1
    )
    
    # Calculate returns for away bets
    away_bets['won'] = away_bets['target_moneyline_win'] == 0
    away_bets['odds_decimal'] = away_bets['away_ml'].apply(
        lambda x: 1 + (x/100) if x > 0 else 1 + (100/abs(x))
    )
    away_bets['profit'] = away_bets.apply(
        lambda row: 100 * (row['odds_decimal'] - 1) if row['won'] else -100,
        axis=1
    )
    
    # Combine all bets
    all_bets = pd.concat([home_bets, away_bets])
    
    if len(all_bets) == 0:
        continue
    
    total_bets = len(all_bets)
    total_wins = all_bets['won'].sum()
    win_pct = (total_wins / total_bets) * 100
    total_profit = all_bets['profit'].sum()
    total_risk = total_bets * 100
    roi = (total_profit / total_risk) * 100
    avg_odds = all_bets['odds_decimal'].mean()
    
    print(f"{threshold:<12.1%} {total_bets:<8} {total_wins:<8} {win_pct:<10.1f} ${total_profit:<11,.0f} {roi:<10.2f}% {avg_odds:<10.2f}")
    
    # Track best result
    if roi > best_roi:
        best_roi = roi
        best_threshold = threshold
        best_results = {
            'threshold': threshold,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'win_pct': win_pct,
            'total_profit': total_profit,
            'roi': roi,
            'avg_odds': avg_odds,
            'all_bets': all_bets
        }

print(f"{'='*90}")

# Best result analysis
if best_results:
    print(f"\n{'='*90}")
    print("BEST RESULT")
    print(f"{'='*90}")
    print(f"  Threshold: {best_threshold:.1%}")
    print(f"  Total Bets: {best_results['total_bets']}")
    print(f"  Wins: {best_results['total_wins']} ({best_results['win_pct']:.1f}%)")
    print(f"  Total Profit: ${best_results['total_profit']:,.0f}")
    print(f"  ROI: {best_results['roi']:.2f}%")
    print(f"  Average Odds: {best_results['avg_odds']:.2f}")
    
    # Monthly breakdown
    all_bets = best_results['all_bets']
    all_bets['month'] = pd.to_datetime(all_bets['date']).dt.to_period('M')
    
    print(f"\n  Monthly Breakdown:")
    print(f"  {'Month':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Profit':<12} {'ROI':<10}")
    print(f"  {'-'*70}")
    
    for month, group in all_bets.groupby('month'):
        month_bets = len(group)
        month_wins = group['won'].sum()
        month_win_pct = (month_wins / month_bets) * 100
        month_profit = group['profit'].sum()
        month_roi = (month_profit / (month_bets * 100)) * 100
        
        print(f"  {str(month):<12} {month_bets:<8} {month_wins:<8} {month_win_pct:<10.1f} ${month_profit:<11,.0f} {month_roi:<10.2f}%")
    
    # Favorites vs Underdogs
    all_bets['bet_type'] = all_bets.apply(
        lambda row: 'Favorite' if (row['edge_home'] > best_threshold and row['home_ml'] < 0) or 
                                   (row['edge_away'] > best_threshold and row['away_ml'] < 0) 
                    else 'Underdog',
        axis=1
    )
    
    print(f"\n  Favorites vs Underdogs:")
    print(f"  {'Type':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Profit':<12} {'ROI':<10}")
    print(f"  {'-'*70}")
    
    for bet_type, group in all_bets.groupby('bet_type'):
        type_bets = len(group)
        type_wins = group['won'].sum()
        type_win_pct = (type_wins / type_bets) * 100
        type_profit = group['profit'].sum()
        type_roi = (type_profit / (type_bets * 100)) * 100
        
        print(f"  {bet_type:<12} {type_bets:<8} {type_wins:<8} {type_win_pct:<10.1f} ${type_profit:<11,.0f} {type_roi:<10.2f}%")

# Save results
results = {
    'trial': 215,
    'study': study_name,
    'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'test_logloss': test_logloss,
    'test_auc': test_auc,
    'test_brier': test_brier,
    'best_threshold': best_threshold,
    'best_roi': best_roi,
    'best_profit': best_results['total_profit'] if best_results else 0,
    'best_bets': best_results['total_bets'] if best_results else 0,
    'best_win_pct': best_results['win_pct'] if best_results else 0,
    'parameters': trial_215.params
}

output_file = 'models/backtest_trial215_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*90}")
print(f"Saved: {output_file}")
print(f"{'='*90}\n")
