"""
Walk-forward backtest: 2024-2025 season with flat unit betting
Train on all data before 2024-25, test on 2024-25 games with calibrated predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from datetime import datetime

print("="*70)
print("WALK-FORWARD BACKTEST: 2024-2025 SEASON")
print("="*70)

# Load data
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Split into train (before 2024-25) and test (2024-25 season)
train_df = df[df['season'] < '2024-25'].copy()
test_df = df[df['season'] == '2024-25'].copy()

print(f"\nData split:")
print(f"  Train: {len(train_df):,} games (seasons < 2024-25)")
print(f"  Train date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
print(f"  Test:  {len(test_df):,} games (2024-25 season)")
print(f"  Test date range: {test_df['date'].min().date()} to {test_df['date'].max().date()}")

# Prepare features
drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in drop_cols]

X_train = train_df[features]
y_train = train_df['target_spread_cover']
X_test = test_df[features]
y_test = test_df['target_spread_cover']

# Load best params from Trial 98
import json
with open('models/single_fold_best_params.json', 'r') as f:
    config = json.load(f)

params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'random_state': 42,
    'verbosity': 0,
    **config['best_params']
}

print("\n" + "="*70)
print("TRAINING MODEL ON PRE-2024-25 DATA")
print("="*70)

# Train XGBoost
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, verbose=False)

# Train isotonic calibrator
raw_train_probs = model.predict_proba(X_train)[:, 1]
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_train_probs, y_train)

print(f"✓ Model trained on {len(X_train):,} games")

# Predict on 2024-25 season
print("\n" + "="*70)
print("PREDICTING 2024-25 SEASON")
print("="*70)

raw_test_probs = model.predict_proba(X_test)[:, 1]
cal_test_probs = iso.transform(raw_test_probs)

# Performance metrics
raw_auc = roc_auc_score(y_test, raw_test_probs)
cal_auc = roc_auc_score(y_test, cal_test_probs)
raw_brier = brier_score_loss(y_test, raw_test_probs)
cal_brier = brier_score_loss(y_test, cal_test_probs)
accuracy = accuracy_score(y_test, (cal_test_probs >= 0.5).astype(int))

print(f"\nPrediction Performance:")
print(f"  Raw AUC:        {raw_auc:.4f}")
print(f"  Calibrated AUC: {cal_auc:.4f}")
print(f"  Raw Brier:      {raw_brier:.4f}")
print(f"  Cal Brier:      {cal_brier:.4f} ({(raw_brier - cal_brier)*100:+.2f}%)")
print(f"  Accuracy:       {accuracy:.2%}")

# Betting simulation - flat 1 unit per bet with edge threshold
print("\n" + "="*70)
print("FLAT UNIT BETTING SIMULATION (REAL CLOSING LINES)")
print("="*70)

# Market implied probability from closing spread
# Standard -110 odds on both sides = 52.38% implied per side (with vig)
# We'll compare our calibrated prob vs 50% fair (vig-removed)
print(f"\nUsing actual target_spread (closing lines) as market reference")
print(f"Betting: Home cover when model > 50%, Away cover when model < 50%")

# Simulate betting with edge thresholds
edge_thresholds = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]

results = []
for threshold in edge_thresholds:
    # Edge = distance from 50% (vig-removed fair probability)
    fair_prob = 0.5
    edge = cal_test_probs - fair_prob
    
    # Bet when edge exceeds threshold
    bets = np.abs(edge) >= threshold
    n_bets = bets.sum()
    
    if n_bets == 0:
        continue
    
    # Flat 1 unit per bet
    # Actual outcome: target_spread_cover (1 = home covered, 0 = away covered)
    # Our bet: Home cover when cal_prob >= 0.5, Away cover when cal_prob < 0.5
    bet_home_cover = cal_test_probs >= 0.5
    actual_home_covered = y_test.values == 1
    
    # Win if our prediction matches actual outcome
    correct = bet_home_cover[bets] == actual_home_covered[bets]
    wins = correct.sum()
    losses = n_bets - wins
    
    # Calculate P&L (assuming -110 odds both sides)
    # Win: +0.909 units (risk 1.1 to win 1)
    # Loss: -1.1 units
    win_payout = 0.909
    loss_cost = 1.1
    
    profit = wins * win_payout - losses * loss_cost
    roi = (profit / (n_bets * loss_cost)) * 100
    win_rate = (wins / n_bets) * 100
    
    results.append({
        'threshold': threshold,
        'n_bets': int(n_bets),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(win_rate),
        'profit': float(profit),
        'roi': float(roi)
    })

print(f"\nEdge Threshold Analysis (-110 odds, real closing spreads):")
print(f"{'Threshold':<12} {'Bets':<8} {'W-L':<12} {'Win%':<8} {'Profit':<10} {'ROI%':<8}")
print("-" * 70)

for r in results:
    print(f"{r['threshold']:.2f}          {r['n_bets']:<8} {r['wins']}-{r['losses']:<10} {r['win_rate']:.1f}%    {r['profit']:+.1f}u      {r['roi']:+.1f}%")

# Best threshold
best = max(results, key=lambda x: x['profit'])
print(f"\nBest Performance:")
print(f"  Edge threshold: {best['threshold']:.2f}")
print(f"  Bets placed:    {best['n_bets']}")
print(f"  Record:         {best['wins']}-{best['losses']}")
print(f"  Win rate:       {best['win_rate']:.1f}%")
print(f"  Total profit:   {best['profit']:+.1f} units")
print(f"  ROI:            {best['roi']:+.1f}%")

# Detailed bet analysis for best threshold
print("\n" + "="*70)
print(f"DETAILED BET LOG (Edge >= {best['threshold']:.2f})")
print("="*70)

bets_mask = np.abs(edge) >= best['threshold']
bet_results = test_df[bets_mask].copy()
bet_results['predicted_prob'] = cal_test_probs[bets_mask]
bet_results['edge'] = edge[bets_mask]
bet_results['bet_side'] = pd.Series(cal_test_probs[bets_mask] >= 0.5).map({True: 'HOME', False: 'AWAY'}).values
bet_results['actual_cover'] = pd.Series(y_test[bets_mask].values).map({1: 'HOME', 0: 'AWAY'}).values
bet_results['correct'] = bet_results['bet_side'] == bet_results['actual_cover']
bet_results['result'] = bet_results['correct'].map({True: 'WIN', False: 'LOSS'})
bet_results['profit'] = bet_results['correct'].map({True: 0.909, False: -1.1})
bet_results['cumulative'] = bet_results['profit'].cumsum()

print(f"\nFirst 20 bets:")
print(f"{'Date':<12} {'Home':<20} {'Away':<20} {'Spread':<7} {'Bet':<5} {'Pred':<7} {'Result':<6} {'P/L':<7} {'Cum':<7}")
print("-" * 120)

for idx, row in bet_results.head(20).iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['home_team'][:19]:<20} {row['away_team'][:19]:<20} "
          f"{row['target_spread']:+6.1f}  {row['bet_side']:<5} {row['predicted_prob']:.3f}   "
          f"{row['result']:<6} {row['profit']:+.2f}u    {row['cumulative']:+.2f}u")

# Save backtest results
backtest_summary = {
    'backtest_date': datetime.now().isoformat(),
    'test_season': '2024-25',
    'train_games': len(X_train),
    'test_games': len(X_test),
    'test_date_range': [test_df['date'].min().isoformat(), test_df['date'].max().isoformat()],
    'performance': {
        'raw_auc': float(raw_auc),
        'calibrated_auc': float(cal_auc),
        'raw_brier': float(raw_brier),
        'calibrated_brier': float(cal_brier),
        'accuracy': float(accuracy)
    },
    'betting_results': results,
    'best_threshold': best
}

with open('models/backtest_2024_25.json', 'w') as f:
    json.dump(backtest_summary, f, indent=2)

print(f"\n✓ Backtest results saved: models/backtest_2024_25.json")

print("\n" + "="*70)
print("✓ WALK-FORWARD BACKTEST COMPLETE")
print("="*70)
