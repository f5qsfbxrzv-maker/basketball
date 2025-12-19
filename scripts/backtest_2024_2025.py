"""
Walk-forward backtest on 2024-25 and 2025-26 seasons
- Train on all data before 2024-25
- Test on 2024-25 and 2025-26 completed games
- Flat unit bets with edge threshold
- Calculate ROI
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from datetime import datetime

print("\n" + "="*80)
print("WALK-FORWARD BACKTEST: 2024-25 & 2025-26 SEASONS")
print("="*80)

# Load data
print("\n[1/6] Loading data...")
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

# Split: Train on everything before 2024-25, Test on 2024-25 and 2025-26
train_cutoff = pd.to_datetime('2024-10-01')
df_train = df[df['date'] < train_cutoff].copy()
df_test = df[df['date'] >= train_cutoff].copy()

X_train = df_train[feature_cols]
y_train = df_train['target_spread_cover']
X_test = df_test[feature_cols]
y_test = df_test['target_spread_cover']

print(f"\n[2/6] Train/Test Split:")
print(f"  Train: {len(df_train):,} games (through Sep 2024)")
print(f"    Date range: {df_train['date'].min().date()} to {df_train['date'].max().date()}")
print(f"    Covers: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Test:  {len(df_test):,} games (2024-25 & 2025-26)")
print(f"    Date range: {df_test['date'].min().date()} to {df_test['date'].max().date()}")
print(f"    Covers: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

# Train model with best params from Trial 306
print("\n[3/6] Training XGBoost with Trial 306 params...")
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

# Get raw predictions
train_probs_raw = model.predict_proba(X_train)[:, 1]
test_probs_raw = model.predict_proba(X_test)[:, 1]

print("  ✓ Model trained")
print(f"    Train AUC: {roc_auc_score(y_train, train_probs_raw):.5f}")
print(f"    Test AUC:  {roc_auc_score(y_test, test_probs_raw):.5f}")

# Calibrate using isotonic regression
print("\n[4/6] Calibrating probabilities...")
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(train_probs_raw, y_train)

train_probs_cal = iso.transform(train_probs_raw)
test_probs_cal = iso.transform(test_probs_raw)

train_brier_raw = brier_score_loss(y_train, train_probs_raw)
train_brier_cal = brier_score_loss(y_train, train_probs_cal)
test_brier_raw = brier_score_loss(y_test, test_probs_raw)
test_brier_cal = brier_score_loss(y_test, test_probs_cal)

print("  ✓ Calibrated")
print(f"    Train Brier: {train_brier_raw:.5f} → {train_brier_cal:.5f} (Δ {(train_brier_raw - train_brier_cal)*100:+.2f}%)")
print(f"    Test Brier:  {test_brier_raw:.5f} → {test_brier_cal:.5f} (Δ {(test_brier_raw - test_brier_cal)*100:+.2f}%)")

# Betting simulation
print("\n[5/6] Simulating flat unit betting...")

# Add predictions to test dataframe
df_test['prob_cover'] = test_probs_cal
df_test['prob_not_cover'] = 1 - test_probs_cal

# Betting parameters
UNIT_SIZE = 100  # $100 flat bets
MIN_EDGE = 0.03  # 3% minimum edge
TYPICAL_ODDS = 1.91  # -110 odds (implied prob ~52.4%)
COMMISSION = 0.048  # Kalshi 4.8% commission on winnings

# Calculate edge and bet sizing
bets = []
for idx, row in df_test.iterrows():
    # Fair probability after removing vig
    # Assume typical -110 both sides (52.4% implied each)
    fair_prob_cover = 0.50  # Vig-removed fair odds
    
    # Edge = our probability - market probability
    edge_cover = row['prob_cover'] - (1 - fair_prob_cover)
    edge_not_cover = row['prob_not_cover'] - fair_prob_cover
    
    # Bet on side with positive edge > threshold
    if edge_cover > MIN_EDGE:
        bet_side = 'cover'
        edge = edge_cover
        win_prob = row['prob_cover']
        actual_result = row['target_spread_cover']
    elif edge_not_cover > MIN_EDGE:
        bet_side = 'not_cover'
        edge = edge_not_cover
        win_prob = row['prob_not_cover']
        actual_result = not row['target_spread_cover']
    else:
        continue  # No bet
    
    # Calculate profit/loss
    if actual_result:
        # Win: get back stake + profit, minus commission on profit
        profit = UNIT_SIZE * 0.91  # Win $91 on $100 bet at -110
        profit_after_commission = profit * (1 - COMMISSION)
        pnl = profit_after_commission
    else:
        # Loss: lose stake
        pnl = -UNIT_SIZE
    
    bets.append({
        'date': row['date'],
        'game': f"{row['away_team']} @ {row['home_team']}",
        'bet_side': bet_side,
        'edge': edge,
        'win_prob': win_prob,
        'stake': UNIT_SIZE,
        'result': 'WIN' if actual_result else 'LOSS',
        'pnl': pnl
    })

bets_df = pd.DataFrame(bets)

# Results
print(f"\n{'='*80}")
print("BACKTEST RESULTS")
print(f"{'='*80}")

if len(bets_df) > 0:
    total_bets = len(bets_df)
    total_staked = total_bets * UNIT_SIZE
    total_wins = (bets_df['result'] == 'WIN').sum()
    total_losses = (bets_df['result'] == 'LOSS').sum()
    win_rate = total_wins / total_bets
    
    total_pnl = bets_df['pnl'].sum()
    roi = (total_pnl / total_staked) * 100
    
    avg_edge = bets_df['edge'].mean()
    avg_win_prob = bets_df['win_prob'].mean()
    
    print(f"\nBetting Summary:")
    print(f"  Total games available: {len(df_test):,}")
    print(f"  Bets placed: {total_bets} ({total_bets/len(df_test)*100:.1f}% of games)")
    print(f"  Total staked: ${total_staked:,.0f}")
    print(f"\nPerformance:")
    print(f"  Wins: {total_wins}")
    print(f"  Losses: {total_losses}")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  Average edge: {avg_edge*100:.2f}%")
    print(f"  Average win probability: {avg_win_prob*100:.1f}%")
    print(f"\nResults:")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"  ROI: {roi:+.2f}%")
    
    # Monthly breakdown
    print(f"\n{'='*80}")
    print("MONTHLY BREAKDOWN")
    print(f"{'='*80}")
    bets_df['month'] = pd.to_datetime(bets_df['date']).dt.to_period('M')
    monthly = bets_df.groupby('month').agg({
        'stake': 'count',
        'pnl': 'sum',
        'result': lambda x: (x == 'WIN').sum()
    }).rename(columns={'stake': 'bets', 'result': 'wins'})
    monthly['win_rate'] = monthly['wins'] / monthly['bets']
    monthly['roi'] = (monthly['pnl'] / (monthly['bets'] * UNIT_SIZE)) * 100
    
    print(f"\n{'Month':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'P&L':<12} {'ROI%':<10}")
    print("-"*80)
    for month, row in monthly.iterrows():
        print(f"{str(month):<12} {row['bets']:<8.0f} {row['wins']:<8.0f} {row['win_rate']*100:<10.1f} ${row['pnl']:<11,.2f} {row['roi']:<+10.2f}")
    
    # Save results
    bets_df.to_csv('models/backtest_2024_2025_results.csv', index=False)
    print(f"\n[6/6] Saved detailed results: models/backtest_2024_2025_results.csv")
    
else:
    print("\n  WARNING: No bets met minimum edge threshold")
    print(f"  Consider lowering MIN_EDGE (currently {MIN_EDGE*100:.1f}%)")

print(f"\n{'='*80}")
