"""
Walk-forward backtest using MONEYLINE odds
- Predict moneyline winners using spread cover model as proxy
- Use real historical moneyline odds from closing_odds_2024_25.csv
- Calculate true ROI based on market prices
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.isotonic import IsotonicRegression
from datetime import datetime

print("\n" + "="*90)
print("WALK-FORWARD BACKTEST - MONEYLINE BETTING")
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

# Split train/test
train_cutoff = pd.to_datetime('2024-10-01')
df_train = df[df['date'] < train_cutoff].copy()
df_test = df[df['date'] >= train_cutoff].copy()

X_train = df_train[feature_cols]
y_train = df_train['target_moneyline_win']  # Use moneyline target
X_test = df_test[feature_cols]
y_test = df_test['target_moneyline_win']

print(f"  Train: {len(df_train):,} games (through Sep 2024)")
print(f"  Test:  {len(df_test):,} games (2024-25 & 2025-26)")
print(f"  Train home win rate: {y_train.mean()*100:.1f}%")
print(f"  Test home win rate: {y_test.mean()*100:.1f}%")

# Load moneyline odds
print("\n[2/7] Loading moneyline odds...")
odds_df = pd.read_csv('data/live/closing_odds_2024_25.csv')
odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

print(f"  Loaded odds for {len(odds_df):,} games")
print(f"  Date range: {odds_df['game_date'].min().date()} to {odds_df['game_date'].max().date()}")

# Merge odds with test data
print("\n[3/7] Merging odds with predictions...")
df_test = df_test.merge(
    odds_df[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='left'
)

n_with_odds = df_test['home_ml_odds'].notna().sum()
print(f"  Matched odds: {n_with_odds} of {len(df_test)} games ({n_with_odds/len(df_test)*100:.1f}%)")

# Filter to games with odds
df_test = df_test[df_test['home_ml_odds'].notna()].copy()
X_test = df_test[feature_cols]
y_test = df_test['target_moneyline_win']

print(f"  Test set with odds: {len(df_test):,} games")

# Train model
print("\n[4/7] Training XGBoost for moneyline prediction...")
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

train_probs_raw = model.predict_proba(X_train)[:, 1]
test_probs_raw = model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, train_probs_raw > 0.5)
test_acc = accuracy_score(y_test, test_probs_raw > 0.5)

print(f"  Train Accuracy: {train_acc*100:.2f}%")
print(f"  Test Accuracy:  {test_acc*100:.2f}%")
print(f"  Train AUC: {roc_auc_score(y_train, train_probs_raw):.5f}")
print(f"  Test AUC:  {roc_auc_score(y_test, test_probs_raw):.5f}")

# Calibrate
print("\n[5/7] Calibrating probabilities...")
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(train_probs_raw, y_train)
test_probs_cal = iso.transform(test_probs_raw)

df_test['prob_home_win'] = test_probs_cal
df_test['prob_away_win'] = 1 - test_probs_cal

# Convert American odds to decimal and probabilities
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

# Betting simulation
print("\n[6/7] Simulating moneyline betting...")

UNIT_SIZE = 100
MIN_EDGE = 0.03
COMMISSION = 0.048  # Kalshi commission

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
    edge_home = row['prob_home_win'] - fair_prob_home
    edge_away = row['prob_away_win'] - fair_prob_away
    
    # Bet on side with positive edge
    if edge_home > MIN_EDGE:
        bet_side = 'home'
        edge = edge_home
        win_prob = row['prob_home_win']
        odds_decimal = row['home_ml_decimal']
        odds_american = row['home_ml_odds']
        actual_result = row['target_moneyline_win']
    elif edge_away > MIN_EDGE:
        bet_side = 'away'
        edge = edge_away
        win_prob = row['prob_away_win']
        odds_decimal = row['away_ml_decimal']
        odds_american = row['away_ml_odds']
        actual_result = not row['target_moneyline_win']
    else:
        continue  # No bet
    
    # Calculate P&L with real odds
    if actual_result:
        # Win: profit = stake * (odds - 1)
        profit = UNIT_SIZE * (odds_decimal - 1)
        profit_after_commission = profit * (1 - COMMISSION)
        pnl = profit_after_commission
    else:
        # Loss
        pnl = -UNIT_SIZE
    
    bets.append({
        'date': row['date'],
        'game': f"{row['away_team']} @ {row['home_team']}",
        'bet_side': bet_side,
        'bet_team': row['home_team'] if bet_side == 'home' else row['away_team'],
        'edge': edge,
        'win_prob': win_prob,
        'odds_american': odds_american,
        'odds_decimal': odds_decimal,
        'stake': UNIT_SIZE,
        'result': 'WIN' if actual_result else 'LOSS',
        'pnl': pnl
    })

bets_df = pd.DataFrame(bets)

# Results
print(f"\n{'='*90}")
print("BACKTEST RESULTS - MONEYLINE BETTING WITH REAL ODDS")
print(f"{'='*90}")

if len(bets_df) > 0:
    total_bets = len(bets_df)
    total_staked = total_bets * UNIT_SIZE
    total_wins = (bets_df['result'] == 'WIN').sum()
    total_losses = (bets_df['result'] == 'LOSS').sum()
    win_rate = total_wins / total_bets
    
    total_pnl = bets_df['pnl'].sum()
    roi = (total_pnl / total_staked) * 100
    
    avg_edge = bets_df['edge'].mean()
    avg_odds = bets_df['odds_decimal'].mean()
    avg_odds_american = bets_df['odds_american'].mean()
    
    print(f"\nBetting Summary:")
    print(f"  Total games available: {len(df_test):,}")
    print(f"  Bets placed: {total_bets} ({total_bets/len(df_test)*100:.1f}% of games)")
    print(f"  Total staked: ${total_staked:,.0f}")
    print(f"\nPerformance:")
    print(f"  Wins: {total_wins}")
    print(f"  Losses: {total_losses}")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  Average edge: {avg_edge*100:.2f}%")
    print(f"  Average odds: {avg_odds:.3f} decimal ({avg_odds_american:+.0f} American)")
    print(f"\nResults:")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"  ROI: {roi:+.2f}%")
    
    # Analyze by bet type (favorite vs underdog)
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
    
    # Sample bets
    print(f"\n{'='*90}")
    print("SAMPLE BETS (First 10)")
    print(f"{'='*90}")
    print(f"\n{'Date':<12} {'Game':<25} {'Bet':<8} {'Odds':<8} {'Result':<8} {'P&L':<10}")
    print("-"*90)
    for idx, row in bets_df.head(10).iterrows():
        odds_str = f"{row['odds_american']:+.0f}"
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['game'][:25]:<25} {row['bet_team']:<8} {odds_str:<8} {row['result']:<8} ${row['pnl']:<9.2f}")
    
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
        print(f"{str(month):<12} {row['bets']:<8.0f} {row['wins']:<8.0f} {row['win_rate']*100:<10.1f} ${row['pnl']:<11,.2f} {row['roi']:<+10.2f}")
    
    # Save results
    bets_df.to_csv('models/backtest_moneyline_results.csv', index=False)
    print(f"\n[7/7] Saved: models/backtest_moneyline_results.csv")
    
else:
    print("\n  WARNING: No bets met minimum edge threshold")

print(f"\n{'='*90}")
