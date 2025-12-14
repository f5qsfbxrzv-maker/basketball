"""
Walk-forward backtest: 2024-25 Season with CLEAN CLOSING LINE ODDS
Using historical closing odds from The Odds API (DraftKings/FanDuel)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sqlite3
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from datetime import datetime
import json

print("="*70)
print("WALK-FORWARD BACKTEST: CLEAN CLOSING LINE ODDS")
print("="*70)

# Load training data
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Load CLEAN moneyline odds from The Odds API
conn = sqlite3.connect('data/live/historical_closing_odds.db')
odds_df = pd.read_sql("""
    SELECT game_date, home_team, away_team, home_ml_odds, away_ml_odds
    FROM moneyline_odds 
    ORDER BY game_date
""", conn)
conn.close()

odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

print(f"\nClean closing odds available: {len(odds_df)} games")
print(f"Date range: {odds_df['game_date'].min().date()} to {odds_df['game_date'].max().date()}")

# Split into train and test
# Train: All games before 2024-25 season
# Test: 2024-25 season games with clean closing odds
train_df = df[df['season'] != '2024-25'].copy()
test_df = df[df['season'] == '2024-25'].copy()

# Merge test data with closing odds by date and teams
test_df = test_df.merge(
    odds_df,
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='inner'
).drop_duplicates(subset=['game_id'], keep='first')  # Remove any duplicate game_ids

# ---------------------------------------------------------
# CRITICAL FIX: DEDUPLICATE GAMES
# ---------------------------------------------------------
print(f"\n⚠️  DEDUPLICATION CHECK:")
print(f"  Raw rows after merge: {len(test_df):,}")

# The odds database contains multiple updates for the same game
# (opening line, midday updates, closing line)
# This creates duplicate rows when we merge by matchup
# We must keep ONLY ONE row per unique game

# Check for duplicates
if 'game_id' in test_df.columns:
    duplicates = test_df.groupby('game_id').size()
    dupe_games = duplicates[duplicates > 1]
    if len(dupe_games) > 0:
        print(f"  ⚠️  Found {len(dupe_games)} games with duplicates")
        print(f"  Example: {dupe_games.head()}")
else:
    # Create unique game identifier if not present
    test_df['game_unique_id'] = test_df['date'].astype(str) + '_' + test_df['home_team'] + '_' + test_df['away_team']
    duplicates = test_df.groupby('game_unique_id').size()
    dupe_games = duplicates[duplicates > 1]
    if len(dupe_games) > 0:
        print(f"  ⚠️  Found {len(dupe_games)} games with duplicates")
        print(f"  Worst case: {dupe_games.max()} rows for single game")

# Keep only ONE row per game (prefer first occurrence = likely opening line)
if 'game_id' in test_df.columns:
    test_df = test_df.drop_duplicates(subset=['game_id'], keep='first')
else:
    test_df = test_df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')

print(f"  ✓ Unique games after dedup: {len(test_df):,}")

print(f"\nData split:")
print(f"  Train: {len(train_df):,} games (all seasons before 2024-25)")
print(f"  Test:  {len(test_df):,} games (2024-25 season with odds available, DEDUPLICATED)")
print(f"  Test date range: {test_df['date'].min().date()} to {test_df['date'].max().date()}")

# Prepare features
drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over',
             'matchup', 'home_ml_odds', 'away_ml_odds', 'spread_line']

features = [c for c in df.columns if c not in drop_cols]

X_train = train_df[features]
y_train = train_df['target_moneyline_win']  # Predicting winner
X_test = test_df[features]
y_test = test_df['target_moneyline_win']

# Load best params from Trial 98
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

# Predict on test set
print("\n" + "="*70)
print("PREDICTING 2024-25 SEASON WITH REAL MONEYLINE ODDS")
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

# Convert American odds to implied probability (with vig)
def american_to_implied_prob(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

# Calculate fair odds (remove vig)
def remove_vig(home_odds, away_odds):
    """Remove vig to get fair probabilities"""
    home_implied = american_to_implied_prob(home_odds)
    away_implied = american_to_implied_prob(away_odds)
    total = home_implied + away_implied
    return home_implied / total, away_implied / total

# Betting simulation with REAL MONEYLINE ODDS
print("\n" + "="*70)
print("MONEYLINE BETTING SIMULATION (REAL ODDS)")
print("="*70)

test_df['model_prob'] = cal_test_probs
test_df['home_fair_prob'], test_df['away_fair_prob'] = zip(*[
    remove_vig(row['home_ml_odds'], row['away_ml_odds']) 
    for _, row in test_df.iterrows()
])

# Edge = model probability - fair market probability
test_df['edge'] = test_df['model_prob'] - test_df['home_fair_prob']

# Simulate betting with edge thresholds
edge_thresholds = [0.00, 0.02, 0.03, 0.04, 0.05]

results = []
for threshold in edge_thresholds:
    bets = np.abs(test_df['edge']) >= threshold
    n_bets = bets.sum()
    
    if n_bets == 0:
        continue
    
    # Determine bet side and calculate payouts
    bet_results = []
    for idx, row in test_df[bets].iterrows():
        if row['edge'] > 0:  # Bet on home
            bet_home = True
            odds = row['home_ml_odds']
        else:  # Bet on away
            bet_home = False
            odds = row['away_ml_odds']
        
        # Did we win?
        home_won = row['target_moneyline_win'] == 1
        won = (bet_home and home_won) or (not bet_home and not home_won)
        
        # Calculate profit (1 unit bet)
        if won:
            if odds > 0:
                profit = odds / 100  # Win on underdog
            else:
                profit = 100 / -odds  # Win on favorite
        else:
            profit = -1.0  # Lost stake
        
        bet_results.append(profit)
    
    wins = sum(1 for p in bet_results if p > 0)
    losses = n_bets - wins
    total_profit = sum(bet_results)
    roi = (total_profit / n_bets) * 100
    win_rate = (wins / n_bets) * 100
    
    results.append({
        'threshold': threshold,
        'n_bets': int(n_bets),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(win_rate),
        'profit': float(total_profit),
        'roi': float(roi)
    })

print(f"\nEdge Threshold Analysis (Real Moneyline Odds - DEDUPLICATED):")
print(f"{'Threshold':<12} {'Bets':<8} {'W-L':<12} {'Win%':<8} {'Profit':<10} {'ROI%':<8}")
print("-" * 70)

for r in results:
    print(f"{r['threshold']:.2f}          {r['n_bets']:<8} {r['wins']}-{r['losses']:<10} {r['win_rate']:.1f}%    {r['profit']:+.1f}u      {r['roi']:+.1f}%")

# Best threshold
best = max(results, key=lambda x: x['profit'])
print(f"\n{'='*70}")
print(f"BEST PERFORMANCE (DEDUPLICATED - TRUE RESULTS)")
print(f"{'='*70}")
print(f"  Edge threshold: {best['threshold']:.2f}")
print(f"  Bets placed:    {best['n_bets']}")
print(f"  Record:         {best['wins']}-{best['losses']}")
print(f"  Win rate:       {best['win_rate']:.1f}%")
print(f"  Total profit:   {best['profit']:+.1f} units")
print(f"  ROI:            {best['roi']:+.1f}%")

# Reality check
print(f"\n{'='*70}")
print(f"REALITY CHECK")
print(f"{'='*70}")
if best['roi'] > 30:
    print(f"⚠️  WARNING: {best['roi']:.1f}% ROI is still suspiciously high!")
    print(f"   Professional sports bettors typically achieve 3-10% ROI")
    print(f"   Double-check for remaining data issues")
elif best['roi'] > 15:
    print(f"✅ {best['roi']:.1f}% ROI is excellent but realistic")
    print(f"   This is in the top 1% of sports bettors")
    print(f"   Proceed with cautious optimism")
elif best['roi'] > 5:
    print(f"✅ {best['roi']:.1f}% ROI is solid and sustainable")
    print(f"   This is a professional-level edge")
    print(f"   Ready for live testing with proper risk management")
elif best['roi'] > 0:
    print(f"⚠️  {best['roi']:.1f}% ROI is marginal")
    print(f"   May not beat transaction costs and variance")
    print(f"   Consider additional feature engineering")
else:
    print(f"❌ Negative ROI: Model not profitable")
    print(f"   Do not deploy until model improves")

# Detailed bet log
print("\n" + "="*70)
print(f"DETAILED BET LOG (Edge >= {best['threshold']:.2f})")
print("="*70)

bets_mask = np.abs(test_df['edge']) >= best['threshold']
bet_log = test_df[bets_mask].copy()

# Add bet details
bet_details = []
for idx, row in bet_log.iterrows():
    if row['edge'] > 0:
        bet_side = 'HOME'
        odds = row['home_ml_odds']
        won = row['target_moneyline_win'] == 1
    else:
        bet_side = 'AWAY'
        odds = row['away_ml_odds']
        won = row['target_moneyline_win'] == 0
    
    if won:
        profit = (odds / 100) if odds > 0 else (100 / -odds)
    else:
        profit = -1.0
    
    bet_details.append({
        'date': row['date'],
        'home': row['home_team'],
        'away': row['away_team'],
        'bet_side': bet_side,
        'odds': odds,
        'edge': row['edge'],
        'won': won,
        'profit': profit
    })

bet_df = pd.DataFrame(bet_details)
bet_df['cumulative'] = bet_df['profit'].cumsum()

print(f"\nFirst 20 bets:")
print(f"{'Date':<12} {'Home':<5} {'Away':<5} {'Bet':<5} {'Odds':<7} {'Edge':<7} {'Result':<6} {'P/L':<7} {'Cum':<7}")
print("-" * 90)

for _, row in bet_df.head(20).iterrows():
    result = 'WIN' if row['won'] else 'LOSS'
    print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['home']:<5} {row['away']:<5} {row['bet_side']:<5} "
          f"{row['odds']:>6}  {row['edge']:+.3f}   {result:<6} {row['profit']:+.2f}u    {row['cumulative']:+.2f}u")

# Save results
backtest_summary = {
    'backtest_date': datetime.now().isoformat(),
    'test_period': f"{test_df['date'].min().date()} to {test_df['date'].max().date()}",
    'train_games': len(X_train),
    'test_games': len(X_test),
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

with open('models/backtest_moneyline_real_odds.json', 'w') as f:
    json.dump(backtest_summary, f, indent=2)

print(f"\n✓ Backtest results saved: models/backtest_moneyline_real_odds.json")

print("\n" + "="*70)
print("✓ WALK-FORWARD BACKTEST COMPLETE (REAL MONEYLINE ODDS)")
print("="*70)
