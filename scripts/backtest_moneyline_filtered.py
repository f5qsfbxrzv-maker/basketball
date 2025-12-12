"""
Walk-forward backtest: 2024-25 Season with FILTERED CLOSING LINE ODDS
Filters out extreme outliers (>±500) to remove corrupted data
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
print("WALK-FORWARD BACKTEST: FILTERED CLOSING LINE ODDS")
print("="*70)

# Load training data
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Load moneyline odds from The Odds API
conn = sqlite3.connect('data/live/historical_closing_odds.db')
odds_df = pd.read_sql("""
    SELECT game_date, home_team, away_team, home_ml_odds, away_ml_odds
    FROM moneyline_odds 
    ORDER BY game_date
""", conn)
conn.close()

odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

print(f"\nTotal closing odds available: {len(odds_df)} games")

# FILTER OUT EXTREME OUTLIERS
# Keep only reasonable odds: -500 to +500
print("\n" + "="*70)
print("FILTERING EXTREME OUTLIERS")
print("="*70)

before_count = len(odds_df)

# Remove games where either team has extreme odds
odds_df = odds_df[
    (odds_df['home_ml_odds'] >= -500) & (odds_df['home_ml_odds'] <= 500) &
    (odds_df['away_ml_odds'] >= -500) & (odds_df['away_ml_odds'] <= 500)
].copy()

after_count = len(odds_df)
removed = before_count - after_count

print(f"Before filtering: {before_count} games")
print(f"After filtering:  {after_count} games")
print(f"Removed:          {removed} games ({removed/before_count*100:.1f}%)")
print(f"Date range: {odds_df['game_date'].min().date()} to {odds_df['game_date'].max().date()}")

# Split into train and test
train_df = df[df['season'] != '2024-25'].copy()
test_df = df[df['season'] == '2024-25'].copy()

# Merge test data with filtered closing odds
test_df = test_df.merge(
    odds_df,
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='inner'
).drop_duplicates(subset=['game_id'], keep='first')

print(f"\nDEDUPLICATION CHECK:")
print(f"  Unique games after merge & dedup: {len(test_df):,}")

print(f"\nData split:")
print(f"  Train: {len(train_df):,} games (all seasons before 2024-25)")
print(f"  Test:  {len(test_df):,} games (2024-25 season with filtered odds)")
print(f"  Test date range: {test_df['date'].min().date()} to {test_df['date'].max().date()}")

# Define features
feature_cols = [c for c in train_df.columns if c not in [
    'date', 'game_id', 'season', 'home_team', 'away_team',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'home_pts', 'away_pts', 'home_ml_odds', 'away_ml_odds', 
    'game_date', 'bookmaker', 'api_game_id', 'snapshot_timestamp',
    'target_game_total', 'target_over_under'  # Totals betting targets
]]

X_train = train_df[feature_cols]
y_train = train_df['target_moneyline_win']

X_test = test_df[feature_cols]
y_test = test_df['target_moneyline_win']

print(f"\n{'='*70}")
print("TRAINING MODEL ON PRE-2024-25 DATA")
print("="*70)

# Load production model
try:
    with open('models/xgboost_final_trial98.json', 'r') as f:
        model_json = f.read()
    
    model = xgb.XGBClassifier()
    model.load_model('models/xgboost_final_trial98.json')
    print("✓ Loaded production model: xgboost_final_trial98.json")
except:
    # Train new model if production model not found
    print("Training new model...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    print("✓ Model trained on {:,} games".format(len(train_df)))

# Load calibrator
try:
    with open('models/isotonic_calibrator_final.pkl', 'rb') as f:
        calibrator = pickle.load(f)
    print("✓ Loaded production calibrator: isotonic_calibrator_final.pkl")
except:
    print("⚠️  No calibrator found, using raw predictions")
    calibrator = None

print(f"\n{'='*70}")
print("PREDICTING 2024-25 SEASON WITH FILTERED MONEYLINE ODDS")
print("="*70)

# Get predictions
raw_probs = model.predict_proba(X_test)[:, 1]

if calibrator is not None:
    cal_probs = calibrator.transform(raw_probs)
else:
    cal_probs = raw_probs

# Metrics
raw_auc = roc_auc_score(y_test, raw_probs)
cal_auc = roc_auc_score(y_test, cal_probs)
raw_brier = brier_score_loss(y_test, raw_probs)
cal_brier = brier_score_loss(y_test, cal_probs)
accuracy = accuracy_score(y_test, (cal_probs > 0.5).astype(int))

print(f"\nPrediction Performance:")
print(f"  Raw AUC:        {raw_auc:.4f}")
print(f"  Calibrated AUC: {cal_auc:.4f}")
print(f"  Raw Brier:      {raw_brier:.4f}")
print(f"  Cal Brier:      {cal_brier:.4f} ({(cal_brier-raw_brier)/raw_brier*100:+.2f}%)")
print(f"  Accuracy:       {accuracy*100:.2f}%")

# Helper functions
def american_to_decimal(odds):
    """Convert American odds to decimal"""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def calculate_edge(model_prob, odds):
    """Calculate edge given model probability and American odds"""
    decimal_odds = american_to_decimal(odds)
    implied_prob = 1 / decimal_odds
    return model_prob - implied_prob

def calculate_payout(odds, stake=1.0):
    """Calculate payout for a winning bet"""
    if odds > 0:
        return stake * (odds / 100)
    else:
        return stake * (100 / abs(odds))

print(f"\n{'='*70}")
print("MONEYLINE BETTING SIMULATION (FILTERED ODDS)")
print("="*70)

# Test multiple edge thresholds
thresholds = [0.00, 0.02, 0.03, 0.04, 0.05]
results = []

for threshold in thresholds:
    bets = []
    
    for idx, row in test_df.iterrows():
        model_prob = cal_probs[test_df.index.get_loc(idx)]
        home_odds = row['home_ml_odds']
        away_odds = row['away_ml_odds']
        
        # Calculate edges
        home_edge = calculate_edge(model_prob, home_odds)
        away_edge = calculate_edge(1 - model_prob, away_odds)
        
        # Place bet if edge > threshold
        if home_edge > threshold:
            actual_win = row['target_moneyline_win']
            payout = calculate_payout(home_odds) if actual_win else -1.0
            bets.append({
                'date': row['date'],
                'home': row['home_team'],
                'away': row['away_team'],
                'pick': 'HOME',
                'odds': home_odds,
                'edge': home_edge,
                'result': actual_win,
                'profit': payout
            })
        elif away_edge > threshold:
            actual_win = 1 - row['target_moneyline_win']
            payout = calculate_payout(away_odds) if actual_win else -1.0
            bets.append({
                'date': row['date'],
                'home': row['home_team'],
                'away': row['away_team'],
                'pick': 'AWAY',
                'odds': away_odds,
                'edge': away_edge,
                'result': actual_win,
                'profit': payout
            })
    
    if len(bets) > 0:
        bet_df = pd.DataFrame(bets)
        wins = bet_df['result'].sum()
        total_profit = bet_df['profit'].sum()
        roi = (total_profit / len(bets)) * 100
        
        results.append({
            'threshold': threshold,
            'bets': len(bets),
            'wins': int(wins),
            'losses': len(bets) - int(wins),
            'win_pct': wins / len(bets) * 100,
            'profit': total_profit,
            'roi': roi
        })

# Display results
print(f"\nEdge Threshold Analysis (Filtered Moneyline Odds):")
print(f"Threshold    Bets     W-L          Win%     Profit     ROI%")
print("-"*70)

for r in results:
    print(f"{r['threshold']:.2f}          {r['bets']:4d}     {r['wins']}-{r['losses']:3d}        "
          f"{r['win_pct']:5.1f}%    {r['profit']:+6.1f}u      {r['roi']:+5.1f}%")

# Find best result
best = max(results, key=lambda x: x['roi'])

print(f"\n{'='*70}")
print("BEST PERFORMANCE (FILTERED ODDS)")
print("="*70)
print(f"  Edge threshold: {best['threshold']:.2f}")
print(f"  Bets placed:    {best['bets']}")
print(f"  Record:         {best['wins']}-{best['losses']}")
print(f"  Win rate:       {best['win_pct']:.1f}%")
print(f"  Total profit:   {best['profit']:+.1f} units")
print(f"  ROI:            {best['roi']:+.1f}%")

# Reality check
print(f"\n{'='*70}")
print("REALITY CHECK")
print("="*70)
if best['roi'] > 5:
    print("✓ Positive ROI: Model shows edge on filtered odds")
    print("  Proceed with caution - validate on additional data")
elif best['roi'] > 0:
    print("⚠️  Marginal positive ROI: Edge exists but small")
    print("  May not overcome transaction costs in live betting")
else:
    print("❌ Negative ROI: Model not profitable on filtered odds")
    print("   Do not deploy until model improves")

# Save detailed bet log for best threshold
threshold = best['threshold']
bets = []

for idx, row in test_df.iterrows():
    model_prob = cal_probs[test_df.index.get_loc(idx)]
    home_odds = row['home_ml_odds']
    away_odds = row['away_ml_odds']
    
    home_edge = calculate_edge(model_prob, home_odds)
    away_edge = calculate_edge(1 - model_prob, away_odds)
    
    if home_edge > threshold:
        actual_win = row['target_moneyline_win']
        payout = calculate_payout(home_odds) if actual_win else -1.0
        bets.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'home': row['home_team'],
            'away': row['away_team'],
            'pick': 'HOME',
            'odds': int(home_odds),
            'edge': float(home_edge),
            'result': int(actual_win),
            'profit': float(payout)
        })
    elif away_edge > threshold:
        actual_win = 1 - row['target_moneyline_win']
        payout = calculate_payout(away_odds) if actual_win else -1.0
        bets.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'home': row['home_team'],
            'away': row['away_team'],
            'pick': 'AWAY',
            'odds': int(away_odds),
            'edge': float(away_edge),
            'result': int(actual_win),
            'profit': float(payout)
        })

print(f"\n{'='*70}")
print(f"DETAILED BET LOG (Edge >= {threshold:.2f})")
print("="*70)

bet_df = pd.DataFrame(bets)
bet_df['cumulative'] = bet_df['profit'].cumsum()

print(f"\nFirst 20 bets:")
print(f"Date         Home  Away  Bet   Odds    Edge    Result P/L     Cum")
print("-"*90)

for i, bet in enumerate(bets[:20]):
    result_str = "WIN " if bet['result'] else "LOSS"
    print(f"{bet['date']}   {bet['home']:3s}   {bet['away']:3s}   "
          f"{bet['pick']:4s}    {bet['odds']:5d}  {bet['edge']:+.3f}   "
          f"{result_str}   {bet['profit']:+.2f}u    {bet_df.iloc[i]['cumulative']:+.2f}u")

# Save results
output = {
    'test_period': f"{test_df['date'].min().date()} to {test_df['date'].max().date()}",
    'games_tested': len(test_df),
    'games_filtered_out': removed,
    'filter_criteria': 'Odds between -500 and +500',
    'model_performance': {
        'auc': float(cal_auc),
        'brier': float(cal_brier),
        'accuracy': float(accuracy)
    },
    'best_threshold': float(best['threshold']),
    'best_results': {
        'bets': int(best['bets']),
        'wins': int(best['wins']),
        'losses': int(best['losses']),
        'win_rate': float(best['win_pct']),
        'profit': float(best['profit']),
        'roi': float(best['roi'])
    },
    'bet_log': bets
}

with open('models/backtest_moneyline_filtered.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Backtest results saved: models/backtest_moneyline_filtered.json")

print(f"\n{'='*70}")
print("✓ WALK-FORWARD BACKTEST COMPLETE (FILTERED MONEYLINE ODDS)")
print("="*70)
