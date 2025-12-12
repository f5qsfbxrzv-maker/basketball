"""
Walk-Forward Backtest with Real Historical Odds
Tests model performance on actual betting lines from 2023-2024
"""
import sys
from pathlib import Path

# Add parent directory to path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import sqlite3
import pandas as pd
import numpy as np
import joblib
from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

print("="*80)
print("WALK-FORWARD BACKTEST - REAL ODDS")
print("="*80)

# Load model
model = joblib.load('models/xgboost_tuned.pkl')  # Using tuned model with conservative params
print(f"\nLoaded model with {len(FEATURE_WHITELIST)} features")

# Initialize feature calculator
calc = FeatureCalculatorV5()
print("Feature calculator ready")

# Load historical odds and game results
conn = sqlite3.connect('data/live/nba_betting_data.db')

# Get games with odds from 2023-2024
backtest_data = pd.read_sql("""
SELECT 
    ho.game_date,
    ho.home_team,
    ho.away_team,
    ho.spread_line,
    ho.spread_home_odds,
    ho.total_line,
    ho.over_odds,
    gr.home_score,
    gr.away_score,
    gr.home_won,
    gr.total_points,
    gr.point_differential
FROM historical_odds ho
INNER JOIN game_results gr 
    ON date(ho.game_date) = date(gr.game_date)
    AND ho.home_team = gr.home_team
    AND ho.away_team = gr.away_team
WHERE date(ho.game_date) >= '2023-01-01' 
  AND date(ho.game_date) < '2024-12-01'
  AND ho.spread_line IS NOT NULL
  AND ho.spread_line != 0
ORDER BY ho.game_date
""", conn)

conn.close()

print(f"\nLoaded {len(backtest_data)} games with odds and outcomes")
print(f"   Date range: {backtest_data['game_date'].min()} to {backtest_data['game_date'].max()}")

# Generate features for each game
print("\n" + "="*80)
print("GENERATING FEATURES")
print("="*80)

features_list = []
valid_games = []

for idx, row in backtest_data.iterrows():
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{len(backtest_data)} ({100*idx/len(backtest_data):.1f}%)", end='\r')
    
    try:
        features = calc.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            season='2023-24' if row['game_date'] < '2024-07-01' else '2024-25',
            game_date=str(row['game_date'])[:10]  # Convert to YYYY-MM-DD
        )
        
        if all(f in features for f in FEATURE_WHITELIST):
            features_list.append([features[f] for f in FEATURE_WHITELIST])
            valid_games.append(idx)
    except Exception as e:
        if idx < 5:  # Print first few errors for debugging
            print(f"\nError on game {idx}: {e}")
        pass

print(f"\nGenerated features for {len(features_list)} games")

# Create feature matrix - use MODEL'S feature order, not whitelist order
X = pd.DataFrame(features_list, columns=FEATURE_WHITELIST)

# Reorder to match model's expected feature order
model_features = model.get_booster().feature_names
X = X[model_features]

# Convert all columns to float (fix XGBoost dtype error)
X = X.astype(float)

backtest_subset = backtest_data.iloc[valid_games].reset_index(drop=True)

# Get model predictions
print("\n" + "="*80)
print("RUNNING PREDICTIONS")
print("="*80)

backtest_subset['model_home_win_prob'] = model.predict_proba(X)[:, 1]
backtest_subset['model_away_win_prob'] = 1 - backtest_subset['model_home_win_prob']

# Calculate implied probabilities from spread (using standard -110 vig)
def spread_to_prob(spread):
    """Convert spread to implied probability (home team to cover)"""
    # Rough approximation: each point ~2.5% probability
    base_prob = 0.50
    adjustment = spread * -0.025  # Negative spread (favorite) increases probability
    return max(0.05, min(0.95, base_prob + adjustment))

backtest_subset['market_home_prob'] = backtest_subset['spread_line'].apply(spread_to_prob)
backtest_subset['market_away_prob'] = 1 - backtest_subset['market_home_prob']

# Calculate edges
backtest_subset['home_edge'] = backtest_subset['model_home_win_prob'] - backtest_subset['market_home_prob']
backtest_subset['away_edge'] = backtest_subset['model_away_win_prob'] - backtest_subset['market_away_prob']

# Identify bets (edge > 3%)
MIN_EDGE = 0.03
backtest_subset['has_edge'] = (backtest_subset['home_edge'].abs() > MIN_EDGE) | (backtest_subset['away_edge'].abs() > MIN_EDGE)
backtest_subset['bet_home'] = backtest_subset['home_edge'] > MIN_EDGE
backtest_subset['bet_away'] = backtest_subset['away_edge'] > MIN_EDGE

# Calculate outcomes (betting spread)
backtest_subset['home_covered'] = backtest_subset['point_differential'] > backtest_subset['spread_line']
backtest_subset['away_covered'] = backtest_subset['point_differential'] < -backtest_subset['spread_line']
backtest_subset['push'] = (backtest_subset['point_differential'] == backtest_subset['spread_line']) | \
                           (backtest_subset['point_differential'] == -backtest_subset['spread_line'])

# Calculate bet results
def calc_bet_result(row):
    if row['bet_home']:
        if row['push']:
            return 0  # Push
        elif row['home_covered']:
            return (100 / 110)  # Win (risking 110 to win 100)
        else:
            return -1.0  # Loss
    elif row['bet_away']:
        if row['push']:
            return 0
        elif row['away_covered']:
            return (100 / 110)
        else:
            return -1.0
    else:
        return 0  # No bet

backtest_subset['bet_result'] = backtest_subset.apply(calc_bet_result, axis=1)
backtest_subset['bet_made'] = backtest_subset['bet_home'] | backtest_subset['bet_away']

# Results
print("\n" + "="*80)
print("BACKTEST RESULTS")
print("="*80)

total_games = len(backtest_subset)
games_with_edge = backtest_subset['has_edge'].sum()
bets_made = backtest_subset['bet_made'].sum()
bets_won = (backtest_subset['bet_result'] > 0).sum()
bets_lost = (backtest_subset['bet_result'] < 0).sum()
pushes = (backtest_subset['bet_result'] == 0).sum() - (total_games - bets_made)

print(f"\nTotal Games Analyzed: {total_games:,}")
print(f"Games with Edge (>3%): {games_with_edge:,} ({100*games_with_edge/total_games:.1f}%)")
print(f"Bets Made: {bets_made:,}")
print(f"  Wins: {bets_won:,}")
print(f"  Losses: {bets_lost:,}")
print(f"  Pushes: {pushes:,}")

if bets_made > 0:
    win_rate = bets_won / (bets_won + bets_lost) if (bets_won + bets_lost) > 0 else 0
    total_profit = backtest_subset['bet_result'].sum()
    roi = (total_profit / bets_made) * 100
    
    print(f"\nWin Rate: {win_rate:.1%}")
    print(f"Total Profit: {total_profit:+.2f} units")
    print(f"ROI: {roi:+.2f}%")
    
    # Monthly breakdown
    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN")
    print("="*80)
    
    backtest_subset['month'] = pd.to_datetime(backtest_subset['game_date']).dt.to_period('M')
    monthly = backtest_subset[backtest_subset['bet_made']].groupby('month').agg({
        'bet_made': 'count',
        'bet_result': ['sum', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
    }).round(2)
    
    monthly.columns = ['Bets', 'Profit', 'Wins', 'Losses']
    monthly['Win%'] = (monthly['Wins'] / (monthly['Wins'] + monthly['Losses']) * 100).round(1)
    monthly['ROI%'] = (monthly['Profit'] / monthly['Bets'] * 100).round(1)
    
    print(monthly)
    
    # Top edges
    print("\n" + "="*80)
    print("BIGGEST EDGES (TOP 10)")
    print("="*80)
    
    top_bets = backtest_subset[backtest_subset['bet_made']].nlargest(10, ['home_edge', 'away_edge'], keep='all')
    
    for idx, row in top_bets.head(10).iterrows():
        bet_side = 'HOME' if row['bet_home'] else 'AWAY'
        edge = row['home_edge'] if row['bet_home'] else row['away_edge']
        result = 'WIN' if row['bet_result'] > 0 else 'LOSS' if row['bet_result'] < 0 else 'PUSH'
        
        print(f"{row['game_date']}: {row['away_team']} @ {row['home_team']}")
        print(f"  Bet {bet_side} | Edge: {edge:+.1%} | Spread: {row['spread_line']:+.1f} | {result}")
        print()

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
