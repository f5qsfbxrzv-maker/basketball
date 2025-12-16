"""
Walk-Forward Backtest - 2023-24 Season
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import json

# Team name mapping
TEAM_MAPPING = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

MODEL_PATH = 'models/xgboost_22features_trial1306_20251215_212306.json'
DATA_PATH = 'data/training_data_matchup_with_injury_advantage_FIXED.csv'
ODDS_PATH = 'data/closing_odds_2023_24.csv'

FEATURES = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff', 'ewma_orb_diff',
    'ewma_vol_3p_diff', 'ewma_chaos_home', 'injury_matchup_advantage',
    'net_fatigue_score', 'ewma_foul_synergy_home', 'total_foul_environment',
    'league_offensive_context', 'season_progress', 'pace_efficiency_interaction',
    'projected_possession_margin', 'three_point_matchup', 'net_free_throw_advantage',
    'star_power_leverage', 'offense_vs_defense_matchup'
]

print("="*90)
print("BACKTEST - 2023-24 SEASON")
print("="*90)

# Load data
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
odds = pd.read_csv(ODDS_PATH)
odds['game_date'] = pd.to_datetime(odds['game_date'])

# Filter test set
test_df = df[df['season'] == '2023-24'].copy()
print(f"\nTest set: {len(test_df)} games (2023-24 season)")

# Map team names
test_df['home_team_full'] = test_df['home_team'].map(TEAM_MAPPING)
test_df['away_team_full'] = test_df['away_team'].map(TEAM_MAPPING)

# Merge odds
test_df = test_df.merge(
    odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    left_on=['date', 'home_team_full', 'away_team_full'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='left',
    suffixes=('', '_odds')
)

print(f"Games with odds: {test_df['home_ml_odds'].notna().sum()}/{len(test_df)}")

# Filter outliers
test_df = test_df[
    (test_df['home_ml_odds'] >= -2000) & (test_df['home_ml_odds'] <= 2000) &
    (test_df['away_ml_odds'] >= -2000) & (test_df['away_ml_odds'] <= 2000)
].dropna(subset=['home_ml_odds']).copy()

print(f"After filtering: {len(test_df)} games\n")

# Load model and predict
model = xgb.Booster()
model.load_model(MODEL_PATH)
X = test_df[FEATURES].values
y = test_df['target_moneyline_win'].values
test_df['home_win_prob'] = model.predict(xgb.DMatrix(X))

# Calculate returns
def american_to_decimal(odds):
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

def calc_return(bet_home, won_home, home_odds, away_odds):
    odds_dec = american_to_decimal(home_odds if bet_home else away_odds)
    won = (bet_home and won_home) or (not bet_home and not won_home)
    return (odds_dec - 1) if won else -1

results = []
for _, row in test_df.iterrows():
    bet_home = row['home_win_prob'] > 0.5
    won_home = row['target_moneyline_win'] == 1
    ret = calc_return(bet_home, won_home, row['home_ml_odds'], row['away_ml_odds'])
    results.append({'date': row['date'], 'return': ret, 'won': ret > 0})

results_df = pd.DataFrame(results)
total = results_df['return'].sum()
roi = (total / len(results_df)) * 100

print("RESULTS:")
print(f"  Total Bets:    {len(results_df)}")
print(f"  Wins:          {results_df['won'].sum()} ({results_df['won'].mean():.1%})")
print(f"  Total Units:   {total:+.2f}")
print(f"  ROI:           {roi:+.2f}%")
print("="*90)

# Save
results_df.to_csv('models/backtest_2023_24_results.csv', index=False)
print("\n✅ Saved to models/backtest_2023_24_results.csv")
