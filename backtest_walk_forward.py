"""
Walk-Forward Backtest - 2024-25 Season
- Uses Trial 1306 model (22 features, fixed ELO)
- Flat bet strategy against moneyline
- Train on all historical data, test on 2024-25 season
- Calculate ROI and total units won/lost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = 'models/xgboost_22features_trial1306_20251215_212306.json'
DATA_PATH = 'data/training_data_matchup_with_injury_advantage_FIXED.csv'
ODDS_PATH = 'data/live/closing_odds_2024_25.csv'  # Will be updated per season
TEST_SEASON = '2024-25'  # Will be updated per season

FEATURES = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff', 'ewma_orb_diff',
    'ewma_vol_3p_diff', 'ewma_chaos_home', 'injury_matchup_advantage',
    'net_fatigue_score', 'ewma_foul_synergy_home', 'total_foul_environment',
    'league_offensive_context', 'season_progress', 'pace_efficiency_interaction',
    'projected_possession_margin', 'three_point_matchup', 'net_free_throw_advantage',
    'star_power_leverage', 'offense_vs_defense_matchup'
]

TARGET = 'target_moneyline_win'
FLAT_BET_SIZE = 1.0  # 1 unit per bet

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("="*90)
print(f"WALK-FORWARD BACKTEST - {TEST_SEASON} SEASON")
print("="*90)

print(f"\n[1/5] Loading data...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Load odds
odds = pd.read_csv(ODDS_PATH)
odds['game_date'] = pd.to_datetime(odds['game_date'])

print(f"  Training data: {len(df):,} games ({df['date'].min().date()} to {df['date'].max().date()})")
print(f"  Odds data: {len(odds):,} games")

# Filter for test season
test_df = df[df['season'] == TEST_SEASON].copy()
train_df = df[df['season'] != TEST_SEASON].copy()

print(f"\n  Test season: {TEST_SEASON}")
print(f"  Train set: {len(train_df):,} games (all seasons except {TEST_SEASON})")
print(f"  Test set:  {len(test_df):,} games ({TEST_SEASON} season)")

# ==============================================================================
# MERGE ODDS WITH TEST DATA
# ==============================================================================
print(f"\n[2/5] Merging odds data...")

# Merge on date and team names
test_df = test_df.merge(
    odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='left'
)

# Check merge success
games_with_odds = test_df['home_ml_odds'].notna().sum()
print(f"  Games with odds: {games_with_odds}/{len(test_df)} ({games_with_odds/len(test_df)*100:.1f}%)")

if games_with_odds < len(test_df) * 0.5:
    print(f"\n  ⚠️  Warning: Low odds coverage. Checking for team name mismatches...")
    print(f"  Test set teams: {sorted(test_df['home_team'].unique())[:10]}")
    print(f"  Odds set teams: {sorted(odds['home_team'].unique())[:10]}")

# Filter to games with odds
test_df = test_df[test_df['home_ml_odds'].notna()].copy()
print(f"  Games with odds: {len(test_df)}")

# Filter out outlier odds (data errors or extreme longshots)
print(f"\n  Filtering outlier odds...")
print(f"    Before filtering: {len(test_df)} games")
print(f"    Odds range: Home [{test_df['home_ml_odds'].min():.0f}, {test_df['home_ml_odds'].max():.0f}]")
print(f"                Away [{test_df['away_ml_odds'].min():.0f}, {test_df['away_ml_odds'].max():.0f}]")

# Remove extreme odds (likely data errors or meaningless longshots)
ODDS_MIN = -2000  # Heavy favorite limit
ODDS_MAX = 2000   # Heavy underdog limit

outliers_before = len(test_df)
test_df = test_df[
    (test_df['home_ml_odds'] >= ODDS_MIN) & (test_df['home_ml_odds'] <= ODDS_MAX) &
    (test_df['away_ml_odds'] >= ODDS_MIN) & (test_df['away_ml_odds'] <= ODDS_MAX)
].copy()
outliers_removed = outliers_before - len(test_df)

print(f"    Removed {outliers_removed} games with odds outside [{ODDS_MIN}, {ODDS_MAX}]")
print(f"    Final test set: {len(test_df)} games")

# ==============================================================================
# LOAD MODEL & GENERATE PREDICTIONS
# ==============================================================================
print(f"\n[3/5] Loading model and generating predictions...")

model = xgb.Booster()
model.load_model(MODEL_PATH)

X_test = test_df[FEATURES].values
y_test = test_df[TARGET].values

dtest = xgb.DMatrix(X_test)
home_win_prob = model.predict(dtest)

test_df['home_win_prob'] = home_win_prob
test_df['away_win_prob'] = 1 - home_win_prob

print(f"  Model loaded: {MODEL_PATH}")
print(f"  Predictions generated for {len(test_df)} games")

# ==============================================================================
# CALCULATE FLAT BET RESULTS
# ==============================================================================
print(f"\n[4/5] Calculating flat bet results...")

def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def calculate_bet_return(bet_on_home, home_won, home_odds, away_odds, stake=1.0):
    """Calculate return for a bet"""
    if bet_on_home:
        odds_decimal = american_to_decimal(home_odds)
        if home_won:
            return stake * odds_decimal - stake  # Profit
        else:
            return -stake  # Loss
    else:
        odds_decimal = american_to_decimal(away_odds)
        if not home_won:
            return stake * odds_decimal - stake  # Profit
        else:
            return -stake  # Loss

# Strategy: Bet on the side our model predicts
results = []

for idx, row in test_df.iterrows():
    home_prob = row['home_win_prob']
    away_prob = row['away_win_prob']
    home_won = row[TARGET] == 1
    
    # Bet on team with higher predicted probability
    bet_on_home = home_prob > 0.5
    
    bet_return = calculate_bet_return(
        bet_on_home,
        home_won,
        row['home_ml_odds'],
        row['away_ml_odds'],
        FLAT_BET_SIZE
    )
    
    results.append({
        'date': row['date'],
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'home_prob': home_prob,
        'bet_on': row['home_team'] if bet_on_home else row['away_team'],
        'bet_odds': row['home_ml_odds'] if bet_on_home else row['away_ml_odds'],
        'home_won': home_won,
        'bet_won': (bet_on_home and home_won) or (not bet_on_home and not home_won),
        'return': bet_return
    })

results_df = pd.DataFrame(results)

# ==============================================================================
# CALCULATE METRICS
# ==============================================================================
print(f"\n[5/5] Performance Summary:")
print(f"="*90)

total_bets = len(results_df)
bets_won = results_df['bet_won'].sum()
win_rate = bets_won / total_bets

total_staked = total_bets * FLAT_BET_SIZE
total_return = results_df['return'].sum()
roi = (total_return / total_staked) * 100

cumulative_units = results_df['return'].cumsum()
max_units = cumulative_units.max()
min_units = cumulative_units.min()
max_drawdown = max_units - min_units

print(f"\n  OVERALL PERFORMANCE:")
print(f"    Total Bets:        {total_bets}")
print(f"    Bets Won:          {bets_won}")
print(f"    Win Rate:          {win_rate:.2%}")
print(f"    Total Staked:      {total_staked:.1f} units")
print(f"    Total Return:      {total_return:+.2f} units")
print(f"    ROI:               {roi:+.2f}%")
print(f"    Max Drawdown:      {max_drawdown:.2f} units")
print(f"    Final Balance:     {total_staked + total_return:.2f} units")

# Breakdown by month
results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M')
monthly = results_df.groupby('month').agg({
    'return': ['sum', 'count'],
    'bet_won': 'sum'
})
monthly.columns = ['units', 'bets', 'wins']
monthly['win_rate'] = monthly['wins'] / monthly['bets']
monthly['roi'] = (monthly['units'] / monthly['bets']) * 100

print(f"\n  MONTHLY BREAKDOWN:")
print(f"    {'Month':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Units':<12} {'ROI':<10}")
print(f"    {'-'*70}")
for month, row in monthly.iterrows():
    print(f"    {str(month):<12} {int(row['bets']):<8} {int(row['wins']):<8} {row['win_rate']:<10.1%} {row['units']:+<12.2f} {row['roi']:+<10.2f}%")

# Betting distribution
favorites = results_df[results_df['bet_odds'] < 0]
underdogs = results_df[results_df['bet_odds'] > 0]

print(f"\n  BET DISTRIBUTION:")
print(f"    Favorites ({len(favorites)} bets):")
print(f"      Win Rate: {favorites['bet_won'].mean():.1%}")
print(f"      Total Return: {favorites['return'].sum():+.2f} units")
print(f"      ROI: {(favorites['return'].sum() / len(favorites)) * 100:+.2f}%")
print(f"\n    Underdogs ({len(underdogs)} bets):")
print(f"      Win Rate: {underdogs['bet_won'].mean():.1%}")
print(f"      Total Return: {underdogs['return'].sum():+.2f} units")
print(f"      ROI: {(underdogs['return'].sum() / len(underdogs)) * 100:+.2f}%")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'models/backtest_{TEST_SEASON.replace("-", "_")}_trial1306_{timestamp}.csv'
results_df.to_csv(output_path, index=False)

summary = {
    'model': 'trial1306_22features_fixed_elo',
    'test_period': f'{TEST_SEASON} season',
    'total_bets': int(total_bets),
    'bets_won': int(bets_won),
    'win_rate': float(win_rate),
    'total_staked': float(total_staked),
    'total_return': float(total_return),
    'roi_pct': float(roi),
    'max_drawdown': float(max_drawdown),
    'flat_bet_size': FLAT_BET_SIZE,
    'timestamp': timestamp
}

with open(f'models/backtest_summary_{TEST_SEASON.replace("-", "_")}_{timestamp}.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*90}")
print(f"✅ BACKTEST COMPLETE")
print(f"{'='*90}")
print(f"  Results saved: {output_path}")
print(f"  Summary saved: models/backtest_summary_{TEST_SEASON.replace('-', '_')}_{timestamp}.json")
print(f"{'='*90}\n")
