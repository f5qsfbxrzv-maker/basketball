"""
Walk-Forward Backtest with Team Name Mapping for 2023-24 Season
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from datetime import datetime

# ==================== CONFIGURATION ====================
MODEL_PATH = 'models/xgboost_22features_trial1306_20251215_212306.json'
TRAINING_DATA_PATH = 'data/training_data_matchup_with_injury_advantage_FIXED.csv'
ODDS_PATH = 'data/closing_odds_2023_24.csv'
TEST_SEASON = '2023-24'
FLAT_BET_SIZE = 1.0
ODDS_FILTER_MIN = -2000
ODDS_FILTER_MAX = 2000

# Team name mapping (abbreviation -> full name)
TEAM_NAME_MAP = {
    'ATL': 'Atlanta Hawks',
    'BKN': 'Brooklyn Nets', 
    'BOS': 'Boston Celtics',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}

# 22 features from Trial 1306
FEATURE_COLS = [
    # ELO (4)
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    # EWMA (6)
    'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff', 'ewma_orb_diff', 
    'ewma_vol_3p_diff', 'ewma_chaos_home',
    # Injuries (1)
    'injury_matchup_advantage',
    # Advanced (11)
    'net_fatigue_score', 'ewma_foul_synergy_home', 'total_foul_environment',
    'league_offensive_context', 'season_progress', 'pace_efficiency_interaction',
    'projected_possession_margin', 'three_point_matchup', 'net_free_throw_advantage',
    'star_power_leverage', 'offense_vs_defense_matchup'
]

def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def calculate_payout(odds, bet_size):
    """Calculate payout from American odds"""
    decimal_odds = american_to_decimal(odds)
    return bet_size * (decimal_odds - 1)  # Profit only

# ==================== LOAD DATA ====================
print("=" * 90)
print(f"WALK-FORWARD BACKTEST - {TEST_SEASON} SEASON")
print("=" * 90)
print()

print("[1/5] Loading data...")
df = pd.read_csv(TRAINING_DATA_PATH)
df['game_date'] = pd.to_datetime(df['game_date'])

# Load odds data
odds_df = pd.read_csv(ODDS_PATH)
odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

print(f"  Training data: {len(df):,} games ({df['game_date'].min().date()} to {df['game_date'].max().date()})")
print(f"  Odds data: {len(odds_df):,} games")
print()

# ==================== SPLIT DATA ====================
# Split by season
df['season'] = df['game_date'].apply(lambda x: 
    f"{x.year}-{str(x.year+1)[-2:]}" if x.month >= 10 else f"{x.year-1}-{str(x.year)[-2:]}"
)

train = df[df['season'] != TEST_SEASON].copy()
test = df[df['season'] == TEST_SEASON].copy()

print(f"  Test season: {TEST_SEASON}")
print(f"  Train set: {len(train):,} games (all seasons except {TEST_SEASON})")
print(f"  Test set:  {len(test):,} games ({TEST_SEASON} season)")
print()

# ==================== MERGE ODDS ====================
print("[2/5] Merging odds data...")

# Map team names in test set
test['home_team_full'] = test['home_team'].map(TEAM_NAME_MAP)
test['away_team_full'] = test['away_team'].map(TEAM_NAME_MAP)

# Merge with odds
test = test.merge(
    odds_df[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    left_on=['game_date', 'home_team_full', 'away_team_full'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='left',
    suffixes=('', '_odds')
)

# Count successful merges
games_with_odds = test['home_ml_odds'].notna().sum()
print(f"  Games with odds: {games_with_odds}/{len(test)} ({100*games_with_odds/len(test):.1f}%)")
print()

# Filter outlier odds
print("  Filtering outlier odds...")
before_filter = games_with_odds
print(f"    Before filtering: {before_filter} games")
print(f"    Odds range: Home [{test['home_ml_odds'].min():.0f}, {test['home_ml_odds'].max():.0f}]")
print(f"                Away [{test['away_ml_odds'].min():.0f}, {test['away_ml_odds'].max():.0f}]")

valid_odds_mask = (
    (test['home_ml_odds'] >= ODDS_FILTER_MIN) & (test['home_ml_odds'] <= ODDS_FILTER_MAX) &
    (test['away_ml_odds'] >= ODDS_FILTER_MIN) & (test['away_ml_odds'] <= ODDS_FILTER_MAX)
)
test = test[valid_odds_mask].copy()

removed = before_filter - len(test)
print(f"    Removed {removed} games with odds outside [{ODDS_FILTER_MIN}, {ODDS_FILTER_MAX}]")
print(f"    Final test set: {len(test)} games")
print()

if len(test) == 0:
    print("⚠️  No games remaining after filtering. Exiting.")
    exit(0)

# ==================== LOAD MODEL & PREDICT ====================
print("[3/5] Loading model and generating predictions...")
bst = xgb.Booster()
bst.load_model(MODEL_PATH)
print(f"  Model loaded: {MODEL_PATH}")

# Generate predictions
X_test = test[FEATURE_COLS].values
dtest = xgb.DMatrix(X_test, feature_names=FEATURE_COLS)
test['home_win_prob'] = bst.predict(dtest)
print(f"  Predictions generated for {len(test)} games")
print()

# ==================== CALCULATE FLAT BETS ====================
print("[4/5] Calculating flat bet results...")

# Calculate bet on favorite (higher implied probability)
test['home_implied_prob'] = 1 / american_to_decimal(test['home_ml_odds'])
test['away_implied_prob'] = 1 / american_to_decimal(test['away_ml_odds'])

# Bet on team with higher model probability
test['bet_home'] = test['home_win_prob'] > 0.5
test['bet_away'] = test['home_win_prob'] <= 0.5

# Calculate results
test['bet_won'] = (
    (test['bet_home'] & test['home_won']) | 
    (test['bet_away'] & ~test['home_won'])
)

test['bet_odds'] = test.apply(
    lambda row: row['home_ml_odds'] if row['bet_home'] else row['away_ml_odds'],
    axis=1
)

test['payout'] = test.apply(
    lambda row: calculate_payout(row['bet_odds'], FLAT_BET_SIZE) if row['bet_won'] else -FLAT_BET_SIZE,
    axis=1
)

test['cumulative_profit'] = test['payout'].cumsum()

# ==================== PERFORMANCE SUMMARY ====================
print()
print("[5/5] Performance Summary:")
print("=" * 90)
print()

total_bets = len(test)
wins = test['bet_won'].sum()
win_rate = wins / total_bets
total_return = test['payout'].sum()
roi = (total_return / (total_bets * FLAT_BET_SIZE)) * 100

print(f"💰 OVERALL PERFORMANCE")
print(f"  Total bets:    {total_bets:,}")
print(f"  Wins:          {wins:,}")
print(f"  Win rate:      {win_rate:.2%}")
print(f"  Total return:  {total_return:+.2f} units")
print(f"  ROI:           {roi:+.2f}%")
print(f"  Max drawdown:  {(test['cumulative_profit'] - test['cumulative_profit'].cummax()).min():.2f} units")
print()

# Favorite vs underdog breakdown
favorite_bets = test[
    ((test['bet_home']) & (test['home_implied_prob'] > test['away_implied_prob'])) |
    ((test['bet_away']) & (test['away_implied_prob'] > test['home_implied_prob']))
]

underdog_bets = test[
    ((test['bet_home']) & (test['home_implied_prob'] < test['away_implied_prob'])) |
    ((test['bet_away']) & (test['away_implied_prob'] < test['home_implied_prob']))
]

print(f"📊 BETTING BREAKDOWN")
print(f"  Favorites ({len(favorite_bets)} bets):")
print(f"    Win rate: {favorite_bets['bet_won'].mean():.1%}")
print(f"    ROI:      {(favorite_bets['payout'].sum() / (len(favorite_bets) * FLAT_BET_SIZE)) * 100:+.2f}%")
print()
print(f"  Underdogs ({len(underdog_bets)} bets):")
print(f"    Win rate: {underdog_bets['bet_won'].mean():.1%}")
print(f"    ROI:      {(underdog_bets['payout'].sum() / (len(underdog_bets) * FLAT_BET_SIZE)) * 100:+.2f}%")
print()

# Monthly breakdown
test['month'] = test['game_date'].dt.to_period('M')
monthly = test.groupby('month').agg({
    'payout': ['count', 'sum'],
    'bet_won': 'mean'
})
monthly.columns = ['bets', 'profit', 'win_rate']
monthly['roi'] = (monthly['profit'] / (monthly['bets'] * FLAT_BET_SIZE)) * 100

print(f"📅 MONTHLY BREAKDOWN")
for month, row in monthly.iterrows():
    print(f"  {month}: {row['bets']:3.0f} bets | Win Rate: {row['win_rate']:.1%} | ROI: {row['roi']:+6.2f}%")
print()

print("=" * 90)
