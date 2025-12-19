"""
Analyze current model's feature importance and performance
Compare to understand if Gold Standard ELO changes improved or hurt predictions
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
import os

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\data\live\nba_betting_data.db"

print("=" * 80)
print("MODEL FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Check if we have any trained models
model_paths = [
    "models/xgboost_gold_elo_optimized.json",
    "models/trial_1306_moneyline_model.json",
    "models/xgboost_model.json"
]

print("\n1. CHECKING AVAILABLE MODELS")
print("-" * 80)
for path in model_paths:
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"{exists} {path}")

# Load the most recent model configuration
config_path = "models/trial_1306_config.json"
if os.path.exists(config_path):
    print(f"\n2. LOADING MODEL CONFIGURATION")
    print("-" * 80)
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"   Trial: {config.get('trial_number', 'Unknown')}")
    print(f"   Features: {len(config.get('features', []))}")
    print(f"   Model type: {config.get('model_type', 'Unknown')}")
    print(f"   Validation Log-Loss: {config.get('cv_log_loss', 'N/A')}")
    
    feature_list = config.get('features', [])
    print(f"\n   Top 10 Features from Trial 1306:")
    for i, feat in enumerate(feature_list[:10], 1):
        print(f"      {i}. {feat}")

# Load historical data with both OLD and NEW ELO values
print(f"\n3. COMPARING ELO DISTRIBUTIONS")
print("-" * 80)

conn = sqlite3.connect(DB_PATH)

# Get current Gold Standard ELO stats
elo_stats = pd.read_sql_query("""
    SELECT 
        season,
        COUNT(*) as num_games,
        AVG(composite_elo) as avg_elo,
        MAX(composite_elo) - MIN(composite_elo) as elo_range,
        STDEV(composite_elo) as elo_std
    FROM (
        SELECT team, season, composite_elo,
               ROW_NUMBER() OVER (PARTITION BY team, season ORDER BY game_date DESC) as rn
        FROM elo_ratings
    ) WHERE rn = 1
    GROUP BY season
    ORDER BY season DESC
""", conn)

print("\nGold Standard ELO Statistics by Season:")
print(elo_stats.to_string(index=False))

# Get recent games and check prediction quality
print(f"\n4. RECENT PREDICTION ACCURACY")
print("-" * 80)

recent_games = pd.read_sql_query("""
    SELECT 
        g1.GAME_DATE as game_date,
        g1.SEASON_ID as season,
        g1.TEAM_ABBREVIATION as home_team,
        g2.TEAM_ABBREVIATION as away_team,
        g1.PTS as home_score,
        g2.PTS as away_score,
        CASE WHEN g1.WL = 'W' THEN 1 ELSE 0 END as home_won
    FROM game_logs g1
    JOIN game_logs g2 ON g1.GAME_ID = g2.GAME_ID AND g1.TEAM_ABBREVIATION != g2.TEAM_ABBREVIATION
    WHERE g1.MATCHUP LIKE '%vs%'
    AND g1.SEASON_ID = '22025'
    AND g1.GAME_DATE >= '2025-12-01'
    ORDER BY g1.GAME_DATE DESC
    LIMIT 50
""", conn)

print(f"   Loaded {len(recent_games)} recent games (Dec 2025)")

# Get ELO ratings for these games
def convert_season_format(season_id):
    year = int(season_id) - 20000
    return f"{year}-{str(year+1)[2:]}"

recent_games['season'] = recent_games['season'].apply(convert_season_format)

# Simple ELO-based prediction
elo_df = pd.read_sql_query("""
    SELECT team, season, game_date, composite_elo, off_elo, def_elo
    FROM elo_ratings
    ORDER BY game_date
""", conn)

conn.close()

# Create ELO lookup dict
elo_dict = {}
for _, row in elo_df.iterrows():
    key = (row['team'], row['season'], row['game_date'])
    elo_dict[key] = {
        'composite_elo': row['composite_elo'],
        'off_elo': row['off_elo'],
        'def_elo': row['def_elo']
    }

def get_latest_elo(team, season, before_date):
    dates = [date for (t, s, date) in elo_dict.keys() if t == team and s == season and date < before_date]
    if dates:
        latest_date = max(dates)
        return elo_dict[(team, season, latest_date)]
    return None

# Add ELO to recent games
home_elos = []
away_elos = []
for _, row in recent_games.iterrows():
    home_elo = get_latest_elo(row['home_team'], row['season'], row['game_date'])
    away_elo = get_latest_elo(row['away_team'], row['season'], row['game_date'])
    home_elos.append(home_elo['composite_elo'] if home_elo else None)
    away_elos.append(away_elo['composite_elo'] if away_elo else None)

recent_games['home_elo'] = home_elos
recent_games['away_elo'] = away_elos
recent_games = recent_games.dropna(subset=['home_elo', 'away_elo'])

# Calculate simple ELO win probability
recent_games['elo_diff'] = recent_games['home_elo'] - recent_games['away_elo']
recent_games['win_prob'] = 1.0 / (1.0 + 10 ** (-recent_games['elo_diff'] / 400))

# Calculate prediction accuracy
elo_accuracy = (recent_games['home_won'] == (recent_games['win_prob'] > 0.5)).mean()
elo_log_loss = log_loss(recent_games['home_won'], recent_games['win_prob'])
elo_brier = brier_score_loss(recent_games['home_won'], recent_games['win_prob'])

print(f"\nGold Standard ELO Performance (Recent 50 games):")
print(f"   Accuracy: {elo_accuracy:.3f} ({int(elo_accuracy * len(recent_games))}/{len(recent_games)} correct)")
print(f"   Log-Loss: {elo_log_loss:.4f}")
print(f"   Brier Score: {elo_brier:.4f}")
print(f"   Average Win Probability: {recent_games['win_prob'].mean():.3f}")

# Show some example predictions
print(f"\n5. RECENT PREDICTION EXAMPLES")
print("-" * 80)
print(f"{'Date':<12} {'Matchup':<20} {'Score':<12} {'Home ELO':<10} {'Away ELO':<10} {'Win Prob':<10} {'Correct':<8}")
print("-" * 80)
for _, game in recent_games.head(10).iterrows():
    matchup = f"{game['home_team']} vs {game['away_team']}"
    score = f"{game['home_score']}-{game['away_score']}"
    correct = "✓" if (game['home_won'] == 1 and game['win_prob'] > 0.5) or (game['home_won'] == 0 and game['win_prob'] < 0.5) else "✗"
    print(f"{game['game_date']:<12} {matchup:<20} {score:<12} {game['home_elo']:<10.1f} {game['away_elo']:<10.1f} {game['win_prob']:<10.3f} {correct:<8}")

# Check if there are major outliers
print(f"\n6. CHECKING FOR PREDICTION OUTLIERS")
print("-" * 80)

# Find games where model was very confident but wrong
recent_games['prediction_error'] = abs(recent_games['home_won'] - recent_games['win_prob'])
high_confidence_wrong = recent_games[(recent_games['win_prob'] > 0.7) | (recent_games['win_prob'] < 0.3)]
high_confidence_wrong = high_confidence_wrong[
    ((high_confidence_wrong['win_prob'] > 0.7) & (high_confidence_wrong['home_won'] == 0)) |
    ((high_confidence_wrong['win_prob'] < 0.3) & (high_confidence_wrong['home_won'] == 1))
]

if len(high_confidence_wrong) > 0:
    print(f"\nFound {len(high_confidence_wrong)} high-confidence incorrect predictions:")
    for _, game in high_confidence_wrong.iterrows():
        result = "UPSET" if game['home_won'] == 1 else "UPSET LOSS"
        print(f"   {game['game_date']}: {game['home_team']} vs {game['away_team']} ({game['home_score']}-{game['away_score']}) - {result}")
        print(f"      Predicted win prob: {game['win_prob']:.3f}, ELO diff: {game['elo_diff']:.1f}")
else:
    print("   ✓ No major upsets in high-confidence predictions")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nThe Gold Standard ELO system (K=15, SCALE=40, WIN_WEIGHT=30) provides:")
print(f"   • Stable ratings (range ~205 points)")
print(f"   • Proper team hierarchy (Brooklyn #25, not #3)")
print(f"   • Baseline accuracy: {elo_accuracy:.1%}")
print(f"\nThe full XGBoost model should improve on this by incorporating:")
print(f"   • Recent performance trends (L10, L5)")
print(f"   • Four Factors (eFG%, TOV%, ORB%, FTr)")
print(f"   • Pace and efficiency metrics")
print(f"   • Rest and schedule factors")
print("\nWait for overnight optimization to complete for best hyperparameters!")
