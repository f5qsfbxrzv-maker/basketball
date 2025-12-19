"""
Check what's happening with model performance
Compare old model vs new Gold Standard ELO features
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import json
import os

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\data\live\nba_betting_data.db"

print("=" * 80)
print("MODEL PERFORMANCE ANALYSIS")
print("=" * 80)

# Check if we have saved model configurations
configs = [
    "models/trial_1306_config.json",
    "models/gold_elo_hyperparameters.json"
]

print("\n1. CHECKING MODEL CONFIGURATIONS")
print("-" * 80)

for config_path in configs:
    if os.path.exists(config_path):
        print(f"\n✓ Found: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"   Features: {len(config.get('features', []))}")
        print(f"   Validation Score: {config.get('cv_log_loss', config.get('validation_log_loss', 'N/A'))}")
        
        if 'features' in config:
            print(f"\n   Top 10 Features:")
            for i, feat in enumerate(config['features'][:10], 1):
                print(f"      {i}. {feat}")

# Load recent games to understand baseline
print(f"\n2. BASELINE PERFORMANCE CHECK")
print("-" * 80)

conn = sqlite3.connect(DB_PATH)

# Get recent games from December 2025
recent_games = pd.read_sql_query("""
    SELECT 
        g1.GAME_DATE,
        g1.TEAM_ABBREVIATION as home_team,
        g2.TEAM_ABBREVIATION as away_team,
        g1.PTS as home_score,
        g2.PTS as away_score,
        CASE WHEN g1.WL = 'W' THEN 1 ELSE 0 END as home_won,
        g1.SEASON_ID
    FROM game_logs g1
    JOIN game_logs g2 ON g1.GAME_ID = g2.GAME_ID AND g1.TEAM_ABBREVIATION != g2.TEAM_ABBREVIATION
    WHERE g1.MATCHUP LIKE '%vs%'
    AND g1.SEASON_ID = '22025'
    AND g1.GAME_DATE >= '2025-12-01'
    ORDER BY g1.GAME_DATE DESC
    LIMIT 100
""", conn)

print(f"   Loaded {len(recent_games)} games from December 2025")
print(f"   Home team win rate: {recent_games['home_won'].mean():.3f}")

# Simple 50/50 baseline
baseline_acc = max(recent_games['home_won'].mean(), 1 - recent_games['home_won'].mean())
print(f"   50/50 prediction accuracy: {baseline_acc:.3f}")

# Check if home court advantage exists
print(f"   Home wins: {recent_games['home_won'].sum()}/{len(recent_games)} = {recent_games['home_won'].mean():.1%}")

# Get ELO data
def convert_season_format(season_id):
    year = int(season_id) - 20000
    return f"{year}-{str(year+1)[2:]}"

recent_games['season'] = recent_games['SEASON_ID'].apply(convert_season_format)

elo_df = pd.read_sql_query("""
    SELECT team, season, game_date, composite_elo
    FROM elo_ratings
    WHERE season = '2025-26'
    ORDER BY game_date
""", conn)

conn.close()

# Create ELO lookup
elo_dict = {}
for _, row in elo_df.iterrows():
    key = (row['team'], row['season'], row['game_date'])
    elo_dict[key] = row['composite_elo']

def get_latest_elo(team, season, before_date):
    dates = [date for (t, s, date) in elo_dict.keys() if t == team and s == season and date < before_date]
    if dates:
        return elo_dict[(team, season, max(dates))]
    return None

# Add ELO to games
home_elos = []
away_elos = []
for _, row in recent_games.iterrows():
    home_elo = get_latest_elo(row['home_team'], row['season'], row['GAME_DATE'])
    away_elo = get_latest_elo(row['away_team'], row['season'], row['GAME_DATE'])
    home_elos.append(home_elo)
    away_elos.append(away_elo)

recent_games['home_elo'] = home_elos
recent_games['away_elo'] = away_elos
recent_games = recent_games.dropna(subset=['home_elo', 'away_elo'])

# Calculate ELO-based predictions
recent_games['elo_diff'] = recent_games['home_elo'] - recent_games['away_elo']
recent_games['win_prob'] = 1.0 / (1.0 + 10 ** (-recent_games['elo_diff'] / 400))

print(f"\n3. GOLD STANDARD ELO PERFORMANCE")
print("-" * 80)
print(f"   Games with ELO: {len(recent_games)}")
print(f"   Average ELO diff (home - away): {recent_games['elo_diff'].mean():.1f}")
print(f"   Average win probability: {recent_games['win_prob'].mean():.3f}")

elo_accuracy = (recent_games['home_won'] == (recent_games['win_prob'] > 0.5)).mean()
elo_log_loss = log_loss(recent_games['home_won'], recent_games['win_prob'])

print(f"\n   ELO Accuracy: {elo_accuracy:.1%} ({int(elo_accuracy * len(recent_games))}/{len(recent_games)})")
print(f"   ELO Log-Loss: {elo_log_loss:.4f}")

# Compare to 50/50 baseline log-loss
baseline_prob = 0.5
baseline_log_loss = log_loss(recent_games['home_won'], [baseline_prob] * len(recent_games))
print(f"   50/50 Baseline Log-Loss: {baseline_log_loss:.4f}")
print(f"   ELO Improvement: {((baseline_log_loss - elo_log_loss) / baseline_log_loss * 100):.1f}% better")

# Check calibration
print(f"\n4. PREDICTION DISTRIBUTION")
print("-" * 80)
prob_bins = [0, 0.4, 0.5, 0.6, 1.0]
prob_labels = ['<40%', '40-50%', '50-60%', '>60%']
recent_games['prob_bin'] = pd.cut(recent_games['win_prob'], bins=prob_bins, labels=prob_labels)

print("\n   Win Probability Distribution:")
for label in prob_labels:
    bin_games = recent_games[recent_games['prob_bin'] == label]
    if len(bin_games) > 0:
        actual_rate = bin_games['home_won'].mean()
        print(f"      {label}: {len(bin_games)} games, {actual_rate:.1%} actual win rate")

# Check if there's a pattern in upsets
print(f"\n5. UPSET ANALYSIS")
print("-" * 80)
high_conf_home = recent_games[recent_games['win_prob'] > 0.65]
high_conf_away = recent_games[recent_games['win_prob'] < 0.35]

if len(high_conf_home) > 0:
    home_upsets = high_conf_home[high_conf_home['home_won'] == 0]
    print(f"   High-confidence home favorites (>65%): {len(high_conf_home)} games")
    print(f"   Upsets: {len(home_upsets)} ({len(home_upsets)/len(high_conf_home):.1%})")

if len(high_conf_away) > 0:
    away_upsets = high_conf_away[high_conf_away['home_won'] == 1]
    print(f"   High-confidence away favorites (<35%): {len(high_conf_away)} games")
    print(f"   Upsets: {len(away_upsets)} ({len(away_upsets)/len(high_conf_away):.1%})")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"""
The Gold Standard ELO system provides:
   • {elo_accuracy:.1%} accuracy (vs {baseline_acc:.1%} baseline)
   • Log-Loss: {elo_log_loss:.4f} (vs {baseline_log_loss:.4f} baseline)
   • {((baseline_log_loss - elo_log_loss) / baseline_log_loss * 100):.1f}% improvement over random guessing

Optuna optimization found:
   • Best log-loss: 0.63711 (Trial #286)
   • This is slightly WORSE than simple ELO baseline ({elo_log_loss:.4f})
   
HYPOTHESIS: The "fixed" feature (away_composite_elo) was already working fine.
The REAL issue is that our XGBoost model might be OVERFITTING or using
outdated hyperparameters that don't match the new ELO scale.

Next Steps:
1. Check if old model was trained on different ELO scale (K=32 vs K=15)
2. Verify feature list hasn't changed
3. Test if simpler model (just ELO features) performs better than complex ensemble
""")
