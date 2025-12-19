"""
DEEP ANALYSIS: Dallas vs Detroit - Why is the model favoring Dallas?
Examines: ELO ratings, all 22 features, injury data, recent stats
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*100)
print("DEEP DIVE ANALYSIS: DET @ DAL - 12/18/2025")
print("="*100)

# Database path
db_path = Path('data/live/nba_betting_data.db')

# ===== 1. CHECK ELO RATINGS =====
print("\n" + "="*100)
print("1. ELO RATINGS")
print("="*100)

conn = sqlite3.connect(db_path)

# Check composite ELO
print("\n📊 COMPOSITE ELO (from elo_ratings table):")
try:
    query = """
    SELECT team, composite_elo
    FROM elo_ratings
    WHERE team IN ('DAL', 'DET')
    ORDER BY composite_elo DESC
    """
    elo_df = pd.read_sql_query(query, conn)
    print(elo_df.to_string(index=False))
except Exception as e:
    print(f"  Error: {e}")
    # Try alternate query
    try:
        query = "SELECT * FROM elo_ratings WHERE team IN ('DAL', 'DET') ORDER BY composite_elo DESC"
        elo_df = pd.read_sql_query(query, conn)
        print(elo_df.to_string(index=False))
    except:
        print("  elo_ratings table not available")

# Check offensive/defensive ELO
print("\n⚔️ OFFENSIVE/DEFENSIVE ELO (from off_def_elo table):")
try:
    query = """
    SELECT team, offensive_elo, defensive_elo, 
           (offensive_elo + defensive_elo) / 2 as avg_elo,
           last_updated
    FROM off_def_elo
    WHERE team IN ('DAL', 'DET')
    ORDER BY offensive_elo DESC
    """
    off_def_df = pd.read_sql_query(query, conn)
    print(off_def_df.to_string(index=False))
except Exception as e:
    print(f"  Error: {e}")
    try:
        query = "SELECT * FROM off_def_elo WHERE team IN ('DAL', 'DET')"
        off_def_df = pd.read_sql_query(query, conn)
        print(off_def_df.to_string(index=False))
    except:
        print("  off_def_elo table not available")

# ===== 2. CHECK INJURY DATA =====
print("\n" + "="*100)
print("2. INJURY DATA")
print("="*100)

print("\n🏥 ACTIVE INJURIES (from active_injuries table):")
query = """
SELECT team, player_name, status, pie_score
FROM active_injuries
WHERE team IN ('DAL', 'DET')
ORDER BY team, pie_score DESC
"""
injury_df = pd.read_sql_query(query, conn)
print(injury_df.to_string(index=False))

# ===== 3. RECENT TEAM STATS =====
print("\n" + "="*100)
print("3. RECENT TEAM PERFORMANCE")
print("="*100)

print("\n📈 LAST 10 GAMES (from team_stats):")
query = """
SELECT team, 
       AVG(points) as avg_pts,
       AVG(opp_points) as avg_opp_pts,
       AVG(fg_pct) as avg_fg_pct,
       AVG(fg3_pct) as avg_3pt_pct,
       AVG(rebounds) as avg_reb,
       COUNT(*) as games
FROM team_stats
WHERE team IN ('DAL', 'DET')
  AND date >= date('now', '-30 days')
GROUP BY team
ORDER BY avg_pts DESC
"""
recent_df = pd.read_sql_query(query, conn)
if not recent_df.empty:
    print(recent_df.to_string(index=False))
else:
    print("  No recent stats in team_stats table")

# ===== 4. CALCULATE FEATURES MANUALLY =====
print("\n" + "="*100)
print("4. TRIAL 1306 FEATURES (22 features)")
print("="*100)

# Import feature calculator
from feature_calculator_v5 import FeatureCalculatorV5

calc = FeatureCalculatorV5()

try:
    features = calc.calculate_game_features(
        home_team='DAL',
        away_team='DET',
        game_date='2025-12-18'
    )
    
    print("\n🎯 ALL 22 FEATURES:")
    print(f"{'Feature Name':<40} {'Value':>12} {'Interpretation'}")
    print("-"*100)
    
    # Group by type
    elo_features = [(k, v) for k, v in features.items() if 'elo' in k.lower()]
    injury_features = [(k, v) for k, v in features.items() if 'injury' in k.lower()]
    other_features = [(k, v) for k, v in features.items() if 'elo' not in k.lower() and 'injury' not in k.lower()]
    
    print("\n🏀 ELO-BASED FEATURES:")
    for feat, val in elo_features:
        if 'home' in feat:
            team = 'DAL'
        elif 'away' in feat:
            team = 'DET'
        elif 'diff' in feat:
            team = 'DAL-DET'
        else:
            team = ''
        
        print(f"  {feat:<38} {val:>12.2f}  ({team})")
    
    print("\n🏥 INJURY-BASED FEATURES:")
    for feat, val in injury_features:
        print(f"  {feat:<38} {val:>12.2f}")
    
    print("\n📊 OTHER FEATURES:")
    for feat, val in other_features:
        print(f"  {feat:<38} {val:>12.2f}")
        
except Exception as e:
    print(f"\n❌ ERROR calculating features: {e}")
    import traceback
    traceback.print_exc()

# ===== 5. CHECK HISTORICAL HEAD-TO-HEAD =====
print("\n" + "="*100)
print("5. HISTORICAL HEAD-TO-HEAD")
print("="*100)

query = """
SELECT date, home_team, away_team, home_score, away_score,
       (home_score - away_score) as margin
FROM game_results
WHERE (home_team = 'DAL' AND away_team = 'DET')
   OR (home_team = 'DET' AND away_team = 'DAL')
ORDER BY date DESC
LIMIT 5
"""
try:
    h2h_df = pd.read_sql_query(query, conn)
    if not h2h_df.empty:
        print(h2h_df.to_string(index=False))
    else:
        print("  No historical games found")
except:
    print("  game_results table not available")

# ===== 6. PREDICT WITH MODEL =====
print("\n" + "="*100)
print("6. MODEL PREDICTION")
print("="*100)

from nba_prediction_engine_v5 import NBAPredictionEngine

predictor = NBAPredictionEngine()
prediction = predictor.predict_game(
    home_team='DAL',
    away_team='DET',
    game_date='2025-12-18',
    game_time='01:30:00',
    home_ml_odds=-110,
    away_ml_odds=-110
)

print(f"\n📊 RAW MODEL OUTPUT:")
print(f"  DAL (Home) Win Prob: {prediction['home_win_prob']*100:.2f}%")
print(f"  DET (Away) Win Prob: {prediction['away_win_prob']*100:.2f}%")

if 'features' in prediction:
    features = prediction['features']
    
    # Find the most impactful features
    print(f"\n🔝 TOP 10 FEATURES BY ABSOLUTE VALUE:")
    sorted_feats = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for i, (feat, val) in enumerate(sorted_feats, 1):
        direction = "favors DAL" if val > 0 else "favors DET"
        print(f"  {i:2d}. {feat:<38} = {val:>8.2f}  ({direction})")

# ===== 7. CHECK FOR DATA ISSUES =====
print("\n" + "="*100)
print("7. DATA QUALITY CHECKS")
print("="*100)

# Check if injury data is correct
print("\n🔍 INJURY DATA VALIDATION:")
query = """
SELECT team, COUNT(*) as injury_count, SUM(pie_score) as total_pie
FROM active_injuries
WHERE team IN ('DAL', 'DET')
GROUP BY team
"""
injury_summary = pd.read_sql_query(query, conn)
print(injury_summary.to_string(index=False))

# Check for duplicate or misattributed injuries
query = """
SELECT player_name, team, status, pie_score
FROM active_injuries
WHERE player_name IN ('Anthony Davis', 'Kyrie Irving', 'D''Angelo Russell')
ORDER BY player_name
"""
print("\n🔍 SPECIFIC PLAYER CHECK (potential misattribution):")
player_check = pd.read_sql_query(query, conn)
if not player_check.empty:
    print(player_check.to_string(index=False))
else:
    print("  None of these players found in active_injuries")

conn.close()

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print("\n⚠️  KEY QUESTIONS:")
print("  1. Are the ELO ratings correct? (Detroit should be higher)")
print("  2. Are injuries correctly attributed? (Anthony Davis shouldn't be on Dallas)")
print("  3. Which features are driving Dallas to 62.5% win probability?")
print("  4. Is there a bug in feature calculation or model input?")
