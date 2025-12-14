"""
Investigate SAS vs OKC prediction - why is SAS at 34%?
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Check what the dashboard is actually predicting
print("=" * 60)
print("INVESTIGATING SAS @ OKC PREDICTION")
print("=" * 60)

# Check injuries
db_path = 'data/live/nba_betting_data.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("\n1. CHECKING ACTIVE INJURIES:")
print("-" * 60)

# Get table schema first
cursor.execute("PRAGMA table_info(active_injuries)")
columns = [col[1] for col in cursor.fetchall()]
print(f"Available columns: {columns}")

# Check for Spurs injuries
cursor.execute("""
    SELECT * FROM active_injuries 
    WHERE LOWER(team_name) LIKE '%spurs%' OR LOWER(team_name) LIKE '%san antonio%'
""")
spurs_injuries = cursor.fetchall()

if spurs_injuries:
    print(f"\nFound {len(spurs_injuries)} Spurs injuries:")
    for inj in spurs_injuries:
        print(f"  {inj}")
else:
    print("No Spurs injuries found in active_injuries table")

# Check for Thunder injuries
cursor.execute("""
    SELECT * FROM active_injuries 
    WHERE LOWER(team_name) LIKE '%thunder%' OR LOWER(team_name) LIKE '%oklahoma%'
""")
thunder_injuries = cursor.fetchall()

if thunder_injuries:
    print(f"\nFound {len(thunder_injuries)} Thunder injuries:")
    for inj in thunder_injuries:
        print(f"  {inj}")
else:
    print("No Thunder injuries found in active_injuries table")

# Check ELO ratings
print("\n2. CHECKING ELO RATINGS:")
print("-" * 60)

cursor.execute("""
    SELECT team_name, off_elo, def_elo, composite_elo, last_updated
    FROM elo_ratings
    WHERE LOWER(team_name) LIKE '%thunder%' OR LOWER(team_name) LIKE '%oklahoma%'
    ORDER BY last_updated DESC
    LIMIT 1
""")
okc_elo = cursor.fetchone()

cursor.execute("""
    SELECT team_name, off_elo, def_elo, composite_elo, last_updated
    FROM elo_ratings
    WHERE LOWER(team_name) LIKE '%spurs%' OR LOWER(team_name) LIKE '%san antonio%'
    ORDER BY last_updated DESC
    LIMIT 1
""")
sas_elo = cursor.fetchone()

if okc_elo:
    print(f"\nOKC: Off={okc_elo[1]:.1f}, Def={okc_elo[2]:.1f}, Composite={okc_elo[3]:.1f} (updated: {okc_elo[4]})")
else:
    print("OKC ELO not found")

if sas_elo:
    print(f"SAS: Off={sas_elo[1]:.1f}, Def={sas_elo[2]:.1f}, Composite={sas_elo[3]:.1f} (updated: {sas_elo[4]})")
else:
    print("SAS ELO not found")

# Check recent games
print("\n3. CHECKING RECENT FORM (Last 5 games):")
print("-" * 60)

cursor.execute("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_logs
    WHERE (home_team LIKE '%Thunder%' OR away_team LIKE '%Thunder%')
    ORDER BY game_date DESC
    LIMIT 5
""")
okc_games = cursor.fetchall()

print("\nOKC Recent Games:")
for game in okc_games:
    date, home, away, home_score, away_score = game
    if 'Thunder' in home:
        result = "W" if home_score > away_score else "L"
        print(f"  {date}: {away} @ {home} - {away_score}-{home_score} ({result})")
    else:
        result = "W" if away_score > home_score else "L"
        print(f"  {date}: {home} @ {away} - {home_score}-{away_score} ({result})")

cursor.execute("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_logs
    WHERE (home_team LIKE '%Spurs%' OR away_team LIKE '%Spurs%')
    ORDER BY game_date DESC
    LIMIT 5
""")
sas_games = cursor.fetchall()

print("\nSAS Recent Games:")
for game in sas_games:
    date, home, away, home_score, away_score = game
    if 'Spurs' in home:
        result = "W" if home_score > away_score else "L"
        print(f"  {date}: {away} @ {home} - {away_score}-{home_score} ({result})")
    else:
        result = "W" if away_score > home_score else "L"
        print(f"  {date}: {home} @ {away} - {home_score}-{away_score} ({result})")

# Check if Victor Wembanyama is listed as a key player
print("\n4. CHECKING FOR WEMBANYAMA:")
print("-" * 60)

cursor.execute("""
    SELECT player_name, team_abbrev, pie_score 
    FROM player_stats
    WHERE LOWER(player_name) LIKE '%wembanyama%'
    ORDER BY pie_score DESC
    LIMIT 1
""")
wemby = cursor.fetchone()

if wemby:
    print(f"Found: {wemby[0]} ({wemby[1]}) - PIE: {wemby[2]:.3f}")
else:
    print("Wembanyama not found in player_stats")

# Check model features
print("\n5. CHECKING MODEL:")
print("-" * 60)

model_path = Path('models/xgboost_final_trial98.json')
if model_path.exists():
    with open(model_path) as f:
        model_data = json.load(f)
    
    # Try to get feature names
    if 'feature_names' in model_data:
        features = model_data['feature_names']
        injury_features = [f for f in features if 'injury' in f.lower() or 'star' in f.lower()]
        print(f"\nModel has {len(features)} features")
        print(f"Injury-related features ({len(injury_features)}):")
        for f in injury_features:
            print(f"  - {f}")
    else:
        print("Could not extract feature names from model")
else:
    print("Model not found")

# Check if there's a paper prediction for this game
print("\n6. CHECKING LOGGED PREDICTIONS:")
print("-" * 60)

cursor.execute("""
    SELECT game_date, home_team, away_team, predicted_winner, 
           model_probability, edge, stake, timestamp
    FROM paper_predictions
    WHERE game_date = '2025-12-13'
    AND (home_team LIKE '%Thunder%' OR home_team LIKE '%Oklahoma%')
    AND (away_team LIKE '%Spurs%' OR away_team LIKE '%San Antonio%')
""")
logged_pred = cursor.fetchone()

if logged_pred:
    print(f"\nLogged prediction found:")
    print(f"  Predicted: {logged_pred[3]}")
    print(f"  Model Prob: {logged_pred[4]:.1%}")
    print(f"  Edge: {logged_pred[5]:+.1%}")
    print(f"  Stake: ${logged_pred[6]:.2f}")
    print(f"  Timestamp: {logged_pred[7]}")
else:
    print("No logged prediction for this game")

conn.close()

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
print("\nIf SAS is showing 34% win probability:")
print("- Check if Wembanyama injury is in active_injuries table")
print("- Verify away_star_missing feature = 1 for SAS")
print("- Check injury_impact_diff feature (should heavily favor OKC)")
print("- Verify ELO differential (OKC should be ~100+ points higher)")
print("\nPossible issues:")
print("1. Injury data not updated in active_injuries table")
print("2. Feature calculator not detecting Wembanyama as star player")
print("3. Model not weighting injury features heavily enough")
print("4. ELO not properly accounting for strength differential")
print("\nTo fix:")
print("- Run: python src/services/injury_updater.py")
print("- Verify Wembanyama PIE score > 0.15 (star threshold)")
print("- Check feature_calculator_v5.py injury impact calculation")
