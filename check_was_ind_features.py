"""Spot check WAS @ IND features and ELO data"""
import sqlite3
import json
from pathlib import Path

print("="*80)
print("ELO RATINGS CHECK")
print("="*80)

db_path = Path("data/live/nba_betting_data.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check ELO ratings for WAS and IND
print("\nELO ratings for WAS and IND:")
cursor.execute("""
    SELECT team, off_elo, def_elo, composite_elo, game_date 
    FROM elo_ratings 
    WHERE team IN ('WAS', 'IND') 
    ORDER BY team, game_date DESC
    LIMIT 20
""")
rows = cursor.fetchall()
print(f"{'Team':<5} {'Off ELO':<10} {'Def ELO':<10} {'Comp ELO':<10} {'Game Date'}")
for row in rows:
    print(f"{row[0]:<5} {row[1]:<10.1f} {row[2]:<10.1f} {row[3]:<10.1f} {row[4]}")

# Check all teams to see if 1500 is default
print("\n" + "="*80)
print("ALL TEAMS LATEST ELO RATINGS")
print("="*80)
cursor.execute("""
    SELECT team, off_elo, def_elo, composite_elo, game_date
    FROM elo_ratings 
    WHERE game_date = (SELECT MAX(game_date) FROM elo_ratings WHERE team = elo_ratings.team)
    ORDER BY composite_elo DESC
""")
all_rows = cursor.fetchall()
print(f"{'Team':<5} {'Off ELO':<10} {'Def ELO':<10} {'Comp ELO':<10} {'Latest Date'}")
for row in all_rows:
    print(f"{row[0]:<5} {row[1]:<10.1f} {row[2]:<10.1f} {row[3]:<10.1f} {row[4]}")

conn.close()

# Check predictions cache
print("\n" + "="*80)
print("PREDICTIONS CACHE - WAS @ IND GAME")
print("="*80)

cache_path = Path("predictions_cache.json")
if cache_path.exists():
    with open(cache_path, 'r') as f:
        cache = json.load(f)
    
    # Find WAS @ IND game
    was_ind_game = None
    for key, pred in cache.items():
        if 'WAS' in key and 'IND' in key:
            was_ind_game = pred
            print(f"\nGame: {key}")
            print(f"Home: {pred['home_team']}, Away: {pred['away_team']}")
            print(f"Home Win Prob: {pred['home_win_prob']:.2%}")
            print(f"Away Win Prob: {pred['away_win_prob']:.2%}")
            
            if pred.get('best_bet'):
                print(f"\nBest Bet: {pred['best_bet']['pick']}")
                print(f"  Edge: {pred['best_bet']['edge']:.2%}")
                print(f"  Model Prob: {pred['best_bet']['model_prob']:.2%}")
                print(f"  Market Prob: {pred['best_bet']['market_prob']:.2%}")
            
            print("\nKey Features:")
            features = pred.get('features', {})
            
            # ELO features
            print("\n  ELO Features:")
            for k, v in sorted(features.items()):
                if 'elo' in k.lower():
                    print(f"    {k}: {v:.2f}")
            
            # Win percentage
            print("\n  Win Percentage:")
            for k, v in sorted(features.items()):
                if 'win_pct' in k.lower():
                    print(f"    {k}: {v:.4f}")
            
            # Points per game
            print("\n  Points Per Game:")
            for k, v in sorted(features.items()):
                if 'ppg' in k.lower() or 'points' in k.lower():
                    print(f"    {k}: {v:.2f}")
            
            # Rest days
            print("\n  Rest:")
            for k, v in sorted(features.items()):
                if 'rest' in k.lower():
                    print(f"    {k}: {v}")
            
            # Injury impact
            print("\n  Injury Impact:")
            for k, v in sorted(features.items()):
                if 'injury' in k.lower() or 'shock' in k.lower():
                    print(f"    {k}: {v}")
            
            break
    
    if not was_ind_game:
        print("\nWAS @ IND game not found in predictions cache")
else:
    print("\nPredictions cache file not found")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("\nIf ELO shows 1500 for any team, that means:")
print("  1. The team has no ELO rating in the database")
print("  2. FeatureCalculatorLive is using default 1500 value")
print("  3. This indicates missing or incomplete ELO data")
print("\nExpected ELO range: ~1400-1600 for active NBA teams")
print("1500 = league average / default fallback")
