import sqlite3
import os

db_files = [
    'nba_betting_data.db',
    'data/live/nba_betting_PRED.db',
    'data/backups/nba_PREDICTIONS.db',
    'models/conservative_3_pct_spread.db'
]

for db_path in db_files:
    if not os.path.exists(db_path):
        continue
    
    print(f"\n{'='*80}")
    print(f"Database: {db_path}")
    print('='*80)
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Check for game_results table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_results'")
        if cur.fetchone():
            cur.execute("SELECT season, COUNT(*) FROM game_results GROUP BY season ORDER BY season")
            seasons = cur.fetchall()
            print(f"\nSeasons in game_results ({len(seasons)} seasons):")
            for season, count in seasons:
                print(f"  {season}: {count} games")
            
            cur.execute("SELECT COUNT(*) FROM game_results")
            total = cur.fetchone()[0]
            print(f"\nTotal games: {total}")
        else:
            print("No game_results table")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")
