"""
Debug injury feature calculation to see what's happening
"""

import pandas as pd
import sqlite3

DB_PATH = "data/live/nba_betting_data.db"

# Test query
conn = sqlite3.connect(DB_PATH)

# Get a recent game
game = pd.read_sql("""
    SELECT game_date, home_team, away_team
    FROM game_results
    WHERE game_date >= '2024-10-22'
    ORDER BY game_date
    LIMIT 1
""", conn)

print(f"Test game: {game['game_date'].values[0]} - {game['away_team'].values[0]} @ {game['home_team'].values[0]}")

game_date = game['game_date'].values[0]
home_team = game['home_team'].values[0]
away_team = game['away_team'].values[0]

# Check if injuries exist for this date
injuries = pd.read_sql("""
    SELECT game_date, player_name, team_abbreviation
    FROM historical_inactives
    WHERE game_date = ?
""", conn, params=(game_date,))

print(f"\nInjuries on {game_date}:")
print(f"  Total: {len(injuries)}")
if len(injuries) > 0:
    print(f"\n  Sample:")
    print(injuries.head(10))
    
    # Check for our specific teams
    home_inj = injuries[injuries['team_abbreviation'] == home_team]
    away_inj = injuries[injuries['team_abbreviation'] == away_team]
    
    print(f"\n  {home_team} injuries: {len(home_inj)}")
    print(f"  {away_team} injuries: {len(away_inj)}")
else:
    print("  NO INJURIES FOUND!")
    
    # Check if any injuries exist at all
    any_injuries = pd.read_sql("""
        SELECT COUNT(*) as count, MIN(game_date) as min_date, MAX(game_date) as max_date
        FROM historical_inactives
    """, conn)
    print(f"\n  Total injuries in DB: {any_injuries['count'].values[0]}")
    print(f"  Date range: {any_injuries['min_date'].values[0]} to {any_injuries['max_date'].values[0]}")

conn.close()
