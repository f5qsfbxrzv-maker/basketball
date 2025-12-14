"""Check database schema and player data sources"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

# List all tables
print("=== DATABASE TABLES ===")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [t[0] for t in cursor.fetchall()]
for table in tables:
    print(f"  - {table}")

# Check for player-related tables with PIE data
print("\n=== CHECKING PLAYER DATA SOURCES ===")

for table in ['player_stats', 'player_season_metrics', 'historical_inactives']:
    if table not in tables:
        print(f"\n{table}: DOES NOT EXIST")
        continue
    
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"\n{table}: {count} rows")
    
    if count > 0:
        # Show structure
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"  Columns: {', '.join(columns)}")
        
        # Show sample
        df = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", conn)
        print(f"  Sample:")
        print(df.to_string(index=False))

conn.close()
