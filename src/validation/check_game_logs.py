"""
Check game logs database coverage
"""
import sqlite3
from datetime import datetime

db_path = 'data/live/nba_betting_data.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all game-related tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%game%'")
tables = cursor.fetchall()
print("=" * 70)
print("GAME-RELATED TABLES")
print("=" * 70)
for table in tables:
    print(f"  - {table[0]}")

print("\n" + "=" * 70)
print("DATA COVERAGE")
print("=" * 70)

# Check game_logs
try:
    cursor.execute("SELECT MIN(game_date), MAX(game_date), COUNT(*) FROM game_logs")
    min_date, max_date, count = cursor.fetchone()
    print(f"\ngame_logs:")
    print(f"  Date Range: {min_date} to {max_date}")
    print(f"  Total Rows: {count:,}")
    
    # Check recent data (last 30 days)
    cursor.execute("SELECT COUNT(*) FROM game_logs WHERE game_date >= date('now', '-30 days')")
    recent_count = cursor.fetchone()[0]
    print(f"  Last 30 Days: {recent_count:,} games")
    
    # Check current season (2024-25)
    cursor.execute("SELECT COUNT(*) FROM game_logs WHERE game_date >= '2024-10-01'")
    season_count = cursor.fetchone()[0]
    print(f"  2024-25 Season: {season_count:,} games")
    
except Exception as e:
    print(f"\ngame_logs: ERROR - {e}")

# Check game_advanced_stats
try:
    cursor.execute("SELECT MIN(game_date), MAX(game_date), COUNT(*) FROM game_advanced_stats")
    min_date, max_date, count = cursor.fetchone()
    print(f"\ngame_advanced_stats:")
    print(f"  Date Range: {min_date} to {max_date}")
    print(f"  Total Rows: {count:,}")
    
    # Check recent data
    cursor.execute("SELECT COUNT(*) FROM game_advanced_stats WHERE game_date >= date('now', '-30 days')")
    recent_count = cursor.fetchone()[0]
    print(f"  Last 30 Days: {recent_count:,} games")
    
    # Check current season
    cursor.execute("SELECT COUNT(*) FROM game_advanced_stats WHERE game_date >= '2024-10-01'")
    season_count = cursor.fetchone()[0]
    print(f"  2024-25 Season: {season_count:,} games")
    
except Exception as e:
    print(f"\ngame_advanced_stats: ERROR - {e}")

print("\n" + "=" * 70)
print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
print("=" * 70)

conn.close()
