import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables in database:")
for t in tables:
    print(f"  - {t[0]}")
    
# Check team_stats structure
print("\nteam_stats columns:")
cols = conn.execute("PRAGMA table_info(team_stats)").fetchall()
for col in cols:
    print(f"  {col[1]} ({col[2]})")
    
print(f"\nteam_stats row count: {conn.execute('SELECT COUNT(*) FROM team_stats').fetchone()[0]}")

# Check if game_advanced_stats exists
try:
    print("\ngame_advanced_stats columns:")
    cols = conn.execute("PRAGMA table_info(game_advanced_stats)").fetchall()
    for col in cols:
        print(f"  {col[1]} ({col[2]})")
    print(f"\ngame_advanced_stats row count: {conn.execute('SELECT COUNT(*) FROM game_advanced_stats').fetchone()[0]}")
except:
    print("\n❌ game_advanced_stats table does NOT exist")

conn.close()
