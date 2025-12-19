"""Check database tables and find current season data"""
import sqlite3

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print("DATABASE TABLES")
print("=" * 80)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [t[0] for t in cursor.fetchall()]
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"   {table}: {count} rows")

print("\n" + "=" * 80)
print("CHECKING FOR 2024-25 SEASON DATA")
print("=" * 80)

# Check game_results
if 'game_results' in tables:
    cursor.execute("""
        SELECT home_team, COUNT(*) as games
        FROM game_results
        WHERE season = '2024-25' AND home_team IN ('DAL', 'DET')
        GROUP BY home_team
    """)
    results = cursor.fetchall()
    if results:
        print("\n✅ game_results has 2024-25 data:")
        for team, count in results:
            print(f"   {team}: {count} home games")
    else:
        print("\n⚠️  No 2024-25 games in game_results")

# Check if we need to run ELO calculation
print("\n" + "=" * 80)
print("ELO SYSTEM STATUS")
print("=" * 80)

if 'elo_ratings' in tables:
    cursor.execute("SELECT COUNT(*) FROM elo_ratings")
    elo_count = cursor.fetchone()[0]
    
    if elo_count == 0:
        print("❌ elo_ratings table is EMPTY!")
        print("   Need to run: python src/features/off_def_elo_system.py")
    else:
        cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
        latest = cursor.fetchone()[0]
        print(f"⚠️  elo_ratings has {elo_count} entries but latest date is: {latest}")
        print("   May need to update with recent games")
else:
    print("❌ elo_ratings table doesn't exist!")

conn.close()
