"""Check database tables and team_records structure"""
import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("📊 Tables in database:")
for t in tables:
    print(f"  - {t[0]}")

# Check if team_records exists
if ('team_records',) in tables:
    print("\n✅ team_records table exists")
    cursor.execute("PRAGMA table_info(team_records)")
    columns = cursor.fetchall()
    print("\nColumns:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Sample data
    cursor.execute("SELECT * FROM team_records LIMIT 3")
    rows = cursor.fetchall()
    print("\nSample data:")
    for row in rows:
        print(f"  {row}")
else:
    print("\n❌ team_records table does NOT exist")
    print("Checking game_logs for team stats...")
    cursor.execute("SELECT DISTINCT team FROM game_logs LIMIT 5")
    teams = cursor.fetchall()
    print(f"Sample teams in game_logs: {teams}")

# Check for head-to-head data in game_logs
print("\n🔍 Checking game_logs for head-to-head matchups...")
cursor.execute("""
    SELECT game_date, team, opponent, team_score, opponent_score, wl
    FROM game_logs 
    WHERE (team = 'GSW' AND opponent = 'MIN') OR (team = 'MIN' AND opponent = 'GSW')
    ORDER BY game_date DESC
    LIMIT 5
""")
h2h = cursor.fetchall()
print(f"Recent GSW vs MIN games: {len(h2h)} found")
for game in h2h:
    print(f"  {game}")

conn.close()
