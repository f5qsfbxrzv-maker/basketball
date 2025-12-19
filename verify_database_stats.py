"""
Verify database has current stats and check what data we have
"""
import sqlite3
from datetime import datetime

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print("DATABASE VERIFICATION - December 17, 2025")
print("=" * 80)

# Check all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()
print(f"\n📊 TABLES IN DATABASE ({len(tables)}):")
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    print(f"   {table[0]}: {count} rows")

# Check if team_stats table exists
print("\n" + "=" * 80)
print("TEAM STATS FROM GAME RESULTS")
print("=" * 80)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_results'")
if cursor.fetchone():
    # Calculate records from game_results
    cursor.execute("""
        SELECT 
            home_team as team,
            SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins,
            SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END) as home_losses
        FROM game_results
        WHERE season = '2024-25'
        GROUP BY home_team
    """)
    home_records = {team: (wins, losses) for team, wins, losses in cursor.fetchall()}
    
    cursor.execute("""
        SELECT 
            away_team as team,
            SUM(CASE WHEN away_score > home_score THEN 1 ELSE 0 END) as away_wins,
            SUM(CASE WHEN away_score < home_score THEN 1 ELSE 0 END) as away_losses
        FROM game_results
        WHERE season = '2024-25'
        GROUP BY away_team
    """)
    away_records = {team: (wins, losses) for team, wins, losses in cursor.fetchall()}
    
    # Combine
    all_teams = set(home_records.keys()) | set(away_records.keys())
    team_records = []
    for team in all_teams:
        h_wins, h_losses = home_records.get(team, (0, 0))
        a_wins, a_losses = away_records.get(team, (0, 0))
        total_wins = h_wins + a_wins
        total_losses = h_losses + a_losses
        team_records.append((team, total_wins, total_losses))
    
    team_records.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n2024-25 SEASON RECORDS (from game_results):")
    for team, wins, losses in team_records:
        print(f"   {team}: {wins}-{losses}")
else:
    print("   ⚠️ game_results table does NOT exist")

# Check game_results table
print("\n" + "=" * 80)
print("GAME RESULTS TABLE")
print("=" * 80)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_results'")
if cursor.fetchone():
    cursor.execute("""
        SELECT 
            MIN(game_date) as first_game,
            MAX(game_date) as last_game,
            COUNT(*) as total_games
        FROM game_results
        WHERE season = '2024-25'
    """)
    result = cursor.fetchone()
    if result and result[0]:
        first_game, last_game, total_games = result
        print(f"   First game: {first_game}")
        print(f"   Last game: {last_game}")
        print(f"   Total games: {total_games}")
        
        # Check recent games
        cursor.execute("""
            SELECT game_date, home_team, away_team, home_score, away_score
            FROM game_results
            WHERE season = '2024-25'
            ORDER BY game_date DESC
            LIMIT 10
        """)
        recent = cursor.fetchall()
        print(f"\n   📅 MOST RECENT GAMES:")
        for game_date, home, away, home_score, away_score in recent:
            print(f"      {game_date}: {away} @ {home} ({away_score}-{home_score})")
    else:
        print("   ⚠️ No 2024-25 season data")
else:
    print("   ⚠️ game_results table does NOT exist")

# Check ELO ratings
print("\n" + "=" * 80)
print("ELO RATINGS TABLE")
print("=" * 80)
cursor.execute("""
    SELECT team, MAX(game_date) as latest_date, composite_elo
    FROM elo_ratings
    WHERE season = '2024-25'
    GROUP BY team
    ORDER BY composite_elo DESC
""")
elo_data = cursor.fetchall()
if elo_data:
    print(f"\n   LATEST ELO RATINGS (as of latest game date):")
    for team, latest_date, elo in elo_data:
        print(f"      {team}: {elo:.1f} (last updated: {latest_date})")
else:
    print("   ⚠️ No ELO data for 2024-25")

# Check injury data
print("\n" + "=" * 80)
print("INJURY DATA")
print("=" * 80)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='injuries'")
if cursor.fetchone():
    cursor.execute("""
        SELECT COUNT(*), MAX(last_updated)
        FROM injuries
    """)
    count, last_update = cursor.fetchone()
    print(f"   Total injuries: {count}")
    print(f"   Last updated: {last_update}")
    
    cursor.execute("""
        SELECT team, COUNT(*) as injury_count
        FROM injuries
        WHERE status IN ('Out', 'Doubtful')
        GROUP BY team
        ORDER BY injury_count DESC
        LIMIT 10
    """)
    injuries = cursor.fetchall()
    print(f"\n   TEAMS WITH MOST INJURIES (Out/Doubtful):")
    for team, count in injuries:
        print(f"      {team}: {count}")
else:
    print("   ⚠️ injuries table does NOT exist")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"   Current Date: December 17, 2025")
print(f"   Expected: Games through 12/17/2025")
print(f"   Expected: DET 20-5, DAL 10-16")

conn.close()
