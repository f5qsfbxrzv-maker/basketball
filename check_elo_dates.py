"""Check latest ELO update dates and team records"""
import sqlite3
from datetime import datetime

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print("ELO RATINGS STATUS CHECK")
print("=" * 80)

# Check latest dates in elo_ratings
cursor.execute("""
    SELECT team, MAX(game_date) as latest_date, MAX(composite_elo) as latest_elo
    FROM elo_ratings
    WHERE team IN ('DAL', 'DET')
    GROUP BY team
    ORDER BY team
""")

print("\n📅 LATEST ELO DATES:")
for team, date, elo in cursor.fetchall():
    print(f"   {team}: {date} → ELO {elo:.1f}")

# Check if there are any recent entries (Dec 2025)
cursor.execute("""
    SELECT COUNT(*) FROM elo_ratings 
    WHERE game_date >= '2025-12-01'
""")
dec_count = cursor.fetchone()[0]
print(f"\n📊 December 2025 ELO entries: {dec_count}")

# Check latest date overall
cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
latest_overall = cursor.fetchone()[0]
print(f"📊 Latest ELO date (any team): {latest_overall}")

# Check team_stats for current records
cursor.execute("""
    SELECT team, season, wins, losses, last_updated
    FROM team_stats
    WHERE team IN ('DAL', 'DET')
    AND season = '2024-25'
""")

print("\n" + "=" * 80)
print("TEAM RECORDS (from team_stats)")
print("=" * 80)
records = cursor.fetchall()
if records:
    for team, season, wins, losses, updated in records:
        print(f"   {team}: {wins}-{losses} (updated: {updated})")
else:
    print("   ⚠️  No team_stats found for 2024-25 season")

# Check game_results for actual records
cursor.execute("""
    SELECT 
        home_team as team,
        SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins,
        SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END) as home_losses
    FROM game_results
    WHERE season = '2024-25' AND home_team IN ('DAL', 'DET')
    GROUP BY home_team
""")
home_records = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

cursor.execute("""
    SELECT 
        away_team as team,
        SUM(CASE WHEN away_score > home_score THEN 1 ELSE 0 END) as away_wins,
        SUM(CASE WHEN away_score < home_score THEN 1 ELSE 0 END) as away_losses
    FROM game_results
    WHERE season = '2024-25' AND away_team IN ('DAL', 'DET')
    GROUP BY away_team
""")
away_records = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

print("\n" + "=" * 80)
print("ACTUAL RECORDS (from game_results)")
print("=" * 80)
for team in ['DAL', 'DET']:
    home_w, home_l = home_records.get(team, (0, 0))
    away_w, away_l = away_records.get(team, (0, 0))
    total_w = home_w + away_w
    total_l = home_l + away_l
    print(f"   {team}: {total_w}-{total_l} ({home_w}-{home_l} home, {away_w}-{away_l} away)")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print("Expected ELO based on records:")
print("   DET (20-5): ~1700 ELO")
print("   DAL (10-16): ~1450 ELO")
print("\nActual ELO in database:")

cursor.execute("""
    SELECT team, composite_elo, game_date
    FROM elo_ratings
    WHERE team IN ('DAL', 'DET')
    ORDER BY game_date DESC
    LIMIT 2
""")
for team, elo, date in cursor.fetchall():
    print(f"   {team}: {elo:.1f} (as of {date})")

print("\n⚠️  ELO ratings are STALE and don't reflect current season performance!")
print("⚠️  Need to run ELO update process to recalculate based on 2024-25 games")

conn.close()
