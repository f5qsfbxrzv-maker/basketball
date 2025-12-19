"""
Fix database: Remove future games and recalculate ELO through today (Dec 17, 2025)
"""
import sqlite3
from datetime import datetime
from src.features.off_def_elo_system import OffDefEloSystem

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print("DATABASE CLEANUP - Removing Future Games")
print("=" * 80)

# Check current data
cursor.execute("""
    SELECT 
        MIN(game_date) as first_game,
        MAX(game_date) as last_game,
        COUNT(*) as total_games
    FROM game_results
    WHERE season = '2024-25'
""")
first, last, total = cursor.fetchone()
print(f"\nBEFORE CLEANUP:")
print(f"   First game: {first}")
print(f"   Last game: {last}")
print(f"   Total games: {total}")

# Delete games after today (Dec 17, 2025)
cursor.execute("""
    DELETE FROM game_results
    WHERE game_date > '2025-12-17'
""")
deleted = cursor.rowcount
conn.commit()

print(f"\n✅ Deleted {deleted} future games")

# Check after cleanup
cursor.execute("""
    SELECT 
        MIN(game_date) as first_game,
        MAX(game_date) as last_game,
        COUNT(*) as total_games
    FROM game_results
    WHERE season = '2024-25'
""")
first, last, total = cursor.fetchone()
print(f"\nAFTER CLEANUP:")
print(f"   First game: {first}")
print(f"   Last game: {last}")
print(f"   Total games: {total}")

# Calculate actual records through today
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

all_teams = set(home_records.keys()) | set(away_records.keys())
team_records = []
for team in all_teams:
    h_wins, h_losses = home_records.get(team, (0, 0))
    a_wins, a_losses = away_records.get(team, (0, 0))
    total_wins = h_wins + a_wins
    total_losses = h_losses + a_losses
    team_records.append((team, total_wins, total_losses))

team_records.sort(key=lambda x: x[1], reverse=True)

print(f"\n📊 CURRENT RECORDS (through 12/17/2025):")
for team, wins, losses in team_records:
    if team in ['DET', 'DAL', 'CLE', 'OKC']:  # Highlight key teams
        print(f"   {team}: {wins}-{losses} ⭐")
    else:
        print(f"   {team}: {wins}-{losses}")

# Clear old ELO data
print("\n" + "=" * 80)
print("RECALCULATING ELO RATINGS")
print("=" * 80)

cursor.execute("DELETE FROM elo_ratings")
conn.commit()
print("✅ Cleared old ELO data")

# Initialize ELO system and process all games chronologically
elo_system = OffDefEloSystem(db_path=DB_PATH)

cursor.execute("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_results
    WHERE season = '2024-25'
    ORDER BY game_date, ROWID
""")
all_games = cursor.fetchall()

print(f"📊 Processing {len(all_games)} games chronologically...")

for i, (game_date, home_team, away_team, home_score, away_score) in enumerate(all_games):
    if (i + 1) % 100 == 0:
        print(f"   Processed {i+1}/{len(all_games)} games...")
    
    elo_system.update_game(
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        game_date=game_date,
        season='2024-25'
    )

print(f"✅ Processed all {len(all_games)} games")

# Show final ELO ratings
print("\n" + "=" * 80)
print("FINAL ELO RATINGS (through 12/17/2025)")
print("=" * 80)

cursor.execute("""
    SELECT team, MAX(game_date) as latest_date, composite_elo, off_elo, def_elo
    FROM elo_ratings
    WHERE season = '2024-25'
    GROUP BY team
    ORDER BY composite_elo DESC
""")
elo_data = cursor.fetchall()

for team, latest_date, composite, off_elo, def_elo in elo_data:
    # Get record
    h_wins, h_losses = home_records.get(team, (0, 0))
    a_wins, a_losses = away_records.get(team, (0, 0))
    wins = h_wins + a_wins
    losses = h_losses + a_losses
    
    if team in ['DET', 'DAL']:
        print(f"   {team}: {composite:.1f} (O:{off_elo:.0f} D:{def_elo:.0f}) [{wins}-{losses}] ⭐")
    else:
        print(f"   {team}: {composite:.1f} (O:{off_elo:.0f} D:{def_elo:.0f}) [{wins}-{losses}]")

conn.close()

print("\n" + "=" * 80)
print("✅ DATABASE FIXED!")
print("=" * 80)
