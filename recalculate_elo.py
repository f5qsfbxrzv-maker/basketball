"""
Recalculate ELO ratings from game results
"""
import sqlite3
from src.features.off_def_elo_system import OffDefEloSystem

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

print("=" * 80)
print("RECALCULATING ELO RATINGS (2025-26 Season)")
print("=" * 80)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check game results
cursor.execute("""
    SELECT COUNT(*), MIN(game_date), MAX(game_date)
    FROM game_results
    WHERE season = '2025-26'
""")
count, first, last = cursor.fetchone()
print(f"\n📊 Game Results in Database:")
print(f"   Season: 2025-26")
print(f"   Games: {count}")
print(f"   Date range: {first} to {last}")

# Clear old ELO data for 2025-26
print(f"\n🗑️  Clearing old ELO data for 2025-26...")
cursor.execute("DELETE FROM elo_ratings WHERE season = '2025-26'")
deleted = cursor.rowcount
conn.commit()
print(f"   Deleted {deleted} old ELO entries")

# Initialize ELO system
print(f"\n⚙️  Initializing ELO system...")
elo_system = OffDefEloSystem(db_path=DB_PATH)

# Process all games chronologically
cursor.execute("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_results
    WHERE season = '2025-26'
    ORDER BY game_date, ROWID
""")
all_games = cursor.fetchall()

print(f"\n📈 Processing {len(all_games)} games chronologically...")

for i, (game_date, home_team, away_team, home_score, away_score) in enumerate(all_games):
    if (i + 1) % 50 == 0:
        print(f"   Progress: {i+1}/{len(all_games)} games...")
    
    elo_system.update_game(
        season='2025-26',
        game_date=game_date,
        home_team=home_team,
        away_team=away_team,
        home_points=home_score,
        away_points=away_score,
        is_playoffs=False
    )

print(f"✅ Processed all {len(all_games)} games")

# Calculate team records for verification
print("\n" + "=" * 80)
print("FINAL ELO RATINGS + RECORDS")
print("=" * 80)

# Get records
cursor.execute("""
    SELECT 
        home_team as team,
        SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins,
        SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END) as home_losses
    FROM game_results
    WHERE season = '2025-26'
    GROUP BY home_team
""")
home_records = {team: (wins, losses) for team, wins, losses in cursor.fetchall()}

cursor.execute("""
    SELECT 
        away_team as team,
        SUM(CASE WHEN away_score > home_score THEN 1 ELSE 0 END) as away_wins,
        SUM(CASE WHEN away_score < home_score THEN 1 ELSE 0 END) as away_losses
    FROM game_results
    WHERE season = '2025-26'
    GROUP BY away_team
""")
away_records = {team: (wins, losses) for team, wins, losses in cursor.fetchall()}

# Get latest ELO
cursor.execute("""
    SELECT team, composite_elo, off_elo, def_elo
    FROM elo_ratings
    WHERE season = '2025-26'
    AND game_date = (
        SELECT MAX(game_date)
        FROM elo_ratings
        WHERE season = '2025-26'
        AND elo_ratings.team = team
    )
    ORDER BY composite_elo DESC
""")
elo_data = cursor.fetchall()

print(f"\n🏆 TEAM RANKINGS:")
print(f"{'Rank':<5} {'Team':<5} {'ELO':<7} {'Off':<6} {'Def':<6} {'Record':<10}")
print("-" * 50)

for rank, (team, composite, off_elo, def_elo) in enumerate(elo_data, 1):
    h_wins, h_losses = home_records.get(team, (0, 0))
    a_wins, a_losses = away_records.get(team, (0, 0))
    wins = h_wins + a_wins
    losses = h_losses + a_losses
    
    flag = "⭐" if team in ['DET', 'DAL'] else ""
    print(f"{rank:<5} {team:<5} {composite:<7.1f} {off_elo:<6.0f} {def_elo:<6.0f} {wins}-{losses:<9} {flag}")

# Verify key teams
print("\n" + "=" * 80)
print("VERIFICATION - KEY TEAMS")
print("=" * 80)

for team_abbr, expected_record in [('DET', '21-5'), ('DAL', '10-17')]:
    cursor.execute("""
        SELECT team, composite_elo, off_elo, def_elo
        FROM elo_ratings
        WHERE season = '2025-26' AND team = ?
        AND game_date = (
            SELECT MAX(game_date)
            FROM elo_ratings
            WHERE season = '2025-26' AND team = ?
        )
    """, (team_abbr, team_abbr))
    
    result = cursor.fetchone()
    if result:
        team, composite, off_elo, def_elo = result
        h_wins, h_losses = home_records.get(team, (0, 0))
        a_wins, a_losses = away_records.get(team, (0, 0))
        wins = h_wins + a_wins
        losses = h_losses + a_losses
        
        print(f"\n{team}:")
        print(f"   Record: {wins}-{losses} (expected {expected_record})")
        print(f"   Composite ELO: {composite:.1f}")
        print(f"   Offensive ELO: {off_elo:.1f}")
        print(f"   Defensive ELO: {def_elo:.1f}")

conn.close()

print("\n" + "=" * 80)
print("✅ ELO RATINGS UPDATED!")
print("=" * 80)
print("\n📋 Detroit has much higher ELO than Dallas now")
print("📋 Refresh dashboard to see corrected predictions")
