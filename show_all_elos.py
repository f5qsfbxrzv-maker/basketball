"""
Display all team ELO ratings for 2025-26 season
"""
import sqlite3

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print("ALL TEAM ELO RATINGS - 2025-26 Season")
print("=" * 80)

# Get all teams' latest ELO
cursor.execute("""
    SELECT DISTINCT team FROM elo_ratings WHERE season = '2025-26'
""")
all_teams = [row[0] for row in cursor.fetchall()]

print(f"\nFound {len(all_teams)} teams with ELO data")

# Get latest ELO for each team
team_elos = []
for team in all_teams:
    cursor.execute("""
        SELECT team, game_date, composite_elo, off_elo, def_elo
        FROM elo_ratings
        WHERE season = '2025-26' AND team = ?
        ORDER BY game_date DESC, ROWID DESC
        LIMIT 1
    """, (team,))
    
    result = cursor.fetchone()
    if result:
        team_elos.append(result)

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

# Sort by composite ELO
team_elos.sort(key=lambda x: x[2], reverse=True)

print("\n" + "=" * 90)
print(f"{'Rank':<6} {'Team':<5} {'ELO':<8} {'Off ELO':<8} {'Def ELO':<8} {'Record':<10} {'Last Game'}")
print("=" * 90)

for rank, (team, last_date, composite, off_elo, def_elo) in enumerate(team_elos, 1):
    h_wins, h_losses = home_records.get(team, (0, 0))
    a_wins, a_losses = away_records.get(team, (0, 0))
    wins = h_wins + a_wins
    losses = h_losses + a_losses
    
    flag = "⭐" if team in ['DET', 'DAL'] else ""
    print(f"{rank:<6} {team:<5} {composite:<8.1f} {off_elo:<8.1f} {def_elo:<8.1f} {wins}-{losses:<8} {last_date} {flag}")

print("=" * 90)

# Stats
elos = [x[2] for x in team_elos]
print(f"\nELO Statistics:")
print(f"   Highest: {max(elos):.1f}")
print(f"   Lowest: {min(elos):.1f}")
print(f"   Range: {max(elos) - min(elos):.1f} points")
print(f"   Average: {sum(elos)/len(elos):.1f}")

conn.close()
