"""
Recalculate ALL ELO ratings with new Syndicate-Level logic
This updates the features to match the new ELO system
"""
import sqlite3
from src.features.off_def_elo_system import OffDefEloSystem

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

print("=" * 80)
print("RECALCULATING ELO WITH SYNDICATE-LEVEL ALGORITHM")
print("=" * 80)

print("""
New Features Implemented:
✓ Auto-Regressive K-Factor (K=32 first 20 games, K=20 after)
✓ Logarithmic Margin Dampening (prevents Brooklyn blowout issue)
✓ 75/25 Season Regression (regress toward 1505)
✓ Garbage Time Filter (via log formula)
""")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get game count
cursor.execute("""
    SELECT COUNT(*), MIN(game_date), MAX(game_date)
    FROM game_results
    WHERE season = '2025-26'
""")
count, first, last = cursor.fetchone()
print(f"\n📊 Games to Process:")
print(f"   Season: 2025-26")
print(f"   Games: {count}")
print(f"   Date range: {first} to {last}")

# Clear old ELO
print(f"\n🗑️  Clearing old ELO ratings...")
cursor.execute("DELETE FROM elo_ratings WHERE season = '2025-26'")
deleted = cursor.rowcount
conn.commit()
print(f"   Deleted {deleted} old ELO entries")

# Initialize ELO system with NEW logic
print(f"\n⚙️  Initializing Syndicate-Level ELO system...")
elo_system = OffDefEloSystem(db_path=DB_PATH)

# Process all games chronologically
cursor.execute("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_results
    WHERE season = '2025-26'
    ORDER BY game_date, ROWID
""")
all_games = cursor.fetchall()

print(f"\n📈 Processing {len(all_games)} games with NEW syndicate logic...")

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

print(f"✅ Processed all {len(all_games)} games with syndicate algorithm")

# Show NEW ELO rankings
print("\n" + "=" * 80)
print("NEW ELO RANKINGS (Syndicate-Level)")
print("=" * 80)

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

print(f"\n{'Rank':<6} {'Team':<5} {'ELO':<8} {'Off':<7} {'Def':<7} {'Record':<10}")
print("=" * 55)

for rank, (team, composite, off_elo, def_elo) in enumerate(elo_data, 1):
    h_wins, h_losses = home_records.get(team, (0, 0))
    a_wins, a_losses = away_records.get(team, (0, 0))
    wins = h_wins + a_wins
    losses = h_losses + a_losses
    
    flag = "⭐" if team in ['DET', 'DAL', 'BKN'] else ""
    print(f"{rank:<6} {team:<5} {composite:<8.1f} {off_elo:<7.1f} {def_elo:<7.1f} {wins}-{losses:<9} {flag}")

# Check Brooklyn specifically
print("\n" + "=" * 80)
print("BROOKLYN VERIFICATION")
print("=" * 80)

cursor.execute("""
    SELECT team, composite_elo, off_elo, def_elo
    FROM elo_ratings
    WHERE season = '2025-26' AND team = 'BKN'
    ORDER BY game_date DESC LIMIT 1
""")
bkn = cursor.fetchone()

if bkn:
    h_wins, h_losses = home_records.get('BKN', (0, 0))
    a_wins, a_losses = away_records.get('BKN', (0, 0))
    wins = h_wins + a_wins
    losses = h_losses + a_losses
    
    print(f"\nBrooklyn Nets:")
    print(f"   Record: {wins}-{losses}")
    print(f"   Composite ELO: {bkn[1]:.1f}")
    print(f"   Off ELO: {bkn[2]:.1f}")
    print(f"   Def ELO: {bkn[3]:.1f}")
    print(f"\n   Expected: Should be MUCH LOWER than old 1595")
    print(f"   Expected: Should rank around 20-25th, not 2nd")

# Check ELO spread
elos = [x[1] for x in elo_data]
print("\n" + "=" * 80)
print("ELO RANGE ANALYSIS")
print("=" * 80)
print(f"\n   Highest ELO: {max(elos):.1f}")
print(f"   Lowest ELO: {min(elos):.1f}")
print(f"   Range: {max(elos) - min(elos):.1f} points")
print(f"   Average: {sum(elos)/len(elos):.1f}")
print(f"\n   Expected: Range should be WIDER (200-250 points)")
print(f"   Old Range: 169.5 points (too condensed)")

conn.close()

print("\n" + "=" * 80)
print("✅ SYNDICATE-LEVEL ELO UPDATED!")
print("=" * 80)
print("\n📋 Next Steps:")
print("   1. Check if Brooklyn dropped out of top 5 (should be ~rank 20-25)")
print("   2. Check if Detroit moved up (should be top 3-5)")
print("   3. Run validation test to see if model retraining is needed")
print("   4. Refresh dashboard to see corrected predictions")
