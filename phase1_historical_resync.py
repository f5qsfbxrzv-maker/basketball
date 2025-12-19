"""
Phase 1: Historical Re-Sync with Gold Standard ELO Parameters
Re-processes ALL historical games (2024-25 + 2025-26) using optimized parameters.
This prevents data leakage and distribution shift in the ML model.
"""
import sqlite3
from src.features.off_def_elo_system import OffDefEloSystem
from datetime import datetime

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\data\live\nba_betting_data.db"

print("=" * 80)
print("PHASE 1: HISTORICAL RE-SYNC WITH GOLD STANDARD ELO")
print("=" * 80)
print("\nGold Standard Parameters (Grid Search Optimized):")
print("   K-Factor: 15.0")
print("   ELO_SCALE: 40.0")
print("   WIN_WEIGHT: 30.0")
print("   MOV_BIAS: 0.5")
print("\nThis ensures Feature Consistency across all historical data.")

# Get historical game counts
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT season, COUNT(DISTINCT GAME_ID) as game_count, MIN(GAME_DATE), MAX(GAME_DATE)
    FROM game_logs
    WHERE season IS NOT NULL
    GROUP BY season
    ORDER BY season
""")
seasons = cursor.fetchall()

print("\n📊 Historical Data Summary:")
for season, count, first, last in seasons:
    print(f"   {season}: {count} games ({first} to {last})")

total_games = sum(s[1] for s in seasons)
print(f"\n   Total Games: {total_games}")

# Confirm re-sync
print("\n⚠️  Starting historical re-sync:")
print("   1. Deleting ALL existing ELO ratings")
print("   2. Recalculating ELO for ALL games using Gold Standard parameters")
print("   3. This will take approximately 2-3 minutes for 24,706 games\n")

# Clear ALL ELO ratings
print("\n🗑️  Clearing all historical ELO ratings...")
cursor.execute("DELETE FROM elo_ratings")
deleted = cursor.rowcount
conn.commit()
print(f"   Deleted {deleted} old ELO entries")

# Initialize ELO system with Gold Standard parameters
print("\n⚙️  Initializing Gold Standard ELO system...")
print(f"   Database: {DB_PATH}")
elo_system = OffDefEloSystem(db_path=DB_PATH)

# Process ALL games chronologically (all seasons)
# game_logs has 2 rows per game (home and away), so we need to aggregate
cursor.execute("""
    WITH game_pairs AS (
        SELECT 
            g1.season,
            g1.GAME_DATE as game_date,
            CASE 
                WHEN g1.MATCHUP LIKE '%vs.%' THEN g1.TEAM_ABBREVIATION
                ELSE g2.TEAM_ABBREVIATION
            END as home_team,
            CASE 
                WHEN g1.MATCHUP LIKE '%vs.%' THEN g2.TEAM_ABBREVIATION
                ELSE g1.TEAM_ABBREVIATION
            END as away_team,
            CASE 
                WHEN g1.MATCHUP LIKE '%vs.%' THEN g1.PTS
                ELSE g2.PTS
            END as home_score,
            CASE 
                WHEN g1.MATCHUP LIKE '%vs.%' THEN g2.PTS
                ELSE g1.PTS
            END as away_score,
            g1.GAME_ID
        FROM game_logs g1
        JOIN game_logs g2 ON g1.GAME_ID = g2.GAME_ID AND g1.TEAM_ABBREVIATION < g2.TEAM_ABBREVIATION
        WHERE g1.season IS NOT NULL
        AND g1.PTS IS NOT NULL
        AND g2.PTS IS NOT NULL
    )
    SELECT season, game_date, home_team, away_team, home_score, away_score
    FROM game_pairs
    ORDER BY game_date, GAME_ID
""")
all_games = cursor.fetchall()
conn.close()

print(f"\n📈 Processing {len(all_games)} games with Gold Standard logic...")
print("   This may take 30-60 seconds...\n")

current_season = None
season_count = 0

for i, (season, game_date, home_team, away_team, home_score, away_score) in enumerate(all_games):
    # Track progress by season
    if season != current_season:
        if current_season:
            print(f"   ✅ {current_season}: {season_count} games processed")
        current_season = season
        season_count = 0
    
    season_count += 1
    
    # Show progress every 100 games
    if (i + 1) % 100 == 0:
        print(f"   Progress: {i+1}/{len(all_games)} games ({100*(i+1)/len(all_games):.1f}%)")
    
    elo_system.update_game(
        season=season,
        game_date=game_date,
        home_team=home_team,
        away_team=away_team,
        home_points=home_score,
        away_points=away_score,
        is_playoffs=False
    )

# Final season
if current_season:
    print(f"   ✅ {current_season}: {season_count} games processed")

print(f"\n✅ Historical re-sync complete! Processed {len(all_games)} games")

# Verify ELO statistics
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        team,
        MAX(composite_elo) as latest_elo,
        COUNT(*) as game_count
    FROM elo_ratings
    WHERE season = '2025-26'
    GROUP BY team
    ORDER BY latest_elo DESC
    LIMIT 5
""")
top_5 = cursor.fetchall()

print("\n" + "=" * 80)
print("VERIFICATION: Top 5 Teams (2025-26)")
print("=" * 80)
print(f"{'Team':<5} {'ELO':<10} {'Games':<8}")
print("-" * 30)
for team, elo, games in top_5:
    print(f"{team:<5} {elo:<10.1f} {games:<8}")

cursor.execute("""
    SELECT COUNT(DISTINCT team) as team_count,
           COUNT(*) as total_entries
    FROM elo_ratings
""")
team_count, total_entries = cursor.fetchone()

print(f"\n📊 Database Statistics:")
print(f"   Total Teams: {team_count}")
print(f"   Total ELO Entries: {total_entries}")

# Get ELO range for latest ratings
cursor.execute("""
    WITH latest_ratings AS (
        SELECT team, composite_elo,
               ROW_NUMBER() OVER (PARTITION BY team ORDER BY game_date DESC) as rn
        FROM elo_ratings
        WHERE season = '2025-26'
    )
    SELECT MAX(composite_elo) - MIN(composite_elo) as elo_range
    FROM latest_ratings
    WHERE rn = 1
""")
elo_range = cursor.fetchone()[0]

print(f"   Current ELO Range: {elo_range:.1f} points")
print(f"   {'✅ HEALTHY' if 200 <= elo_range <= 250 else '⚠️ CHECK'}: Expected 200-250 points")

conn.close()

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE: HISTORICAL DATA RE-SYNCED")
print("=" * 80)
print("\n📋 Next Steps:")
print("   1. ✅ Historical ELO ratings updated with Gold Standard parameters")
print("   2. ⏭️  Run Phase 2: Regenerate training features with new ELO values")
print("   3. ⏭️  Run Phase 3: Optuna hyperparameter optimization + model retraining")
print("\nTo proceed: python phase2_regenerate_training_data.py")
