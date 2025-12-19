"""
EMERGENCY FULL DATA REBUILD
- Clears corrupted ELO table
- Fetches ALL games from Oct 2023 to present
- Rebuilds ELO from scratch
- Verifies Away ELO is not 0.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
from datetime import datetime
from config.settings import DB_PATH

print("=" * 80)
print("EMERGENCY FULL DATA REBUILD")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\n⚠️  WARNING: This will DELETE all ELO ratings and rebuild from scratch!")
print("   Press Ctrl+C within 5 seconds to cancel...")

import time
for i in range(5, 0, -1):
    print(f"   Starting in {i}...")
    time.sleep(1)

print("\n✓ Starting full rebuild...\n")

# Step 1: Backup and clear ELO table
print("[1/5] Backing up and clearing ELO table...")
try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Count existing records
    cursor.execute("SELECT COUNT(*) FROM elo_ratings")
    old_count = cursor.fetchone()[0]
    print(f"  Found {old_count:,} existing ELO records")
    
    # Backup to CSV
    backup_file = f"data/backups/elo_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"  Backing up to: {backup_file}")
    cursor.execute("SELECT * FROM elo_ratings")
    import pandas as pd
    backup_df = pd.DataFrame(cursor.fetchall(), 
                            columns=['team', 'season', 'game_date', 'off_elo', 'def_elo', 'composite_elo', 'is_playoffs'])
    backup_df.to_csv(backup_file, index=False)
    print(f"  ✓ Backup saved")
    
    # Clear table
    cursor.execute("DELETE FROM elo_ratings")
    conn.commit()
    print(f"  ✓ Cleared ELO table")
    
    conn.close()
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Step 2: Update game logs (ALL seasons)
print("\n[2/5] Fetching ALL game logs from Oct 2023 to present...")
try:
    from src.collectors.update_game_logs import update_game_logs
    
    # Fetch 2023-24 season
    print("\n  Fetching 2023-24 season...")
    update_game_logs(season='2023-24', last_n_days=None)
    
    # Fetch 2024-25 season (current)
    print("\n  Fetching 2024-25 season...")
    update_game_logs(season='2024-25', last_n_days=None)
    
    print("\n  ✓ All game logs fetched")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Update advanced stats
print("\n[3/5] Fetching advanced stats...")
try:
    from src.collectors.update_game_advanced_stats import update_game_advanced_stats
    update_game_advanced_stats()
    print("  ✓ Advanced stats updated")
except Exception as e:
    print(f"  ✗ WARNING: {e}")

# Step 4: Rebuild ELO from scratch
print("\n[4/5] Rebuilding ELO from EVERY game since Oct 2023...")
try:
    from src.features.off_def_elo_system import OffDefEloSystem
    
    elo_system = OffDefEloSystem(db_path=str(DB_PATH))
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Get ALL games, ordered chronologically
    cursor.execute("""
        SELECT game_date, home_team, away_team, home_score, away_score
        FROM game_logs
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY game_date, game_id
    """)
    
    games = cursor.fetchall()
    total_games = len(games)
    print(f"  Found {total_games:,} completed games to process")
    
    if total_games == 0:
        print("  ✗ ERROR: No games found in database!")
        sys.exit(1)
    
    # Initialize all teams for 2023-24 season
    cursor.execute("SELECT DISTINCT home_team FROM game_logs")
    all_teams = [row[0] for row in cursor.fetchall()]
    print(f"  Initializing {len(all_teams)} teams...")
    elo_system.initialize_season('2023-24', all_teams)
    
    # Process each game
    print(f"  Processing games...")
    processed = 0
    errors = 0
    
    for i, (game_date, home, away, home_score, away_score) in enumerate(games):
        if i % 100 == 0:
            pct = (i / total_games) * 100
            print(f"    Progress: {i:,}/{total_games:,} ({pct:.1f}%) - Latest: {game_date}")
        
        try:
            # Determine season
            year = int(game_date[:4])
            month = int(game_date[5:7])
            season = f"{year}-{str(year+1)[-2:]}" if month >= 10 else f"{year-1}-{str(year)[-2:]}"
            
            elo_system.update_elo(
                home_team=home,
                away_team=away,
                home_score=home_score,
                away_score=away_score,
                game_date=game_date,
                season=season
            )
            processed += 1
            
        except Exception as e:
            errors += 1
            if errors < 10:  # Only print first 10 errors
                print(f"    ⚠️  Error processing {away}@{home} on {game_date}: {e}")
    
    conn.close()
    
    print(f"\n  ✓ ELO rebuild complete!")
    print(f"    Processed: {processed:,} games")
    print(f"    Errors: {errors}")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Verify ELO integrity
print("\n[5/5] Verifying ELO integrity...")
try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Count ELO records
    cursor.execute("SELECT COUNT(*) FROM elo_ratings")
    new_count = cursor.fetchone()[0]
    print(f"  ✓ Created {new_count:,} ELO records")
    
    # Check for impossible values (outside 500-2500 range)
    cursor.execute("""
        SELECT team, game_date, off_elo, def_elo, composite_elo
        FROM elo_ratings
        WHERE off_elo < 500 OR off_elo > 2500 
           OR def_elo < 500 OR def_elo > 2500
           OR composite_elo < 500 OR composite_elo > 2500
    """)
    
    bad_elos = cursor.fetchall()
    if bad_elos:
        print(f"  ⚠️  WARNING: Found {len(bad_elos)} records with impossible ELO values!")
        for team, date, off, def_, comp in bad_elos[:5]:
            print(f"     {team} on {date}: off={off:.1f}, def={def_:.1f}, comp={comp:.1f}")
    else:
        print(f"  ✓ All ELO values within reasonable range (500-2500)")
    
    # Show sample of latest ELO values
    cursor.execute("""
        SELECT team, game_date, off_elo, def_elo, composite_elo
        FROM elo_ratings
        WHERE game_date = (SELECT MAX(game_date) FROM elo_ratings)
        ORDER BY composite_elo DESC
        LIMIT 5
    """)
    
    print("\n  Latest ELO ratings (Top 5 teams):")
    for team, date, off, def_, comp in cursor.fetchall():
        print(f"    {team}: off={off:.1f}, def={def_:.1f}, comp={comp:.1f} (as of {date})")
    
    conn.close()
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")

# Step 6: Update injuries
print("\n[6/6] Updating injury data...")
try:
    from src.services.live_injury_updater import LiveInjuryUpdater
    
    updater = LiveInjuryUpdater(db_path=str(DB_PATH))
    injury_count = updater.update_active_injuries()
    print(f"  ✓ Updated {injury_count} active injuries from ESPN")
except Exception as e:
    print(f"  ⚠️  WARNING: {e}")

# Final summary
print("\n" + "=" * 80)
print("REBUILD COMPLETE!")
print("=" * 80)

try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT MAX(game_date) FROM game_logs")
    latest_game = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
    latest_elo = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT team) FROM elo_ratings WHERE game_date = ?", (latest_elo,))
    team_count = cursor.fetchone()[0]
    
    print(f"\n✓ Latest game in database: {latest_game}")
    print(f"✓ Latest ELO update: {latest_elo}")
    print(f"✓ Teams with current ELO: {team_count}")
    
    # Check freshness
    today = datetime.now().date()
    if latest_elo:
        elo_date = datetime.strptime(latest_elo, '%Y-%m-%d').date()
        days_old = (today - elo_date).days
        if days_old <= 1:
            print(f"\n✓✓✓ SUCCESS: Data is FRESH (last update: {days_old} day(s) ago)")
        else:
            print(f"\n⚠️  WARNING: Data is {days_old} days old")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Delete predictions_cache.json to force fresh predictions")
print("2. Run: python validate_features.py")
print("3. Verify Away ELO is NO LONGER 0.0")
print("4. Launch dashboard: python nba_gui_dashboard_v2.py")
print("\nIf ELO values still look wrong, check feature_calculator_v5.py")
