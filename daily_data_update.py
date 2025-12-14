"""
DAILY DATA UPDATE - Run this every morning before making predictions
Updates: ELO ratings, game logs, advanced stats, injuries
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
import sqlite3
from config.settings import DB_PATH

print("="*80)
print("DAILY DATA UPDATE - NBA Betting System")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Step 1: Update game logs (recent results)
print("\n[1/4] Updating game logs...")
try:
    from src.collectors.update_game_logs import update_game_logs
    update_game_logs(last_n_days=7)
    print("[OK] Game logs updated")
except Exception as e:
    print(f"[ERROR] Failed to update game logs: {e}")

# Step 2: Update advanced stats
print("\n[2/4] Updating advanced stats...")
try:
    from src.collectors.update_game_advanced_stats import update_game_advanced_stats
    update_game_advanced_stats()
    print("[OK] Advanced stats updated")
except Exception as e:
    print(f"[ERROR] Failed to update advanced stats: {e}")

# Step 3: Update ELO ratings
print("\n[3/4] Updating ELO ratings...")
try:
    from src.features.off_def_elo_system import OffDefEloSystem
    
    elo_system = OffDefEloSystem(db_path=str(DB_PATH))
    
    # Get date range for ELO update
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Find last ELO update date
    cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
    last_elo_date = cursor.fetchone()[0]
    
    if last_elo_date:
        start_date = datetime.strptime(last_elo_date, '%Y-%m-%d') + timedelta(days=1)
        print(f"  Last ELO update: {last_elo_date}")
        print(f"  Updating from: {start_date.strftime('%Y-%m-%d')}")
        
        # Get games to process
        cursor.execute("""
            SELECT game_date, home_team, away_team, home_score, away_score
            FROM game_logs
            WHERE game_date >= ?
            ORDER BY game_date
        """, (start_date.strftime('%Y-%m-%d'),))
        
        games = cursor.fetchall()
        print(f"  Found {len(games)} games to process")
        
        if games:
            for game_date, home, away, home_score, away_score in games:
                try:
                    elo_system.update_elo(
                        home_team=home,
                        away_team=away,
                        home_score=home_score,
                        away_score=away_score,
                        game_date=game_date,
                        season='2024-25'
                    )
                except Exception as e:
                    print(f"  ⚠️  Failed to update ELO for {away}@{home} on {game_date}: {e}")
            
            print(f"[OK] ELO ratings updated through {games[-1][0]}")
        else:
            print("[OK] ELO ratings already up to date")
    else:
        print("[WARNING] No existing ELO data found - need to initialize ELO system")
        print("   Run: python -m src.features.off_def_elo_system --initialize")
    
    conn.close()
    
except Exception as e:
    print(f"[ERROR] Failed to update ELO: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Update injuries
print("\n[4/4] Updating injury data...")
try:
    from src.services.live_injury_updater import LiveInjuryUpdater
    
    updater = LiveInjuryUpdater(db_path=str(DB_PATH))
    injury_count = updater.update_active_injuries()
    print(f"[OK] Updated {injury_count} active injuries from ESPN")
except Exception as e:
    print(f"[ERROR] Failed to update injuries: {e}")

# Summary
print("\n" + "="*80)
print("DATA UPDATE COMPLETE")
print("="*80)

# Show data freshness
try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT MAX(game_date) FROM game_logs")
    latest_game = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
    latest_elo = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM active_injuries")
    injury_count = cursor.fetchone()[0]
    
    print(f"Latest game in database: {latest_game}")
    print(f"Latest ELO update: {latest_elo}")
    print(f"Active injuries tracked: {injury_count}")
    
    # Check if data is fresh
    today = datetime.now().date()
    if latest_elo:
        elo_date = datetime.strptime(latest_elo, '%Y-%m-%d').date()
        days_old = (today - elo_date).days
        if days_old > 1:
            print(f"\n[WARNING] ELO data is {days_old} days old!")
            print("   Predictions may be using stale ratings")
        else:
            print(f"\n[OK] Data is fresh (last update: {days_old} day(s) ago)")
    
    conn.close()
    
except Exception as e:
    print(f"Error checking data freshness: {e}")

print("\n[OK] Ready to make predictions with fresh data!")
print("\nNext steps:")
print("  1. Run dashboard: python nba_gui_dashboard_v2.py")
print("  2. Or run main_predict.py for today's games")
