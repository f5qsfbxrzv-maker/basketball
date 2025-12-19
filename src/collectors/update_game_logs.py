"""
Update Game Logs - Simple NBA API Data Collector
Updates game_logs and game_advanced_stats tables with recent games
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import time

try:
    from nba_api.stats.endpoints import leaguegamelog
    from nba_api.stats.static import teams
except ImportError:
    print("âŒ ERROR: nba_api not installed")
    print("   Run: pip install nba-api")
    exit(1)

from config.settings import DB_PATH

def get_season_string(year: int) -> str:
    """Convert year to NBA season string (e.g., 2024 -> '2024-25')"""
    return f"{year}-{str(year + 1)[-2:]}"

def update_game_logs(season: str = None, last_n_days: int = None):
    """
    Update game_logs table with recent games
    
    Args:
        season: NBA season string (e.g., '2024-25'). Defaults to current season.
        last_n_days: Only show games from last N days. Defaults to all.
    """
    
    if season is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        # NBA season runs Oct-June, so if we're in Jan-June, it's the previous year's season
        season_year = current_year - 1 if current_month <= 6 else current_year
        season = get_season_string(season_year)
    
    print("=" * 70)
    print(f"UPDATING GAME LOGS - Season: {season}")
    print("=" * 70)
    
    try:
        # Fetch game logs from NBA API
        print("\nðŸ“¥ Fetching game logs from NBA API...")
        game_log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season'
        )
        
        # Rate limit
        time.sleep(0.6)
        
        df = game_log.get_data_frames()[0]
        
        if df.empty:
            print(f"âš ï¸  No games found for season {season}")
            return
        
        print(f"âœ… Fetched {len(df):,} game records")
        
        # Add game_date column (convert GAME_DATE string to date)
        df['game_date'] = pd.to_datetime(df['GAME_DATE']).dt.date
        
        # Filter by last_n_days if specified
        if last_n_days:
            cutoff_date = datetime.now().date() - timedelta(days=last_n_days)
            df = df[df['game_date'] >= cutoff_date]
            print(f"ðŸ“… Filtered to last {last_n_days} days: {len(df):,} records")
        
        # Show date range
        print(f"ðŸ“Š Date Range: {df['game_date'].min()} to {df['game_date'].max()}")
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        
        # Save to database (replace existing data for this season)
        print(f"\nðŸ’¾ Saving to database: {DB_PATH}")
        
        # Delete existing records for this date range to avoid duplicates
        cursor = conn.cursor()
        min_date = df['game_date'].min()
        max_date = df['game_date'].max()
        
        cursor.execute(
            "DELETE FROM game_logs WHERE game_date >= ? AND game_date <= ?",
            (str(min_date), str(max_date))
        )
        deleted = cursor.rowcount
        print(f"ðŸ—‘ï¸  Deleted {deleted:,} existing records in date range")
        
        # Insert new records
        df.to_sql('game_logs', conn, if_exists='append', index=False)
        print(f"âœ… Inserted {len(df):,} new records")
        
        # Show summary
        cursor.execute("SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM game_logs")
        total, db_min, db_max = cursor.fetchone()
        print(f"\nðŸ“Š DATABASE SUMMARY:")
        print(f"   Total Records: {total:,}")
        print(f"   Date Range: {db_min} to {db_max}")
        
        conn.commit()
        conn.close()
        
        print("\nâœ… UPDATE COMPLETE")
        
    except Exception as e:
        print(f"\nâŒ ERROR: {e}")
        import traceback
        traceback.print_exc()

def show_status():
    """Show current database status"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("=" * 70)
    print("CURRENT DATABASE STATUS")
    print("=" * 70)
    
    # game_logs
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            MIN(game_date) as first_game,
            MAX(game_date) as last_game,
            COUNT(DISTINCT SEASON_ID) as seasons
        FROM game_logs
    """)
    total, first, last, seasons = cursor.fetchone()
    print(f"\ngame_logs:")
    print(f"  Total Games: {total:,}")
    print(f"  Date Range: {first} to {last}")
    print(f"  Seasons: {seasons}")
    
    # Recent games
    cursor.execute("SELECT COUNT(*) FROM game_logs WHERE game_date >= date('now', '-7 days')")
    recent = cursor.fetchone()[0]
    print(f"  Last 7 Days: {recent:,} games")
    
    # game_advanced_stats
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            MIN(game_date) as first_game,
            MAX(game_date) as last_game
        FROM game_advanced_stats
    """)
    total, first, last = cursor.fetchone()
    print(f"\ngame_advanced_stats:")
    print(f"  Total Games: {total:,}")
    print(f"  Date Range: {first} to {last}")
    
    cursor.execute("SELECT COUNT(*) FROM game_advanced_stats WHERE game_date >= date('now', '-7 days')")
    recent = cursor.fetchone()[0]
    print(f"  Last 7 Days: {recent:,} games")
    
    print(f"\nðŸ“… Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Update NBA game logs')
    parser.add_argument('--season', type=str, help='NBA season (e.g., 2024-25)')
    parser.add_argument('--last-n-days', type=int, help='Only update games from last N days')
    parser.add_argument('--status', action='store_true', help='Show database status only')
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    else:
        update_game_logs(season=args.season, last_n_days=args.last_n_days)
        print("\n")
        show_status()

