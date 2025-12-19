"""
REBUILD ELO WITH SEASON RESETS
Fixes:
1. Away ELO = 0.0 bug
2. Impossible ratings (1039, etc.)
3. Implements soft reset between seasons (75% carry-over)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
from datetime import datetime, timedelta
from config.settings import DB_PATH
import pandas as pd

print("=" * 80)
print("REBUILDING ELO WITH SEASON RESETS")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Configuration
SEASON_RESET_FACTOR = 0.75  # Carry 75% of prior season rating
DEFAULT_ELO_OFF = 1500.0
DEFAULT_ELO_DEF = 1500.0
K_FACTOR = 20
HOME_ADVANTAGE = 100

class SeasonAwareEloEngine:
    def __init__(self):
        self.off_elo = {}
        self.def_elo = {}
        self.last_game_date = None
        self.current_season = None
        
    def get_elo(self, team, rating_type='off'):
        """Get ELO rating (fixes 0.0 bug by always returning valid value)"""
        if rating_type == 'off':
            return self.off_elo.get(team, DEFAULT_ELO_OFF)
        else:
            return self.def_elo.get(team, DEFAULT_ELO_DEF)
    
    def detect_season_change(self, game_date):
        """Detect if we've crossed into a new season (>90 day gap)"""
        if self.last_game_date is None:
            return False
        
        days_gap = (game_date - self.last_game_date).days
        return days_gap > 90
    
    def soft_reset_all_teams(self):
        """Apply soft reset: 75% old rating + 25% baseline (1500)"""
        print(f"\n   🔄 SEASON RESET DETECTED - Applying soft reset to all teams...")
        reset_count = 0
        
        for team in self.off_elo.keys():
            old_off = self.off_elo[team]
            old_def = self.def_elo[team]
            
            # Soft reset formula: 0.75 * old + 0.25 * 1500
            new_off = (old_off * SEASON_RESET_FACTOR) + (DEFAULT_ELO_OFF * (1 - SEASON_RESET_FACTOR))
            new_def = (old_def * SEASON_RESET_FACTOR) + (DEFAULT_ELO_DEF * (1 - SEASON_RESET_FACTOR))
            
            self.off_elo[team] = new_off
            self.def_elo[team] = new_def
            reset_count += 1
        
        print(f"   ✓ Reset {reset_count} teams (prevented rating drift)")
    
    def update_from_game(self, game_date, home_team, away_team, home_score, away_score):
        """Update ELO based on game result"""
        # Check for season change
        if self.detect_season_change(game_date):
            self.soft_reset_all_teams()
        
        self.last_game_date = game_date
        
        # Get current ratings (with defaults if team is new)
        home_off = self.get_elo(home_team, 'off')
        home_def = self.get_elo(home_team, 'def')
        away_off = self.get_elo(away_team, 'off')
        away_def = self.get_elo(away_team, 'def')
        
        # Expected points
        exp_home_pts = 110 + (home_off - away_def + HOME_ADVANTAGE) / 10
        exp_away_pts = 110 + (away_off - home_def) / 10
        
        # Actual errors
        home_off_error = home_score - exp_home_pts
        away_off_error = away_score - exp_away_pts
        
        # Margin multiplier (dampen blowouts)
        margin = abs(home_score - away_score)
        if margin > 20:
            margin = 20 + (margin - 20) * 0.5
        margin_mult = 1 + (margin / 40)
        
        k = K_FACTOR * margin_mult
        
        # Update ratings
        self.off_elo[home_team] = home_off + k * (home_off_error / 10)
        self.def_elo[away_team] = away_def - k * (home_off_error / 10)
        self.off_elo[away_team] = away_off + k * (away_off_error / 10)
        self.def_elo[home_team] = home_def - k * (away_off_error / 10)
        
        return {
            'home_off': self.off_elo[home_team],
            'home_def': self.def_elo[home_team],
            'away_off': self.off_elo[away_team],
            'away_def': self.def_elo[away_team],
            'composite_home': (self.off_elo[home_team] + self.def_elo[home_team]) / 2,
            'composite_away': (self.off_elo[away_team] + self.def_elo[away_team]) / 2
        }

# Step 1: Backup existing ELO
print("\n[1/4] Backing up existing ELO table...")
try:
    conn = sqlite3.connect(str(DB_PATH))
    backup_file = f"data/backups/elo_corrupt_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df_backup = pd.read_sql_query("SELECT * FROM elo_ratings", conn)
    print(f"  Backing up {len(df_backup):,} records to: {backup_file}")
    df_backup.to_csv(backup_file, index=False)
    
    conn.close()
    print("  ✓ Backup complete")
except Exception as e:
    print(f"  ⚠️  Warning: {e}")

# Step 2: Clear ELO table
print("\n[2/4] Clearing corrupted ELO table...")
try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM elo_ratings")
    conn.commit()
    
    cleared = cursor.rowcount
    print(f"  ✓ Cleared {cleared:,} corrupted records")
    
    conn.close()
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Step 3: Rebuild ELO from all games
print("\n[3/4] Rebuilding ELO from all game history...")
try:
    conn = sqlite3.connect(str(DB_PATH))
    
    # Get ALL games chronologically
    query = """
        SELECT game_date, home_team, away_team, home_score, away_score
        FROM game_logs
        WHERE home_score IS NOT NULL 
          AND away_score IS NOT NULL
        ORDER BY game_date, game_id
    """
    
    df_games = pd.read_sql_query(query, conn)
    df_games['game_date'] = pd.to_datetime(df_games['game_date'])
    
    total_games = len(df_games)
    print(f"  Found {total_games:,} completed games")
    
    if total_games == 0:
        print("  ✗ ERROR: No games in database!")
        sys.exit(1)
    
    # Initialize engine
    engine = SeasonAwareEloEngine()
    
    # Process each game
    print("  Processing games with season resets...")
    cursor = conn.cursor()
    processed = 0
    season_resets = 0
    
    for idx, row in df_games.iterrows():
        if idx % 100 == 0 and idx > 0:
            pct = (idx / total_games) * 100
            print(f"    Progress: {idx:,}/{total_games:,} ({pct:.1f}%) - {row['game_date'].strftime('%Y-%m-%d')}")
        
        # Check if season reset will happen
        will_reset = engine.detect_season_change(row['game_date'])
        if will_reset:
            season_resets += 1
        
        # Update ELO
        result = engine.update_from_game(
            game_date=row['game_date'],
            home_team=row['home_team'],
            away_team=row['away_team'],
            home_score=row['home_score'],
            away_score=row['away_score']
        )
        
        # Determine season
        year = row['game_date'].year
        month = row['game_date'].month
        season = f"{year}-{str(year+1)[-2:]}" if month >= 10 else f"{year-1}-{str(year)[-2:]}"
        
        # Insert into database
        cursor.execute("""
            INSERT INTO elo_ratings 
            (team, season, game_date, off_elo, def_elo, composite_elo, is_playoffs)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (row['home_team'], season, row['game_date'].strftime('%Y-%m-%d'),
              result['home_off'], result['home_def'], result['composite_home']))
        
        cursor.execute("""
            INSERT INTO elo_ratings 
            (team, season, game_date, off_elo, def_elo, composite_elo, is_playoffs)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (row['away_team'], season, row['game_date'].strftime('%Y-%m-%d'),
              result['away_off'], result['away_def'], result['composite_away']))
        
        processed += 1
        
        if idx % 500 == 0:
            conn.commit()
    
    conn.commit()
    conn.close()
    
    print(f"\n  ✓ Rebuild complete!")
    print(f"    Games processed: {processed:,}")
    print(f"    Season resets applied: {season_resets}")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Verify integrity
print("\n[4/4] Verifying ELO integrity...")
try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Count records
    cursor.execute("SELECT COUNT(*) FROM elo_ratings")
    total_records = cursor.fetchone()[0]
    print(f"  ✓ Total ELO records: {total_records:,}")
    
    # Check for zeros (Away ELO bug)
    cursor.execute("SELECT COUNT(*) FROM elo_ratings WHERE off_elo = 0 OR def_elo = 0")
    zero_count = cursor.fetchone()[0]
    
    if zero_count == 0:
        print(f"  ✓ NO ZERO ELOS - Bug fixed!")
    else:
        print(f"  ✗ WARNING: Found {zero_count} zero ELO records")
    
    # Check for impossible values
    cursor.execute("""
        SELECT COUNT(*) FROM elo_ratings 
        WHERE off_elo < 800 OR off_elo > 2200 
           OR def_elo < 800 OR def_elo > 2200
    """)
    impossible_count = cursor.fetchone()[0]
    
    if impossible_count == 0:
        print(f"  ✓ All ELO values in reasonable range (800-2200)")
    else:
        print(f"  ⚠️  Found {impossible_count} records with extreme values")
    
    # Show latest ratings (top 5)
    cursor.execute("""
        SELECT team, game_date, off_elo, def_elo, composite_elo
        FROM elo_ratings
        WHERE game_date = (SELECT MAX(game_date) FROM elo_ratings)
        ORDER BY composite_elo DESC
        LIMIT 5
    """)
    
    print("\n  Latest ELO ratings (Top 5 teams):")
    for team, date, off, def_, comp in cursor.fetchall():
        print(f"    {team:4s}: off={off:7.1f}, def={def_:7.1f}, comp={comp:7.1f} ({date})")
    
    # Check sample teams (ORL, WAS, IND)
    print("\n  Sample teams (checking for corruption):")
    for team in ['ORL', 'WAS', 'IND']:
        cursor.execute("""
            SELECT off_elo, def_elo, composite_elo, game_date
            FROM elo_ratings
            WHERE team = ? AND game_date = (SELECT MAX(game_date) FROM elo_ratings WHERE team = ?)
        """, (team, team))
        
        row = cursor.fetchone()
        if row:
            print(f"    {team:4s}: off={row[0]:7.1f}, def={row[1]:7.1f}, comp={row[2]:7.1f} ({row[3]})")
    
    conn.close()
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")

# Summary
print("\n" + "=" * 80)
print("REBUILD COMPLETE!")
print("=" * 80)
print("\n✓✓✓ ELO system rebuilt with season resets")
print("✓✓✓ Away ELO = 0.0 bug should be FIXED")
print("✓✓✓ Impossible ratings (1039, etc.) should be CORRECTED")
print("\nNEXT STEPS:")
print("  1. Delete predictions_cache.json")
print("  2. Run: python validate_features.py")
print("  3. Verify Away ELO is no longer 0.0")
print("  4. Launch dashboard")
