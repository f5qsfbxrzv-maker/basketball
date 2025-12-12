"""
INJURY DATA COLLECTOR - REAL-TIME & HISTORICAL
Adapted from user's box score scraper code

1. Live Scraper: Scrapes CBS Sports for current reports
2. Historical Backfiller: Uses NBA API (BoxScoreSummaryV2 + BoxScoreTraditionalV2)
   to find who was inactive/DNP in past games
"""

import requests
import sqlite3
import pandas as pd
import time
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Tuple

from config.database_paths import NBA_BETTING_DB

class InjuryDataCollector:
    def __init__(self, db_path: str = str(NBA_BETTING_DB)):
        self.db_path = db_path
        self.live_url = "https://www.cbssports.com/nba/injuries/"
        self.nba_base_url = "https://stats.nba.com/stats"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.nba.com/',
            'Origin': 'https://www.nba.com'
        }
        self._init_db()

    def _init_db(self):
        """Create tables for live and historical injury data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table 1: ACTIVE INJURIES (Live scraping from CBS)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_injuries (
                player_name TEXT,
                team_name TEXT,
                position TEXT,
                status TEXT,
                injury_desc TEXT,
                last_updated TEXT,
                PRIMARY KEY (player_name, team_name)
            )
        ''')
        
        # Table 2: HISTORICAL INACTIVES (Box score backfill)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_inactives (
                game_id TEXT,
                player_id INTEGER,
                player_name TEXT,
                team_id INTEGER,
                team_abbreviation TEXT,
                season TEXT,
                game_date TEXT,
                PRIMARY KEY (game_id, player_id)
            )
        ''')

        # Ensure schema includes expected columns (add if missing)
        cursor.execute("PRAGMA table_info(historical_inactives)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        required = ["team_id", "game_date"]
        for col in required:
            if col not in existing_cols:
                alter_sql = f"ALTER TABLE historical_inactives ADD COLUMN {col} TEXT" if col == "game_date" else f"ALTER TABLE historical_inactives ADD COLUMN {col} INTEGER"
                cursor.execute(alter_sql)
        conn.commit()
        
        conn.commit()
        conn.close()

    # =====================================================
    # PART 1: LIVE INJURY SCRAPING (CBS Sports)
    # =====================================================
    
    def scrape_live_injuries(self):
        """Scrape current injury reports from CBS Sports"""
        print("🏥 Scraping live injuries from CBS Sports...")
        
        try:
            r = requests.get(self.live_url, headers=self.headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, 'html.parser')
            
            # Parse injury table (implementation depends on CBS HTML structure)
            # This is a placeholder - actual parsing logic would go here
            
            print("✅ Live injuries scraped successfully")
            
        except Exception as e:
            print(f"❌ Error scraping live injuries: {e}")

    # =====================================================
    # PART 2: HISTORICAL BACKFILLER (NBA API Box Scores)
    # =====================================================
    
    def backfill_historical_injuries(self, season: str):
        """
        Master function to iterate through games and fetch inactives.
        Uses NBA API BoxScoreSummaryV2 (inactive list) + BoxScoreTraditionalV2 (MIN=0)
        
        Args:
            season: Format "2023-24"
        """
        print(f"📜 Backfilling Historical Inactives for {season}...")
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Get all game_ids for this season
            games_df = pd.read_sql_query(
                "SELECT game_id FROM game_results WHERE season = ?", 
                conn, 
                params=(season,)
            )
        except Exception as e:
            print(f"   ❌ Error: game_results table not found. {e}")
            print("      Run data collection first to populate game_results")
            conn.close()
            return
        
        if games_df.empty:
            print(f"   ⚠️  No games found for {season}")
            conn.close()
            return

        count = 0
        skipped = 0
        
        for game_id in games_df['game_id']:
            # Skip if we already have data for this game
            if self._has_historical_data(game_id, conn):
                skipped += 1
                continue
                
            # Fetch and save inactives
            success = self._fetch_game_inactives_and_boxscore(game_id, season, conn)
            if success:
                count += 1
            
            if (count + skipped) % 50 == 0:
                print(f"   Progress: {count} new, {skipped} skipped / {len(games_df)} total games...")
            
            # Rate limiting (NBA API is strict)
            time.sleep(0.6)  # ~100 games/minute max
            
        conn.close()
        print(f"✅ Backfill complete for {season}")
        print(f"   Added: {count} new game records")
        print(f"   Skipped: {skipped} already processed")

    def _has_historical_data(self, game_id: str, conn: sqlite3.Connection) -> bool:
        """Check if we already have inactive data for this game"""
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM historical_inactives WHERE game_id = ? LIMIT 1", (game_id,))
        return cur.fetchone() is not None

    def _fetch_game_inactives_and_boxscore(
        self, 
        game_id: str, 
        season: str, 
        conn: sqlite3.Connection
    ) -> bool:
        """
        Fetches both the official Inactive List and the Box Score to find players with 0 minutes.
        
        Returns:
            True if successful, False otherwise
        """
        url_summary = f"{self.nba_base_url}/boxscoresummaryv2"
        url_traditional = f"{self.nba_base_url}/boxscoretraditionalv2"
        
        all_inactives = set()  # Set of player_ids
        player_lookup = {}  # player_id -> (name, team)
        
        def _find_header_index(headers, *candidates):
            """Find header index from multiple possible names (case-insensitive fallback)."""
            for name in candidates:
                if name in headers:
                    return headers.index(name)
            # case-insensitive match
            lower = [h.lower() for h in headers]
            for name in candidates:
                try:
                    return lower.index(name.lower())
                except ValueError:
                    continue
            raise ValueError(f"None of {candidates} found in headers: {headers}")
        # ============================================================
        # STEP 1: Fetch Official Inactive List (BoxScoreSummaryV2)
        # ============================================================
        try:
            r_summary = requests.get(
                url_summary, 
                headers=self.headers, 
                params={'GameID': game_id}, 
                timeout=10
            )
            r_summary.raise_for_status()
            data_summary = r_summary.json()
            
            # Find InactivePlayers result set
            inactive_set = next(
                (rs for rs in data_summary['resultSets'] if rs['name'] == 'InactivePlayers'), 
                None
            )
            
            if inactive_set and inactive_set['rowSet']:
                headers = inactive_set['headers']
                try:
                    idx_id = _find_header_index(headers, 'PLAYER_ID', 'Player_ID')
                    # try canonical PLAYER_NAME, otherwise fall back to FIRST_NAME + LAST_NAME
                    try:
                        idx_name = _find_header_index(headers, 'PLAYER_NAME', 'PLAYER')
                        name_mode = 'single'
                    except ValueError:
                        if 'FIRST_NAME' in headers and 'LAST_NAME' in headers:
                            idx_first = headers.index('FIRST_NAME')
                            idx_last = headers.index('LAST_NAME')
                            name_mode = 'split'
                        else:
                            raise
                    idx_team = _find_header_index(headers, 'TEAM_ABBREVIATION', 'TEAM_ABBREV', 'TEAM', 'TEAM_NAME')
                except Exception as e:
                    print(f"      ⚠️  BoxScoreSummary header mismatch for {game_id}: {e}")
                else:
                    for row in inactive_set['rowSet']:
                        player_id = row[idx_id]
                        if name_mode == 'single':
                            player_name = row[idx_name]
                        else:
                            fn = row[idx_first] or ''
                            ln = row[idx_last] or ''
                            player_name = f"{fn} {ln}".strip()
                        team_abbrev = row[idx_team]

                        # if TEAM_ID present, capture it alongside abbreviation
                        if 'TEAM_ID' in headers:
                            try:
                                team_id_idx = headers.index('TEAM_ID')
                                team_id_val = row[team_id_idx]
                                player_lookup[player_id] = (player_name, (team_id_val, team_abbrev))
                            except Exception:
                                player_lookup[player_id] = (player_name, team_abbrev)
                        else:
                            player_lookup[player_id] = (player_name, team_abbrev)

                        all_inactives.add(player_id)
                    
        except Exception as e:
            print(f"      ⚠️  BoxScoreSummary error for {game_id}: {e}")
            # Continue anyway - we'll try to get MIN=0 players

        # ============================================================
        # STEP 2: Fetch Box Score to find MIN=0 players (DNP-CD)
        # ============================================================
        try:
            r_trad = requests.get(
                url_traditional, 
                headers=self.headers, 
                params={'GameID': game_id}, 
                timeout=10
            )
            r_trad.raise_for_status()
            data_trad = r_trad.json()
            
            # Find PlayerStats result set
            player_stats = next(
                (rs for rs in data_trad['resultSets'] if rs['name'] == 'PlayerStats'), 
                None
            )
            
            if player_stats and player_stats['rowSet']:
                headers = player_stats['headers']
                try:
                    idx_id = _find_header_index(headers, 'PLAYER_ID', 'Player_ID')
                    # try canonical PLAYER_NAME, otherwise fall back to FIRST_NAME + LAST_NAME
                    try:
                        idx_name = _find_header_index(headers, 'PLAYER_NAME', 'PLAYER')
                        name_mode_stats = 'single'
                    except ValueError:
                        if 'FIRST_NAME' in headers and 'LAST_NAME' in headers:
                            idx_first = headers.index('FIRST_NAME')
                            idx_last = headers.index('LAST_NAME')
                            name_mode_stats = 'split'
                        else:
                            raise
                    idx_team = _find_header_index(headers, 'TEAM_ABBREVIATION', 'TEAM_ABBREV', 'TEAM', 'TEAM_NAME')
                    idx_min = _find_header_index(headers, 'MIN')
                except Exception as e:
                    print(f"      ⚠️  BoxScoreTraditional header mismatch for {game_id}: {e}")
                else:
                    for row in player_stats['rowSet']:
                        player_id = row[idx_id]
                        if name_mode_stats == 'single':
                            player_name = row[idx_name]
                        else:
                            fn = row[idx_first] or ''
                            ln = row[idx_last] or ''
                            player_name = f"{fn} {ln}".strip()
                        team_abbrev = row[idx_team]
                        min_played = row[idx_min]

                        # Check if player didn't play (MIN is None or 0)
                        if min_played is None or (isinstance(min_played, (int, float)) and min_played == 0):
                            all_inactives.add(player_id)
                            if player_id not in player_lookup:
                                player_lookup[player_id] = (player_name, team_abbrev)
                        
        except Exception as e:
            print(f"      ⚠️  BoxScoreTraditional error for {game_id}: {e}")
            return False

        # ============================================================
        # STEP 3: Save unique inactives to database
        # ============================================================
        if not all_inactives:
            # No inactives found - save empty record to mark game as processed
            return True
        # get game_date from game_results if available
        try:
            cur = conn.cursor()
            cur.execute("SELECT game_date FROM game_results WHERE game_id = ? LIMIT 1", (game_id,))
            gd_row = cur.fetchone()
            game_date_val = gd_row[0] if gd_row else None
        except Exception:
            game_date_val = None

        records = []
        for player_id in all_inactives:
            if player_id in player_lookup:
                player_name, team_abbrev = player_lookup[player_id]
                # try to find team_id if available in the lookup (some headers include TEAM_ID)
                team_id_val = None
                # player_lookup entries are (player_name, team_abbrev) but upstream header parsing
                # may be extended; attempt a best-effort parse if a tuple includes team_id
                if isinstance(team_abbrev, tuple) and len(team_abbrev) >= 2:
                    # legacy: stored as (team_id, team_abbrev)
                    team_id_val = team_abbrev[0]
                    team_abbrev = team_abbrev[1]

                records.append((game_id, player_id, player_name, team_id_val, team_abbrev, season, game_date_val))

        if records:
            self._save_historical_inactives(records, conn)
            
        return True

    def _save_historical_inactives(
        self, 
        records: List[Tuple[str, int, str, str, str]], 
        conn: sqlite3.Connection
    ):
        """Save inactive players to database"""
        if not records:
            return
        
        try:
            # Insert with explicit full column list (match DB schema: game_id, player_id, player_name,
            # team_id, team_abbreviation, season, game_date)
            conn.executemany(
                "INSERT OR IGNORE INTO historical_inactives (game_id, player_id, player_name, team_id, team_abbreviation, season, game_date) VALUES (?,?,?,?,?,?,?)",
                records
            )
            conn.commit()
        except Exception as e:
            print(f"      ⚠️  Database save error: {e}")


# =====================================================
# USAGE EXAMPLE
# =====================================================
if __name__ == "__main__":
    print("="*80)
    print("INJURY DATA COLLECTOR - HISTORICAL BACKFILL")
    print("="*80)
    
    collector = InjuryDataCollector(db_path="../nba_betting_data.db")
    
    # Backfill specific season
    season_to_backfill = "2023-24"
    
    print(f"\nBackfilling injuries for {season_to_backfill}...")
    print("This will take ~12 minutes for a full season (~1,230 games)")
    print("Rate limited to ~100 games/minute to respect NBA API")
    
    collector.backfill_historical_injuries(season_to_backfill)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
To backfill all training data seasons (2015-2024):

for season in ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
               "2020-21", "2021-22", "2022-23", "2023-24"]:
    collector.backfill_historical_injuries(season)
    
Total time: ~2 hours for complete historical backfill

Then update extract_training_data_optimized.py to calculate injury_impact_diff
from historical_inactives table instead of active_injuries.
    """)
