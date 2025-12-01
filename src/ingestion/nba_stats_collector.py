"""
NBA Stats Collector V2 - Using Official nba_api Library
Much more reliable than raw requests to stats.nba.com
Includes automatic rate limiting and retry logic built-in
"""

import sqlite3
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
from v2.logger_setup import get_structured_adapter, classify_error

# Official nba_api library
from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats
from nba_api.stats.static import teams

warnings.filterwarnings('ignore')

class NBAStatsCollectorV2:
    """
    Enhanced NBA statistics collector using official nba_api library
    More reliable than raw requests with built-in rate limiting
    """
    
    def __init__(self, db_path: str = "data/database/nba_betting_data.db"):
        self.db_path = db_path
        self.event_logger = get_structured_adapter(component='collector', prediction_version='v5.0')
        self._init_database()
        self.event_logger.event('info', "NBA Stats Collector V2 initialized (using nba_api)", category='lifecycle')
    
    def _init_database(self):
        """Initialize SQLite database with minimal schema - let nba_api define columns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Game results table (consolidated game outcomes) - our main table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE,
                game_date TEXT,
                season TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                home_won INTEGER,
                total_points INTEGER,
                point_differential INTEGER
            )
        ''')
        
        # Bankroll history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                bankroll REAL,
                change REAL,
                reason TEXT
            )
        ''')

        # Player impact stats (PIE / usage) for injury valuation (gold standard)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_stats (
                player_id INTEGER,
                player_name TEXT,
                team_id INTEGER,
                team_abbreviation TEXT,
                season TEXT,
                pie REAL,
                usg_pct REAL,
                net_rating REAL,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season)
            )
        ''')
        
        # Note: team_stats and game_logs tables will be created automatically by pandas
        # when we save data with to_sql(). This way nba_api defines the schema.
        
        conn.commit()
        conn.close()
        self.event_logger.event('info', f"Database initialized: {self.db_path}", category='lifecycle')
    
    def get_team_stats(self, season: str = "2024-25") -> pd.DataFrame:
        """
        Fetch team statistics using nba_api with ALL necessary metrics
        Gold Standard Features: Four Factors + Pace + Ratings
        """
        self.event_logger.event('info', f"Fetching Team Stats for {season}...", category='network')
        
        try:
            # Fetch multiple stat types to ensure we have everything
            # 1. Advanced stats (includes pace, off/def ratings, etc.)
            advanced_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )
            df_advanced = advanced_stats.get_data_frames()[0]
            
            # 2. Four Factors stats (eFG%, TOV%, REB%, FT Rate)
            four_factors = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Four Factors',
                per_mode_detailed='PerGame'
            )
            df_four_factors = four_factors.get_data_frames()[0]
            
            # Merge the dataframes on TEAM_ID
            df = pd.merge(
                df_advanced,
                df_four_factors,
                on='TEAM_ID',
                suffixes=('', '_ff'),
                how='left'
            )
            
            # Save to database
            self._save_team_stats(df, season)
            
            self.event_logger.event('info', f"Team Stats fetched: {len(df)} teams", category='network')
            return df
            
        except Exception as e:
            self.event_logger.event('error', f"Error fetching team stats: {e}", category=classify_error(e))
            return pd.DataFrame()
    
    def _save_team_stats(self, df: pd.DataFrame, season: str):
        """Save team stats to database - flexible schema with context manager"""
        if df.empty:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Use INSERT OR REPLACE to update existing records (preserve historical data)
            cursor = conn.cursor()
            
            # Add season column
            df = df.copy()
            df['season'] = season
            
            # Save all columns nba_api gives us
            # Delete only the specific TEAM_IDs for this season (not entire season history)
            try:
                # Check if records exist for this season
                cursor.execute("SELECT COUNT(*) FROM team_stats WHERE season = ?", (season,))
                existing_count = cursor.fetchone()[0]
                
                if existing_count > 0:
                    # Delete only the teams we're about to update (not all historical data)
                    team_ids = df['TEAM_ID'].unique().tolist()
                    placeholders = ','.join('?' * len(team_ids))
                    cursor.execute(f"DELETE FROM team_stats WHERE season = ? AND TEAM_ID IN ({placeholders})", 
                                 [season] + team_ids)
                    updated = cursor.rowcount
                    self.event_logger.event('info', f"Updating {updated} team_stats for {season} (preserving historical data)", category='data_integrity')
                
                # Now insert the fresh data
                df.to_sql('team_stats', conn, if_exists='append', index=False)
                self.event_logger.event('info', f"Saved {len(df)} team_stats for {season}", category='data_integrity')
            except Exception as e:
                self.event_logger.event('error', f"team_stats write failed: {e}", category='data_integrity')
                raise
            
            conn.commit()
    
    def get_game_logs(self, season: str = "2024-25") -> pd.DataFrame:
        """
        Fetch game logs for all teams using nba_api
        Returns per-team-per-game statistics
        """
        self.event_logger.event('info', f"Fetching Game Logs for {season}...", category='network')
        
        try:
            # Use nba_api endpoint (much more reliable)
            game_log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star='Regular Season',
                player_or_team_abbreviation='T'
            )
            
            # Get dataframe
            df = game_log.get_data_frames()[0]
            
            # Save to database
            self._save_game_logs(df, season)
            
            self.event_logger.event('info', f"Game Logs fetched: {len(df)} team-games", category='network')
            return df
        except Exception as e:
            self.event_logger.event('error', f"Failed to fetch game logs: {e}", category='network')
            return pd.DataFrame()

    def get_player_impact_stats(self, season: str = "2024-25") -> pd.DataFrame:
        """Fetch player-level impact metrics (PIE, usage, net rating) for injury weighting."""
        self.event_logger.event('info', f"Fetching Player Impact Stats for {season}...", category='network')
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            # Some nfl/nba_api versions differ in accepted kwargs. Try the full, then a simplified fallback.
            try:
                stats = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    season_type_all_star='Regular Season',
                    measure_type_detailed='Advanced',
                    per_mode_detailed='PerGame'
                )
            except TypeError:
                # Fallback to a simpler argument set supported by older/newer nba_api versions
                stats = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    season_type_all_star='Regular Season',
                    per_mode_detailed='PerGame'
                )
            df = stats.get_data_frames()[0]
            # Normalize columns of interest
            cols_map = {
                'PLAYER_ID': 'player_id',
                'PLAYER_NAME': 'player_name',
                'TEAM_ID': 'team_id',
                'TEAM_ABBREVIATION': 'team_abbreviation',
                'PIE': 'pie',
                'USG_PCT': 'usg_pct',
                'NET_RATING': 'net_rating'
            }
            subset = df[list(cols_map.keys())].rename(columns=cols_map)
            subset['season'] = season
            with sqlite3.connect(self.db_path) as conn:
                subset.to_sql('player_stats', conn, if_exists='replace', index=False)
            self.event_logger.event('info', f"Saved {len(subset)} player impact rows", category='network')
            return subset
        except Exception as e:
            self.event_logger.event('error', f"Player impact fetch failed: {e}", category=classify_error(e))
            return pd.DataFrame()
            
        except Exception as e:
            self.event_logger.event('error', f"Error fetching game logs: {e}", category=classify_error(e))
            return pd.DataFrame()
    
    def get_pbp_logs(self, season: str = "2024-25", max_games: int = None, resume: bool = True) -> pd.DataFrame:
        """
        Fetch play-by-play data for all games in a season with resume capability.
        This is needed for live win probability backtesting.
        
        Args:
            season: Season string (e.g., "2024-25")
            max_games: Optional limit on games to fetch (for testing)
            resume: If True, skip games that already have PBP data
        
        Returns:
            DataFrame with columns: game_id, event_num, period, clock, home_score, 
                                   away_score, event_type, event_description
        """
        from nba_api.stats.endpoints import playbyplay  # Use non-v2 endpoint
        
        self.event_logger.event('info', f"Fetching PBP data for season {season}", category='lifecycle')
        
        try:
            # First, get all game IDs for the season from game_logs
            conn = sqlite3.connect(self.db_path)
            
            # Get distinct game IDs (game_logs has 2 rows per game - one for each team)
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT DISTINCT GAME_ID 
                FROM game_logs 
                WHERE season = '{season}'
                ORDER BY GAME_ID
            """)
            game_ids = [row[0] for row in cursor.fetchall()]
            
            # Get game IDs that already have PBP data (for resume)
            existing_game_ids = set()
            if resume:
                try:
                    cursor.execute("SELECT DISTINCT game_id FROM pbp_logs")
                    existing_game_ids = {row[0] for row in cursor.fetchall()}
                    self.event_logger.event('info', f"Found {len(existing_game_ids)} games with existing PBP data", category='lifecycle')
                except:
                    pass  # Table might not exist yet
            
            conn.close()
            
            if not game_ids:
                self.event_logger.event('warning', f"No games found for season {season} - download game logs first", category='data_integrity')
                return pd.DataFrame()
            
            # Filter out games we already have
            if resume and existing_game_ids:
                game_ids = [gid for gid in game_ids if gid not in existing_game_ids]
                self.event_logger.event('info', f"Resuming: {len(game_ids)} games remaining to download", category='lifecycle')
            
            if max_games:
                game_ids = game_ids[:max_games]
            
            if not game_ids:
                self.event_logger.event('info', f"All games already have PBP data - nothing to download", category='lifecycle')
                return pd.DataFrame()
            
            self.event_logger.event('info', f"Downloading PBP for {len(game_ids)} games", category='lifecycle')
            
            all_pbp_data = []
            successful_downloads = 0
            failed_downloads = 0
            
            for idx, game_id in enumerate(game_ids, 1):
                try:
                    # Fetch PBP for this game (use non-v2 endpoint)
                    pbp = playbyplay.PlayByPlay(game_id=game_id)
                    pbp_df = pbp.get_data_frames()[0]  # First dataframe has the PBP data
                    
                    if not pbp_df.empty:
                        # Parse SCORE column (format: "5 - 2" or "TIE")
                        def parse_score(score_str):
                            if pd.isna(score_str) or score_str is None:
                                return 0, 0
                            score_str = str(score_str).strip()
                            if score_str == 'TIE' or score_str == '0 - 0':
                                # Find previous non-null score in this game
                                return None, None  # Will forward-fill
                            try:
                                parts = score_str.split('-')
                                if len(parts) == 2:
                                    away_score = int(parts[0].strip())
                                    home_score = int(parts[1].strip())
                                    return home_score, away_score
                            except:
                                return 0, 0
                            return 0, 0
                        
                        # Parse scores
                        scores = pbp_df['SCORE'].apply(parse_score)
                        pbp_df['home_score'] = scores.apply(lambda x: x[0] if x else None)
                        pbp_df['away_score'] = scores.apply(lambda x: x[1] if x else None)
                        
                        # Forward fill scores (handles None values from TIE)
                        pbp_df['home_score'] = pbp_df['home_score'].ffill().fillna(0).astype(int)
                        pbp_df['away_score'] = pbp_df['away_score'].ffill().fillna(0).astype(int)
                        
                        # Extract key columns we need for backtesting
                        pbp_clean = pd.DataFrame({
                            'game_id': game_id,
                            'event_num': pbp_df['EVENTNUM'],
                            'period': pbp_df['PERIOD'],
                            'clock': pbp_df['PCTIMESTRING'],
                            'home_score': pbp_df['home_score'],
                            'away_score': pbp_df['away_score'],
                            'event_type': pbp_df['EVENTMSGTYPE'],
                            'event_description': pbp_df['HOMEDESCRIPTION'].fillna('') + ' ' + pbp_df['VISITORDESCRIPTION'].fillna('')
                        })
                        
                        all_pbp_data.append(pbp_clean)
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                        self.event_logger.event('warning', f"Empty PBP data for game {game_id}", category='data_integrity')
                    
                    # Progress logging every 50 games
                    if idx % 50 == 0:
                        self.event_logger.event('info', f"PBP Progress: {idx}/{len(game_ids)} games ({successful_downloads} successful, {failed_downloads} failed)", category='lifecycle')
                        
                        # INCREMENTAL SAVE every 50 games to prevent data loss
                        if all_pbp_data:
                            batch_df = pd.concat(all_pbp_data, ignore_index=True)
                            self._save_pbp_logs(batch_df, season)
                            self.event_logger.event('info', f"Saved batch: {len(batch_df):,} events", category='lifecycle')
                            all_pbp_data = []  # Clear batch after saving
                    
                    # Rate limiting - NBA API is strict
                    time.sleep(0.6)  # ~100 requests per minute
                    
                except Exception as e:
                    failed_downloads += 1
                    self.event_logger.event('warning', f"PBP fetch failed for game {game_id}: {e}", category='data_integrity')
                    time.sleep(1)  # Extra delay on error
                    continue
            
            # Save remaining data
            if all_pbp_data:
                combined_df = pd.concat(all_pbp_data, ignore_index=True)
                self._save_pbp_logs(combined_df, season)
                self.event_logger.event('info', f"Final batch saved: {len(combined_df):,} events", category='lifecycle')
            
            # Summary
            total_downloaded = successful_downloads
            self.event_logger.event('info', f"PBP download complete: {total_downloaded} games successful, {failed_downloads} failed", category='lifecycle')
            
            # Return combined result for verification
            conn = sqlite3.connect(self.db_path)
            try:
                result_df = pd.read_sql_query(f"SELECT * FROM pbp_logs WHERE game_id IN ({','.join(['?' for _ in game_ids])})", conn, params=game_ids)
                conn.close()
                return result_df
            except:
                conn.close()
                return pd.DataFrame()
                
        except Exception as e:
            self.event_logger.event('error', f"PBP collection failed: {e}", category=classify_error(e))
            return pd.DataFrame()
    
    def _save_pbp_logs(self, df: pd.DataFrame, season: str):
        """Save play-by-play data to database"""
        if df.empty:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Don't add season column - game_id already identifies it
            # pbp_logs table was created by live_model_backtester with specific schema
            try:
                df.to_sql('pbp_logs', conn, if_exists='append', index=False)
                self.event_logger.event('info', f"Saved {len(df):,} PBP events to database", category='lifecycle')
            except Exception as e:
                self.event_logger.event('error', f"pbp_logs write failed: {e}", category='data_integrity')
                raise  # Re-raise to see the error
            
            conn.commit()
    
    def _save_game_logs(self, df: pd.DataFrame, season: str):
        """
        Save game logs to database with SAFE PACE calculation (no division by zero)
        PACE = Possessions per 48 minutes
        """
        if df.empty:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Use selective DELETE + INSERT to update existing records (preserve historical data)
            cursor = conn.cursor()
            
            # Add season column
            df = df.copy()
            df['season'] = season
            
            # Check how many records exist for this season
            cursor.execute("SELECT COUNT(*) FROM game_logs WHERE season = ?", (season,))
            existing_count = cursor.fetchone()[0]
            
            if existing_count > 0:
                # Delete only the specific GAME_IDs we're about to update (not entire season)
                game_ids = df['GAME_ID'].unique().tolist()
                if game_ids:
                    placeholders = ','.join('?' * len(game_ids))
                    cursor.execute(f"DELETE FROM game_logs WHERE GAME_ID IN ({placeholders})", game_ids)
                    deleted = cursor.rowcount
                    self.event_logger.event('info', f"Updating {deleted} game_logs (preserving {existing_count - deleted} historical records)", category='data_integrity')
            else:
                self.event_logger.event('info', f"Inserting {len(df)} new game_logs for {season}", category='data_integrity')
            
            # CRITICAL: Calculate PACE for each game with SAFE division
            # Pace = 48 * ((Team Possessions + Opp Possessions) / (2 * (Minutes / 5)))
            # Estimate possessions from box score:
            # Poss ≈ FGA + 0.44*FTA - OREB + TOV
            if all(col in df.columns for col in ['FGA', 'FTA', 'OREB', 'TOV', 'MIN']):
                # Calculate estimated possessions
                df['POSS_EST'] = df['FGA'] + (0.44 * df['FTA']) - df['OREB'] + df['TOV']
                
                # SAFE pace calculation with division by zero protection
                df['pace'] = np.where(
                    (df['MIN'].notna()) & (df['MIN'] > 0),  # Check MIN is valid and positive
                    (df['POSS_EST'] * 48) / (df['MIN'] / 5),
                    100.0  # League average fallback if MIN is 0 or NaN
                )
                
                # Sanity check: pace should be between 90-115
                df['pace'] = np.clip(df['pace'], 90, 115)
            else:
                # Fallback: use league average if columns missing
                df['pace'] = 100.0
                self.event_logger.event('warning', "Missing columns for pace calculation, using default 100", category='data_integrity')
            
            # Save everything nba_api gives us PLUS our calculated pace
            try:
                df.to_sql('game_logs', conn, if_exists='append', index=False)
            except Exception as e:
                self.event_logger.event('warning', f"game_logs write note: {e}", category='data_integrity')
            
            conn.commit()
        
        # Populate game_results from the game logs
        self._populate_game_results_v2(df)
    
    def _populate_game_results_v2(self, game_logs_df: pd.DataFrame):
        """
        Populate game_results table from game logs (nba_api format)
        Consolidates two team-game records into one game record
        """
        if game_logs_df.empty:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Group by game_id to get both teams
            for game_id, game_df in game_logs_df.groupby('GAME_ID'):
                if len(game_df) != 2:
                    continue  # Skip if not exactly 2 teams
                
                # Sort so home team is first (has 'vs.' in matchup)
                game_df = game_df.sort_values('MATCHUP', ascending=False)
                teams = game_df.to_dict('records')
                
                # Determine home/away
                if 'vs.' in teams[0]['MATCHUP']:
                    home_idx, away_idx = 0, 1
                elif 'vs.' in teams[1]['MATCHUP']:
                    home_idx, away_idx = 1, 0
                elif '@' in teams[0]['MATCHUP']:
                    home_idx, away_idx = 1, 0
                else:
                    home_idx, away_idx = 0, 1
                
                home = teams[home_idx]
                away = teams[away_idx]
                
                # Extract data
                game_date = home['GAME_DATE']
                season = home['season']
                home_team = home['TEAM_ABBREVIATION']
                away_team = away['TEAM_ABBREVIATION']
                home_score = home['PTS']
                away_score = away['PTS']
                home_won = 1 if home['WL'] == 'W' else 0
                total_points = home_score + away_score
                point_diff = home_score - away_score
                
                # Insert into game_results
                cursor.execute('''
                    INSERT OR IGNORE INTO game_results 
                    (game_id, game_date, season, home_team, away_team, home_score, away_score, 
                     home_won, total_points, point_differential)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_id, game_date, season, home_team, away_team, home_score, away_score,
                      home_won, total_points, point_diff))
            
            conn.commit()
    
    def download_historical_seasons(self, start_year: int, end_year: int):
        """
        Download multiple seasons of data using nba_api
        Much more reliable than raw requests
        
        Args:
            start_year: Starting year (e.g., 2015 for 2015-16 season)
            end_year: Ending year (e.g., 2024 for 2024-25 season)
        """
        self.event_logger.event('info', f"DOWNLOADING NBA DATA (Using nba_api) Seasons: {start_year}-{end_year}", category='lifecycle')
        
        for year in range(start_year, end_year + 1):
            season = f"{year}-{str(year+1)[-2:]}"
            self.event_logger.event('info', f"Processing Season: {season}", category='lifecycle')
            
            try:
                # Fetch team stats (advanced metrics)
                team_stats_df = self.get_team_stats(season)
                
                # Small delay between requests
                time.sleep(2)
                
                # Fetch game logs (per-game data)
                game_logs_df = self.get_game_logs(season)

                # Fetch player impact (PIE) stats for injury valuation
                player_stats_df = self.get_player_impact_stats(season)
                
                # Longer delay between seasons
                time.sleep(3)
                
                self.event_logger.event('info', f"Season {season} completed successfully", category='lifecycle')
                
            except Exception as e:
                self.event_logger.event('error', f"Error processing season {season}: {e}", category=classify_error(e))
                continue
            self.event_logger.event('info', "DOWNLOAD COMPLETE", category='lifecycle')
        
        # Show summary
        self._show_data_summary()
    
    def _show_data_summary(self):
        """Display summary of downloaded data"""
        with sqlite3.connect(self.db_path) as conn:
            # Count records
            team_stats_count = pd.read_sql("SELECT COUNT(*) as count FROM team_stats", conn).iloc[0]['count']
            game_logs_count = pd.read_sql("SELECT COUNT(*) as count FROM game_logs", conn).iloc[0]['count']
            game_results_count = pd.read_sql("SELECT COUNT(*) as count FROM game_results", conn).iloc[0]['count']
            
            self.event_logger.event('info', f"Data Summary: team_stats={team_stats_count:,} game_logs={game_logs_count:,} game_results={game_results_count:,}", category='lifecycle')
            
            # Show season breakdown
            season_df = pd.read_sql("SELECT season, COUNT(*) as games FROM game_results GROUP BY season ORDER BY season", conn)
            if not season_df.empty:
                for _, row in season_df.iterrows():
                    self.event_logger.event('info', f"Season {row['season']}: {row['games']} games", category='lifecycle')
    
    def export_to_csv(self, output_path: str = "data/master_training_data_v5.csv"):
        """
        Export consolidated data to CSV for model training
        """
        self.event_logger.event('info', "Exporting data to CSV...", category='lifecycle')
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all game results with scores
            query = """
                SELECT 
                    game_id,
                    game_date as date,
                    season,
                    home_team,
                    away_team,
                    home_score,
                    away_score,
                    home_won,
                    total_points,
                    CASE 
                        WHEN point_differential IS NOT NULL THEN point_differential
                        ELSE (home_score - away_score)
                    END as actual_spread
                FROM game_results
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL
                ORDER BY game_date, game_id
            """
            
            df = pd.read_sql(query, conn)
        
        if df.empty:
            self.event_logger.event('warning', "No data to export", category='data_integrity')
            return
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        self.event_logger.event('info', f"Exported {len(df):,} games to {output_path}", category='lifecycle')
        self.event_logger.event('info', f"Date range: {df['date'].min()} to {df['date'].max()} | Seasons: {df['season'].nunique()}", category='lifecycle')


if __name__ == "__main__":
    collector = NBAStatsCollectorV2()
    collector.download_historical_seasons(2022, 2024)
    collector.export_to_csv()
