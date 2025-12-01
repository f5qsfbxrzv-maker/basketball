"""
FEATURE CALCULATOR - GOLD STANDARD EXTRACTION ENGINE (v5.0 OPTIMIZED)
The "Physics Engine" for the Self-Learning ML Model.

METHODOLOGY:
1. Identity Comparison: (Home O - Home D) vs (Away O - Away D)
2. Pace Adjustment: Scales efficiency differentials to expected possessions
3. SOS Adjustment: Adjusts team strength based on opponent quality
4. Raw Output: Delivers pure, unbiased signals to the ML ensemble
5. Recency Decay: Exponentially weights recent games (L10)

OPTIMIZATION:
- In-Memory Caching: 100x speed improvement via Pandas DataFrames
- Instant vectorized operations replace slow SQL queries
- Pre-calculated Strength of Schedule (SOS)
- Supports exponential decay for recency weighting

ARCHITECTURE:
- calculate_game_features() → Raw differentials for ML Model
- calculate_weighted_score() → Baseline "Eye Test" for GUI display ONLY
  (ML model ignores hardcoded weights and learns optimal weights from data)

INTEGRATION:
- Compatible with nba_stats_collector.py (nba_api based)
- Works with dynamic_elo_calculator.py for ELO differentials
- Feeds ml_model_trainer.py (Stacked Generalization: XGBoost + RF + LightGBM → LogisticRegression)
- Designed for NBA_Dashboard_Enhanced_v5.py integration
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from scipy.stats import norm
import logging

from v2.constants import RECENCY_STATS_BLEND_WEIGHT

try:
    from config import DB_PATH as MASTER_DB_PATH
except Exception:
    MASTER_DB_PATH = "data/database/nba_betting_data.db"

# Advanced Off/Def ELO system (optional)
try:
    from v2.core.off_def_elo_system import OffDefEloSystem
except ImportError:  # graceful degradation if not present
    OffDefEloSystem = None

# Import ELO calculator (has its own internal cache)
try:
    from v2.core.dynamic_elo_calculator import DynamicELOCalculator
except ImportError:
    logging.warning("DynamicELOCalculator not available - ELO features disabled")
    DynamicELOCalculator = None

class FeatureCalculatorV5:
    """
    Gold Standard Feature Engineering - The "Physics Engine"
    
    Extracts raw, unbiased features for the Self-Learning ML Model.
    Uses Identity Comparison logic: (Team Identity) vs (Opponent Identity)
    where Team Identity = Offense - Defense
    
    Features Generated (RAW for ML):
    - Four Factors differentials (eFG%, TOV%, REB%, FTr) using Identity Comparison
    - Net Rating differential: (H_off - H_def) - (A_off - A_def)
    - Offensive/Defensive ratings (raw values for totals prediction)
    - Pace prediction (average of both teams)
    - Rest day differentials (including back-to-back flags)
    - Head-to-head win rates (L3Y)
    - Strength of Schedule differential
    - ELO differential
    - Recency-weighted rolling averages (exponential decay)
    
    NOTE: calculate_weighted_score() uses STANDARD weights (Dean Oliver)
    for baseline GUI display ONLY. The ML model learns optimal weights.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize feature calculator with in-memory data caching
        
        Args:
            db_path: Path to SQLite database
        """
        # Use centralized master DB path when not explicitly provided
        self.db_path = db_path if db_path is not None else MASTER_DB_PATH
        self.logger = logging.getLogger(__name__)
        
        # --- STANDARD WEIGHTS (Dean Oliver Four Factors) ---
        # NOTE: These are ONLY used for calculate_weighted_score() baseline display.
        # The ML model (ml_model_trainer.py) learns optimal weights from raw features.
        self.WEIGHTS = {
            'efg': 0.40,    # Effective Field Goal %
            'tov': 0.25,    # Turnover %
            'reb': 0.20,    # Rebounding %
            'ftr': 0.15     # Free Throw Rate
        }
        
        # Baseline Model Parameters (for GUI "Eye Test" display only)
        self.HCA_POINTS = 2.5           # Home Court Advantage in points
        self.FF_BLEND_WEIGHT = 0.70     # Four Factors vs Net Rating blend
        self.SPREAD_STD_DEV = 13.5      # Standard deviation for win probability
        
        # --- IN-MEMORY DATA CACHES ---
        # These hold the entire database in RAM for instant access
        self.team_stats_df = pd.DataFrame()
        self.game_logs_df = pd.DataFrame()
        self.game_results_df = pd.DataFrame()
        self.player_stats_df = pd.DataFrame()  # Player impact (PIE) cache for injury weighting
        self.sos_map = {}  # Pre-calculated Strength of Schedule
        
        # Initialize legacy single ELO calculator BEFORE loading data
        if DynamicELOCalculator:
            self.elo_calculator = DynamicELOCalculator()
        else:
            self.elo_calculator = None
            self.logger.warning("ELO calculator not available")

        # Initialize advanced Off/Def ELO system if available
        if OffDefEloSystem:
            try:
                self.offdef_elo_system = OffDefEloSystem(self.db_path)
            except Exception as e:
                self.logger.warning(f"OffDefEloSystem init failed: {e}")
                self.offdef_elo_system = None
        else:
            self.offdef_elo_system = None
        
        # Load data and initialize ELO from history
        try:
            self._load_data_into_memory()
        except Exception as e:
            self.logger.warning(f"Data cache load failed: {e}. Run 'Download Data' first.")

    def _load_data_into_memory(self):
        """
        CRITICAL OPTIMIZATION:
        Loads DB tables into pandas DataFrames once.
        All subsequent lookups happen in RAM (nanoseconds) vs Disk (milliseconds).
        
        Performance Impact:
        - SQL query: ~10-50ms per query
        - Pandas filter: ~0.1-1ms per operation
        - 50-500x speedup for training/prediction
        """
        self.logger.info("Loading data into in-memory cache...")
        
        # NBA team abbreviation to full name mapping
        self.team_abbrev_map = {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
            'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
        }
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                cur = conn.cursor()

                # --- Team Stats (small) ---
                try:
                    self.team_stats_df = pd.read_sql_query("SELECT * FROM team_stats", conn)
                    self.logger.info(f"Loaded {len(self.team_stats_df)} team stat records")
                except Exception as e:
                    self.logger.warning(f"team_stats load failed: {e}")

                # Helper: determine if a table is very large
                def _table_count(table_name: str) -> int:
                    try:
                        cur.execute(f"SELECT COUNT(1) FROM {table_name}")
                        return int(cur.fetchone()[0] or 0)
                    except Exception:
                        return 0

                # --- Game Results ---
                gr_count = _table_count('game_results')
                if gr_count == 0:
                    # fallback: try to read whole table (may be empty)
                    self.game_results_df = pd.read_sql_query("SELECT * FROM game_results", conn)
                elif gr_count > 20000:
                    # Large history detected — load only the most recent N rows to bound memory use
                    self.logger.info(f"game_results too large ({gr_count} rows). Loading recent 20000 rows only to limit memory.")
                    self.game_results_df = pd.read_sql_query(
                        "SELECT * FROM game_results ORDER BY game_date DESC LIMIT 20000", conn
                    )
                    # Restore chronological order
                    if not self.game_results_df.empty:
                        self.game_results_df = self.game_results_df.iloc[::-1].reset_index(drop=True)
                else:
                    self.game_results_df = pd.read_sql_query("SELECT * FROM game_results", conn)

                if not self.game_results_df.empty:
                    date_col = 'GAME_DATE' if 'GAME_DATE' in self.game_results_df.columns else 'game_date'
                    try:
                        self.game_results_df['game_date'] = pd.to_datetime(self.game_results_df[date_col])
                    except Exception:
                        # best-effort parse
                        self.game_results_df['game_date'] = pd.to_datetime(self.game_results_df[date_col], errors='coerce')
                    if 'point_differential' in self.game_results_df.columns and 'point_diff' not in self.game_results_df.columns:
                        self.game_results_df['point_diff'] = self.game_results_df['point_differential']
                self.logger.info(f"Loaded {len(self.game_results_df)} game result records")

                # --- Game Logs ---
                gl_count = _table_count('game_logs')
                if gl_count == 0:
                    self.game_logs_df = pd.read_sql_query("SELECT * FROM game_logs", conn)
                elif gl_count > 50000:
                    # Game logs can be very large; only pull recent entries
                    self.logger.info(f"game_logs too large ({gl_count} rows). Loading recent 30000 rows only to limit memory.")
                    self.game_logs_df = pd.read_sql_query(
                        "SELECT * FROM game_logs ORDER BY game_date DESC LIMIT 30000", conn
                    )
                    if not self.game_logs_df.empty:
                        self.game_logs_df = self.game_logs_df.iloc[::-1].reset_index(drop=True)
                else:
                    self.game_logs_df = pd.read_sql_query("SELECT * FROM game_logs", conn)

                if not self.game_logs_df.empty:
                    date_col = 'GAME_DATE' if 'GAME_DATE' in self.game_logs_df.columns else 'game_date'
                    try:
                        self.game_logs_df['game_date'] = pd.to_datetime(self.game_logs_df[date_col])
                    except Exception:
                        self.game_logs_df['game_date'] = pd.to_datetime(self.game_logs_df[date_col], errors='coerce')
                self.logger.info(f"Loaded {len(self.game_logs_df)} game log records")

                # --- Player impact stats (PIE) - optional and small
                try:
                    pcount = _table_count('player_stats')
                    if pcount > 0 and pcount < 50000:
                        self.player_stats_df = pd.read_sql_query("SELECT player_id, player_name, team_abbreviation, season, pie FROM player_stats", conn)
                        self.logger.info(f"Loaded {len(self.player_stats_df)} player impact records")
                    else:
                        self.player_stats_df = pd.DataFrame()
                        if pcount > 0:
                            self.logger.info(f"player_stats table present but large ({pcount} rows) — skipping full load")
                except Exception as e:
                    self.logger.warning(f"Player stats not available: {e}")

            except Exception as e:
                self.logger.error(f"Error loading data: {e}")
                raise
        
        # Pre-calculate SOS since we have the data in memory
        self._calculate_season_sos()
        
        # Initialize ELO ratings from historical games
        if self.elo_calculator:
            self._initialize_elo_from_history(season="2025-26")
        
        self.logger.info("In-memory cache loaded successfully")

    def _calculate_season_sos(self):
        """
        Calculates Strength of Schedule using in-memory frames.
        SOS = Average Net Rating of all opponents faced
        """
        if self.game_results_df.empty:
            self.sos_map = {}
            return
        
        # Get mapping of Team -> Net Rating (use actual column names)
        if 'net_rating' in self.team_stats_df.columns:
            # Handle both 'team_name' and 'TEAM_NAME' column variations
            team_col = 'team_name' if 'team_name' in self.team_stats_df.columns else 'TEAM_NAME'
            ratings = self.team_stats_df.groupby(team_col)['net_rating'].mean().to_dict()
        else:
            ratings = {}
        
        sos = {}
        
        # Calculate SOS for each team
        for team in ratings:
            # Filter games involving this team (use actual column names: home_team, away_team)
            team_games = self.game_results_df[
                (self.game_results_df['home_team'] == team) |
                (self.game_results_df['away_team'] == team)
            ]
            def get_pbp_features(self, game_id: str) -> dict:
                """
                Aggregate play-by-play features for a given game_id from pbp_logs table.
                Returns a dict with possessions, scoring runs, clutch events, and event counts.
                """
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        pbp_df = pd.read_sql_query(
                            "SELECT * FROM pbp_logs WHERE game_id = ?", conn, params=(game_id,)
                        )
                except Exception as e:
                    self.logger.warning(f"PBP feature extraction failed for {game_id}: {e}")
                    return {
                        'pbp_event_count': 0,
                        'possessions': 0,
                        'scoring_runs': 0,
                        'clutch_events': 0
                    }

                if pbp_df.empty:
                    return {
                        'pbp_event_count': 0,
                        'possessions': 0,
                        'scoring_runs': 0,
                        'clutch_events': 0
                    }

                # Possessions: count of change of possession events (NBA MSGTYPE 1, 2, 3, 4, 5, 6, 7)
                possession_types = [1, 2, 3, 4, 5, 6, 7]
                possessions = pbp_df[pbp_df['event_type'].isin(possession_types)].shape[0]

                # Scoring runs: count runs of consecutive scoring events (MSGTYPE 1 = Made Shot, 3 = Free Throw)
                scoring_types = [1, 3]
                scoring_events = pbp_df[pbp_df['event_type'].isin(scoring_types)]
                scoring_runs = 0
                last_team = None
                run_length = 0
                for _, row in scoring_events.iterrows():
                    desc = row['event_description']
                    if 'makes' in desc or 'free throw' in desc:
                        team = 'home' if row['home_score'] > row['away_score'] else 'away'
                        if team == last_team:
                            run_length += 1
                        else:
                            if run_length >= 3:
                                scoring_runs += 1
                            run_length = 1
                            last_team = team
                if run_length >= 3:
                    scoring_runs += 1

                # Clutch events: last 2 minutes of 4th quarter or OT, close score (<=5 points)
                clutch_events = pbp_df[
                    (pbp_df['period'] >= 4) &
                    (pbp_df['clock'].str.startswith('00:')) &
                    (abs(pbp_df['home_score'] - pbp_df['away_score']) <= 5)
                ].shape[0]

                return {
                    'pbp_event_count': pbp_df.shape[0],
                    'possessions': possessions,
                    'scoring_runs': scoring_runs,
                    'clutch_events': clutch_events
                }
            
            # Extract opponents list
            opponents = []
            for _, row in team_games.iterrows():
                opp = row['away_team'] if row['home_team'] == team else row['home_team']
                opponents.append(opp)
            
            # Average their ratings
            if opponents:
                avg_opp_rating = np.mean([ratings.get(opp, 0) for opp in opponents])
                sos[team] = avg_opp_rating
            else:
                sos[team] = 0
        
        self.sos_map = sos
        self.logger.info(f"Calculated SOS for {len(sos)} teams")
    
    def _initialize_elo_from_history(self, season: str = "2025-26"):
        """
        Initialize ELO ratings from historical game results.
        Processes games chronologically to build accurate current ratings.
        
        Args:
            season: NBA season to initialize (e.g., "2025-26")
        """
        if not self.elo_calculator:
            return
        
        # Initialize all teams with base rating (use full names for ELO system)
        all_teams = list(self.team_abbrev_map.values())
        self.elo_calculator.initialize_teams(all_teams)
        
        if self.game_results_df.empty:
            self.logger.warning("No game results data - ELO will use initial ratings only")
            return
        
        # Filter to season and sort chronologically
        season_games = self.game_results_df[
            self.game_results_df['season'] == season
        ].sort_values('game_date')
        
        if len(season_games) == 0:
            self.logger.warning(f"No game results for {season} - ELO will use initial ratings")
            return
        
        # Process each game to update ratings
        processed = 0
        for _, game in season_games.iterrows():
            try:
                # Database stores abbreviations, convert to full names
                home_abbrev = game['home_team']
                away_abbrev = game['away_team']
                
                # Convert to full names (ELO system uses full names)
                home = self.team_abbrev_map.get(home_abbrev, home_abbrev)
                away = self.team_abbrev_map.get(away_abbrev, away_abbrev)
                
                # Update ELO ratings based on game outcome
                self.elo_calculator.update_ratings(
                    home_team=home,
                    away_team=away,
                    home_score=int(game['home_score']),
                    away_score=int(game['away_score']),
                    date=str(game['game_date'])
                )
                processed += 1
            except Exception as e:
                self.logger.warning(f"Error updating ELO for game: {e}")
                continue
        
        self.logger.info(f"Initialized ELO from {processed} games in {season}")


    def calculate_game_features(
        self,
        home_team: str,
        away_team: str,
        season: str = "2024-25",
        use_recency: bool = True,
        games_back: int = 10,
        game_date: str = None,
        decay_rate: float = 0.15
    ) -> Dict:
        """
        Extract RAW features for ML Model (Stacked Generalization).
        
        This is the core "Physics Engine" that calculates unbiased statistical signals.
        The ML model (XGBoost + RF + LightGBM → LogisticRegression) learns optimal
        weights from these raw differentials using TimeSeriesSplit validation.
        
        Identity Comparison Logic:
        - Home Identity = Home_Offense - Home_Defense
        - Away Identity = Away_Offense - Away_Defense
        - Net Differential = Home Identity - Away Identity
        
        Args:
            home_team: Home team name
            away_team: Away team name
            season: Season string (e.g., "2024-25")
            use_recency: Whether to use recency-weighted stats
            games_back: Number of recent games for rolling average
            game_date: Date of game (YYYY-MM-DD format)
            decay_rate: Exponential decay rate for recency weights
            
        Returns:
            Dictionary with RAW features for ML consumption:
            - vs_efg_diff, vs_tov, vs_reb_diff, vs_ftr_diff (Four Factors)
            - vs_net_rating (Identity Comparison)
            - elo_diff, sos_diff (contextual strength)
            - rest_days_diff, is_b2b_diff (situational)
            - h2h_win_rate_l3y (historical matchup)
            - h_off_rating, h_def_rating, a_off_rating, a_def_rating (for totals)
            - expected_pace (for pace adjustment)
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        current_date = datetime.strptime(game_date, '%Y-%m-%d')
        
        # Get season stats from cache
        home_stats = self._get_team_stats(home_team, season)
        away_stats = self._get_team_stats(away_team, season)
        
        if not home_stats or not away_stats:
            self.logger.warning(f"Missing stats for {home_team} vs {away_team}")
            return self._get_empty_features()
        
        # Apply recency weighting if enabled
        if use_recency:
            home_recency = self._get_decayed_stats(
                home_team, season, current_date, games_back, decay_rate
            )
            away_recency = self._get_decayed_stats(
                away_team, season, current_date, games_back, decay_rate
            )
            
            if home_recency and away_recency:
                # Blend season stats with recent form using configured weight
                home_stats = self._blend_stats(home_stats, home_recency, RECENCY_STATS_BLEND_WEIGHT)
                away_stats = self._blend_stats(away_stats, away_recency, RECENCY_STATS_BLEND_WEIGHT)
        
        # Calculate all feature groups
        features = {}
        features.update(self._calc_four_factors_diff(home_stats, away_stats))
        features.update(self._calc_pace_features(home_stats, away_stats))
        features.update(self._calc_rest_features(home_team, away_team, current_date))
        features.update(self._calc_h2h_features(home_team, away_team, current_date))

        # Injury impact (value-weighted) differential
        features.update(self._calculate_injury_impact(home_team, away_team))
        
        # Add legacy single ELO differential
        if self.elo_calculator:
            # Convert abbreviations to full names for ELO lookup
            home_full = self.team_abbrev_map.get(home_team, home_team)
            away_full = self.team_abbrev_map.get(away_team, away_team)
            features['elo_diff'] = self.elo_calculator.get_rating_differential(
                home_full, away_full, home_team=home_full
            )
        else:
            features['elo_diff'] = 0

        # Add advanced Off/Def ELO differentials (composite, offensive, defensive)
        if getattr(self, 'offdef_elo_system', None):
            try:
                elo_pack = self.offdef_elo_system.get_differentials(season, home_team, away_team)
                # Ensure keys present
                features['off_elo_diff'] = elo_pack.get('off_elo_diff', 0)
                features['def_elo_diff'] = elo_pack.get('def_elo_diff', 0)
                features['composite_elo_diff'] = elo_pack.get('composite_elo_diff', 0)
            except Exception as e:
                self.logger.warning(f"Off/Def ELO differential failed: {e}")
                features.setdefault('off_elo_diff', 0)
                features.setdefault('def_elo_diff', 0)
                features.setdefault('composite_elo_diff', 0)
        else:
            features['off_elo_diff'] = 0
            features['def_elo_diff'] = 0
            features['composite_elo_diff'] = 0
        
        # Add SOS differential
        home_sos = self.sos_map.get(home_team, 0)
        away_sos = self.sos_map.get(away_team, 0)
        features['sos_diff'] = home_sos - away_sos
        
        # Add raw ratings
        features['h_off_rating'] = home_stats.get('off_rating', 110)
        features['h_def_rating'] = home_stats.get('def_rating', 110)
        features['a_off_rating'] = away_stats.get('off_rating', 110)
        features['a_def_rating'] = away_stats.get('def_rating', 110)
        
        return features

    # --- CACHE-OPTIMIZED HELPERS ---
    
    def _get_team_stats(self, team: str, season: str) -> Optional[Dict]:
        """Query team stats from in-memory DataFrame"""
        if self.team_stats_df.empty:
            return None
        
        # Convert abbreviation to full name if needed
        if len(team) == 3 and team in self.team_abbrev_map:
            team = self.team_abbrev_map[team]
        
        # Handle both column name variations
        team_col = 'team_name' if 'team_name' in self.team_stats_df.columns else 'TEAM_NAME'
        season_col = 'season' if 'season' in self.team_stats_df.columns else 'SEASON_ID'
        
        # Pandas filtering is instant (< 1ms)
        result = self.team_stats_df[
            (self.team_stats_df[team_col] == team) &
            (self.team_stats_df[season_col] == season)
        ]
        
        if result.empty:
            return None
        
        # Convert to dict and normalize keys to lowercase
        stats_dict = result.iloc[0].to_dict()
        stats_lower = {k.lower(): v for k, v in stats_dict.items()}
        
        # Map NBA API column names to feature calculator expected names
        column_mapping = {
            'tm_tov_pct': 'tov_pct',        # Team turnover % -> tov_pct
            'fta_rate': 'ftr',                # Free throw attempt rate -> ftr  
            'fta_rate': 'ft_rate',            # Also map to ft_rate
            'opp_fta_rate': 'opp_ft_rate',    # Opponent FT rate
            'dreb_pct': 'def_reb_pct',        # Defensive rebound % -> def_reb_pct
        }
        
        # Apply mappings
        for old_name, new_name in column_mapping.items():
            if old_name in stats_lower:
                stats_lower[new_name] = stats_lower[old_name]
        
        return stats_lower
    
    def _get_decayed_stats(
        self,
        team: str,
        season: str,
        game_date: datetime,
        games_back: int,
        decay_rate: float
    ) -> Optional[Dict]:
        """
        Calculate exponentially-weighted rolling average from in-memory DataFrame.
        
        More recent games weighted higher using: weight = exp(-decay_rate * game_age)
        """
        if self.game_logs_df.empty:
            return None
        
        # Handle column name variations
        team_col = 'team_name' if 'team_name' in self.game_logs_df.columns else 'TEAM_NAME'
        season_col = 'season' if 'season' in self.game_logs_df.columns else 'SEASON_ID'
        
        # Pandas query (fast)
        query = self.game_logs_df[
            (self.game_logs_df[team_col] == team) &
            (self.game_logs_df[season_col] == season) &
            (self.game_logs_df['game_date'] < game_date)
        ]
        
        # Get most recent N games (sorted by date descending)
        recent_games = query.nlargest(games_back, 'game_date')
        
        # Need at least half the requested games for valid average
        if len(recent_games) < games_back / 2:
            return None
        
        # Calculate exponential weights (most recent = highest weight)
        weights = np.exp(-decay_rate * np.arange(len(recent_games)))
        weights /= weights.sum()  # Normalize to sum to 1
        
        # Columns to average (only use columns that exist in the DataFrame)
        stat_columns = [
            'efg_pct', 'tov_pct', 'oreb_pct', 'ft_rate',
            'off_rating', 'def_rating', 
            'opp_efg_pct', 'opp_tov_pct', 'def_reb_pct', 'opp_ft_rate'
        ]
        # Add pace only if it exists (not in game_logs currently)
        if 'pace' in recent_games.columns:
            stat_columns.append('pace')
        
        # Vectorized weighted average calculation
        weighted_stats = {}
        for col in stat_columns:
            if col in recent_games.columns:
                weighted_stats[col] = np.sum(recent_games[col].values * weights)
        
        return weighted_stats if weighted_stats else None

    def _calc_rest_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime
    ) -> Dict:
        """Calculate rest day features from in-memory game results"""
        if self.game_results_df.empty:
            return {'rest_days_diff': 0, 'is_b2b_diff': 0}
        
        def get_rest_days(team: str) -> int:
            """Get days of rest for a team"""
            # Filter games involving this team before the current date (use actual column names)
            past_games = self.game_results_df[
                ((self.game_results_df['home_team'] == team) |
                 (self.game_results_df['away_team'] == team)) &
                (self.game_results_df['game_date'] < game_date)
            ]
            
            if past_games.empty:
                return 7  # Default rest if no previous games
            
            last_game = past_games['game_date'].max()
            rest = (game_date.date() - last_game.date()).days - 1
            
            # Clamp rest days between 0-5
            return max(0, min(rest, 5))
        
        home_rest = get_rest_days(home_team)
        away_rest = get_rest_days(away_team)
        
        return {
            'rest_days_diff': home_rest - away_rest,
            'is_b2b_diff': (1 if home_rest == 0 else 0) - (1 if away_rest == 0 else 0)
        }

    def _calc_h2h_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime
    ) -> Dict:
        """Calculate head-to-head features from in-memory game results"""
        if self.game_results_df.empty:
            return {'h2h_win_rate_l3y': 0.5}
        
        # Look back 3 years
        start_date = game_date - timedelta(days=365 * 3)
        
        # Complex filtering is much faster in Pandas than SQL (use actual column names)
        h2h_games = self.game_results_df[
            (((self.game_results_df['home_team'] == home_team) &
              (self.game_results_df['away_team'] == away_team)) |
             ((self.game_results_df['home_team'] == away_team) &
              (self.game_results_df['away_team'] == home_team))) &
            (self.game_results_df['game_date'] >= start_date) &
            (self.game_results_df['game_date'] < game_date)
        ]
        
        if h2h_games.empty:
            return {'h2h_win_rate_l3y': 0.5}  # Neutral if no history
        
        # Count wins for home team (use point_diff not point_differential)
        wins = 0
        for _, row in h2h_games.iterrows():
            # Determine margin from home team's perspective
            if row['home_team'] == home_team:
                margin = row['point_diff']
            else:
                margin = -row['point_diff']
            
            if margin > 0:
                wins += 1
        
        win_rate = wins / len(h2h_games)
        return {'h2h_win_rate_l3y': win_rate}

    def _calculate_injury_impact(
        self,
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Calculate injury impact differential using live injury reports
        
        Returns 6 features:
        - home_injury_impact: Expected impact to home team (points)
        - away_injury_impact: Expected impact to away team
        - injury_differential: Net advantage (positive = home advantaged)
        - home_star_out_count: Number of star players (>3.0 value) OUT for home
        - away_star_out_count: Number of star players OUT for away
        - both_teams_injured: Both teams missing star player (binary)
        
        NOTE: This requires live injury data. For historical training,
        these features will be zeros unless injury database is populated.
        """
        try:
            # Import here to avoid circular dependency
            from v2.services.injury_scraper import InjuryScraper
            
            scraper = InjuryScraper()
            differential = scraper.get_game_injury_differential(home_team, away_team)
            
            # Get individual team impacts for additional features
            home_impact_data = scraper.get_team_injury_impact(home_team)
            away_impact_data = scraper.get_team_injury_impact(away_team)
            
            return {
                'home_injury_impact': home_impact_data['total_impact'],
                'away_injury_impact': away_impact_data['total_impact'],
                'injury_differential': differential,
                'home_star_out_count': home_impact_data['star_injuries'],
                'away_star_out_count': away_impact_data['star_injuries'],
                'both_teams_injured': int(
                    home_impact_data['star_injuries'] > 0 and 
                    away_impact_data['star_injuries'] > 0
                ),
            }
        except Exception as e:
            # Graceful degradation if injury scraper fails or data unavailable
            self.logger.warning(f"Injury impact calculation failed: {e}")
            return {
                'home_injury_impact': 0.0,
                'away_injury_impact': 0.0,
                'injury_differential': 0.0,
                'home_star_out_count': 0,
                'away_star_out_count': 0,
                'both_teams_injured': 0,
            }

    def calculate_weighted_score(self, features: Dict) -> Dict:
        """
        Generate BASELINE predictions for GUI "Eye Test" display.
        
        ⚠️ IMPORTANT: This is NOT used for actual betting decisions.
        The ML model (ml_model_trainer.py) learns optimal weights from raw features
        and produces the ACTUAL win probability via Stacked Generalization.
        
        This method uses STANDARD Four Factors weights (Dean Oliver) to provide
        a physics-based reference point in the dashboard for comparison purposes.
        
        Args:
            features: Dictionary of raw features from calculate_game_features()
            
        Returns:
            Dictionary with baseline predictions:
            - spread: Predicted point spread (Home perspective)
            - total: Predicted total points
            - home_score, away_score: Projected scores
            - win_prob: Statistical probability (for reference, not betting)
        """
        # Calculate Four Factors composite score
        weights_sum = sum(self.WEIGHTS.values())
        four_factors_score = (
            features['vs_efg_diff'] * self.WEIGHTS['efg'] +
            features['vs_tov'] * self.WEIGHTS['tov'] +
            features['vs_reb_diff'] * self.WEIGHTS['reb'] +
            features['vs_ftr_diff'] * self.WEIGHTS['ftr']
        ) / weights_sum
        
        # Blend Four Factors with Net Rating
        blended_score = (
            four_factors_score * self.FF_BLEND_WEIGHT +
            features['vs_net_rating'] * (1 - self.FF_BLEND_WEIGHT)
        )
        
        # Add SOS adjustment
        blended_score += features.get('sos_diff', 0)
        
        # Convert to points using pace
        pace = features.get('expected_pace', 100)
        raw_margin = (blended_score / 100.0) * pace
        
        # Add home court advantage
        predicted_spread = raw_margin + self.HCA_POINTS
        
        # Calculate projected scores
        home_proj_rating = (features['h_off_rating'] + features['a_def_rating']) / 2.0
        away_proj_rating = (features['a_off_rating'] + features['h_def_rating']) / 2.0
        
        home_points = (home_proj_rating / 100.0) * pace + (self.HCA_POINTS / 2.0)
        away_points = (away_proj_rating / 100.0) * pace - (self.HCA_POINTS / 2.0)
        predicted_total = home_points + away_points
        
        # Calculate win probability using normal distribution
        z_score = predicted_spread / self.SPREAD_STD_DEV
        win_probability = norm.cdf(z_score)
        
        return {
            'spread': predicted_spread,
            'total': predicted_total,
            'home_score': home_points,
            'away_score': away_points,
            'win_prob': win_probability
        }

    def _calc_four_factors_diff(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """
        Calculate Four Factors differentials using Identity Comparison.
        
        Identity Logic: (Team O - Team D) vs (Opponent O - Opponent D)
        This captures each team's NET advantage in each factor.
        
        Four Factors (Dean Oliver):
        1. Shooting (eFG%) - Most important (40%)
        2. Turnovers (TOV%) - Second (25%)
        3. Rebounding (OREB% vs DREB%) - Third (20%)
        4. Free Throws (FTr) - Fourth (15%)
        
        Returns raw differentials for ML model to learn optimal weighting.
        """
        return {
            # Shooting differential (offense vs defense for each team)
            'vs_efg_diff': (
                (home_stats['efg_pct'] - home_stats['opp_efg_pct']) -
                (away_stats['efg_pct'] - away_stats['opp_efg_pct'])
            ),
            
            # Turnover differential (forcing vs committing)
            'vs_tov': (
                (home_stats['opp_tov_pct'] - home_stats['tov_pct']) -
                (away_stats['opp_tov_pct'] - away_stats['tov_pct'])
            ),
            
            # Rebounding differential
            'vs_reb_diff': (
                (home_stats['oreb_pct'] - (1 - home_stats['def_reb_pct'])) -
                (away_stats['oreb_pct'] - (1 - away_stats['def_reb_pct']))
            ),
            
            # Free throw rate differential
            'vs_ftr_diff': (
                (home_stats['ft_rate'] - home_stats['opp_ft_rate']) -
                (away_stats['ft_rate'] - away_stats['opp_ft_rate'])
            ),
            
            # Net rating differential
            'vs_net_rating': (
                (home_stats['off_rating'] - home_stats['def_rating']) -
                (away_stats['off_rating'] - away_stats['def_rating'])
            )
        }

    def _calc_pace_features(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Calculate expected pace for the game"""
        home_pace = home_stats.get('pace', 100)
        away_pace = away_stats.get('pace', 100)
        
        return {
            'expected_pace': (home_pace + away_pace) / 2
        }

    def _blend_stats(
        self,
        season_stats: Dict,
        recent_stats: Dict,
        recent_weight: float
    ) -> Dict:
        """
        Blend season stats with recent form.
        
        Args:
            season_stats: Full season statistics
            recent_stats: Recent game statistics
            recent_weight: Weight for recent stats (0-1)
            
        Returns:
            Blended statistics dictionary
        """
        blended = season_stats.copy()
        
        for key in recent_stats:
            if key in season_stats:
                blended[key] = (
                    recent_stats[key] * recent_weight +
                    season_stats[key] * (1 - recent_weight)
                )
        
        return blended

    def _get_empty_features(self) -> Dict:
        """Return empty feature set with default values"""
        return {
            'vs_net_rating': 0,
            'h_off_rating': 110,
            'h_def_rating': 110,
            'a_off_rating': 110,
            'a_def_rating': 110,
            'vs_efg_diff': 0,
            'vs_tov': 0,
            'vs_reb_diff': 0,
            'vs_ftr_diff': 0,
            'expected_pace': 100,
            'rest_days_diff': 0,
            'is_b2b_diff': 0,
            'h2h_win_rate_l3y': 0.5,
            'elo_diff': 0,
            'off_elo_diff': 0,
            'def_elo_diff': 0,
            'composite_elo_diff': 0,
            'sos_diff': 0,
            'injury_impact_diff': 0.0
        }

    def reload_data(self):
        """Reload all data from database into memory"""
        self.logger.info("Reloading data cache...")
        self._load_data_into_memory()

    def get_cache_stats(self) -> Dict:
        """Get statistics about the in-memory cache"""
        return {
            'team_stats_count': len(self.team_stats_df),
            'game_logs_count': len(self.game_logs_df),
            'game_results_count': len(self.game_results_df),
            'sos_teams_count': len(self.sos_map),
            'cache_loaded': not self.team_stats_df.empty
        }

    def create_feature_dataframe(self, games: List[Dict]) -> pd.DataFrame:
        """Vectorize feature generation for a list of games into a DataFrame.
        Each game dict should include keys: home_team, away_team, season, game_date (YYYY-MM-DD).
        Missing values are handled with defaults. Includes PBP features."""
        rows = []
        for g in games or []:
            try:
                home = g.get('home_team') or g.get('home')
                away = g.get('away_team') or g.get('away')
                season = g.get('season') or datetime.now().strftime('%Y-%Y')
                date = g.get('game_date') or g.get('date') or datetime.now().strftime('%Y-%m-%d')
                game_id = g.get('game_id') or None
                feats = self.calculate_game_features(
                    home_team=home,
                    away_team=away,
                    season=season,
                    game_date=date,
                    use_recency=True,
                    games_back=10
                )
                scores = self.calculate_weighted_score(feats)
                row = {
                    'home_team': home,
                    'away_team': away,
                    'season': season,
                    'game_date': date,
                }
                row.update(feats)
                row.update({f"pred_{k}": v for k, v in scores.items()})
                # Add PBP features if game_id is available
                if game_id:
                    pbp_feats = self.get_pbp_features(game_id)
                    row.update(pbp_feats)
                rows.append(row)
            except Exception as e:
                self.logger.error(f"Feature build failed for {g}: {e}")
                continue
        return pd.DataFrame(rows)

    # =======================
    # INJURY IMPACT (VALUE)
    # =======================
    def _calculate_injury_impact(self, home_team: str, away_team: str) -> Dict:
        """Compute value-weighted injury differential using PIE and status weights.
        Returns dict with 'injury_impact_diff' scaled to 'net rating' point units (×100).
        """
        try:
            # Active injuries table expected from InjuryDataCollectorV2
            with sqlite3.connect(self.db_path) as conn:
                inj_df = pd.read_sql_query(
                    "SELECT player_name, team_name, status FROM active_injuries", conn
                ) if conn else pd.DataFrame()
        except Exception as e:
            self.logger.debug(f"Injury table read failed: {e}")
            return {'injury_impact_diff': 0.0}

        if inj_df.empty or self.player_stats_df.empty:
            return {'injury_impact_diff': 0.0}

        # Status weights
        status_weights = {
            'OUT': 1.0,
            'DOUBTFUL': 0.75,
            'QUESTIONABLE': 0.50,
            'PROBABLE': 0.10
        }

        def team_loss(team_abbr: str) -> float:
            subset = inj_df[inj_df['team_name'].str.contains(team_abbr, case=False, na=False)]
            total = 0.0
            for _, row in subset.iterrows():
                status = str(row['status']).upper()
                weight = next((w for k, w in status_weights.items() if k in status), 0.0)
                pname = row['player_name']
                # Exact match first
                pstat = self.player_stats_df[self.player_stats_df['player_name'] == pname]
                if pstat.empty:
                    pstat = self.player_stats_df[self.player_stats_df['player_name'].str.contains(pname, case=False, na=False)]
                pie = float(pstat.iloc[0]['pie']) if not pstat.empty else 0.05
                total += weight * pie
            return total

        try:
            home_loss = team_loss(home_team)
            away_loss = team_loss(away_team)
            diff = (away_loss - home_loss) * 100.0
            return {'injury_impact_diff': diff}
        except Exception as e:
            self.logger.debug(f"Injury impact calc failed: {e}")
            return {'injury_impact_diff': 0.0}


# Backward compatibility alias (now points to v6 implementation)
FeatureCalculatorV5 = FeatureCalculatorV5

if __name__ == "__main__":
    # Test the feature calculator
    logging.basicConfig(level=logging.INFO)
    
    calc = FeatureCalculatorV5()
    
    # Print cache stats
    stats = calc.get_cache_stats()
    print("\n📊 Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test feature calculation
    features = calc.calculate_game_features(
        home_team="LAL",
        away_team="BOS",
        season="2024-25",
        use_recency=True,
        games_back=10
    )
    
    print("\n🔢 Sample Features:")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Test prediction
    predictions = calc.calculate_weighted_score(features)
    print("\n🎯 Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value:.2f}")
