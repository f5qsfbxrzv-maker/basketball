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
- calculate_game_features() ΓåÆ Raw differentials for ML Model
- calculate_weighted_score() ΓåÆ Baseline "Eye Test" for GUI display ONLY
  (ML model ignores hardcoded weights and learns optimal weights from data)

INTEGRATION:
- Compatible with nba_stats_collector.py (nba_api based)
- Works with dynamic_elo_calculator.py for ELO differentials
- Feeds ml_model_trainer.py (Stacked Generalization: XGBoost + RF + LightGBM ΓåÆ LogisticRegression)
- Designed for NBA_Dashboard_Enhanced_v5.py integration

FEATURE PRUNING:
- SHAP-based whitelist reduces 107 ΓåÆ 33 features
- Filters out collinear noise (ewma_foul_rate vs foul_rate)
- Surfaces injury signal from rank #69 to top 5
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from scipy.stats import norm
import logging

from config.constants import RECENCY_STATS_BLEND_WEIGHT

try:
    from config import DB_PATH as MASTER_DB_PATH
except Exception:
    MASTER_DB_PATH = "nba_betting_data.db"

# Feature whitelist for SHAP-based pruning
try:
    from config.feature_whitelist import FEATURE_WHITELIST
except ImportError:
    FEATURE_WHITELIST = None  # Fall back to all features if not available

# Advanced Off/Def ELO system (optional)
try:
    from src.features.off_def_elo_system import OffDefEloSystem
except ImportError:  # graceful degradation if not present
    OffDefEloSystem = None

# Note: Legacy DynamicELOCalculator has been replaced by OffDefEloSystem
# All ELO features now come from the Off/Def ELO system above

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
        # ΓÜá∩╕Å CRITICAL: team_stats is DEPRECATED - use game_advanced_stats with date filtering!
        self.team_stats_df = pd.DataFrame()  # DEPRECATED - causes data leakage
        self.game_advanced_stats_df = pd.DataFrame()  # Γ£ô CORRECT - game-by-game with dates
        self.game_logs_df = pd.DataFrame()
        self.game_results_df = pd.DataFrame()
        self.player_stats_df = pd.DataFrame()  # Player impact (PIE) cache for injury weighting
        self.injury_history = {}  # Track injury_impact over time for EWMA calculation
        self.sos_map = {}  # Pre-calculated Strength of Schedule
        
        # Legacy DynamicELOCalculator has been replaced by OffDefEloSystem
        self.elo_calculator = None

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

                # --- Game Advanced Stats (CORRECT - game-by-game with dates) ---
                try:
                    self.game_advanced_stats_df = pd.read_sql_query(
                        "SELECT * FROM game_advanced_stats ORDER BY game_date", conn
                    )
                    if not self.game_advanced_stats_df.empty:
                        self.game_advanced_stats_df['game_date'] = pd.to_datetime(
                            self.game_advanced_stats_df['game_date']
                        )
                    self.logger.info(f"Loaded {len(self.game_advanced_stats_df)} game advanced stat records")
                except Exception as e:
                    self.logger.warning(f"game_advanced_stats load failed: {e}")
                    self.logger.warning("ΓÜá∩╕Å CRITICAL: Cannot calculate features without game_advanced_stats!")

                # --- Team Stats (DEPRECATED - DO NOT USE for predictions) ---
                # This table has FULL SEASON averages with NO date filtering = DATA LEAKAGE
                try:
                    self.team_stats_df = pd.read_sql_query("SELECT * FROM team_stats", conn)
                    self.logger.warning(f"ΓÜá∩╕Å Loaded team_stats ({len(self.team_stats_df)} rows) - DEPRECATED, use game_advanced_stats instead")
                except Exception as e:
                    self.logger.info(f"team_stats not loaded (OK - should use game_advanced_stats): {e}")

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
                    # Large history detected ΓÇö load only the most recent N rows to bound memory use
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
                            self.logger.info(f"player_stats table present but large ({pcount} rows) ΓÇö skipping full load")
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

    def get_team_stats_as_of_date(self, team_abb: str, as_of_date: str, lookback_games: int = 10) -> Dict[str, float]:
        """
        Γ£à CORRECT WAY: Calculate team stats using game_advanced_stats WITH date filtering
        
        This prevents data leakage by only using games BEFORE as_of_date.
        Also calculates opponent stats using "mirror" rows (same game_id, different team).
        
        Args:
            team_abb: Team abbreviation (e.g., 'LAL', 'BOS')
            as_of_date: Only use games before this date (YYYY-MM-DD or datetime)
            lookback_games: Number of recent games to average (default: 10)
        
        Returns:
            Dictionary with team stats AND opponent stats (opp_efg_pct, opp_tov_pct, etc.)
        """
        if self.game_advanced_stats_df.empty:
            self.logger.warning(f"ΓÜá∩╕Å game_advanced_stats not loaded - cannot calculate stats for {team_abb}")
            return {}
        
        # Convert as_of_date to datetime if string or date object
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)
        elif hasattr(as_of_date, 'year'):  # datetime.date object
            as_of_date = pd.to_datetime(as_of_date)
        
        # Filter: team games BEFORE as_of_date only (prevent data leakage!)
        team_games = self.game_advanced_stats_df[
            (self.game_advanced_stats_df['team_abb'] == team_abb) &
            (self.game_advanced_stats_df['game_date'] < as_of_date)
        ].sort_values('game_date', ascending=False)
        
        if len(team_games) == 0:
            self.logger.warning(f"No games found for {team_abb} before {as_of_date}")
            return {}
        
        # Take last N games (most recent)
        recent_games = team_games.head(lookback_games)
        
        # Get opponent stats using "mirror" logic
        # For each game, find the opponent's row (same game_id, different team)
        opp_stats_list = []
        for _, game in recent_games.iterrows():
            # Find mirror row: same game_id, different team_abb
            mirror = self.game_advanced_stats_df[
                (self.game_advanced_stats_df['game_id'] == game['game_id']) &
                (self.game_advanced_stats_df['team_abb'] != team_abb)
            ]
            if len(mirror) > 0:
                opp_stats_list.append(mirror.iloc[0])
        
        # Calculate team averages
        stats = {
            'off_rating': recent_games['off_rating'].mean(),
            'def_rating': recent_games['def_rating'].mean(),
            'net_rating': recent_games['net_rating'].mean(),
            'pace': recent_games['pace'].mean(),
            'efg_pct': recent_games['efg_pct'].mean(),
            'tov_pct': recent_games['tov_pct'].mean(),
            'orb_pct': recent_games['orb_pct'].mean(),
            'fta_rate': recent_games['fta_rate'].mean(),
            'fg3a_per_100': recent_games['fg3a_per_100'].mean() if 'fg3a_per_100' in recent_games.columns else 0,
            'fg3_pct': recent_games['fg3_pct'].mean() if 'fg3_pct' in recent_games.columns else 0,
            'games_used': len(recent_games)
        }
        
        # Calculate opponent averages (from mirror rows)
        if len(opp_stats_list) > 0:
            opp_df = pd.DataFrame(opp_stats_list)
            stats['opp_efg_pct'] = opp_df['efg_pct'].mean()
            stats['opp_tov_pct'] = opp_df['tov_pct'].mean()
            stats['opp_orb_pct'] = opp_df['orb_pct'].mean()
            stats['opp_fta_rate'] = opp_df['fta_rate'].mean()
            stats['opp_off_rating'] = opp_df['off_rating'].mean()
            stats['opp_def_rating'] = opp_df['def_rating'].mean()
        else:
            # Fallback to league average estimates if no mirror rows found
            stats['opp_efg_pct'] = 0.520
            stats['opp_tov_pct'] = 0.135
            stats['opp_orb_pct'] = 0.250
            stats['opp_fta_rate'] = 0.240
            stats['opp_off_rating'] = 110.0
            stats['opp_def_rating'] = 110.0
        
        return stats

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
        season: str = None,  # Auto-calculate from game_date if None
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
            season: Season string (e.g., "2025-26"). If None, auto-calculated from game_date.
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
        
        # Auto-calculate season from game_date if not provided
        if season is None:
            year = current_date.year
            month = current_date.month
            # NBA season runs Oct-June, so Oct-Dec uses current year, Jan+ uses previous year
            season = f"{year}-{str(year + 1)[-2:]}" if month >= 10 else f"{year - 1}-{str(year)[-2:]}"
        
        # Get stats WITH DATE FILTERING to prevent data leakage
        home_stats = self.get_team_stats_as_of_date(home_team, game_date, lookback_games=10)
        away_stats = self.get_team_stats_as_of_date(away_team, game_date, lookback_games=10)
        
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
        
        # NEW WHITELISTED FEATURES (EWMA-based, high signal)
        features.update(self._calc_ewma_features(home_team, away_team, game_date))
        features.update(self._calc_rest_fatigue_features(home_team, away_team, current_date))
        features.update(self._calc_baseline_features(home_team, away_team, game_date))
        features.update(self._calc_altitude_features(home_team, away_team))
        
        # OLD FEATURES (kept for backwards compatibility, will be filtered by whitelist)
        features.update(self._calc_four_factors_diff(home_stats, away_stats))
        features.update(self._calc_pace_features(home_stats, away_stats))
        features.update(self._calc_rest_features(home_team, away_team, current_date))
        features.update(self._calc_h2h_features(home_team, away_team, current_date))

        # Injury impact (value-weighted) differential
        # Use LIVE injuries for today/future games, historical for backtesting past games
        from datetime import datetime as dt
        today = dt.now().date()
        game_dt = current_date.date() if hasattr(current_date, 'date') else current_date
        
        if game_dt >= today:
            # Future or today's game -> use LIVE injury data from ESPN
            injury_data = self._calculate_injury_impact(home_team, away_team)
        else:
            # Historical game -> use database injury records
            injury_data = self._calculate_historical_injury_impact(home_team, away_team, game_date)
        
        features.update(injury_data)
        
        # Injury features (8 total) now calculated in _calculate_injury_impact():
        # injury_impact_diff, injury_impact_abs, injury_shock_home/away/diff, 
        # home/away_star_missing, star_mismatch
        
        # Add temporal features (7 features for time context)
        try:
            from datetime import datetime as dt
            if isinstance(current_date, str):
                current_date = dt.strptime(current_date, '%Y-%m-%d')
            
            # Season features - 2025-26 season
            features['season_year'] = 2025  # 2025-26 season
            features['season_year_normalized'] = 2025 / 2026.0
            
            # Game progression features  
            # 2025-26 NBA season started October 21, 2025
            season_start = dt(2025, 10, 21)
            days_since_start = (current_date - season_start).days
            
            # Typical NBA season: ~82 games over ~170 days (Oct-April)
            # Teams average ~3.5 games per week = ~0.5 games per day
            features['games_into_season'] = max(0, days_since_start * 0.48)  # ~0.48 games/day average
            features['season_progress'] = min(max(0, days_since_start / 170.0), 1.0)  # 170-day regular season
            features['endgame_phase'] = 1.0 if days_since_start > 140 else 0.0  # Last ~30 days
            
            # Calendar features
            features['season_month'] = current_date.month
            features['is_season_opener'] = 1.0 if days_since_start < 7 else 0.0
        except Exception as e:
            # Fallback defaults
            features.setdefault('season_year', 2024)
            features.setdefault('season_year_normalized', 2024/2025.0)
            features.setdefault('games_into_season', 30.0)
            features.setdefault('season_progress', 0.5)
            features.setdefault('endgame_phase', 0.0)
            features.setdefault('season_month', 12)
            features.setdefault('is_season_opener', 0.0)
        
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
        
        # SHAP-based feature pruning: Filter to whitelist only
        if FEATURE_WHITELIST is not None:
            features = {k: v for k, v in features.items() if k in FEATURE_WHITELIST}
            self.logger.info(f"Feature pruning applied: {len(features)} features retained from whitelist")
        
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
        Calculate injury impact using LIVE ESPN injury data
        
        Returns injury_impact features that match model expectations:
        - injury_impact_diff, injury_impact_abs
        - injury_shock_home, injury_shock_away, injury_shock_diff  
        - home_star_missing, away_star_missing, star_mismatch
        """
        # Import the LIVE injury calculator
        from injury_impact_live import calculate_team_injury_impact_simple
        
        # Calculate raw impacts using live injuries
        home_impact = calculate_team_injury_impact_simple(home_team, None, self.db_path)
        away_impact = calculate_team_injury_impact_simple(away_team, None, self.db_path)
        
        # Get EWMA baselines for shock calculation
        home_ewma_inj = 0.0  # TODO: Calculate historical average if needed
        away_ewma_inj = 0.0
        
        # STAR_THRESHOLD: PIE >= 0.15 implies ~3.0+ impact
        STAR_THRESHOLD = 4.0
        
        return {
            'home_injury_impact': home_impact,
            'away_injury_impact': away_impact,
            'injury_impact_diff': home_impact - away_impact,
            'injury_impact_abs': abs(home_impact) + abs(away_impact),
            'injury_shock_home': home_impact - home_ewma_inj,
            'injury_shock_away': away_impact - away_ewma_inj,
            'injury_shock_diff': (home_impact - home_ewma_inj) - (away_impact - away_ewma_inj),
            'home_star_missing': 1 if home_impact >= STAR_THRESHOLD else 0,
            'away_star_missing': 1 if away_impact >= STAR_THRESHOLD else 0,
            'star_mismatch': (1 if home_impact >= STAR_THRESHOLD else 0) - (1 if away_impact >= STAR_THRESHOLD else 0),
        }

    def calculate_weighted_score(self, features: Dict) -> Dict:
        """
        Generate BASELINE predictions for GUI "Eye Test" display.
        
        ΓÜá∩╕Å IMPORTANT: This is NOT used for actual betting decisions.
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
                (home_stats.get('efg_pct', 0.5) - home_stats.get('opp_efg_pct', 0.5)) -
                (away_stats.get('efg_pct', 0.5) - away_stats.get('opp_efg_pct', 0.5))
            ),
            
            # Turnover differential (forcing vs committing)
            'vs_tov': (
                (home_stats.get('opp_tov_pct', 0.135) - home_stats.get('tov_pct', 0.135)) -
                (away_stats.get('opp_tov_pct', 0.135) - away_stats.get('tov_pct', 0.135))
            ),
            
            # Rebounding differential (use orb_pct from database, estimate drb_pct)
            'vs_reb_diff': (
                (home_stats.get('orb_pct', 0.25) - (1 - home_stats.get('opp_orb_pct', 0.25))) -
                (away_stats.get('orb_pct', 0.25) - (1 - away_stats.get('opp_orb_pct', 0.25)))
            ),
            
            # Free throw rate differential (use fta_rate from database)
            'vs_ftr_diff': (
                (home_stats.get('fta_rate', 0.24) - home_stats.get('opp_fta_rate', 0.24)) -
                (away_stats.get('fta_rate', 0.24) - away_stats.get('opp_fta_rate', 0.24))
            ),
            
            # Net rating differential
            'vs_net_rating': (
                (home_stats.get('off_rating', 110) - home_stats.get('def_rating', 110)) -
                (away_stats.get('off_rating', 110) - away_stats.get('def_rating', 110))
            )
        }

    # =======================
    # NEW WHITELISTED FEATURES  
    # =======================
    
    def _calc_ewma_features(self, home_team: str, away_team: str, game_date: str) -> Dict:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) features
        These capture recent form with recency bias (recent games weighted higher)
        """
        features = {}
        
        # Get EWMA stats for both teams (last 10 games, alpha=0.3 for strong recency bias)
        home_ewma = self._get_ewma_stats(home_team, game_date, span=10)
        away_ewma = self._get_ewma_stats(away_team, game_date, span=10)
        
        if not home_ewma or not away_ewma:
            # Return zeros if data missing
            return {
                'ewma_efg_diff': 0,
                'ewma_tov_diff': 0,
                'ewma_orb_diff': 0,
                'ewma_pace_diff': 0,
                'ewma_vol_3p_diff': 0,
                'home_ewma_3p_pct': 0.35,
                'away_ewma_3p_pct': 0.35,
                'away_ewma_tov_pct': 0.135,
                'home_orb': 0.25,
                'away_orb': 0.25,
                'away_ewma_fta_rate': 0.24,
                'ewma_foul_synergy_home': 0,
                'ewma_foul_synergy_away': 0,
                'total_foul_environment': 0,
                'ewma_chaos_home': 0,
                'ewma_net_chaos': 0
            }
        
        # Shooting efficiency diff (eFG%)
        features['ewma_efg_diff'] = home_ewma.get('efg_pct', 0.5) - away_ewma.get('efg_pct', 0.5)
        
        # Turnover control diff
        features['ewma_tov_diff'] = away_ewma.get('tov_pct', 0.135) - home_ewma.get('tov_pct', 0.135)
        
        # Rebounding diff
        features['ewma_orb_diff'] = home_ewma.get('orb_pct', 0.25) - away_ewma.get('orb_pct', 0.25)
        
        # Pace diff
        features['ewma_pace_diff'] = home_ewma.get('pace', 100) - away_ewma.get('pace', 100)
        
        # 3-point volume diff
        features['ewma_vol_3p_diff'] = home_ewma.get('fg3a_per_100', 30) - away_ewma.get('fg3a_per_100', 30)
        
        # Key absolutes (baselines)
        features['home_ewma_3p_pct'] = home_ewma.get('fg3_pct', 0.35)
        features['away_ewma_3p_pct'] = away_ewma.get('fg3_pct', 0.35)
        features['away_ewma_tov_pct'] = away_ewma.get('tov_pct', 0.135)
        features['home_orb'] = home_ewma.get('orb_pct', 0.25)
        features['away_orb'] = away_ewma.get('orb_pct', 0.25)
        features['away_ewma_fta_rate'] = away_ewma.get('fta_rate', 0.24)
        
        # Foul synergy (placeholder - requires play-by-play data)
        features['ewma_foul_synergy_home'] = home_ewma.get('fta_rate', 0.24) * 100
        features['ewma_foul_synergy_away'] = away_ewma.get('fta_rate', 0.24) * 100
        features['total_foul_environment'] = features['ewma_foul_synergy_home'] + features['ewma_foul_synergy_away']
        
        # Chaos metrics (variance in performance)
        features['ewma_chaos_home'] = home_ewma.get('std_net_rating', 5.0)
        features['ewma_net_chaos'] = (home_ewma.get('std_net_rating', 5.0) + away_ewma.get('std_net_rating', 5.0)) / 2
        
        return features
    
    def _get_ewma_injury_impact(self, team_abb: str, as_of_date: str, span: int = 10) -> float:
        """
        Calculate EWMA of injury_impact for a team to detect "shock" (new injuries).
        Returns rolling average injury impact over last 10 games.
        """
        # Build cache key
        cache_key = f"{team_abb}_{as_of_date}_{span}"
        if cache_key in self.injury_history:
            return self.injury_history[cache_key]
        
        # Query historical injuries for this team
        try:
            conn = sqlite3.connect(self.db_path)
            inj_df = pd.read_sql_query(
                """
                SELECT game_date, player_name, team_abbreviation
                FROM historical_inactives
                WHERE team_abbreviation = ?
                  AND game_date < ?
                ORDER BY game_date DESC
                LIMIT 300
                """,
                conn,
                params=(team_abb, as_of_date)
            )
            conn.close()
        except Exception as e:
            self.logger.debug(f"EWMA injury query failed: {e}")
            return 0.0
        
        if inj_df.empty:
            return 0.0
        
        # Group by game_date and calculate daily injury impact
        daily_impacts = []
        for game_date in inj_df['game_date'].unique():
            day_injuries = inj_df[inj_df['game_date'] == game_date]
            
            # Calculate impact using same logic as _calculate_historical_injury_impact
            day_impact = 0.0
            for _, row in day_injuries.iterrows():
                player_name = self._normalize_player_name(row['player_name'])
                if self.player_stats_df is not None and not self.player_stats_df.empty and 'player_name' in self.player_stats_df.columns:
                    try:
                        player_match = self.player_stats_df[
                            self.player_stats_df['player_name'].str.lower() == player_name.lower()
                        ]
                        if not player_match.empty:
                            pie = player_match.iloc[0]['pie']
                            if pie >= 0.08:  # MIN_PIE_THRESHOLD
                                base_impact = pie * 20.0
                                gravity = self._calculate_dynamic_gravity_multiplier(pie, 0.0855, 0.0230)
                                day_impact += base_impact * gravity
                    except (KeyError, AttributeError) as e:
                        # Skip this player if lookup fails
                        continue
            
            daily_impacts.append(min(day_impact, 15.0))
        
        # Calculate EWMA
        if len(daily_impacts) == 0:
            ewma_value = 0.0
        else:
            series = pd.Series(daily_impacts[:span])  # Use last N games
            ewma_value = series.ewm(span=span, adjust=False).mean().iloc[-1] if len(series) > 0 else 0.0
        
        # Cache result
        self.injury_history[cache_key] = ewma_value
        return ewma_value
    
    def _get_ewma_stats(self, team_abb: str, as_of_date: str, span: int = 10) -> Dict:
        """Calculate exponentially weighted moving average stats for a team"""
        if self.game_advanced_stats_df.empty:
            return {}
        
        # Convert as_of_date
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)
        elif hasattr(as_of_date, 'year'):
            as_of_date = pd.to_datetime(as_of_date)
        
        # Get team's games before as_of_date
        team_games = self.game_advanced_stats_df[
            (self.game_advanced_stats_df['team_abb'] == team_abb) &
            (self.game_advanced_stats_df['game_date'] < as_of_date)
        ].sort_values('game_date', ascending=True)  # Oldest first for EWMA
        
        if len(team_games) < 3:
            return {}
        
        # Take last 20 games for EWMA calculation
        recent = team_games.tail(20)
        
        # Calculate EWMA (span=10 means alpha Γëê 0.18, giving ~50% weight to last 3.8 games)
        ewma_stats = {
            'efg_pct': recent['efg_pct'].ewm(span=span).mean().iloc[-1],
            'tov_pct': recent['tov_pct'].ewm(span=span).mean().iloc[-1],
            'orb_pct': recent['orb_pct'].ewm(span=span).mean().iloc[-1],
            'pace': recent['pace'].ewm(span=span).mean().iloc[-1],
            'fg3a_per_100': recent['fg3a_per_100'].ewm(span=span).mean().iloc[-1] if 'fg3a_per_100' in recent.columns else 30,
            'fg3_pct': recent['fg3_pct'].ewm(span=span).mean().iloc[-1] if 'fg3_pct' in recent.columns else 0.35,
            'fta_rate': recent['fta_rate'].ewm(span=span).mean().iloc[-1],
            'net_rating': recent['net_rating'].ewm(span=span).mean().iloc[-1],
            'std_net_rating': recent['net_rating'].std()  # Chaos metric
        }
        
        return ewma_stats
    
    def _calc_rest_fatigue_features(self, home_team: str, away_team: str, current_date) -> Dict:
        """Calculate rest and fatigue features"""
        features = {}
        
        # Get rest days for each team
        home_rest = self._get_rest_days(home_team, current_date)
        away_rest = self._get_rest_days(away_team, current_date)
        
        features['home_rest_days'] = home_rest
        features['away_rest_days'] = away_rest
        features['rest_advantage'] = home_rest - away_rest
        features['fatigue_mismatch'] = 1 if abs(home_rest - away_rest) >= 2 else 0
        
        # Back-to-back flags (STRICT: 0 days rest = played yesterday)
        features['home_back_to_back'] = 1 if home_rest == 0 else 0
        features['away_back_to_back'] = 1 if away_rest == 0 else 0
        
        # 3-in-4 nights (schedule compression)
        features['home_3in4'] = self._check_3in4(home_team, current_date)
        features['away_3in4'] = self._check_3in4(away_team, current_date)
        
        return features
    
    def _get_rest_days(self, team_abb: str, game_date) -> int:
        """Calculate days of rest since last game"""
        if self.game_logs_df.empty:
            return 2  # Default to 2 days rest
        
        # Convert game_date to datetime
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        elif not isinstance(game_date, pd.Timestamp):
            game_date = pd.to_datetime(game_date)
        
        # Find team's last game before this date
        team_games = self.game_logs_df[
            (self.game_logs_df['TEAM_ABBREVIATION'] == team_abb) &
            (pd.to_datetime(self.game_logs_df['GAME_DATE']) < game_date)
        ].sort_values('GAME_DATE', ascending=False)
        
        if len(team_games) == 0:
            return 3  # Season start default
        
        last_game_date = pd.to_datetime(team_games.iloc[0]['GAME_DATE'])
        rest_days = (game_date - last_game_date).days
        
        return max(0, rest_days - 1)  # Subtract 1 because game day doesn't count as rest
    
    def _check_3in4(self, team_abb: str, game_date) -> int:
        """
        Check if team is playing 3 games in 4 nights - AND this is NOT the first game.
        CRITICAL: Only flag games where fatigue is ACUTE (2nd or 3rd game in the stretch).
        
        The first game of a 3-in-4 stretch is fresh - don't flag it.
        The 2nd and 3rd games are where fatigue matters.
        """
        if self.game_logs_df.empty:
            return 0
        
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        elif not isinstance(game_date, pd.Timestamp):
            game_date = pd.to_datetime(game_date)
        
        # Get all team games up to and including current date
        team_games = self.game_logs_df[
            (self.game_logs_df['TEAM_ABBREVIATION'] == team_abb) &
            (pd.to_datetime(self.game_logs_df['GAME_DATE']) <= game_date)
        ].copy()
        
        if len(team_games) < 3:
            return 0  # Need history
        
        # Sort and deduplicate
        team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
        team_games = team_games.sort_values('GAME_DATE').drop_duplicates(subset=['GAME_DATE'])
        dates = team_games['GAME_DATE'].tolist()
        
        if game_date not in dates:
            return 0
        
        current_idx = dates.index(game_date)
        
        if current_idx < 2:
            return 0  # Need at least 2 previous games
        
        # Check if previous 2 games + today = 3 games in 4 days or less
        date_2_ago = dates[current_idx - 2]
        date_1_ago = dates[current_idx - 1]
        
        span = (game_date - date_2_ago).days
        
        # TRUE 3-in-4: Today is the 2nd or 3rd game in a <=3 day span
        # This means: span <= 3 AND we're not the first game (we already checked current_idx >= 2)
        if span <= 3:
            # Additional check: Make sure the previous game was recent (not weeks ago)
            # If game 1 was 3 days ago but game 2 was yesterday, this IS a 3-in-4
            # If game 1 was 3 days ago and game 2 was 2 days ago, this is also 3-in-4
            # But if all 3 games are spread out evenly, it's less fatiguing
            
            # Simplest rule: If span <= 3, flag it (the acute fatigue is real)
            return 1
        else:
            return 0
    
    def _calc_baseline_features(self, home_team: str, away_team: str, game_date: str) -> Dict:
        """Calculate baseline features including composite ELO"""
        features = {}
        
        # Composite ELO from OffDefEloSystem
        if hasattr(self, 'offdef_elo_system') and self.offdef_elo_system:
            try:
                # Determine season from game_date
                if isinstance(game_date, str):
                    game_dt = pd.to_datetime(game_date)
                else:
                    game_dt = game_date
                
                year = game_dt.year
                month = game_dt.month
                season = f"{year - 1}-{str(year)[-2:]}" if month < 10 else f"{year}-{str(year + 1)[-2:]}"
                
                # Get home team's ELO rating as of game_date
                home_team_elo = self.offdef_elo_system.get_latest(
                    home_team, 
                    season=season,
                    before_date=str(game_dt.date())
                )
                
                if home_team_elo:
                    features['home_composite_elo'] = home_team_elo.composite
                else:
                    features['home_composite_elo'] = 1500  # Fallback to baseline
            except Exception as e:
                self.logger.warning(f"Failed to get composite ELO: {e}")
                features['home_composite_elo'] = 1500
        else:
            features['home_composite_elo'] = 1500  # Fallback if ELO system not available
        
        return features
    
    def _calc_altitude_features(self, home_team: str, away_team: str) -> Dict:
        """Calculate altitude game indicator (Denver/Utah advantage)"""
        altitude_teams = ['DEN', 'UTA']
        
        is_altitude_game = 1 if home_team in altitude_teams and away_team not in altitude_teams else 0
        
        return {'altitude_game': is_altitude_game}

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
    def _normalize_player_name(self, name: str) -> str:
        """
        Normalize player name to handle different formats.
        
        CSV format: "Last, First" (e.g., "Antetokounmpo, Giannis")
        Stats format: "First Last" (e.g., "Giannis Antetokounmpo")
        
        Returns both formats for fuzzy matching.
        """
        name = name.strip()
        
        # If comma-separated, convert to space-separated
        if ',' in name:
            parts = name.split(',')
            if len(parts) == 2:
                last, first = parts[0].strip(), parts[1].strip()
                return f"{first} {last}"
        
        return name
    
    def _calculate_dynamic_gravity_multiplier(self, player_pie: float, 
                                               league_avg_pie: float = 0.0855, 
                                               league_std_pie: float = 0.0230) -> float:
        """
        Calculate impact multiplier based purely on statistical dominance (Z-score).
        NO HARDCODED PLAYER NAMES - fully mathematical and self-adjusting.
        
        This implements the "2-Stage Slope" Dynamic Gravity Model:
        - Stage 1 (Z Γëñ 1.0): Average/role players ΓåÆ 1.0x baseline
        - Stage 2 (1.0 < Z Γëñ 2.5): Star Zone ΓåÆ Aggressive ramp (guards/wings)
        - Stage 3 (Z > 2.5): MVP Zone ΓåÆ Continued slope (statistical giants)
        
        CALIBRATED CONSTANTS (from actual data - Dec 2024):
        - Mean PIE: 0.0855 (7,714 rotation players, PIE >= 0.05)
        - Std Dev: 0.0230
        
        Args:
            player_pie: Player's PIE value (0-0.30 range)
            league_avg_pie: League average PIE (calibrated: 0.0855)
            league_std_pie: Standard deviation of PIE (calibrated: 0.0230)
            
        Returns:
            Multiplier for injury impact (1.0x baseline ΓåÆ 5.25x+ for MVPs)
            
        Examples (with calibrated constants):
            - PIE 0.086 (Average/Z=0.0) ΓåÆ 1.00x
            - PIE 0.125 (Brunson/Z=1.72) ΓåÆ 1.96x Γ¡É Auto-discovered
            - PIE 0.140 (LeBron/Z=2.39) ΓåÆ 2.85x
            - PIE 0.171 (Luka/Z=3.70) ΓåÆ 4.80x 
            - PIE 0.184 (Giannis/Z=4.27) ΓåÆ 5.66x (capped at 4.5x)
        """
        # Protect against division by zero
        if league_std_pie == 0:
            return 1.0
        
        # Calculate Z-score (statistical dominance)
        z_score = (player_pie - league_avg_pie) / league_std_pie
        
        # STAGE 1: AVERAGE & ROLE PLAYERS (Z Γëñ 1.0)
        # PIE ~0.109 or less ΓåÆ Replacement-level impact
        if z_score <= 1.0:
            return 1.0
        
        # STAGE 2: THE STAR ZONE (1.0 < Z Γëñ 2.5)
        # Captures: Brunson (1.7), Haliburton (1.9), LeBron (2.4), Curry (2.4+)
        # Aggressive ramp to catch guards/wings who drive winning despite lower PIE
        # Slope: +1.33 per sigma ΓåÆ Z=2.5 reaches 3.0x multiplier
        elif z_score <= 2.5:
            return 1.0 + ((z_score - 1.0) * 1.33)
        
        # STAGE 3: THE MVP ZONE (Z > 2.5)
        # Captures: SGA (2.7), Luka (3.7), Embiid (4.0), Jokic (4.1), Giannis (4.3)
        # Continue from 3.0x base with gentler slope to prevent runaway values
        # Slope: +1.5 per sigma ΓåÆ Z=4.0 (Jokic) = 5.25x
        else:
            base_star_boost = 3.0  # Continuation from Stage 2 peak
            mvp_boost = (z_score - 2.5) * 1.5
            multiplier = base_star_boost + mvp_boost
            
            # Soft cap at 4.5x to prevent extreme outliers
            # (Jokic/Giannis in peak seasons could theoretically hit 5.5x+)
            return min(multiplier, 4.5)
    
    def _calculate_historical_injury_impact(self, home_team: str, away_team: str, game_date: str) -> Dict:
        """
        Compute injury impact for historical games using historical_inactives table.
        For backtesting - uses game_date to query injuries on that specific date.
        
        Uses DYNAMIC GRAVITY MODEL: PIE-weighted impact with Z-score multipliers.
        Elite players (3+ sigma) get exponential weight due to systemic importance.
        
        Args:
            home_team: Team abbreviation (e.g., 'LAL')
            away_team: Team abbreviation (e.g., 'BOS') 
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Dict with 'home_injury_impact' and 'away_injury_impact'
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query historical_inactives for injuries on this game date
                inj_df = pd.read_sql_query(
                    """
                    SELECT player_name, team_abbreviation, season, game_date
                    FROM historical_inactives
                    WHERE game_date = ?
                    """,
                    conn,
                    params=(game_date,)
                )
        except Exception as e:
            self.logger.debug(f"Historical injury query failed: {e}")
            return {'home_injury_impact': 0.0, 'away_injury_impact': 0.0}
        
        if inj_df.empty:
            return {'home_injury_impact': 0.0, 'away_injury_impact': 0.0}
        
        # NBA PIE distribution constants (CALIBRATED 2024-12-08)
        # Based on 7,714 rotation players (PIE >= 0.05)
        # Mean: 0.0855, Std: 0.0230
        LEAGUE_AVG_PIE = 0.0855
        LEAGUE_STD_PIE = 0.0230
        MIN_PIE_THRESHOLD = 0.08  # Only count rotation players (approx 8-9 man rotation)
        
        # Calculate PIE-weighted impact with dynamic gravity
        home_impact = 0.0
        away_impact = 0.0
        
        for _, inj_row in inj_df.iterrows():
            player_name = self._normalize_player_name(inj_row['player_name'])
            team = inj_row['team_abbreviation']
            
            # Look up PIE from player_stats
            if self.player_stats_df is not None and not self.player_stats_df.empty:
                player_match = self.player_stats_df[
                    self.player_stats_df['player_name'].str.lower() == player_name.lower()
                ]
                
                if not player_match.empty:
                    pie = player_match.iloc[0]['pie']
                    
                    # CRITICAL FIX: Skip bench warmers (only count rotation players)
                    if pie < MIN_PIE_THRESHOLD:
                        self.logger.debug(f"Skipping bench player {player_name} (PIE: {pie:.3f})")
                        continue
                    
                    # Calculate dynamic gravity multiplier based on Z-score
                    gravity_multiplier = self._calculate_dynamic_gravity_multiplier(
                        pie, LEAGUE_AVG_PIE, LEAGUE_STD_PIE
                    )
                    
                    # Base impact: PIE scaled to points (0.15 PIE = 3.0 points)
                    base_impact = pie * 20.0
                    
                    # Apply gravity multiplier
                    total_impact = base_impact * gravity_multiplier
                    
                    # Log superstar detection for verification
                    if gravity_multiplier > 2.0:
                        z_score = (pie - LEAGUE_AVG_PIE) / LEAGUE_STD_PIE
                        self.logger.debug(
                            f"≡ƒîƒ SUPERSTAR GRAVITY: {player_name} | "
                            f"PIE: {pie:.3f} | Z: {z_score:.2f}╧â | "
                            f"Mult: {gravity_multiplier:.2f}x | Impact: {total_impact:.2f}"
                        )
                    
                    # Add to appropriate team
                    if team == home_team:
                        home_impact += total_impact
                    elif team == away_team:
                        away_impact += total_impact
                else:
                    # No PIE data found - skip this player (don't default to 0.5)
                    self.logger.debug(f"No PIE data for {player_name}, skipping")
                    continue
            else:
                # No player_stats_df loaded - skip
                continue
        
        # Cap at 15.0 to prevent extreme outliers
        home_impact = min(home_impact, 15.0)
        away_impact = min(away_impact, 15.0)
        
        return {
            'home_injury_impact': home_impact,
            'away_injury_impact': away_impact
        }
    

# Backward compatibility alias (now points to v6 implementation)
FeatureCalculatorV5 = FeatureCalculatorV5

if __name__ == "__main__":
    # Test the feature calculator
    logging.basicConfig(level=logging.INFO)
    
    calc = FeatureCalculatorV5()
    
    # Print cache stats
    stats = calc.get_cache_stats()
    print("\n≡ƒôè Cache Statistics:")
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
    
    print("\n≡ƒöó Sample Features:")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Test prediction
    predictions = calc.calculate_weighted_score(features)
    print("\n≡ƒÄ» Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value:.2f}")
