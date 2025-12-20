"""
NBA BETTING SYSTEM - COMPREHENSIVE GUI DASHBOARD V2
World-class professional grade with all features integrated:
- Kalshi odds display with implied probabilities
- Comprehensive matchup breakdown (10 stats)
- Basketball-Reference injury integration
- White text/dark green highlights
- Show All Games checkbox
- Bankroll persistence (save/load to JSON)
- Game detail dialog with full stats
- Performance tab with paper trading
- Settings tab with bankroll management
- Auto-refresh capability
- Edge filtering
"""
import sys
import json
import traceback
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QTextEdit, QGroupBox,
    QHeaderView, QMessageBox, QProgressBar, QDialog, QGridLayout, QCheckBox,
    QLineEdit, QScrollArea, QDateEdit
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QFont, QColor
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# === SANITIZED ARCHITECTURE - All paths from config ===
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import (
    MONEYLINE_MODEL, TOTALS_MODEL, DB_PATH as CONFIG_DB_PATH,
    ISOTONIC_CALIBRATOR, PLATT_CALIBRATOR
)

# Import MDP Production Config (1.5% fav / 8.0% dog - 29.1% ROI)
try:
    from production_config_mdp import (
        MODEL_PATH as MDP_MODEL_PATH,
        ACTIVE_FEATURES,
        MIN_EDGE_FAVORITE,
        MIN_EDGE_UNDERDOG,
        FILTER_MIN_OFF_ELO,
        NBA_STD_DEV,
        KELLY_FRACTION_MULTIPLIER,
        MODEL_VERSION as MDP_VERSION,
        N_ESTIMATORS,
        BACKTEST_ROI as MDP_BACKTEST_ROI
    )
    FAVORITE_EDGE_THRESHOLD = MIN_EDGE_FAVORITE
    UNDERDOG_EDGE_THRESHOLD = MIN_EDGE_UNDERDOG
    STRATEGY_KELLY_FRACTION = KELLY_FRACTION_MULTIPLIER
    MODEL_VERSION = MDP_VERSION
    MDP_FEATURES = ACTIVE_FEATURES
    BACKTEST_ROI = MDP_BACKTEST_ROI
    MODEL_PATH = Path(MDP_MODEL_PATH) if isinstance(MDP_MODEL_PATH, str) else MDP_MODEL_PATH
    print(f"[OK] Loaded MDP {MDP_VERSION}: {FAVORITE_EDGE_THRESHOLD*100:.1f}% fav / {UNDERDOG_EDGE_THRESHOLD*100:.1f}% dog")
    print(f"[OK] Features: {len(MDP_FEATURES)}, NBA_STD_DEV: {NBA_STD_DEV}")
except ImportError as e:
    print(f"[ERROR] production_config_mdp not found: {e}")
    print("[WARNING] Falling back to legacy config")
    FAVORITE_EDGE_THRESHOLD = 0.015  # 1.5%
    UNDERDOG_EDGE_THRESHOLD = 0.080  # 8.0%
    STRATEGY_KELLY_FRACTION = 0.25
    MODEL_VERSION = "legacy"
    NBA_STD_DEV = 13.42
    FILTER_MIN_OFF_ELO = -90
    MDP_FEATURES = []  # Will need manual specification
    BACKTEST_ROI = 0.291  # Default to MDP ROI
    MODEL_PATH = MONEYLINE_MODEL

TOTAL_MODEL_PATH = TOTALS_MODEL
BANKROLL_SETTINGS_FILE = PROJECT_ROOT / "bankroll_settings.json"
DATABASE_PATH = CONFIG_DB_PATH
PREDICTIONS_CACHE_FILE = PROJECT_ROOT / "predictions_cache.json"
MIN_EDGE = 0.015  # MDP v2.2 minimum threshold (1.5% for favorites)
MAX_EDGE = 1.00
KELLY_FRACTION = STRATEGY_KELLY_FRACTION  # Use production Kelly (0.25)
MAX_BET_PCT = 0.05
BANKROLL = 10000.0

# Try importing optional dependencies
try:
    # Use regular feature calculator (v5 is production-ready)
    import sys
    sys.path.insert(0, 'src')
    sys.path.insert(0, 'src/features')
    sys.path.insert(0, 'src/services')
    
    from src.features.feature_calculator_v5 import FeatureCalculatorV5 as FeatureCalculator
    from injury_impact_live import calculate_team_injury_impact_simple
    
    # LiveInjuryUpdater is optional
    try:
        from src.services.live_injury_updater import LiveInjuryUpdater
    except ImportError:
        print("[WARNING] LiveInjuryUpdater not available, using basic injury data")
        LiveInjuryUpdater = None
    
    # Use nba_api for schedule (built-in, no ESPN needed)
    from nba_api.live.nba.endpoints import scoreboard
    from nba_api.stats.static import teams as nba_teams
    
    PREDICTION_ENGINE_AVAILABLE = True
    print("[OK] Loaded prediction engine with feature_calculator_v5")
except ImportError as e:
    PREDICTION_ENGINE_AVAILABLE = False
    print(f"[ERROR] Could not load prediction engine: {e}")
    print(f"[WARNING] Prediction engine not available")
    import traceback
    traceback.print_exc()
    FeatureCalculator = None
    LiveInjuryUpdater = None

try:
    from src.core.paper_trading_tracker import PaperTradingTracker
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False
    print("[WARNING] paper_trading_tracker not available")

# Import new enhanced systems
try:
    from src.core.bet_tracker import BetTracker
    BET_TRACKER_AVAILABLE = True
    print("[OK] BetTracker loaded")
except ImportError:
    BET_TRACKER_AVAILABLE = False
    BetTracker = None
    print("[WARNING] BetTracker not available")

try:
    from src.services.live_odds_fetcher import LiveOddsFetcher
    LIVE_ODDS_AVAILABLE = True
    print("[OK] LiveOddsFetcher loaded")
except ImportError as e:
    LIVE_ODDS_AVAILABLE = False
    LiveOddsFetcher = None
    print(f"[WARNING] LiveOddsFetcher not available: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.services.espn_schedule_service import ESPNScheduleService
    ESPN_SCHEDULE_AVAILABLE = True
    print("[OK] ESPNScheduleService loaded")
except ImportError as e:
    ESPN_SCHEDULE_AVAILABLE = False
    ESPNScheduleService = None
    print(f"[WARNING] ESPNScheduleService not available: {e}")

try:
    from src.core.daily_prediction_logger import DailyPredictionLogger
    DAILY_LOGGER_AVAILABLE = True
    print("[OK] DailyPredictionLogger loaded")
except ImportError:
    DAILY_LOGGER_AVAILABLE = False
    DailyPredictionLogger = None
    print("[WARNING] DailyPredictionLogger not available")


class NBAPredictionEngine:
    """Production prediction engine using MDP Regressor with 19 features"""
    
    def __init__(self):
        self.bankroll = BANKROLL
        self.predictions_cache = {}
        
        if PREDICTION_ENGINE_AVAILABLE:
            # Load MDP PRODUCTION model (19 features, 29.1% ROI)
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            
            # Load XGBoost REGRESSOR model (MDP uses .json Booster format)
            import xgboost as xgb
            from scipy.stats import norm
            self.norm = norm  # Store for probability conversion
            self.model = xgb.Booster()
            self.model.load_model(str(MODEL_PATH))
            self.nba_std_dev = NBA_STD_DEV  # 13.42 - model's empirical RMSE
            
            # Use MDP's 19 features (in correct order)
            if MDP_FEATURES:
                self.features = MDP_FEATURES
                print(f"[OK] Using MDP feature list: {len(self.features)} features")
            else:
                # Fallback to model's feature names if config not loaded
                try:
                    self.features = list(self.model.feature_names)
                except:
                    self.features = MDP_FEATURES  # Use hardcoded list
            
            print(f"[OK] Loaded MDP model: {MODEL_PATH.name}")
            print(f"[OK] Features: {len(self.features)} (should be 19)")
            print(f"[OK] Thresholds: {FAVORITE_EDGE_THRESHOLD*100:.1f}% FAV / {UNDERDOG_EDGE_THRESHOLD*100:.1f}% DOG")
            print(f"[OK] Expected ROI: 29.1% (2024-25 validation)")
            print(f"[OK] NBA_STD_DEV: {self.nba_std_dev} (empirical RMSE)")
            
            # Use consolidated database for all services
            db_path = str(DATABASE_PATH)
            self.feature_calculator = FeatureCalculator(db_path=db_path)
            if LiveInjuryUpdater is not None:
                self.injury_updater = LiveInjuryUpdater(db_path=db_path)
            else:
                self.injury_updater = None
            self.injury_service = None  # Not used - we have feature_calculator
            
            # Load total model if available (not used in MDP v2.2, but keep for compatibility)
            if TOTAL_MODEL_PATH.exists():
                self.total_model = joblib.load(TOTAL_MODEL_PATH)
                print(f"[OK] Loaded total model: {TOTAL_MODEL_PATH.name}")
            else:
                self.total_model = None
                print(f"[WARNING] Total model not found: {TOTAL_MODEL_PATH}")
            
            # Initialize enhanced systems
            if BET_TRACKER_AVAILABLE:
                self.bet_tracker = BetTracker(db_path=db_path)
                print("[OK] BetTracker initialized")
            else:
                self.bet_tracker = None
            
            if LIVE_ODDS_AVAILABLE:
                self.odds_fetcher = LiveOddsFetcher()
                print("[OK] LiveOddsFetcher initialized")
            else:
                self.odds_fetcher = None
            
            if ESPN_SCHEDULE_AVAILABLE:
                self.schedule_service = ESPNScheduleService(db_path=str(db_path))
                print("[OK] ESPNScheduleService initialized")
            else:
                self.schedule_service = None
            
            if DAILY_LOGGER_AVAILABLE:
                self.daily_logger = DailyPredictionLogger(db_path=db_path)
                print("[OK] DailyPredictionLogger initialized")
            else:
                self.daily_logger = None
        else:
            self.model = None
            self.total_model = None
            self.feature_calculator = None
            self.injury_service = None
        
        # Load cached predictions after everything else is initialized
        self.load_predictions_cache()
    
    def load_model(self):
        """Compatibility - model is part of predictor"""
        return None
    
    def save_bankroll(self):
        """Save bankroll to JSON file for persistence"""
        try:
            with open(BANKROLL_SETTINGS_FILE, 'w') as f:
                json.dump({'bankroll': self.bankroll, 'last_updated': datetime.now().isoformat()}, f, indent=2)
            print(f"[SUCCESS] Bankroll saved: ${self.bankroll:,.2f}")
            return True
        except Exception as e:
            print(f"[ERROR] Saving bankroll: {e}")
            return False
    
    def load_bankroll(self):
        """Load bankroll from JSON file"""
        try:
            if BANKROLL_SETTINGS_FILE.exists():
                with open(BANKROLL_SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    self.bankroll = settings.get('bankroll', BANKROLL)
                    print(f"[SUCCESS] Bankroll loaded: ${self.bankroll:,.2f}")
                    return True
            else:
                print(f"[INFO] No saved bankroll found, using default: ${BANKROLL:,.2f}")
                self.bankroll = BANKROLL
                return False
        except Exception as e:
            print(f"[ERROR] Loading bankroll: {e}")
            self.bankroll = BANKROLL
            return False
    
    def save_predictions_cache(self):
        """Save predictions to cache file"""
        try:
            with open(PREDICTIONS_CACHE_FILE, 'w') as f:
                json.dump(self.predictions_cache, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"[ERROR] Saving predictions cache: {e}")
            return False
    
    def load_predictions_cache(self):
        """Load predictions from cache file"""
        try:
            if PREDICTIONS_CACHE_FILE.exists():
                with open(PREDICTIONS_CACHE_FILE, 'r') as f:
                    self.predictions_cache = json.load(f)
                print(f"[OK] Loaded {len(self.predictions_cache)} cached predictions")
                return True
        except Exception as e:
            print(f"[ERROR] Loading predictions cache: {e}")
            self.predictions_cache = {}
        return False
    
    def is_valid_odds(self, home_odds: int, away_odds: int) -> bool:
        """
        Filter out corrupted/extreme odds (validated filter)
        Relaxed to allow bigger favorites and underdogs with value
        """
        return (-1000 <= home_odds <= 1000) and (-1000 <= away_odds <= 1000)
    
    def predict_game(self, home_team: str, away_team: str, game_date: str, game_time: str = "19:00",
                     home_ml_odds: Optional[int] = None, away_ml_odds: Optional[int] = None) -> Dict:
        """Make prediction using FeatureCalculatorV5 with live injuries - REQUIRES REAL ODDS"""
        try:
            if not self.model or not self.feature_calculator:
                return {'error': 'Model not available'}
            
            # Update live injuries from ESPN before prediction (if available)
            if self.injury_updater is not None:
                try:
                    injury_count = self.injury_updater.update_active_injuries()
                    print(f"[OK] Updated {injury_count} live injuries from ESPN")
                except Exception as inj_err:
                    print(f"[WARNING] Could not update live injuries: {inj_err}")
            else:
                print(f"[INFO] Using cached injury data (LiveInjuryUpdater not available)")
            
            # Get real market odds from LiveOddsFetcher (Kalshi API)
            kalshi_home_prob = None
            kalshi_away_prob = None
            yes_price = None
            no_price = None
            odds_source = 'None'
            has_real_odds = False
            
            # Try to get live odds from Kalshi
            # Attempt to initialize odds fetcher if not already done
            if not self.odds_fetcher and LIVE_ODDS_AVAILABLE:
                try:
                    from src.services.live_odds_fetcher import LiveOddsFetcher
                    self.odds_fetcher = LiveOddsFetcher()
                    print(f"[ODDS] LiveOddsFetcher initialized for {away_team} @ {home_team}")
                except Exception as init_err:
                    print(f"[WARNING] Could not initialize LiveOddsFetcher: {init_err}")
            
            if self.odds_fetcher and hasattr(self.odds_fetcher, 'kalshi_client') and self.odds_fetcher.kalshi_client:
                try:
                    print(f"[ODDS] Fetching live odds for {away_team} @ {home_team} on {game_date}...")
                    odds_data = self.odds_fetcher.get_moneyline_odds(home_team, away_team, game_date)
                    
                    # Check if odds_data is None (no real market data available)
                    if odds_data is None:
                        print(f"[ODDS] No real market data returned from fetcher")
                        has_real_odds = False
                    else:
                        home_ml_odds = odds_data.get('home_ml')
                        away_ml_odds = odds_data.get('away_ml')
                        yes_price = odds_data.get('yes_price')
                        no_price = odds_data.get('no_price')
                        odds_source = odds_data.get('source', 'unknown')
                        
                        # VALIDATED ODDS QUALITY CHECK
                        if home_ml_odds and away_ml_odds:
                            odds_valid = self.is_valid_odds(home_ml_odds, away_ml_odds)
                            has_real_odds = (odds_source == 'kalshi' and yes_price is not None)
                            
                            if not odds_valid:
                                print(f"[WARNING ODDS] Extreme odds filtered: {home_ml_odds}/{away_ml_odds}")
                                has_real_odds = False
                            
                            # Calculate fair probabilities from Kalshi prices
                            if yes_price and no_price:
                                kalshi_home_prob, kalshi_away_prob = self.odds_fetcher.remove_vig(home_ml_odds, away_ml_odds)
                            
                            print(f"[ODDS] {away_team} @ {home_team}: {odds_source} | {home_ml_odds}/{away_ml_odds}")
                        else:
                            print(f"[WARNING] No valid odds in returned data from {odds_source}")
                    
                except Exception as e:
                    print(f"[WARNING] LiveOddsFetcher error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if not self.odds_fetcher:
                    print(f"[INFO] LiveOddsFetcher not available (self.odds_fetcher is None)")
                elif not hasattr(self.odds_fetcher, 'kalshi_client'):
                    print(f"[INFO] LiveOddsFetcher has no kalshi_client attribute")
                elif not self.odds_fetcher.kalshi_client:
                    print(f"[INFO] LiveOddsFetcher kalshi_client is None")
            
            # CRITICAL: Block predictions without real market odds
            if not has_real_odds or home_ml_odds is None or away_ml_odds is None:
                return {
                    'error': 'NO_REAL_ODDS',
                    'message': f'Cannot generate prediction without live market odds. Odds source: {odds_source}',
                    'home_team': home_team,
                    'away_team': away_team,
                    'game_date': game_date,
                    'odds_source': odds_source
                }
            
            # Extract features using FeatureCalculatorV5
            # MDP expects 19 features (optimized feature set with VIF < 2.34)
            features_dict = self.feature_calculator.calculate_game_features(
                home_team=home_team,
                away_team=away_team,
                game_date=game_date if isinstance(game_date, str) else game_date.strftime('%Y-%m-%d')
            )
            
            # Convert to DataFrame for XGBoost
            X = pd.DataFrame([features_dict])
            
            # MDP uses 19 clean features
            # Ensure all 19 required features are present
            missing_features = [f for f in self.features if f not in X.columns]
            if missing_features:
                print(f"[WARNING] Missing {len(missing_features)} features: {missing_features[:5]}...")
                # Add missing features with default value 0
                for feat in missing_features:
                    X[feat] = 0.0
            
            # Filter to only the 19 features MDP expects (in correct order)
            X = X[self.features]
            
            # Apply physics filter (broken offense check)
            off_elo_diff = features_dict.get('off_elo_diff', 0)
            if off_elo_diff < FILTER_MIN_OFF_ELO:
                print(f"[FILTER] Off ELO diff {off_elo_diff:.1f} < {FILTER_MIN_OFF_ELO} (broken offense)")
                return {
                    'error': f'Physics filter: Broken offense (off_elo_diff={off_elo_diff:.1f})'
                }
            
            print(f"[INFO] Prepared {len(X.columns)} features for MDP: {X.columns.tolist()[:5]}...")
            
            # MDP REGRESSOR: Predict margin, then convert to probability
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X, feature_names=self.features)
            predicted_margin = float(self.model.predict(dmatrix)[0])
            
            # Convert margin to probability using Normal CDF with empirical RMSE
            home_prob = float(self.norm.cdf(predicted_margin / self.nba_std_dev))
            away_prob = 1 - home_prob
            
            print(f"[MDP] Predicted margin: {predicted_margin:.2f}, Home prob: {home_prob:.3f}")
            
            # Calculate edges using American odds
            home_ml_prob = self.odds_to_prob(home_ml_odds)
            away_ml_prob = self.odds_to_prob(away_ml_odds)
            
            home_edge = home_prob - home_ml_prob
            away_edge = away_prob - away_ml_prob
            
            print(f"[DEBUG] {away_team} @ {home_team}: model_home={home_prob:.4f}, market_home={home_ml_prob:.4f}, home_edge={home_edge:.4f}")
            print(f"[DEBUG]   model_away={away_prob:.4f}, market_away={away_ml_prob:.4f}, away_edge={away_edge:.4f}")
            
            # Apply MDP ASYMMETRIC THRESHOLD LOGIC
            # Favorites (negative odds): require 1.5% edge (low variance)
            # Underdogs (positive odds): require 8.0% edge (high variance)
            
            # Determine if home/away are favorites or underdogs based on odds sign
            home_is_favorite = home_ml_odds < 0
            away_is_favorite = away_ml_odds < 0
            
            # Apply appropriate threshold (asymmetric)
            home_threshold = FAVORITE_EDGE_THRESHOLD if home_is_favorite else UNDERDOG_EDGE_THRESHOLD
            away_threshold = FAVORITE_EDGE_THRESHOLD if away_is_favorite else UNDERDOG_EDGE_THRESHOLD
            
            home_qualifies = home_edge >= home_threshold
            away_qualifies = away_edge >= away_threshold
            
            # Calculate stakes (only for qualifying bets)
            home_stake = self.kelly_stake(home_edge, home_ml_odds, home_prob) if home_qualifies else 0
            away_stake = self.kelly_stake(away_edge, away_ml_odds, away_prob) if away_qualifies else 0
            
            # Build bets with classification
            all_bets = [
                {
                    'type': 'Moneyline',
                    'pick': home_team,
                    'edge': home_edge,
                    'model_prob': home_prob,
                    'market_prob': home_ml_prob,
                    'odds': home_ml_odds,
                    'stake': home_stake,
                    'bet_class': 'FAVORITE' if home_is_favorite else 'UNDERDOG',
                    'threshold': home_threshold,
                    'qualifies': home_qualifies
                },
                {
                    'type': 'Moneyline',
                    'pick': away_team,
                    'edge': away_edge,
                    'model_prob': away_prob,
                    'market_prob': away_ml_prob,
                    'odds': away_ml_odds,
                    'stake': away_stake,
                    'bet_class': 'FAVORITE' if away_is_favorite else 'UNDERDOG',
                    'threshold': away_threshold,
                    'qualifies': away_qualifies
                }
            ]
            
            all_bets.sort(key=lambda x: x['edge'], reverse=True)
            best_bet = all_bets[0] if all_bets[0]['qualifies'] else None
            
            # Get injured players from database (for display)
            home_injuries = []
            away_injuries = []
            try:
                import sqlite3
                conn = sqlite3.connect(str(DATABASE_PATH))
                cursor = conn.cursor()
                
                # Map team abbreviations to full names for injury lookup
                team_name_map = {
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
                
                home_full_name = team_name_map.get(home_team, home_team)
                away_full_name = team_name_map.get(away_team, away_team)
                
                # Fetch home team injuries using full team name
                cursor.execute("""
                    SELECT player_name, status, injury_desc, position 
                    FROM active_injuries 
                    WHERE team_name = ?
                """, (home_full_name,))
                home_rows = cursor.fetchall()
                for row in home_rows:
                    home_injuries.append({
                        'player': row[0],
                        'status': row[1],
                        'injury': row[2],
                        'position': row[3]
                    })
                print(f"[DEBUG] Home team ({home_team}/{home_full_name}): {len(home_rows)} injuries")
                
                # Fetch away team injuries using full team name
                cursor.execute("""
                    SELECT player_name, status, injury_desc, position 
                    FROM active_injuries 
                    WHERE team_name = ?
                """, (away_full_name,))
                away_rows = cursor.fetchall()
                for row in away_rows:
                    away_injuries.append({
                        'player': row[0],
                        'status': row[1],
                        'injury': row[2],
                        'position': row[3]
                    })
                print(f"[DEBUG] Away team ({away_team}/{away_full_name}): {len(away_rows)} injuries")
                
                conn.close()
                print(f"[OK] Found {len(home_injuries)} injuries for {home_team}, {len(away_injuries)} for {away_team}")
            except Exception as e:
                print(f"[WARNING] Could not get injured players from database: {e}")
                import traceback
                traceback.print_exc()
            
            # Calculate PIE-based injury impact (used for model features, matches training data)
            home_injury_impact = 0.0
            away_injury_impact = 0.0
            try:
                # Use the simplified PIE-based injury model (works with live database)
                home_injury_impact = calculate_team_injury_impact_simple(
                    home_full_name, 
                    game_date, 
                    str(DATABASE_PATH)
                )
                away_injury_impact = calculate_team_injury_impact_simple(
                    away_full_name, 
                    game_date, 
                    str(DATABASE_PATH)
                )
                print(f"[INJURY PIE] Home: {home_injury_impact:.2f}, Away: {away_injury_impact:.2f}")
            except Exception as e:
                print(f"[WARNING] PIE injury calculation failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Predict total if model available
            total_prediction = None
            if self.total_model is not None:
                try:
                    X_total = pd.DataFrame([features_dict])
                    if hasattr(self.total_model, 'feature_names_in_'):
                        # Add missing features with defaults (same as moneyline model)
                        missing_total_features = [f for f in self.total_model.feature_names_in_ if f not in X_total.columns]
                        if missing_total_features:
                            print(f"[INFO] Adding {len(missing_total_features)} missing features to total model")
                            for feat in missing_total_features:
                                X_total[feat] = 0  # Add with default value
                        
                        # Reorder to match model expectations
                        X_total = X_total[self.total_model.feature_names_in_]
                    total_prediction = self.total_model.predict(X_total)[0]
                except Exception as e:
                    print(f"[WARNING] Total model prediction failed: {e}")
            
            result = {
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date,
                'game_time': game_time,
                'home_win_prob': home_prob,
                'away_win_prob': away_prob,
                'all_bets': all_bets,
                'best_bet': best_bet,
                'features': features_dict,
                'kalshi_home_prob': kalshi_home_prob,
                'kalshi_away_prob': kalshi_away_prob,
                'kalshi_total_line': None,  # Not used in MDP v2.2
                'odds_source': odds_source,
                'has_real_odds': has_real_odds,
                'home_injuries': home_injuries,
                'away_injuries': away_injuries,
                'home_injury_impact': home_injury_impact,
                'away_injury_impact': away_injury_impact,
                'predicted_total': total_prediction,
                'yes_price': yes_price,
                'no_price': no_price
            }
            
            # Cache the prediction
            cache_key = f"{game_date}_{away_team}@{home_team}"
            self.predictions_cache[cache_key] = result
            self.save_predictions_cache()
            
            # Log bet to BetTracker if there's a qualifying bet
            if self.bet_tracker and best_bet and best_bet.get('qualifies', False):
                try:
                    picked_team = best_bet['pick']
                    is_home = (picked_team == home_team)
                    
                    # Get fair probability from vig removal
                    if kalshi_home_prob and kalshi_away_prob:
                        fair_prob = kalshi_home_prob if is_home else kalshi_away_prob
                    else:
                        fair_prob = best_bet['market_prob']
                    
                    # Log to Trial 1306 bet tracker
                    logged = self.bet_tracker.log_bet(
                        game_date=game_date,
                        home_team=home_team,
                        away_team=away_team,
                        bet_type='Moneyline',
                        predicted_winner=picked_team,
                        model_probability=best_bet['model_prob'],
                        fair_probability=fair_prob,
                        market_odds=best_bet['odds'],
                        edge=best_bet['edge'],
                        stake_amount=best_bet['stake'],
                        bankroll=self.bankroll,
                        kelly_fraction=0.25,  # Quarter Kelly
                        threshold_type=best_bet['bet_class'],
                        market_source=odds_source,
                        yes_price=yes_price,
                        no_price=no_price,
                        home_elo=features_dict.get('home_composite_elo'),
                        away_elo=features_dict.get('away_composite_elo'),
                        injury_advantage=features_dict.get('injury_matchup_advantage')
                    )
                    
                    if logged:
                        print(f"[BET TRACKED] {picked_team} {best_bet['bet_class']} @ {best_bet['odds']}")
                    
                except Exception as e:
                    print(f"[WARNING] Could not log bet to tracker: {e}")
            
            # Legacy paper trading tracker (keep for backward compatibility)
            if PAPER_TRADING_AVAILABLE and best_bet and best_bet.get('qualifies', False):
                try:
                    from paper_trading_tracker import PaperTradingTracker
                    tracker = PaperTradingTracker()
                    prediction_id = tracker.log_prediction(
                        game_date=game_date,
                        game_time=game_time,
                        home_team=home_team,
                        away_team=away_team,
                        prediction_result=result,
                        features=features_dict,
                        bankroll=10000.0,
                        model_version="Trial1306",
                        notes=f"Edge: {best_bet['edge']:.1%}, {best_bet['bet_class']}, Threshold: {best_bet['threshold']:.1%}"
                    )
                    print(f"[LOGGED] Legacy tracker #{prediction_id}")
                except Exception as e:
                    print(f"[WARNING] Legacy tracker failed: {e}")
            
            # Log ALL predictions to daily logger (bet or not) for model performance tracking
            if self.daily_logger:
                try:
                    self.daily_logger.log_prediction(
                        game_date=game_date,
                        home_team=home_team,
                        away_team=away_team,
                        model_home_prob=home_prob,
                        model_away_prob=away_prob,
                        home_odds=home_ml_odds,
                        away_odds=away_ml_odds,
                        odds_source=odds_source,
                        best_bet=best_bet,
                        features=features_dict
                    )
                except Exception as e:
                    print(f"[WARNING] Daily logger failed: {e}")
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': f'Prediction failed: {str(e)}'}
    
    def odds_to_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability (with vig)"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def kelly_stake(self, edge: float, odds: int, win_prob: float) -> float:
        """Calculate Kelly criterion stake (production formula)"""
        if edge <= 0:
            return 0
        
        # Convert to decimal odds
        decimal_odds = self.american_to_decimal(odds)
        b = decimal_odds - 1  # Net odds
        
        # Use actual model win probability (not approximation)
        p = win_prob
        q = 1 - p
        
        # Kelly formula: f* = (bp - q) / b
        full_kelly = (b * p - q) / b
        full_kelly = max(0, full_kelly)
        
        # Apply fractional Kelly and max bet limits
        stake = full_kelly * KELLY_FRACTION * self.bankroll
        stake = min(stake, self.bankroll * MAX_BET_PCT)
        
        return stake


class GameDetailDialog(QDialog):
    """Dialog showing comprehensive game breakdown with matchup stats"""
    
    def __init__(self, prediction: Dict, parent=None):
        super().__init__(parent)
        self.prediction = prediction
        self.init_ui()
    
    def init_ui(self):
        try:
            matchup = f"{self.prediction.get('away_team', 'Unknown')} @ {self.prediction.get('home_team', 'Unknown')}"
            self.setWindowTitle(f"Game Details - {matchup}")
            self.setGeometry(200, 100, 1000, 800)
            
            # Main layout
            main_layout = QVBoxLayout()
            
            # Create scroll area for all content
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            # Content widget inside scroll area
            content_widget = QWidget()
            layout = QVBoxLayout(content_widget)
            
            # Header
            header = QLabel(
                f"<h2 style='color: white; background-color: #2c3e50; padding: 10px;'>{matchup}</h2>"
                f"<p style='color: #ecf0f1; background-color: #34495e; padding: 5px;'>"
                f"{self.prediction.get('game_date', 'TBD')} at {self.prediction.get('game_time', 'TBD')}</p>"
            )
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(header)
            
            # Model Probabilities
            prob_group = QGroupBox("Model Predictions")
            prob_layout = QGridLayout()
            
            prob_layout.addWidget(QLabel("<b>Team</b>"), 0, 0)
            prob_layout.addWidget(QLabel("<b>Win Probability</b>"), 0, 1)
            
            home_prob = self.prediction.get('home_win_prob', 0.5)
            away_prob = self.prediction.get('away_win_prob', 0.5)
            
            prob_layout.addWidget(QLabel(f"<b>{self.prediction.get('home_team', 'Home')}</b>"), 1, 0)
            prob_layout.addWidget(QLabel(f"<span style='color: #27ae60;'>{home_prob:.1%}</span>"), 1, 1)
            
            prob_layout.addWidget(QLabel(f"<b>{self.prediction.get('away_team', 'Away')}</b>"), 2, 0)
            prob_layout.addWidget(QLabel(f"<span style='color: #e74c3c;'>{away_prob:.1%}</span>"), 2, 1)
            
            prob_group.setLayout(prob_layout)
            layout.addWidget(prob_group)
            
            # Market Odds (Kalshi)
            odds_group = QGroupBox("Market Odds (Kalshi)")
            odds_layout = QGridLayout()
            
            odds_layout.addWidget(QLabel("<b>Team</b>"), 0, 0)
            odds_layout.addWidget(QLabel("<b>Implied Probability</b>"), 0, 1)
            odds_layout.addWidget(QLabel("<b>Price</b>"), 0, 2)
            odds_layout.addWidget(QLabel("<b>Source</b>"), 0, 3)
            
            kalshi_home_prob = self.prediction.get('kalshi_home_prob')
            kalshi_away_prob = self.prediction.get('kalshi_away_prob')
            yes_price = self.prediction.get('yes_price')
            no_price = self.prediction.get('no_price')
            odds_source = self.prediction.get('odds_source', 'Unknown')
            
            # Home team odds
            odds_layout.addWidget(QLabel(f"<b>{self.prediction.get('home_team', 'Home')}</b>"), 1, 0)
            if kalshi_home_prob is not None:
                odds_layout.addWidget(QLabel(f"<span style='color: #3498db;'>{kalshi_home_prob:.1%}</span>"), 1, 1)
            else:
                odds_layout.addWidget(QLabel("<span style='color: gray;'>N/A</span>"), 1, 1)
            
            if yes_price is not None:
                odds_layout.addWidget(QLabel(f"{yes_price}¢"), 1, 2)
            else:
                odds_layout.addWidget(QLabel("<span style='color: gray;'>N/A</span>"), 1, 2)
            
            # Away team odds
            odds_layout.addWidget(QLabel(f"<b>{self.prediction.get('away_team', 'Away')}</b>"), 2, 0)
            if kalshi_away_prob is not None:
                odds_layout.addWidget(QLabel(f"<span style='color: #e74c3c;'>{kalshi_away_prob:.1%}</span>"), 2, 1)
            else:
                odds_layout.addWidget(QLabel("<span style='color: gray;'>N/A</span>"), 2, 1)
            
            if no_price is not None:
                odds_layout.addWidget(QLabel(f"{no_price}¢"), 2, 2)
            else:
                odds_layout.addWidget(QLabel("<span style='color: gray;'>N/A</span>"), 2, 2)
            
            # Source (spans both rows)
            source_label = QLabel(f"<b>{odds_source.upper()}</b>")
            if odds_source == 'kalshi':
                source_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            else:
                source_label.setStyleSheet("color: #95a5a6; font-weight: bold;")
            odds_layout.addWidget(source_label, 1, 3, 2, 1)
            
            odds_group.setLayout(odds_layout)
            layout.addWidget(odds_group)
            
            # All Available Bets
            bets_group = QGroupBox("Available Bets (Sorted by Edge)")
            bets_layout = QVBoxLayout()
            
            bets_table = QTableWidget()
            bets_table.setColumnCount(7)
            bets_table.setHorizontalHeaderLabels([
                'Bet Type', 'Pick', 'Edge', 'Model Prob', 'Market Prob', 'Odds', 'Stake'
            ])
            
            all_bets = self.prediction.get('all_bets', [])
            bets_table.setRowCount(len(all_bets))
            
            for row, bet in enumerate(all_bets):
                bets_table.setItem(row, 0, QTableWidgetItem(bet['type']))
                bets_table.setItem(row, 1, QTableWidgetItem(bet['pick']))
                
                # Enhanced 5-level color gradient for better distinguishability
                edge = bet['edge']
                edge_item = QTableWidgetItem(f"{edge:+.1%}")
                
                if edge >= 0.15:
                    edge_item.setBackground(QColor(34, 139, 34))
                    edge_item.setForeground(QColor(255, 255, 255))
                elif edge >= 0.10:
                    edge_item.setBackground(QColor(50, 205, 50))
                    edge_item.setForeground(QColor(0, 0, 0))
                elif edge >= 0.07:
                    edge_item.setBackground(QColor(255, 215, 0))
                    edge_item.setForeground(QColor(0, 0, 0))
                elif edge >= 0.05:
                    edge_item.setBackground(QColor(255, 140, 0))
                    edge_item.setForeground(QColor(0, 0, 0))
                elif edge >= 0.03:
                    edge_item.setBackground(QColor(255, 200, 150))
                    edge_item.setForeground(QColor(0, 0, 0))
                
                bets_table.setItem(row, 2, edge_item)
                bets_table.setItem(row, 3, QTableWidgetItem(f"{bet['model_prob']:.1%}"))
                bets_table.setItem(row, 4, QTableWidgetItem(f"{bet['market_prob']:.1%}"))
                
                # Format odds
                odds_val = bet['odds']
                if isinstance(odds_val, int):
                    odds_str = f"{odds_val:+d}"
                else:
                    odds_str = f"{int(odds_val):+d}" if odds_val == int(odds_val) else f"{odds_val:+.0f}"
                bets_table.setItem(row, 5, QTableWidgetItem(odds_str))
                bets_table.setItem(row, 6, QTableWidgetItem(f"${bet['stake']:.2f}"))
            
                bets_table.setItem(row, 6, QTableWidgetItem(f"${bet['stake']:.2f}"))
            
            bets_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            bets_layout.addWidget(bets_table)
            bets_group.setLayout(bets_layout)
            layout.addWidget(bets_group)
            
            # Matchup Breakdown - Comprehensive Stats
            breakdown_group = QGroupBox("🏀 Matchup Breakdown")
            breakdown_layout = QVBoxLayout()
            
            comparison_table = QTableWidget()
            comparison_table.setColumnCount(4)
            comparison_table.setHorizontalHeaderLabels(['Stat', self.prediction['home_team'], self.prediction['away_team'], 'Advantage'])
            
            features = self.prediction.get('features', {})
            stats_rows = []
            
            # Extract team names and features
            home_team = self.prediction['home_team']
            away_team = self.prediction['away_team']
            features = self.prediction.get('features', {})
            
            # Records - use features from prediction instead of database query
            home_win_pct = features.get('home_win_pct')
            away_win_pct = features.get('away_win_pct')
            
            # Estimate record from win percentage (82 games in season)
            if home_win_pct is not None and away_win_pct is not None:
                home_wins = int(home_win_pct * 82)
                home_losses = 82 - home_wins
                away_wins = int(away_win_pct * 82)
                away_losses = 82 - away_wins
            else:
                # Fallback: no record data available
                home_wins = away_wins = 0
                home_losses = away_losses = 0
            
            stats_rows.append((
                'Record', 
                (home_wins, home_losses),
                (away_wins, away_losses),
                'record',
                f"{home_wins}-{home_losses}", 
                f"{away_wins}-{away_losses}"
            ))
            
            # Last 10 (estimate from recent form if available, otherwise from win pct)
            if home_win_pct is not None and away_win_pct is not None:
                home_last_10_wins = int(home_win_pct * 10)
                away_last_10_wins = int(away_win_pct * 10)
            else:
                home_last_10_wins = away_last_10_wins = 0
            
            stats_rows.append((
                'Last 10', 
                f"{home_last_10_wins}-{10-home_last_10_wins}",
                f"{away_last_10_wins}-{10-away_last_10_wins}",
                'record',
                f"{home_last_10_wins}-{10-home_last_10_wins}", 
                f"{away_last_10_wins}-{10-away_last_10_wins}"
            ))
            
            # TEAM STATISTICS - Comprehensive breakdown
            stats_rows.append(('═══ OFFENSE ═══', None, None, 'header', '', ''))
            
            # Offensive Rating & PPG
            home_off_rating = features.get('home_off_rating')
            away_off_rating = features.get('away_off_rating')
            home_pace = features.get('home_pace')
            away_pace = features.get('away_pace')
            
            if home_off_rating and away_off_rating:
                stats_rows.append(('Offensive Rating', home_off_rating, away_off_rating, 'higher', f"{home_off_rating:.1f}", f"{away_off_rating:.1f}"))
                
                if home_pace and away_pace:
                    home_ppg = home_off_rating * home_pace / 100
                    away_ppg = away_off_rating * away_pace / 100
                    stats_rows.append(('Est. Points Per Game', home_ppg, away_ppg, 'higher', f"{home_ppg:.1f}", f"{away_ppg:.1f}"))
            
            # Advanced offensive metrics
            home_efg = features.get('home_efg_pct')
            away_efg = features.get('away_efg_pct')
            if home_efg is not None and away_efg is not None:
                stats_rows.append(('Effective FG%', home_efg, away_efg, 'higher', f"{home_efg:.1%}", f"{away_efg:.1%}"))
            
            home_3p_vol = features.get('home_three_point_volume')
            away_3p_vol = features.get('away_three_point_volume')
            if home_3p_vol is not None and away_3p_vol is not None:
                stats_rows.append(('3PT Attempt Rate', home_3p_vol, away_3p_vol, 'neutral', f"{home_3p_vol:.1%}", f"{away_3p_vol:.1%}"))
            
            # Turnovers
            home_tov = features.get('home_turnover_pct')
            away_tov = features.get('away_turnover_pct')
            if home_tov is not None and away_tov is not None:
                stats_rows.append(('Turnover %', home_tov, away_tov, 'lower', f"{home_tov:.1%}", f"{away_tov:.1%}"))
            
            stats_rows.append(('═══ DEFENSE ═══', None, None, 'header', '', ''))
            
            # Defensive Rating & Points Allowed
            home_def_rating = features.get('home_def_rating')
            away_def_rating = features.get('away_def_rating')
            if home_def_rating and away_def_rating:
                stats_rows.append(('Defensive Rating', home_def_rating, away_def_rating, 'lower', f"{home_def_rating:.1f}", f"{away_def_rating:.1f}"))
                
                if home_pace and away_pace:
                    home_papg = home_def_rating * home_pace / 100
                    away_papg = away_def_rating * away_pace / 100
                    stats_rows.append(('Est. Points Allowed', home_papg, away_papg, 'lower', f"{home_papg:.1f}", f"{away_papg:.1f}"))
            
            # Defensive EFG allowed
            home_def_efg = features.get('home_opp_efg_pct')
            away_def_efg = features.get('away_opp_efg_pct')
            if home_def_efg is not None and away_def_efg is not None:
                stats_rows.append(('Opp Effective FG%', home_def_efg, away_def_efg, 'lower', f"{home_def_efg:.1%}", f"{away_def_efg:.1%}"))
            
            stats_rows.append(('═══ TEMPO & ELO ═══', None, None, 'header', '', ''))
            
            # Pace
            if home_pace and away_pace:
                stats_rows.append(('Pace (Poss/Game)', home_pace, away_pace, 'neutral', f"{home_pace:.1f}", f"{away_pace:.1f}"))
            
            # Composite ELO
            home_elo = features.get('home_composite_elo')
            away_elo = features.get('away_composite_elo')
            if home_elo is not None and away_elo is not None:
                stats_rows.append(('Composite ELO', home_elo, away_elo, 'higher', f"{home_elo:.0f}", f"{away_elo:.0f}"))
            
            # Offensive/Defensive ELO
            home_off_elo = features.get('home_off_elo')
            away_off_elo = features.get('away_off_elo')
            if home_off_elo is not None and away_off_elo is not None:
                stats_rows.append(('Offensive ELO', home_off_elo, away_off_elo, 'higher', f"{home_off_elo:.0f}", f"{away_off_elo:.0f}"))
            
            home_def_elo = features.get('home_def_elo')
            away_def_elo = features.get('away_def_elo')
            if home_def_elo is not None and away_def_elo is not None:
                stats_rows.append(('Defensive ELO', home_def_elo, away_def_elo, 'higher', f"{home_def_elo:.0f}", f"{away_def_elo:.0f}"))
            
            stats_rows.append(('═══ CONTEXT ═══', None, None, 'header', '', ''))
            
            # Rest
            home_rest = features.get('home_rest_days')
            away_rest = features.get('away_rest_days')
            if home_rest is not None and away_rest is not None:
                stats_rows.append(('Rest Days', home_rest, away_rest, 'higher', f"{home_rest:.0f}", f"{away_rest:.0f}"))
            
            # Injuries (PIE-based impact score)
            home_injury_impact = self.prediction.get('home_injury_impact', 0.0)
            away_injury_impact = self.prediction.get('away_injury_impact', 0.0)
            home_injuries = self.prediction.get('home_injuries', [])
            away_injuries = self.prediction.get('away_injuries', [])
            
            stats_rows.append((
                'Injury Impact (PIE)', 
                home_injury_impact, 
                away_injury_impact, 
                'lower',
                f"{home_injury_impact:.1f} pts ({len(home_injuries)} out)",
                f"{away_injury_impact:.1f} pts ({len(away_injuries)} out)"
                ))
            
            # Populate table with highlighting
            comparison_table.setRowCount(len(stats_rows))
            for row, stat_data in enumerate(stats_rows):
                stat_name = stat_data[0]
                home_val_raw = stat_data[1]
                away_val_raw = stat_data[2]
                comparison_type = stat_data[3]
                home_val_str = stat_data[4]
                away_val_str = stat_data[5]
                
                # Check if this is a header row
                if comparison_type == 'header':
                    # Header row - spans all columns
                    stat_item = QTableWidgetItem(stat_name)
                    stat_item.setForeground(QColor(100, 200, 255))
                    stat_item.setBackground(QColor(30, 30, 30))
                    stat_item.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                    stat_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    comparison_table.setItem(row, 0, stat_item)
                    
                    # Empty cells for other columns
                    for col in [1, 2, 3]:
                        empty_item = QTableWidgetItem("")
                        empty_item.setBackground(QColor(30, 30, 30))
                        comparison_table.setItem(row, col, empty_item)
                    continue
                
                # Stat name column
                stat_item = QTableWidgetItem(stat_name)
                stat_item.setForeground(QColor(255, 255, 255))
                stat_item.setBackground(QColor(40, 40, 40))
                stat_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                comparison_table.setItem(row, 0, stat_item)
                
                # Determine leader
                if comparison_type == 'record':
                    if isinstance(home_val_raw, tuple):
                        home_wins = home_val_raw[0]
                        away_wins = away_val_raw[0]
                    else:
                        home_wins = int(home_val_raw.split('-')[0]) if '-' in str(home_val_raw) else 0
                        away_wins = int(away_val_raw.split('-')[0]) if '-' in str(away_val_raw) else 0
                    
                    home_better = home_wins > away_wins
                    away_better = away_wins > home_wins
                    diff = abs(home_wins - away_wins)
                    diff_str = f"+{diff} W" if diff > 0 else "Even"
                elif comparison_type == 'higher':
                    home_better = home_val_raw > away_val_raw
                    away_better = away_val_raw > home_val_raw
                    diff = abs(home_val_raw - away_val_raw)
                    diff_str = f"+{diff:.1f}"
                elif comparison_type == 'lower':
                    home_better = home_val_raw < away_val_raw
                    away_better = away_val_raw < home_val_raw
                    diff = abs(home_val_raw - away_val_raw)
                    diff_str = f"-{diff:.1f}"
                else:  # neutral
                    home_better = False
                    away_better = False
                    diff_str = "—"
                
                # Home value with highlighting
                home_item = QTableWidgetItem(home_val_str)
                home_item.setForeground(QColor(255, 255, 255))
                if home_better:
                    home_item.setBackground(QColor(0, 100, 0))
                    home_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                else:
                    home_item.setBackground(QColor(60, 60, 60))
                comparison_table.setItem(row, 1, home_item)
                
                # Away value with highlighting
                away_item = QTableWidgetItem(away_val_str)
                away_item.setForeground(QColor(255, 255, 255))
                if away_better:
                    away_item.setBackground(QColor(139, 0, 0))
                    away_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                else:
                    away_item.setBackground(QColor(60, 60, 60))
                comparison_table.setItem(row, 2, away_item)
                
                # Advantage/Difference column
                adv_item = QTableWidgetItem(diff_str)
                adv_item.setForeground(QColor(255, 255, 255))
                if home_better:
                    adv_item.setBackground(QColor(0, 80, 0))
                    adv_item.setToolTip(f"{self.prediction['home_team']} advantage")
                elif away_better:
                    adv_item.setBackground(QColor(100, 0, 0))
                    adv_item.setToolTip(f"{self.prediction['away_team']} advantage")
                else:
                    adv_item.setBackground(QColor(60, 60, 60))
                comparison_table.setItem(row, 3, adv_item)
            
            comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            comparison_table.verticalHeader().setVisible(False)
            comparison_table.setMinimumHeight(400)
            breakdown_layout.addWidget(comparison_table)
            
            breakdown_group.setLayout(breakdown_layout)
            layout.addWidget(breakdown_group)
            
            # Injured Players Section
            injuries_group = QGroupBox("🩹 Injury Report")
            injuries_layout = QHBoxLayout()
            
            # Home team injuries
            home_inj_widget = QWidget()
            home_inj_layout = QVBoxLayout()
            home_inj_label = QLabel(f"<b>{self.prediction['home_team']} Injuries</b>")
            home_inj_layout.addWidget(home_inj_label)
            
            home_injuries = self.prediction.get('home_injuries', [])
            if home_injuries:
                for inj in home_injuries:
                    player = inj.get('player') or inj.get('player_name', 'Unknown')
                    status = inj.get('status', 'Questionable')
                    injury = inj.get('injury') or inj.get('injury_desc', 'Unknown')
                    status_color = '#e74c3c' if status.lower() == 'out' else '#f39c12'
                    inj_text = QLabel(f"<span style='color:{status_color};'>●</span> {player} - {status} ({injury})")
                    home_inj_layout.addWidget(inj_text)
            else:
                home_inj_layout.addWidget(QLabel("<i>No injuries reported</i>"))
            
            home_inj_widget.setLayout(home_inj_layout)
            injuries_layout.addWidget(home_inj_widget)
            
            # Away team injuries
            away_inj_widget = QWidget()
            away_inj_layout = QVBoxLayout()
            away_inj_label = QLabel(f"<b>{self.prediction['away_team']} Injuries</b>")
            away_inj_layout.addWidget(away_inj_label)
            
            away_injuries = self.prediction.get('away_injuries', [])
            if away_injuries:
                for inj in away_injuries:
                    player = inj.get('player') or inj.get('player_name', 'Unknown')
                    status = inj.get('status', 'Questionable')
                    injury = inj.get('injury') or inj.get('injury_desc', 'Unknown')
                    status_color = '#e74c3c' if status.lower() == 'out' else '#f39c12'
                    inj_text = QLabel(f"<span style='color:{status_color};'>●</span> {player} - {status} ({injury})")
                    away_inj_layout.addWidget(inj_text)
            else:
                away_inj_layout.addWidget(QLabel("<i>No injuries reported</i>"))
            
            away_inj_widget.setLayout(away_inj_layout)
            injuries_layout.addWidget(away_inj_widget)
            
            injuries_group.setLayout(injuries_layout)
            layout.addWidget(injuries_group)
            
            # Previous Meetings Section
            previous_meetings_group = QGroupBox("📊 Previous Meetings (Last 10)")
            previous_meetings_layout = QVBoxLayout()
            
            try:
                import sqlite3
                conn = sqlite3.connect(str(DATABASE_PATH))
                cursor = conn.cursor()
                
                # Query for previous meetings
                cursor.execute("""
                    SELECT GAME_DATE, TEAM_ABBREVIATION, MATCHUP, PTS, 
                           CASE WHEN MATCHUP LIKE '%vs.%' THEN 'HOME' ELSE 'AWAY' END as location
                    FROM game_logs
                    WHERE (TEAM_ABBREVIATION = ? OR TEAM_ABBREVIATION = ?)
                      AND (MATCHUP LIKE ? OR MATCHUP LIKE ?)
                    ORDER BY GAME_DATE DESC
                    LIMIT 20
                """, (home_team, away_team, f'%{away_team}%', f'%{home_team}%'))
                
                raw_meetings = cursor.fetchall()
                conn.close()
                
                # Process into unique games
                meetings_dict = {}
                for game_date, team, matchup, pts, location in raw_meetings:
                    if game_date not in meetings_dict:
                        meetings_dict[game_date] = {'date': game_date, 'teams': {}, 'matchup': matchup}
                    meetings_dict[game_date]['teams'][team] = pts
                
                # Convert to list format for display
                meetings = []
                for date, data in sorted(meetings_dict.items(), reverse=True)[:10]:
                    teams = list(data['teams'].keys())
                    if len(teams) == 2:
                        # Determine home/away based on which team is in our matchup
                        if teams[0] == home_team:
                            h_team, a_team = teams[0], teams[1]
                        elif teams[1] == home_team:
                            h_team, a_team = teams[1], teams[0]
                        elif teams[0] == away_team:
                            h_team, a_team = teams[1], teams[0]
                        else:
                            h_team, a_team = teams[0], teams[1]
                        
                        h_score = data['teams'].get(h_team, 0)
                        a_score = data['teams'].get(a_team, 0)
                        meetings.append((date, h_team, a_team, h_score, a_score))
                
                if meetings:
                    meetings_table = QTableWidget()
                    meetings_table.setColumnCount(4)
                    meetings_table.setHorizontalHeaderLabels(['Date', 'Matchup', 'Score', 'Winner'])
                    meetings_table.setRowCount(len(meetings))
                    
                    for row, meeting in enumerate(meetings):
                        game_date, h_team, a_team, h_score, a_score = meeting
                        
                        date_str = game_date if isinstance(game_date, str) else game_date.strftime('%Y-%m-%d')
                        meetings_table.setItem(row, 0, QTableWidgetItem(date_str))
                        
                        matchup_str = f"{a_team} @ {h_team}"
                        meetings_table.setItem(row, 1, QTableWidgetItem(matchup_str))
                        
                        if h_score and a_score:
                            score_str = f"{int(h_score)}-{int(a_score)}"
                            meetings_table.setItem(row, 2, QTableWidgetItem(score_str))
                            
                            winner = h_team if h_score > a_score else a_team
                            winner_item = QTableWidgetItem(winner)
                            if winner == home_team:
                                winner_item.setBackground(QColor(0, 100, 0))
                            else:
                                winner_item.setBackground(QColor(139, 0, 0))
                            winner_item.setForeground(QColor(255, 255, 255))
                            meetings_table.setItem(row, 3, winner_item)
                        else:
                            meetings_table.setItem(row, 2, QTableWidgetItem("N/A"))
                            meetings_table.setItem(row, 3, QTableWidgetItem("N/A"))
                    
                    meetings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                    meetings_table.setMaximumHeight(300)
                    previous_meetings_layout.addWidget(meetings_table)
                else:
                    previous_meetings_layout.addWidget(QLabel("<i>No previous meetings found</i>"))
            
            except Exception as e:
                print(f"[WARNING] Could not fetch previous meetings: {e}")
                previous_meetings_layout.addWidget(QLabel(f"<i>Error loading meetings: {e}</i>"))
            
            previous_meetings_group.setLayout(previous_meetings_layout)
            layout.addWidget(previous_meetings_group)
            
            # Predicted Total with Kalshi line comparison
            if self.prediction.get('predicted_total'):
                total_group = QGroupBox("🎯 Total Points Prediction")
                total_layout = QVBoxLayout()
                
                model_total = self.prediction['predicted_total']
                total_label = QLabel(f"<h3 style='color: #3498db;'>Model Prediction: {model_total:.1f} points</h3>")
                total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                total_layout.addWidget(total_label)
                
                # Show Kalshi totals line if available
                kalshi_total = self.prediction.get('kalshi_total_line')
                if kalshi_total:
                    kalshi_label = QLabel(f"<h3 style='color: #e67e22;'>Kalshi Line: {kalshi_total:.1f} points</h3>")
                    kalshi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    total_layout.addWidget(kalshi_label)
                    
                    # Show difference
                    diff = model_total - kalshi_total
                    diff_color = '#27ae60' if abs(diff) > 3 else '#95a5a6'
                    diff_label = QLabel(f"<h4 style='color: {diff_color};'>Difference: {diff:+.1f} points</h4>")
                    diff_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    total_layout.addWidget(diff_label)
                
                total_group.setLayout(total_layout)
                layout.addWidget(total_group)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.accept)
            layout.addWidget(close_btn)
            
            # Set the content widget to the scroll area
            scroll.setWidget(content_widget)
            
            # Add scroll area to main layout
            main_layout.addWidget(scroll)
            
            self.setLayout(main_layout)
        
        except Exception as e:
            print(f"[ERROR] Failed to initialize game details dialog: {e}")
            traceback.print_exc()
            # Create minimal fallback UI
            fallback_layout = QVBoxLayout()
            error_label = QLabel(f"<h3 style='color: red;'>Error Loading Game Details</h3><p>{str(e)}</p>")
            error_label.setWordWrap(True)
            fallback_layout.addWidget(error_label)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.accept)
            fallback_layout.addWidget(close_btn)
            self.setLayout(fallback_layout)


class PredictionsTab(QWidget):
    """Main predictions tab with all filters and Kalshi odds display"""
    
    def __init__(self, predictor: NBAPredictionEngine, main_window, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.main_window = main_window
        self.predictions = []
        self.current_predictions = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model Info Panel (Trial 1306)
        model_info = QGroupBox("📊 Trial 1306 Production Model")
        model_info_layout = QHBoxLayout()
        try:
            version_text = f"Version: {MODEL_VERSION} | Features: 22 | Thresholds: {FAVORITE_EDGE_THRESHOLD*100:.1f}% FAV / {UNDERDOG_EDGE_THRESHOLD*100:.1f}% DOG"
            roi_text = f"Expected ROI: {TRIAL1306_BACKTEST_ROI*100:.1f}% | Kelly: {KELLY_FRACTION*100:.0f}%"
        except:
            version_text = "Version: Unknown | Features: Unknown | Thresholds: Unknown"
            roi_text = "Expected ROI: Unknown | Kelly: Unknown"
        
        model_info_label = QLabel(f"{version_text} | {roi_text}")
        model_info_label.setStyleSheet("color: #2ecc71; font-weight: bold; padding: 5px;")
        model_info_layout.addWidget(model_info_label)
        model_info.setLayout(model_info_layout)
        layout.addWidget(model_info)
        
        # Controls
        controls = QHBoxLayout()
        
        # Date picker
        controls.addWidget(QLabel("Select Date:"))
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setDisplayFormat("yyyy-MM-dd")
        self.date_picker.setToolTip("Select specific date to view games")
        self.date_picker.dateChanged.connect(self.on_date_changed)
        controls.addWidget(self.date_picker)
        
        controls.addWidget(QLabel("Days to Show:"))
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 14)
        self.days_spin.setValue(1)  # Default to 1 day (today only)
        self.days_spin.setToolTip("Number of days to load starting from selected date")
        controls.addWidget(self.days_spin)
        
        controls.addWidget(QLabel("Display Filter (Min Edge %):"))
        self.edge_spin = QDoubleSpinBox()
        self.edge_spin.setRange(0, 100)
        self.edge_spin.setValue(0.0)  # Default to 0 to show all qualified bets
        self.edge_spin.setSingleStep(0.5)
        self.edge_spin.setToolTip("Filter table display only (Trial 1306 uses 2%/10% thresholds for bet qualification)")
        self.edge_spin.valueChanged.connect(self.update_table)
        controls.addWidget(self.edge_spin)
        
        self.refresh_btn = QPushButton("🔄 Refresh Predictions")
        self.refresh_btn.clicked.connect(self.refresh_predictions)
        controls.addWidget(self.refresh_btn)
        
        # Auto-refresh toggle
        self.auto_refresh_checkbox = QCheckBox("Auto-refresh every 5 min")
        self.auto_refresh_checkbox.stateChanged.connect(self.toggle_auto_refresh)
        controls.addWidget(self.auto_refresh_checkbox)
        
        # Show all games toggle
        self.show_all_checkbox = QCheckBox("Show All Games")
        self.show_all_checkbox.setChecked(True)  # Default to showing all games
        self.show_all_checkbox.setToolTip("Display all games, even those without qualifying edge")
        self.show_all_checkbox.stateChanged.connect(self.update_table)
        controls.addWidget(self.show_all_checkbox)
        
        # Kalshi-only filter
        self.kalshi_only_checkbox = QCheckBox("Kalshi Markets Only")
        self.kalshi_only_checkbox.setChecked(False)
        self.kalshi_only_checkbox.setToolTip("Only show games with real Kalshi market odds (filter out placeholder odds)")
        self.kalshi_only_checkbox.stateChanged.connect(self.update_table)
        controls.addWidget(self.kalshi_only_checkbox)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Predictions table
        self.table = QTableWidget()
        self.table.setColumnCount(13)  # Added Class column
        self.table.setHorizontalHeaderLabels([
            'Date', 'Time', 'Matchup', 'Best Bet', 'Type', 'Class', 'Edge', 'Prob', 'Stake', 'Odds', 'Action', 'Wager $', 'Log Bet'
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(self.show_game_details)
        
        # Fix selection colors for better readability
        self.table.setStyleSheet("""
        QTableWidget::item:selected {
            background-color: #2874A6;
            color: white;
        }
        QTableWidget::item:selected:active {
            background-color: #1F618D;
            color: white;
        }
        """)
        layout.addWidget(self.table)
        
        # Summary
        self.summary = QLabel("Click Refresh to load predictions")
        layout.addWidget(self.summary)
        
        self.setLayout(layout)
        
        # Auto-refresh timer
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.refresh_predictions)
        
        # Load cached predictions on startup for instant display
        QTimer.singleShot(100, self.load_cached_predictions_on_startup)
    
    def log_bet_from_button(self, pred: Dict):
        """Log bet from button click - finds row dynamically to handle sorting"""
        if not PAPER_TRADING_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "Paper trading tracker not available")
            return
        
        try:
            # Find which row the sender button is in
            sender = self.sender()
            if not sender:
                QMessageBox.warning(self, "Error", "Could not identify button")
                return
            
            button_row = None
            for row in range(self.table.rowCount()):
                if self.table.cellWidget(row, 11) == sender:
                    button_row = row
                    break
            
            if button_row is None:
                QMessageBox.warning(self, "Error", "Could not find button row")
                return
            
            # Get wager from input field in the same row
            wager_widget = self.table.cellWidget(button_row, 10)
            if not wager_widget:
                QMessageBox.warning(self, "Error", "Could not find wager input")
                return
            
            wager_text = wager_widget.text().strip().replace('$', '').replace(',', '')
            if not wager_text:
                QMessageBox.warning(self, "Invalid Input", "Please enter a wager amount")
                return
            
            wager = float(wager_text)
            
            # Get prediction details
            best_bet = pred.get('best_bet')
            if not best_bet:
                QMessageBox.warning(self, "No Bet", "This game has no recommended bet")
                return
            
            game_date = pred['game_date']
            home_team = pred['home_team']
            away_team = pred['away_team']
            prediction = best_bet['pick']
            pred_type = best_bet['type']
            
            # Get odds and probabilities
            odds = best_bet['odds']
            model_prob = best_bet['model_prob']
            if isinstance(model_prob, str):
                model_prob = float(model_prob.strip('%')) / 100 if '%' in model_prob else float(model_prob)
            
            # Calculate fair probability
            if odds > 0:
                implied_prob = 100.0 / (odds + 100.0)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100.0)
            
            edge = model_prob - implied_prob
            
            # Log to database
            from paper_trading_tracker import PaperTradingTracker
            tracker = PaperTradingTracker()
            tracker.log_prediction(
                game_date=game_date,
                home_team=home_team,
                away_team=away_team,
                prediction_type=pred_type,
                predicted_winner=prediction,
                model_probability=model_prob,
                fair_probability=implied_prob,
                odds=odds,
                edge=edge,
                stake=wager
            )
            
            QMessageBox.information(self, "Success", f"Logged bet: {prediction} for ${wager:.2f}")
            
            # Clear wager input
            wager_widget.clear()
        
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Please enter a valid number: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to log bet: {e}")
    
    def load_cached_predictions_on_startup(self):
        """Load cached predictions on startup for instant display"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            should_auto_refresh = False
            
            if PREDICTIONS_CACHE_FILE.exists():
                with open(PREDICTIONS_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                
                # Get timestamp from file modification time
                cache_time = datetime.fromtimestamp(PREDICTIONS_CACHE_FILE.stat().st_mtime)
                cache_date = cache_time.strftime('%Y-%m-%d')
                time_diff = datetime.now() - cache_time
                
                if time_diff.total_seconds() < 60:
                    time_str = f"{int(time_diff.total_seconds())}s ago"
                elif time_diff.total_seconds() < 3600:
                    time_str = f"{int(time_diff.total_seconds() / 60)}m ago"
                else:
                    time_str = f"{int(time_diff.total_seconds() / 3600)}h ago"
                
                # Convert cache to prediction list format
                # Cache can be dict (old format) or already a list
                self.predictions = []
                if isinstance(cache_data, dict):
                    # Dict format: {"game_key": prediction_dict}
                    for game_key, pred in cache_data.items():
                        if isinstance(pred, dict) and 'home_team' in pred:
                            self.predictions.append(pred)
                elif isinstance(cache_data, list):
                    # List format: [prediction_dict, prediction_dict, ...]
                    self.predictions = [p for p in cache_data if isinstance(p, dict) and 'home_team' in p]
                
                if len(self.predictions) > 0:
                    self.update_table()
                    
                    # Auto-refresh if cache is from a different day
                    if cache_date != today:
                        print(f"[STARTUP] Cache from {cache_date}, today is {today} - auto-refreshing")
                        self.status_label.setText(f"📦 Cache from {cache_date} - refreshing for {today}...")
                        QTimer.singleShot(500, self.refresh_predictions)
                    else:
                        self.status_label.setText(f"📦 Loaded {len(self.predictions)} cached predictions from {time_str} - Click Refresh to update")
                        print(f"[OK] Loaded {len(self.predictions)} predictions from cache ({time_str})")
                    return
            
            # No cache available, do initial refresh for today
            print(f"[STARTUP] No cache found - auto-refreshing for {today}")
            self.status_label.setText(f"Loading predictions for {today}...")
            QTimer.singleShot(500, self.refresh_predictions)
            
        except Exception as e:
            print(f"[WARNING] Could not load cached predictions: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Error loading cache - click Refresh to load predictions")
    
    def _check_game_result(self, home_team: str, away_team: str, game_date: str):
        """Check if game has been completed and fetch result from database"""
        try:
            import sqlite3
            conn = sqlite3.connect(str(DATABASE_PATH))
            cursor = conn.cursor()
            
            # Try game_results table first
            cursor.execute("""
                SELECT home_score, away_score
                FROM game_results
                WHERE game_date = ? AND home_team = ? AND away_team = ?
            """, (game_date, home_team, away_team))
            result = cursor.fetchone()
            
            if not result:
                # Try game_logs table (contains historical game data)
                cursor.execute("""
                    SELECT PTS, MATCHUP
                    FROM game_logs
                    WHERE GAME_DATE = ? AND TEAM_ABBREVIATION IN (?, ?)
                    ORDER BY TEAM_ABBREVIATION
                """, (game_date, home_team, away_team))
                rows = cursor.fetchall()
                if len(rows) == 2:
                    # Match teams to scores based on MATCHUP field
                    home_score = None
                    away_score = None
                    for pts, matchup in rows:
                        if 'vs.' in matchup:  # Home game
                            home_score = pts
                        else:  # Away game (@)
                            away_score = pts
                    
                    if home_score and away_score:
                        result = (home_score, away_score)
            
            conn.close()
            
            if result:
                home_score, away_score = result
                winner = home_team if home_score > away_score else away_team
                print(f"[FINAL] {away_team} @ {home_team}: {away_score}-{home_score} (Winner: {winner})")
                return {
                    'home_score': home_score,
                    'away_score': away_score,
                    'winner': winner
                }
            return None
        except Exception as e:
            print(f"[DEBUG] Could not check game result: {e}")
            return None
    
    # Removed _get_future_games_from_odds - now using ScheduleService with The Odds API
    
    def toggle_auto_refresh(self, state):
        """Toggle auto-refresh timer"""
        if state == Qt.CheckState.Checked.value:
            self.auto_refresh_timer.start(300000)  # 5 minutes
            self.status_label.setText("✅ Auto-refresh enabled (every 5 min)")
        else:
            self.auto_refresh_timer.stop()
            self.status_label.setText("")
    
    def on_date_changed(self):
        """Handle date picker change - auto refresh when date changes"""
        print(f"[DATE] Selected date changed to: {self.date_picker.date().toString('yyyy-MM-dd')}")
        # Optionally auto-refresh when date changes
        # self.refresh_predictions()
    
    def refresh_predictions(self):
        """Load predictions using ESPN schedule service (includes selected date + future days)"""
        self.status_label.setText("⏳ Loading predictions...")
        
        # CRITICAL: Clear predictions AND current_predictions to force fresh data
        self.predictions = []
        self.current_predictions = []
        
        # Get days_ahead from spinner (includes today as day 0)
        days_ahead = self.days_spin.value()
        
        try:
            if not PREDICTION_ENGINE_AVAILABLE:
                self.status_label.setText("❌ Prediction engine not available")
                return
            
            # Get games using ESPN schedule service
            from datetime import datetime, timedelta
            
            all_games = []
            
            # Get selected date from date picker
            selected_qdate = self.date_picker.date()
            start_date = datetime(selected_qdate.year(), selected_qdate.month(), selected_qdate.day())
            today = datetime.now().strftime('%Y-%m-%d')
            
            print(f"\n{'='*60}")
            print(f"[REFRESH] Starting game fetch - Current date: {today}")
            print(f"[REFRESH] Selected start date: {start_date.strftime('%Y-%m-%d')}")
            print(f"[REFRESH] Days to fetch: {days_ahead}")
            print(f"{'='*60}\n")
            
            # Fetch games using ESPN schedule service
            if ESPN_SCHEDULE_AVAILABLE and hasattr(self.predictor, 'schedule_service') and self.predictor.schedule_service:
                # Don't clear cache - let it use CSV schedule for fast lookups
                # self.predictor.schedule_service.clear_cache()
                
                for day_offset in range(days_ahead):
                    target_date = start_date + timedelta(days=day_offset)
                    target_date_str = target_date.strftime('%Y-%m-%d')
                    
                    day_label = "TODAY" if target_date_str == today else f"{target_date_str}"
                    print(f"[ESPN] Fetching games for {day_label}...")
                    
                    try:
                        games = self.predictor.schedule_service.fetch_games_for_date(target_date_str, save_to_db=True)
                        
                        for game in games:
                            print(f"[ESPN]   ✓ {game['away_team']} @ {game['home_team']} on {game['game_date']} at {game['game_time']}")
                            all_games.append({
                                'home_team': game['home_team'],
                                'away_team': game['away_team'],
                                'game_date': game['game_date'],
                                'game_time': game['game_time']
                            })
                        
                        print(f"[ESPN] Found {len(games)} games for {target_date_str}\n")
                    except Exception as api_err:
                        print(f"[WARNING] ESPN error for {target_date_str}: {api_err}")
                        import traceback
                        traceback.print_exc()
            else:
                print("[ERROR] ESPN schedule service not available")
                self.status_label.setText("❌ ESPN schedule service not available")
                return
            
            print(f"{'='*60}")
            print(f"[SCHEDULE] TOTAL GAMES FOUND: {len(all_games)}")
            
            # Show breakdown by date
            if all_games:
                from collections import defaultdict
                games_by_date = defaultdict(int)
                for game in all_games:
                    games_by_date[game['game_date']] += 1
                
                print(f"[SCHEDULE] Breakdown by date:")
                for date in sorted(games_by_date.keys()):
                    date_label = " (TODAY)" if date == today else ""
                    print(f"[SCHEDULE]   {date}{date_label}: {games_by_date[date]} games")
            print(f"{'='*60}\n")
            
            if not all_games:
                self.status_label.setText(f"No games found for next {days_ahead} day(s)")
                return
            
            # Get predictions for each game
            for game in all_games:
                try:
                    # Check if game is completed
                    game_result = self._check_game_result(game['home_team'], game['away_team'], game['game_date'])
                    
                    # Prediction will fetch live odds internally via LiveOddsFetcher
                    # No default odds - predictions require real market data
                    pred = self.predictor.predict_game(
                        home_team=game['home_team'],
                        away_team=game['away_team'],
                        game_date=game['game_date'],
                        game_time=game.get('game_time', 'TBD')
                        # home_ml_odds and away_ml_odds omitted - will be fetched by LiveOddsFetcher
                    )
                    if 'error' not in pred:
                        # Add game result if available
                        if game_result:
                            pred['is_final'] = True
                            pred['final_home_score'] = game_result['home_score']
                            pred['final_away_score'] = game_result['away_score']
                            pred['actual_winner'] = game_result['winner']
                        else:
                            pred['is_final'] = False
                        self.predictions.append(pred)
                    else:
                        print(f"[WARNING] Prediction error for {game['away_team']} @ {game['home_team']}: {pred.get('error')}")
                except Exception as game_error:
                    print(f"[ERROR] Failed to predict {game['away_team']} @ {game['home_team']}: {game_error}")
                    import traceback
                    traceback.print_exc()
            
            self.status_label.setText(f"✅ Loaded {len(self.predictions)} predictions")
            print(f"[DEBUG] Loaded {len(self.predictions)} predictions")
            for p in self.predictions:
                print(f"[DEBUG] Prediction: {p.get('away_team')} @ {p.get('home_team')}, best_bet: {p.get('best_bet')}")
            
            # Update display
            self.update_table()
            refresh_time = datetime.now().strftime('%I:%M %p')
            self.status_label.setText(f"✅ Loaded {len(self.predictions)} predictions at {refresh_time}")
            print(f"[OK] Refresh complete. Loaded {len(self.predictions)} predictions")
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            print(f"[ERROR] Loading predictions: {e}")
            import traceback
            traceback.print_exc()
    
    def update_table(self):
        """Update table with predictions and filters"""
        min_edge = self.edge_spin.value() / 100
        show_all = self.show_all_checkbox.isChecked()
        kalshi_only = self.kalshi_only_checkbox.isChecked()
        
        print(f"\n[TABLE UPDATE] show_all={show_all}, min_edge={min_edge:.1%}, kalshi_only={kalshi_only}")
        print(f"[TABLE UPDATE] Total predictions to filter: {len(self.predictions)}")
        
        # Filter predictions
        filtered = []
        for pred in self.predictions:
            best_bet = pred.get('best_bet')
            all_bets = pred.get('all_bets', [])
            
            game_label = f"{pred['away_team']} @ {pred['home_team']} ({pred.get('game_date', 'N/A')})"
            
            # Filter by Kalshi markets only if enabled
            if kalshi_only and not pred.get('has_real_odds', False):
                print(f"[TABLE UPDATE]   ✗ Filtered out {game_label} (no Kalshi odds)")
                continue
            
            if show_all:
                # Show ALL games regardless of edge or qualifying status
                filtered.append(pred)
                print(f"[TABLE UPDATE]   ✓ Show All: {game_label}")
            elif best_bet:
                # Handle edge as string or float
                edge = best_bet.get('edge', 0)
                if isinstance(edge, str):
                    try:
                        edge = float(edge.strip('%')) / 100 if '%' in edge else float(edge)
                    except:
                        edge = 0
                
                if edge >= min_edge:
                    filtered.append(pred)
                    print(f"[TABLE UPDATE]   ✓ Edge filter: {game_label} (edge={edge:.1%} >= {min_edge:.1%})")
                else:
                    print(f"[TABLE UPDATE]   ✗ Edge too low: {game_label} (edge={edge:.1%} < {min_edge:.1%})")
        
        # Sort by edge
        def sort_key(x):
            best_bet = x.get('best_bet')
            if best_bet:
                edge = best_bet.get('edge', -999)
                # Handle edge as string (from cache) or float
                if isinstance(edge, str):
                    try:
                        edge = float(edge.strip('%')) / 100 if '%' in edge else float(edge)
                    except:
                        edge = -999
                return edge
            # If no best_bet, use highest edge from all_bets
            all_bets = x.get('all_bets', [])
            if all_bets:
                edges = []
                for b in all_bets:
                    e = b.get('edge', -999)
                    if isinstance(e, str):
                        try:
                            e = float(e.strip('%')) / 100 if '%' in e else float(e)
                        except:
                            e = -999
                    edges.append(e)
                return max(edges, default=-999)
            return -999
        
        filtered.sort(key=sort_key, reverse=True)
        self.current_predictions = filtered
        
        self.table.setRowCount(len(filtered))
        print(f"[TABLE UPDATE] Final display: {len(filtered)} games in table")
        if filtered:
            dates = set([p.get('game_date', 'N/A') for p in filtered])
            print(f"[TABLE UPDATE] Dates shown: {sorted(dates)}")
        print()
        
        total_stake = 0
        bet_count = 0
        
        for row, pred in enumerate(filtered):
            best_bet = pred.get('best_bet')
            
            # Date
            date_item = QTableWidgetItem(pred['game_date'])
            date_item.setForeground(QColor(255, 255, 255))  # White text
            self.table.setItem(row, 0, date_item)
            
            # Time
            time_item = QTableWidgetItem(pred.get('game_time', ''))
            time_item.setForeground(QColor(255, 255, 255))
            self.table.setItem(row, 1, time_item)
            
            # Matchup
            matchup = f"{pred['away_team']} @ {pred['home_team']}"
            if pred.get('is_final'):
                matchup = f"⚫ FINAL: {matchup} ({pred['final_away_score']}-{pred['final_home_score']})"
            matchup_item = QTableWidgetItem(matchup)
            if pred.get('is_final'):
                matchup_item.setForeground(QColor(255, 255, 100))  # Yellow for completed
            else:
                matchup_item.setForeground(QColor(135, 206, 250))  # Light blue for upcoming
            matchup_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            matchup_item.setToolTip("Double-click for detailed breakdown")
            # Store actual prediction object in item data for reliable retrieval
            matchup_item.setData(Qt.ItemDataRole.UserRole, pred)
            self.table.setItem(row, 2, matchup_item)
            
            if best_bet:
                bet_count += 1
                
                # Best Bet
                bet_item = QTableWidgetItem(best_bet['pick'])
                bet_item.setForeground(QColor(255, 255, 255))
                bet_item.setBackground(QColor(0, 100, 0))  # Dark green
                bet_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                self.table.setItem(row, 3, bet_item)
                
                # Type
                type_item = QTableWidgetItem(best_bet['type'])
                type_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 4, type_item)
                
                # Class (FAVORITE / UNDERDOG)
                bet_class = best_bet.get('bet_class', 'UNKNOWN')
                class_item = QTableWidgetItem(bet_class)
                class_item.setForeground(QColor(255, 255, 255))
                if bet_class == 'FAVORITE':
                    class_item.setBackground(QColor(0, 51, 102))  # Navy blue for favorites
                else:
                    class_item.setBackground(QColor(128, 0, 128))  # Purple for underdogs
                class_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                class_item.setToolTip(f"Threshold: {best_bet.get('threshold', 0):.1%}")
                self.table.setItem(row, 5, class_item)
                
                # Edge with gradient color system
                edge_pct = best_bet['edge']
                
                # Handle edge as string (from cache) or float
                if isinstance(edge_pct, str):
                    try:
                        edge_pct = float(edge_pct.strip('%')) / 100 if '%' in edge_pct else float(edge_pct)
                    except:
                        edge_pct = 0.0
                
                edge_item = QTableWidgetItem(f"{edge_pct:+.1%}")
                edge_item.setForeground(QColor(255, 255, 255))
                
                # Gradient color system for easy opportunity identification
                if edge_pct >= 0.15:  # 15%+ = Exceptional (brightest green)
                    edge_item.setBackground(QColor(0, 255, 0))  # Bright green
                elif edge_pct >= 0.10:  # 10-15% = Excellent (light green)
                    edge_item.setBackground(QColor(50, 205, 50))  # Lime green
                elif edge_pct >= 0.08:  # 8-10% = Very Good (medium green)
                    edge_item.setBackground(QColor(34, 139, 34))  # Forest green
                elif edge_pct >= 0.05:  # 5-8% = Good (dark green)
                    edge_item.setBackground(QColor(0, 128, 0))  # Dark green
                else:  # 3-5% = Acceptable (darker green)
                    edge_item.setBackground(QColor(0, 100, 0))  # Very dark green
                
                self.table.setItem(row, 6, edge_item)  # Column 6 (was 5)
                
                # Prob
                model_prob = best_bet['model_prob']
                if isinstance(model_prob, str):
                    try:
                        model_prob = float(model_prob.strip('%')) / 100 if '%' in model_prob else float(model_prob)
                    except:
                        model_prob = 0.0
                prob_item = QTableWidgetItem(f"{model_prob:.1%}")
                prob_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 7, prob_item)  # Column 7 (was 6)
                
                # Stake
                stake = best_bet['stake']
                if isinstance(stake, str):
                    try:
                        stake = float(stake.replace('$', '').replace(',', ''))
                    except:
                        stake = 0.0
                stake_item = QTableWidgetItem(f"${stake:.2f}")
                stake_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 8, stake_item)  # Column 8 (was 7)
                
                # Add to total stake (use the cleaned stake value)
                total_stake += stake
                
                # Odds - display Kalshi price (cents) and American odds
                yes_price = pred.get('yes_price')
                no_price = pred.get('no_price')
                odds_source = pred.get('odds_source', 'default')
                
                # Show Kalshi price for the bet pick
                if best_bet['pick'] == pred['home_team']:
                    if yes_price is not None and odds_source == 'kalshi':
                        odds_str = f"{yes_price}\u00a2 ({best_bet['odds']:+d})"
                    else:
                        odds_str = f"{best_bet['odds']:+d}"
                elif best_bet['pick'] == pred['away_team']:
                    if no_price is not None and odds_source == 'kalshi':
                        odds_str = f"{no_price}\u00a2 ({best_bet['odds']:+d})"
                    else:
                        odds_str = f"{best_bet['odds']:+d}"
                else:
                    odds_str = f"{best_bet['odds']:+d}"
                
                odds_item = QTableWidgetItem(odds_str)
                odds_item.setForeground(QColor(255, 255, 255))
                # Highlight Kalshi odds in green
                if odds_source == 'kalshi' and (yes_price or no_price):
                    odds_item.setBackground(QColor(0, 100, 0))
                self.table.setItem(row, 9, odds_item)  # Column 9 (was 8)
                
                # Button
                place_btn = QPushButton("📝 Details")
                place_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
                place_btn.clicked.connect(lambda checked, p=pred: self.show_game_details_from_button(p))
                self.table.setCellWidget(row, 10, place_btn)  # Column 10 (was 9)
                
                # Wager input
                wager_input = QLineEdit()
                wager_input.setPlaceholderText("$")
                wager_input.setMaximumWidth(80)
                self.table.setCellWidget(row, 11, wager_input)  # Column 11 (was 10)
                
                # Log Bet button
                log_bet_btn = QPushButton("💾 Log")
                log_bet_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
                log_bet_btn.clicked.connect(lambda checked, p=pred: self.log_bet_from_button(p))
                self.table.setCellWidget(row, 12, log_bet_btn)  # Column 12 (was 11)
            else:
                # No positive edge bet - show game info without bet details
                # Best Bet column - show highest edge pick even if negative
                all_bets = pred.get('all_bets', [])
                if all_bets and len(all_bets) > 0:
                    highest_edge_bet = max(all_bets, key=lambda x: x['edge'])
                    bet_item = QTableWidgetItem(f"{highest_edge_bet['pick']} (no edge)")
                    bet_item.setForeground(QColor(200, 200, 200))  # Gray text
                    self.table.setItem(row, 3, bet_item)
                    
                    # Type
                    type_item = QTableWidgetItem(highest_edge_bet['type'])
                    type_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 4, type_item)
                    
                    # Class - show N/A
                    class_item = QTableWidgetItem("N/A")
                    class_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 5, class_item)
                    
                    # Edge (negative or low)
                    edge_item = QTableWidgetItem(f"{highest_edge_bet['edge']:+.1%}")
                    edge_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 6, edge_item)
                    
                    # Prob
                    prob_item = QTableWidgetItem(f"{highest_edge_bet['model_prob']:.1%}")
                    prob_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 7, prob_item)
                    
                    # Stake - N/A
                    stake_item = QTableWidgetItem("N/A")
                    stake_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 8, stake_item)
                    
                    # Odds - show Kalshi price if available
                    yes_price = pred.get('yes_price')
                    no_price = pred.get('no_price')
                    odds_source = pred.get('odds_source', 'default')
                    
                    if highest_edge_bet['pick'] == pred['home_team']:
                        if yes_price is not None and odds_source == 'kalshi':
                            odds_str = f"{yes_price}¢ ({highest_edge_bet['odds']:+d})"
                        else:
                            odds_str = f"{highest_edge_bet['odds']:+d}"
                    elif highest_edge_bet['pick'] == pred['away_team']:
                        if no_price is not None and odds_source == 'kalshi':
                            odds_str = f"{no_price}¢ ({highest_edge_bet['odds']:+d})"
                        else:
                            odds_str = f"{highest_edge_bet['odds']:+d}"
                    else:
                        odds_str = f"{highest_edge_bet['odds']:+d}"
                    
                    odds_item = QTableWidgetItem(odds_str)
                    odds_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 9, odds_item)
                
                # Button (column 10)
                view_btn = QPushButton("👁️ View")
                view_btn.setStyleSheet("background-color: #6c757d; color: white;")
                view_btn.clicked.connect(lambda checked, p=pred: self.show_game_details_from_button(p))
                self.table.setCellWidget(row, 10, view_btn)
                
                # Wager input (column 11)
                wager_input = QLineEdit()
                wager_input.setPlaceholderText("$")
                wager_input.setMaximumWidth(80)
                self.table.setCellWidget(row, 11, wager_input)
                
                # Log Bet button (column 12)
                log_bet_btn = QPushButton("💾 Log")
                log_bet_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
                log_bet_btn.clicked.connect(lambda checked, p=pred: self.log_bet_from_button(p))
                self.table.setCellWidget(row, 12, log_bet_btn)
        
        # Update summary
        bankroll_pct = (total_stake / self.predictor.bankroll) * 100 if self.predictor.bankroll > 0 else 0
        self.summary.setText(
            f"📊 Total Games: {len(filtered)} | "
            f"🎯 Recommended Bets: {bet_count} | "
            f"💰 Total Stake: ${total_stake:.2f} ({bankroll_pct:.1f}% of bankroll) | "
            f"⚙️ Strategy: {FAVORITE_EDGE_THRESHOLD*100:.1f}% FAV / {UNDERDOG_EDGE_THRESHOLD*100:.1f}% DOG | "
            f"💡 Double-click any game for full breakdown"
        )
    
    def log_bet_from_row(self, row: int, pred: Dict):
        """Log bet from table row using prediction data"""
        if not PAPER_TRADING_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "Paper trading tracker not available")
            return
        
        try:
            # Get wager from input field in row
            wager_widget = self.table.cellWidget(row, 10)
            if not wager_widget:
                QMessageBox.warning(self, "Error", "Could not find wager input")
                return
            
            wager_text = wager_widget.text().strip().replace('$', '').replace(',', '')
            if not wager_text:
                QMessageBox.warning(self, "Invalid Input", "Please enter a wager amount")
                return
            
            wager = float(wager_text)
            
            # Get prediction details
            best_bet = pred.get('best_bet')
            if not best_bet:
                QMessageBox.warning(self, "No Bet", "This game has no recommended bet")
                return
            
            game_date = pred['game_date']
            home_team = pred['home_team']
            away_team = pred['away_team']
            prediction = best_bet['pick']
            pred_type = best_bet['type']
            
            # Get odds and probabilities
            odds = best_bet['odds']
            model_prob = best_bet['model_prob']
            if isinstance(model_prob, str):
                model_prob = float(model_prob.strip('%')) / 100 if '%' in model_prob else float(model_prob)
            
            # Calculate fair probability
            if odds > 0:
                implied_prob = 100.0 / (odds + 100.0)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100.0)
            
            edge = model_prob - implied_prob
            
            # Log to database
            from paper_trading_tracker import PaperTradingTracker
            tracker = PaperTradingTracker()
            tracker.log_prediction(
                game_date=game_date,
                home_team=home_team,
                away_team=away_team,
                prediction_type=pred_type,
                predicted_winner=prediction,
                model_probability=model_prob,
                fair_probability=implied_prob,
                odds=odds,
                edge=edge,
                stake=wager
            )
            
            QMessageBox.information(self, "Success", f"Logged bet: {prediction} for ${wager:.2f}")
            
            # Clear wager input
            wager_widget.clear()
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Please enter a valid number: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to log bet: {e}")
    
    def log_bet_from_row(self, row: int, pred: Dict):
        """Log bet from table row using prediction data"""
        if not PAPER_TRADING_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "Paper trading tracker not available")
            return
        
        try:
            # Get wager from input field in row
            wager_widget = self.table.cellWidget(row, 10)
            if not wager_widget:
                QMessageBox.warning(self, "Error", "Could not find wager input")
                return
            
            wager_text = wager_widget.text().strip().replace('$', '').replace(',', '')
            if not wager_text:
                QMessageBox.warning(self, "Invalid Input", "Please enter a wager amount")
                return
            
            wager = float(wager_text)
            
            # Get prediction details
            best_bet = pred.get('best_bet')
            if not best_bet:
                QMessageBox.warning(self, "No Bet", "This game has no recommended bet")
                return
            
            game_date = pred['game_date']
            home_team = pred['home_team']
            away_team = pred['away_team']
            prediction = best_bet['pick']
            pred_type = best_bet['type']
            
            # Get odds and probabilities
            odds = best_bet['odds']
            model_prob = best_bet['model_prob']
            if isinstance(model_prob, str):
                model_prob = float(model_prob.strip('%')) / 100 if '%' in model_prob else float(model_prob)
            
            # Calculate fair probability
            if odds > 0:
                implied_prob = 100.0 / (odds + 100.0)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100.0)
            
            edge = model_prob - implied_prob
            
            # Log to database
            from paper_trading_tracker import PaperTradingTracker
            tracker = PaperTradingTracker()
            tracker.log_prediction(
                game_date=game_date,
                home_team=home_team,
                away_team=away_team,
                prediction_type=pred_type,
                predicted_winner=prediction,
                model_probability=model_prob,
                fair_probability=implied_prob,
                odds=odds,
                edge=edge,
                stake=wager
            )
            
            QMessageBox.information(self, "Success", f"Logged bet: {prediction} for ${wager:.2f}")
            
            # Clear wager input
            wager_widget.clear()
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Please enter a valid number: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to log bet: {e}")
    
    def show_game_details(self, row, col):
        """Show detailed game info on double-click"""
        try:
            # Get prediction object directly from item data
            item = self.table.item(row, 2)  # Matchup column
            if item:
                pred = item.data(Qt.ItemDataRole.UserRole)
                if pred and isinstance(pred, dict):
                    dialog = GameDetailDialog(pred, self)
                    dialog.exec()
                    return
            
            # Fallback: use row index if within bounds
            if row < len(self.current_predictions):
                pred = self.current_predictions[row]
                dialog = GameDetailDialog(pred, self)
                dialog.exec()
        except Exception as e:
            print(f"[ERROR] Failed to show game details: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Could not load game details: {e}")
    
    def show_game_details_from_button(self, pred):
        """Show game details from button click"""
        dialog = GameDetailDialog(pred, self)
        dialog.exec()


class PerformanceTab(QWidget):
    """Performance tab with bet log display and performance metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls = QHBoxLayout()
        
        self.load_btn = QPushButton("📈 Load Performance")
        self.load_btn.clicked.connect(self.load_performance)
        controls.addWidget(self.load_btn)
        
        self.update_btn = QPushButton("🔄 Update Outcomes")
        self.update_btn.clicked.connect(self.update_outcomes)
        controls.addWidget(self.update_btn)
        
        self.load_bet_log_btn = QPushButton("📋 Load Bet Log")
        self.load_bet_log_btn.clicked.connect(self.load_bet_log)
        controls.addWidget(self.load_bet_log_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Metrics
        metrics_group = QGroupBox("📊 Overall Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(200)
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Bet Log table
        bet_log_group = QGroupBox("📋 Complete Bet Log")
        bet_log_layout = QVBoxLayout()
        
        self.bet_log_table = QTableWidget()
        self.bet_log_table.setColumnCount(10)
        self.bet_log_table.setHorizontalHeaderLabels([
            'Date', 'Game', 'Prediction', 'Type', 'Stake', 'Odds', 'Model Prob', 'Edge', 'Result', 'P/L'
        ])
        self.bet_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bet_log_table.setSortingEnabled(True)
        bet_log_layout.addWidget(self.bet_log_table)
        
        bet_log_group.setLayout(bet_log_layout)
        layout.addWidget(bet_log_group)
        
        self.setLayout(layout)
    
    def load_performance(self):
        """Load paper trading performance"""
        if not PAPER_TRADING_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "Paper trading tracker not available")
            return
        
        try:
            self.tracker = PaperTradingTracker()
            report = self.tracker.generate_performance_report()
            
            metrics_text = f"""
Total Bets: {report['total_bets']}
Win Rate: {report['win_rate']:.1%}
ROI: {report['roi']:.1%}
Total Profit/Loss: ${report['total_profit']:.2f}
Average Stake: ${report['avg_stake']:.2f}

Calibration:
  Brier Score: {report.get('brier_score', 0):.4f}
  
Edge Buckets:
"""
            for bucket in report.get('edge_buckets', []):
                metrics_text += f"  {bucket['range']}: {bucket['count']} bets, ROI: {bucket['roi']:.1%}\n"
            
            self.metrics_text.setText(metrics_text)
            QMessageBox.information(self, "Success", f"Loaded {report['total_bets']} predictions")
            
        except Exception as e:
            QMessageBox.warning(self, "Info", f"No performance data yet: {e}")
    
    def load_bet_log(self):
        """Load and display complete bet log"""
        if not PAPER_TRADING_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "Paper trading tracker not available")
            return
        
        try:
            import sqlite3
            conn = sqlite3.connect(str(DATABASE_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT game_date, home_team, away_team, predicted_winner, prediction_type,
                       stake, odds, model_probability, edge, outcome, profit_loss
                FROM paper_predictions
                ORDER BY game_date DESC, timestamp DESC
            """)
            
            bets = cursor.fetchall()
            conn.close()
            
            self.bet_log_table.setRowCount(len(bets))
            
            for row, bet in enumerate(bets):
                date, home, away, prediction, pred_type, stake, odds, model_prob, edge, outcome, profit = bet
                
                # Date
                self.bet_log_table.setItem(row, 0, QTableWidgetItem(date))
                
                # Game
                game_str = f"{away} @ {home}"
                self.bet_log_table.setItem(row, 1, QTableWidgetItem(game_str))
                
                # Prediction
                self.bet_log_table.setItem(row, 2, QTableWidgetItem(prediction))
                
                # Type
                self.bet_log_table.setItem(row, 3, QTableWidgetItem(pred_type))
                
                # Stake
                stake_item = QTableWidgetItem(f"${stake:.2f}")
                self.bet_log_table.setItem(row, 4, stake_item)
                
                # Odds
                odds_str = f"{int(odds):+d}" if odds == int(odds) else f"{odds:+.0f}"
                self.bet_log_table.setItem(row, 5, QTableWidgetItem(odds_str))
                
                # Model Probability
                prob_item = QTableWidgetItem(f"{model_prob:.1%}" if model_prob else "N/A")
                self.bet_log_table.setItem(row, 6, prob_item)
                
                # Edge
                edge_item = QTableWidgetItem(f"{edge:+.1%}" if edge else "N/A")
                if edge and edge >= 0.10:
                    edge_item.setBackground(QColor(50, 205, 50))
                elif edge and edge >= 0.05:
                    edge_item.setBackground(QColor(255, 215, 0))
                self.bet_log_table.setItem(row, 7, edge_item)
                
                # Result
                result_item = QTableWidgetItem(outcome if outcome else "Pending")
                if outcome == "WIN":
                    result_item.setBackground(QColor(144, 238, 144))
                    result_item.setForeground(QColor(0, 100, 0))
                elif outcome == "LOSS":
                    result_item.setBackground(QColor(255, 200, 200))
                    result_item.setForeground(QColor(139, 0, 0))
                self.bet_log_table.setItem(row, 8, result_item)
                
                # Profit/Loss
                if profit is not None:
                    pl_item = QTableWidgetItem(f"${profit:+.2f}")
                    if profit > 0:
                        pl_item.setForeground(QColor(0, 128, 0))
                    elif profit < 0:
                        pl_item.setForeground(QColor(255, 0, 0))
                    self.bet_log_table.setItem(row, 9, pl_item)
                else:
                    self.bet_log_table.setItem(row, 9, QTableWidgetItem("-"))
            
            print(f"[BET LOG] Loaded {len(bets)} bets")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load bet log: {e}")
    
    def update_outcomes(self):
        """Update outcomes from API"""
        if not PAPER_TRADING_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "Paper trading tracker not available")
            return
        
        if not self.tracker:
            self.tracker = PaperTradingTracker()
        
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            self.tracker.update_outcomes_from_api(yesterday)
            QMessageBox.information(self, "Success", f"Updated outcomes for {yesterday}")
            self.load_performance()
            self.load_bet_log()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Update failed: {e}")


class SettingsTab(QWidget):
    """Settings tab with bankroll management"""
    
    def __init__(self, predictor: NBAPredictionEngine, main_window, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.main_window = main_window
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Bankroll Management
        bankroll_group = QGroupBox("💰 Bankroll Management")
        bankroll_layout = QVBoxLayout()
        
        # Current bankroll display
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("<b>Current Bankroll:</b>"))
        self.bankroll_display = QLabel(f"${self.predictor.bankroll:,.2f}")
        self.bankroll_display.setStyleSheet("font-size: 18px; color: green; font-weight: bold;")
        current_layout.addWidget(self.bankroll_display)
        current_layout.addStretch()
        bankroll_layout.addLayout(current_layout)
        
        # Edit bankroll
        edit_layout = QHBoxLayout()
        edit_layout.addWidget(QLabel("Update Bankroll: $"))
        
        self.bankroll_input = QDoubleSpinBox()
        self.bankroll_input.setRange(0, 1000000)
        self.bankroll_input.setValue(self.predictor.bankroll)
        self.bankroll_input.setDecimals(2)
        self.bankroll_input.setSingleStep(100)
        edit_layout.addWidget(self.bankroll_input)
        
        update_btn = QPushButton("💾 Update & Save")
        update_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        update_btn.clicked.connect(self.update_bankroll)
        edit_layout.addWidget(update_btn)
        
        edit_layout.addStretch()
        bankroll_layout.addLayout(edit_layout)
        
        # Last saved info
        self.last_saved_label = QLabel("Not yet saved")
        self.last_saved_label.setStyleSheet("font-style: italic; color: #666;")
        bankroll_layout.addWidget(self.last_saved_label)
        
        # Update last saved display
        self.update_last_saved_display()
        
        bankroll_group.setLayout(bankroll_layout)
        layout.addWidget(bankroll_group)
        
        # System info
        info_group = QGroupBox("ℹ️ System Information")
        info_layout = QVBoxLayout()
        
        if self.predictor.feature_calculator and self.predictor.model:
            num_features = len(self.predictor.features)
            injury_features = [f for f in self.predictor.features if 'injury' in f.lower() or 'shock' in f.lower() or 'star' in f.lower()]
            model_name = MODEL_PATH.stem
            info_text = f"""
<b>Model:</b> {model_name} - MDP v2.2 (PRODUCTION - Regression)<br>
<b>Features:</b> {num_features} Features (Including {len(injury_features)} injury/shock features)<br>
<b>Feature Calculator:</b> FeatureCalculatorV5 (PRODUCTION)<br>
<b>Validated Performance:</b> <font color='#90EE90'><b>29.1% ROI</b></font> (1.5%/8.0% thresholds)<br>
<b>Bet Thresholds:</b> {FAVORITE_EDGE_THRESHOLD*100:.1f}% Favorites / {UNDERDOG_EDGE_THRESHOLD*100:.1f}% Underdogs<br>
<b>Model RMSE:</b> 13.42 points | <b>MAE:</b> 11.06 points<br>
<b>Database:</b> {DATABASE_PATH.name}<br>
<b>Last Updated:</b> {datetime.now().strftime('%Y-%m-%d')}<br>
<br>
<b>Bankroll File:</b> {BANKROLL_SETTINGS_FILE.name}<br>
<b>Status:</b> ✓ MDP v2.2 Production Model Loaded (All Systems Active)
"""
        else:
            info_text = "<b>Status:</b> Predictor not loaded"
        
        info_label = QLabel(info_text)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Risk settings - Editable
        risk_group = QGroupBox("⚙️ Risk Management Settings")
        risk_layout = QVBoxLayout()
        
        # Kelly Criterion %
        kelly_layout = QHBoxLayout()
        kelly_layout.addWidget(QLabel("<b>Kelly Criterion %:</b>"))
        self.kelly_spin = QDoubleSpinBox()
        self.kelly_spin.setRange(0.0, 1.0)
        self.kelly_spin.setSingleStep(0.05)
        self.kelly_spin.setValue(KELLY_FRACTION)
        self.kelly_spin.setDecimals(2)
        self.kelly_spin.setSuffix("%")
        self.kelly_spin.valueChanged.connect(self.update_risk_settings)
        kelly_layout.addWidget(self.kelly_spin)
        kelly_layout.addWidget(QLabel("(0.25 = Quarter Kelly, 0.50 = Half Kelly)"))
        kelly_layout.addStretch()
        risk_layout.addLayout(kelly_layout)
        
        # MDP v2.2 Thresholds (Informational - Optimized via Grid Search)
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("<b>MDP v2.2 Bet Thresholds:</b>"))
        threshold_info = QLabel(f"<font color='#90EE90'>{FAVORITE_EDGE_THRESHOLD*100:.1f}% for Favorites | {UNDERDOG_EDGE_THRESHOLD*100:.1f}% for Underdogs</font>")
        threshold_info.setStyleSheet("font-weight: bold;")
        threshold_layout.addWidget(threshold_info)
        threshold_layout.addWidget(QLabel("<font color='gray'>(Optimized via grid search, not adjustable)</font>"))
        threshold_layout.addStretch()
        risk_layout.addLayout(threshold_layout)
        
        # Maximum Wager %
        max_wager_layout = QHBoxLayout()
        max_wager_layout.addWidget(QLabel("<b>Maximum Wager %:</b>"))
        self.max_wager_spin = QDoubleSpinBox()
        self.max_wager_spin.setRange(0.01, 0.20)
        self.max_wager_spin.setSingleStep(0.01)
        self.max_wager_spin.setValue(MAX_BET_PCT)
        self.max_wager_spin.setDecimals(2)
        self.max_wager_spin.setSuffix("%")
        self.max_wager_spin.valueChanged.connect(self.update_risk_settings)
        max_wager_layout.addWidget(self.max_wager_spin)
        max_wager_layout.addWidget(QLabel("(% of bankroll per bet)"))
        max_wager_layout.addStretch()
        risk_layout.addLayout(max_wager_layout)
        
        # Save button
        save_risk_btn = QPushButton("💾 Save Risk Settings")
        save_risk_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 8px;")
        save_risk_btn.clicked.connect(self.save_risk_settings)
        risk_layout.addWidget(save_risk_btn)
        
        # Current values display
        self.risk_display = QLabel()
        self.risk_display.setStyleSheet("color: #28a745; font-style: italic; padding: 5px;")
        self.update_risk_display()
        risk_layout.addWidget(self.risk_display)
        
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        # Warnings
        warning_group = QGroupBox("⚠️ Important Notes")
        warning_layout = QVBoxLayout()
        
        warning_text = QLabel(
            "• Bankroll is automatically saved to JSON file<br>"
            "• Changes persist across sessions<br>"
            "• Run paper trading for 2-3 weeks before real money<br>"
            "• Verify all predictions manually before betting<br>"
            "• Never bet more than you can afford to lose"
        )
        warning_text.setTextFormat(Qt.TextFormat.RichText)
        warning_text.setStyleSheet("color: #856404; background-color: #fff3cd; padding: 10px;")
        warning_layout.addWidget(warning_text)
        warning_group.setLayout(warning_layout)
        layout.addWidget(warning_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_last_saved_display(self):
        """Update the last saved timestamp display"""
        try:
            if BANKROLL_SETTINGS_FILE.exists():
                with open(BANKROLL_SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    last_updated = settings.get('last_updated', 'Unknown')
                    bankroll = settings.get('bankroll', 0)
                    self.last_saved_label.setText(f"Last saved: ${bankroll:,.2f} at {last_updated[:19]}")
            else:
                self.last_saved_label.setText("No saved bankroll file found")
        except Exception as e:
            self.last_saved_label.setText(f"Error reading save file: {e}")
    
    def update_bankroll(self):
        """Update bankroll and save to file"""
        new_bankroll = self.bankroll_input.value()
        old_bankroll = self.predictor.bankroll
        
        self.predictor.bankroll = new_bankroll
        
        # Save to file
        if self.predictor.save_bankroll():
            self.bankroll_display.setText(f"${new_bankroll:,.2f}")
            self.main_window.update_status_bar()
            self.update_last_saved_display()
            QMessageBox.information(
                self, 
                "Bankroll Updated & Saved", 
                f"Bankroll updated from ${old_bankroll:,.2f} to ${new_bankroll:,.2f}\n"
                f"Saved to: {BANKROLL_SETTINGS_FILE.name}"
            )
        else:
            QMessageBox.warning(self, "Save Failed", "Bankroll updated but failed to save to file")
    
    def update_risk_settings(self):
        """Update risk display when spinboxes change"""
        self.update_risk_display()
    
    def update_risk_display(self):
        """Update the current risk settings display"""
        kelly = self.kelly_spin.value() if hasattr(self, 'kelly_spin') else KELLY_FRACTION
        max_wager = self.max_wager_spin.value() if hasattr(self, 'max_wager_spin') else MAX_BET_PCT
        
        # MDP v2.2 uses optimized thresholds (1.5% fav / 8.0% dog)
        self.risk_display.setText(
            f"Current: Kelly {kelly:.0%} | Thresholds: {FAVORITE_EDGE_THRESHOLD*100:.1f}% FAV / {UNDERDOG_EDGE_THRESHOLD*100:.1f}% DOG | Max Wager {max_wager:.1%} of bankroll"
        )
    
    def save_risk_settings(self):
        """Save risk management settings to file"""
        global KELLY_FRACTION, MAX_BET_PCT
        
        new_kelly = self.kelly_spin.value()
        new_max_wager = self.max_wager_spin.value()
        
        # Update global constants (MDP thresholds are built-in and not saved)
        KELLY_FRACTION = new_kelly
        MAX_BET_PCT = new_max_wager
        
        # Save to bankroll settings file
        try:
            settings = {}
            if BANKROLL_SETTINGS_FILE.exists():
                with open(BANKROLL_SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
            
            settings['kelly_fraction'] = new_kelly
            settings['max_bet_pct'] = new_max_wager
            settings['last_updated'] = datetime.now().isoformat()
            
            with open(BANKROLL_SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.update_risk_display()
            QMessageBox.information(
                self, 
                "Settings Saved",
                f"Risk management settings updated:\n\n"
                f"Kelly Criterion: {new_kelly:.0%}\n"
                f"Maximum Wager: {new_max_wager:.1%} of bankroll\n\n"
                f"Note: MDP v2.2 thresholds ({FAVORITE_EDGE_THRESHOLD*100:.1f}%/{UNDERDOG_EDGE_THRESHOLD*100:.1f}%) are optimized"
                f"Settings will apply to new predictions."
            )
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save settings: {e}")
    
    def quick_adjust_removed(self, amount: float):
        """REMOVED - Quick bankroll adjustment with save"""
        # This method has been removed as quick adjust is no longer needed
        pass
        self.predictor.save_bankroll()
        self.main_window.update_status_bar()
        self.update_last_saved_display()
    
    def refresh_display(self):
        """Refresh all displays"""
        self.bankroll_display.setText(f"${self.predictor.bankroll:,.2f}")
        self.bankroll_input.setValue(self.predictor.bankroll)
        self.update_last_saved_display()


class MainWindow(QMainWindow):
    """Main application window with all features"""
    
    def __init__(self):
        super().__init__()
        self.predictor = NBAPredictionEngine()
        self.placed_bets = []
        
        # Load bankroll from file
        self.predictor.load_bankroll()
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Touch of Grey Model - Making Brandi Money! 💰")
        self.setGeometry(100, 100, 1600, 900)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout()
        
        # Header with Touch of Grey branding
        header = QLabel("🏀 TOUCH OF GREY MODEL - MAKING BRANDI MONEY! 💰")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("padding: 15px; background-color: #2c3e50; color: #FFD700;")  # Gold text
        layout.addWidget(header)
        
        # Tabs
        tabs = QTabWidget()
        self.predictions_tab = PredictionsTab(self.predictor, self)
        self.settings_tab = SettingsTab(self.predictor, self)
        
        # Add Unified Analytics Tab (replaces Trial 1306 + Legacy Performance)
        try:
            from src.dashboard.analytics_tab import AnalyticsTab
            self.analytics_tab = AnalyticsTab(self.predictor, self)
            tabs.addTab(self.predictions_tab, "📊 Predictions")
            tabs.addTab(self.analytics_tab, "📈 Analytics & Results")
            tabs.addTab(self.settings_tab, "⚙️ Settings")
            print("[OK] Unified Analytics tab loaded")
        except Exception as e:
            print(f"[WARNING] Could not load Analytics tab: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to legacy tabs
            self.performance_tab = PerformanceTab()
            tabs.addTab(self.predictions_tab, "📊 Predictions")
            tabs.addTab(self.performance_tab, "📈 Performance")
            tabs.addTab(self.settings_tab, "⚙️ Settings")
        
        layout.addWidget(tabs)
        central.setLayout(layout)
        
        # Status bar
        self.update_status_bar()
    
    def update_status_bar(self):
        """Update status bar with current info"""
        status_text = (
            f"💰 Bankroll: ${self.predictor.bankroll:,.2f} | "
            f"📊 Model: GamePredictor V2 | "
            f"✅ Status: Ready"
        )
        self.statusBar().showMessage(status_text)
    
    def closeEvent(self, event):
        """Save bankroll on close"""
        self.predictor.save_bankroll()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark palette for better contrast
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

