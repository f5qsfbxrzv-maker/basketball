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
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QTextEdit, QGroupBox,
    QHeaderView, QMessageBox, QProgressBar, QDialog, QGridLayout, QCheckBox,
    QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
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

# Use centralized config paths
MODEL_PATH = MONEYLINE_MODEL
TOTAL_MODEL_PATH = TOTALS_MODEL
BANKROLL_SETTINGS_FILE = PROJECT_ROOT / "bankroll_settings.json"
DATABASE_PATH = CONFIG_DB_PATH
PREDICTIONS_CACHE_FILE = PROJECT_ROOT / "predictions_cache.json"
MIN_EDGE = 0.03
MAX_EDGE = 1.00
KELLY_FRACTION = 0.50
MAX_BET_PCT = 0.05
BANKROLL = 10000.0

# Try importing optional dependencies
try:
    # Use FeatureCalculatorLive for LIVE PRODUCTION (keeps v5 intact for backtesting)
    import sys
    sys.path.insert(0, 'src')
    from features.feature_calculator_live import FeatureCalculatorV5 as FeatureCalculator
    from espn_schedule_service_live import ESPNScheduleService
    from injury_impact_live import calculate_team_injury_impact_simple
    from src.services.live_injury_updater import LiveInjuryUpdater
    PREDICTION_ENGINE_AVAILABLE = True
    print("[OK] Using FeatureCalculatorLive (LIVE PRODUCTION - v5 preserved for backtesting)")
except ImportError as e:
    PREDICTION_ENGINE_AVAILABLE = False
    print(f"[ERROR] Could not load FeatureCalculatorV5: {e}")
    print(f"[WARNING] Prediction engine not available: {e}")

try:
    from paper_trading_tracker import PaperTradingTracker
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False
    print("[WARNING] paper_trading_tracker not available")


class NBAPredictionEngine:
    """Production prediction engine using FeatureCalculatorV5 with all injury/shock features"""
    
    def __init__(self):
        self.bankroll = BANKROLL
        self.predictions_cache = {}
        
        if PREDICTION_ENGINE_AVAILABLE:
            # Load the PRODUCTION model (Dec 12, 2025 - 36.7% ROI, 70.8% win rate)
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            
            # Load XGBoost model from JSON
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(str(MODEL_PATH))
            self.features = list(self.model.feature_names)
            
            # Use consolidated database for all services
            db_path = str(DATABASE_PATH)
            self.feature_calculator = FeatureCalculator(db_path=db_path)
            self.injury_updater = LiveInjuryUpdater(db_path=db_path)
            self.espn_schedule = ESPNScheduleService(db_path=db_path)
            self.injury_service = None  # Not used - we have feature_calculator
            print(f"[OK] Loaded PRODUCTION model (Dec 12): {len(self.features)} features including {len([f for f in self.features if 'injury' in f.lower() or 'shock' in f.lower()])} injury/shock features")
            
            # Load total model if available
            if TOTAL_MODEL_PATH.exists():
                self.total_model = joblib.load(TOTAL_MODEL_PATH)
                print(f"[OK] Loaded total model: {TOTAL_MODEL_PATH.name}")
            else:
                self.total_model = None
                print(f"[WARNING] Total model not found: {TOTAL_MODEL_PATH}")
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
        Based on forensic audit: 27.4% of raw API data had corrupted odds
        """
        return (-500 <= home_odds <= 500) and (-500 <= away_odds <= 500)
    
    def predict_game(self, home_team: str, away_team: str, game_date: str, game_time: str = "19:00",
                     home_ml_odds: int = -110, away_ml_odds: int = -110) -> Dict:
        """Make prediction using FeatureCalculatorV5 with live injuries"""
        try:
            if not self.model or not self.feature_calculator:
                return {'error': 'Model not available'}
            
            # Update live injuries from ESPN before prediction
            try:
                injury_count = self.injury_updater.update_active_injuries()
                print(f"[OK] Updated {injury_count} live injuries from ESPN")
            except Exception as inj_err:
                print(f"[WARNING] Could not update live injuries: {inj_err}")
                import traceback
                traceback.print_exc()
            
            # Get real market odds from multi-source service
            try:
                from multi_source_odds_service import MultiSourceOddsService
                odds_svc = MultiSourceOddsService()
                game_dt = datetime.strptime(game_date, '%Y-%m-%d') if isinstance(game_date, str) else game_date
                odds_data = odds_svc.get_game_odds(home_team, away_team, game_dt)
                
                # Extract American odds and Kalshi probabilities
                home_ml_odds = odds_data.get('home_ml_odds', -110)
                away_ml_odds = odds_data.get('away_ml_odds', -110)
                
                # Default to -110 if None returned
                if home_ml_odds is None:
                    home_ml_odds = -110
                if away_ml_odds is None:
                    away_ml_odds = -110
                
                kalshi_home_prob = odds_data.get('kalshi_home_prob')
                kalshi_away_prob = odds_data.get('kalshi_away_prob')
                odds_source = odds_data.get('source', 'Unknown')
                has_real_odds = (odds_source == 'Kalshi' and kalshi_home_prob is not None)
                
                # VALIDATED ODDS QUALITY CHECK (from forensic audit)
                odds_valid = self.is_valid_odds(home_ml_odds, away_ml_odds)
                if not odds_valid:
                    print(f"[WARNING ODDS] Extreme odds filtered: {home_ml_odds}/{away_ml_odds}")
                    has_real_odds = False  # Mark as invalid for filtering
                
                print(f"[DEBUG ODDS] {away_team} @ {home_team}: source={odds_source}, valid={odds_valid}, has_real_odds={has_real_odds}")
                print(f"[DEBUG ODDS]   home_ml={home_ml_odds}, away_ml={away_ml_odds}")
                print(f"[DEBUG ODDS]   kalshi_home_prob={kalshi_home_prob}, kalshi_away_prob={kalshi_away_prob}")
            except Exception as e:
                print(f"[WARNING] Could not get odds: {e}")
                home_ml_odds = -110
                away_ml_odds = -110
                kalshi_home_prob = None
                kalshi_away_prob = None
                odds_source = 'Error'
                has_real_odds = False
            
            # Extract features using FeatureCalculatorV5
            features_dict = self.feature_calculator.calculate_game_features(
                home_team=home_team,
                away_team=away_team,
                game_date=game_date if isinstance(game_date, str) else game_date.strftime('%Y-%m-%d')
            )
            
            # Convert to DataFrame and DMatrix for XGBoost Booster
            import xgboost as xgb
            X = pd.DataFrame([features_dict])
            X = X[self.features]  # Use self.features not feature_names_in_
            dmatrix = xgb.DMatrix(X)
            
            # Get model probability (home team win probability)
            home_prob = float(self.model.predict(dmatrix)[0])
            away_prob = 1 - home_prob
            
            # Calculate edges using American odds
            home_ml_prob = self.odds_to_prob(home_ml_odds)
            away_ml_prob = self.odds_to_prob(away_ml_odds)
            
            home_edge = home_prob - home_ml_prob
            away_edge = away_prob - away_ml_prob
            
            print(f"[DEBUG] {away_team} @ {home_team}: model_home={home_prob:.4f}, market_home={home_ml_prob:.4f}, home_edge={home_edge:.4f}")
            print(f"[DEBUG]   model_away={away_prob:.4f}, market_away={away_ml_prob:.4f}, away_edge={away_edge:.4f}")
            
            # Calculate stakes
            home_stake = self.kelly_stake(home_edge, home_ml_odds)
            away_stake = self.kelly_stake(away_edge, away_ml_odds)
            
            # Build bets
            all_bets = [
                {
                    'type': 'Moneyline',
                    'pick': home_team,
                    'edge': home_edge,
                    'model_prob': home_prob,
                    'market_prob': home_ml_prob,
                    'odds': home_ml_odds,
                    'stake': home_stake
                },
                {
                    'type': 'Moneyline',
                    'pick': away_team,
                    'edge': away_edge,
                    'model_prob': away_prob,
                    'market_prob': away_ml_prob,
                    'odds': away_ml_odds,
                    'stake': away_stake
                }
            ]
            
            all_bets.sort(key=lambda x: x['edge'], reverse=True)
            best_bet = all_bets[0] if all_bets[0]['edge'] >= MIN_EDGE else None
            
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
                        # Filter to only features the total model expects
                        total_features = [f for f in self.total_model.feature_names_in_ if f in features_dict]
                        X_total = X_total[total_features]
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
                'kalshi_total_line': odds_data.get('total'),  # Add totals line from odds service
                'odds_source': odds_source,
                'has_real_odds': has_real_odds,
                'home_injuries': home_injuries,
                'away_injuries': away_injuries,
                'home_injury_impact': home_injury_impact,  # PIE-based impact score
                'away_injury_impact': away_injury_impact,  # PIE-based impact score
                'predicted_total': total_prediction
            }
            
            # Cache the prediction
            cache_key = f"{game_date}_{away_team}@{home_team}"
            self.predictions_cache[cache_key] = result
            self.save_predictions_cache()
            
            # Log to paper trading tracker if available and there's a best bet
            if PAPER_TRADING_AVAILABLE and best_bet and best_bet['edge'] >= MIN_EDGE:
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
                        bankroll=10000.0,  # Default bankroll
                        model_version="v6.0",
                        notes=f"Edge: {best_bet['edge']:.1%}, Bet: {best_bet['pick']}"
                    )
                    print(f"[LOGGED] Prediction #{prediction_id} logged: {best_bet['pick']} (edge: {best_bet['edge']:.1%})")
                except Exception as e:
                    print(f"[WARNING] Could not log prediction: {e}")
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': f'Prediction failed: {str(e)}'}
    
    def odds_to_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def kelly_stake(self, edge: float, odds: int) -> float:
        """Calculate Kelly criterion stake"""
        if edge <= 0:
            return 0
        
        if odds > 0:
            b = odds / 100
        else:
            b = 100 / abs(odds)
        
        p = edge + (1 - edge) / 2  # Approximate win prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, kelly_fraction)
        
        # Apply half-Kelly and max bet limits
        stake = kelly_fraction * KELLY_FRACTION * self.bankroll
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
            
            # Extract team names for database queries
            home_team = self.prediction['home_team']
            away_team = self.prediction['away_team']
            
            # Records - calculate from win percentage
            home_win_pct = features.get('home_win_pct', 0.5)
            away_win_pct = features.get('away_win_pct', 0.5)
            
            # Fetch actual records from database
            import sqlite3
            try:
                conn = sqlite3.connect(str(DATABASE_PATH))
                cursor = conn.cursor()
                
                # Get home team record
                cursor.execute("SELECT wins, losses, last_10 FROM team_records WHERE team = ?", (home_team,))
                home_record = cursor.fetchone()
                home_wins, home_losses, home_last_10 = home_record if home_record else (0, 0, '0-0')
                
                # Get away team record
                cursor.execute("SELECT wins, losses, last_10 FROM team_records WHERE team = ?", (away_team,))
                away_record = cursor.fetchone()
                away_wins, away_losses, away_last_10 = away_record if away_record else (0, 0, '0-0')
                
                conn.close()
            except Exception as e:
                print(f"[WARNING] Could not fetch team records: {e}")
                home_wins, home_losses, home_last_10 = 0, 0, '0-0'
                away_wins, away_losses, away_last_10 = 0, 0, '0-0'
            
            stats_rows.append((
                'Record', 
                (int(home_wins), int(home_losses)),
                (int(away_wins), int(away_losses)),
                'record',
                f"{int(home_wins)}-{int(home_losses)}", 
                f"{int(away_wins)}-{int(away_losses)}"
            ))
            
            stats_rows.append((
                'Last 10', 
                home_last_10,
                away_last_10,
                'record',
                home_last_10, 
                away_last_10
                ))
            
            # Points Per Game
            home_off_rating = features.get('home_off_rating') or 110
            away_off_rating = features.get('away_off_rating') or 110
            home_pace = features.get('home_pace') or 100
            away_pace = features.get('away_pace') or 100
            home_ppg = home_off_rating * home_pace / 100
            away_ppg = away_off_rating * away_pace / 100
            stats_rows.append(('Points Per Game', home_ppg, away_ppg, 'higher', f"{home_ppg:.1f}", f"{away_ppg:.1f}"))
            
            # Points Allowed
            home_def_rating = features.get('home_def_rating') or 110
            away_def_rating = features.get('away_def_rating') or 110
            home_papg = home_def_rating * home_pace / 100
            away_papg = away_def_rating * away_pace / 100
            stats_rows.append(('Points Allowed', home_papg, away_papg, 'lower', f"{home_papg:.1f}", f"{away_papg:.1f}"))
            
            # Offensive Rating
            stats_rows.append(('Offensive Rating', home_off_rating, away_off_rating, 'higher', f"{home_off_rating:.1f}", f"{away_off_rating:.1f}"))
            
            # Defensive Rating (lower is better)
            stats_rows.append(('Defensive Rating', home_def_rating, away_def_rating, 'lower', f"{home_def_rating:.1f}", f"{away_def_rating:.1f}"))
            
            # Pace
            stats_rows.append(('Pace (Poss/Game)', home_pace, away_pace, 'neutral', f"{home_pace:.1f}", f"{away_pace:.1f}"))
            
            # ELO
            home_elo = features.get('home_composite_elo') or 1500
            away_elo = features.get('away_composite_elo') or 1500
            stats_rows.append(('Composite ELO', home_elo, away_elo, 'higher', f"{home_elo:.0f}", f"{away_elo:.0f}"))
            
            # Rest
            home_rest = features.get('home_rest_days', 0)
            away_rest = features.get('away_rest_days', 0)
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
                f"{home_injury_impact:.1f} ({len(home_injuries)} out)",
                f"{away_injury_impact:.1f} ({len(away_injuries)} out)"
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
                        if 'vs.' in data['matchup']:
                            h_team, a_team = teams[0], teams[1] if teams[0] in data['matchup'].split('vs.')[0] else teams[1], teams[0]
                        else:
                            h_team, a_team = teams[1], teams[0]
                        
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
        
        # Controls
        controls = QHBoxLayout()
        
        controls.addWidget(QLabel("Days Ahead:"))
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 14)
        self.days_spin.setValue(1)
        controls.addWidget(self.days_spin)
        
        controls.addWidget(QLabel("Min Edge %:"))
        self.edge_spin = QDoubleSpinBox()
        self.edge_spin.setRange(0, 100)
        self.edge_spin.setValue(3.0)
        self.edge_spin.setSingleStep(0.5)
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
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            'Date', 'Time', 'Matchup', 'Best Bet', 'Type', 'Edge', 'Prob', 'Stake', 'Odds', 'Action', 'Wager $', 'Log Bet'
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
            if PREDICTIONS_CACHE_FILE.exists():
                with open(PREDICTIONS_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                
                # Get timestamp from file modification time
                cache_time = datetime.fromtimestamp(PREDICTIONS_CACHE_FILE.stat().st_mtime)
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
                    self.status_label.setText(f"📦 Loaded {len(self.predictions)} cached predictions from {time_str} - Click Refresh to update")
                    print(f"[OK] Loaded {len(self.predictions)} predictions from cache ({time_str})")
                    return
            
            # No cache available, do initial refresh
            self.status_label.setText("No cached predictions - click Refresh to load")
            
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
    
    def refresh_predictions(self):
        """Load predictions using ESPN API for full schedule support"""
        self.status_label.setText("⏳ Loading predictions...")
        self.predictions = []
        
        # Get days_ahead from spinner
        days_ahead = self.days_spin.value()
        
        try:
            if not PREDICTION_ENGINE_AVAILABLE:
                self.status_label.setText("❌ Prediction engine not available")
                return
            
            # Use ESPN schedule API for all dates (supports future games)
            all_games = []
            for offset in range(days_ahead):
                target_date = (datetime.now() + timedelta(days=offset)).strftime('%Y-%m-%d')
                games = self.predictor.espn_schedule.fetch_games_for_date(target_date)
                all_games.extend(games)
            
            if not all_games:
                self.status_label.setText(f"No games found in next {days_ahead} day(s)")
                return
            
            print(f"[ESPN SCHEDULE] Found {len(all_games)} games in next {days_ahead} day(s)")
            
            # Get predictions for each game
            for game in all_games:
                try:
                    # Check if game is completed
                    game_result = self._check_game_result(game['home_team'], game['away_team'], game['game_date'])
                    
                    pred = self.predictor.predict_game(
                        home_team=game['home_team'],
                        away_team=game['away_team'],
                        game_date=game['game_date'],
                        game_time=game.get('game_time', 'TBD'),
                        home_ml_odds=-110,
                        away_ml_odds=-110
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
        
        print(f"[DEBUG] update_table called: show_all={show_all}, min_edge={min_edge:.1%}, total_predictions={len(self.predictions)}")
        
        # Filter predictions
        filtered = []
        for pred in self.predictions:
            best_bet = pred.get('best_bet')
            all_bets = pred.get('all_bets', [])
            
            # Filter by Kalshi markets only if enabled
            if kalshi_only and not pred.get('has_real_odds', False):
                continue
            
            if show_all:
                # Show all games, even without positive edge
                filtered.append(pred)
                print(f"[DEBUG] Show all: Added {pred['away_team']} @ {pred['home_team']}")
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
                    print(f"[DEBUG] Edge filter: Added {pred['away_team']} @ {pred['home_team']} (edge={edge:.1%})")
        
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
        print(f"[DEBUG] Displaying {len(filtered)} predictions in table")
        
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
            # Store prediction index in item data to handle sorting correctly
            matchup_item.setData(Qt.ItemDataRole.UserRole, row)
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
                
                self.table.setItem(row, 5, edge_item)
                
                # Prob
                model_prob = best_bet['model_prob']
                if isinstance(model_prob, str):
                    try:
                        model_prob = float(model_prob.strip('%')) / 100 if '%' in model_prob else float(model_prob)
                    except:
                        model_prob = 0.0
                prob_item = QTableWidgetItem(f"{model_prob:.1%}")
                prob_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 6, prob_item)
                
                # Stake
                stake = best_bet['stake']
                if isinstance(stake, str):
                    try:
                        stake = float(stake.replace('$', '').replace(',', ''))
                    except:
                        stake = 0.0
                stake_item = QTableWidgetItem(f"${stake:.2f}")
                stake_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 7, stake_item)
                
                # Add to total stake (use the cleaned stake value)
                total_stake += stake
                
                # Odds - display Kalshi decimal probability
                kalshi_home_prob = pred.get('kalshi_home_prob')
                kalshi_away_prob = pred.get('kalshi_away_prob')
                
                # Show Kalshi implied probability in decimal format (0.67)
                if best_bet['pick'] == pred['home_team'] and kalshi_home_prob is not None:
                    odds_str = f"{kalshi_home_prob:.2f}"
                elif best_bet['pick'] == pred['away_team'] and kalshi_away_prob is not None:
                    odds_str = f"{kalshi_away_prob:.2f}"
                else:
                    odds_str = "TBD"
                
                odds_item = QTableWidgetItem(odds_str)
                odds_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 8, odds_item)
                
                # Button
                place_btn = QPushButton("📝 Details")
                place_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
                place_btn.clicked.connect(lambda checked, p=pred: self.show_game_details_from_button(p))
                self.table.setCellWidget(row, 9, place_btn)
                
                # Wager input
                wager_input = QLineEdit()
                wager_input.setPlaceholderText("$")
                wager_input.setMaximumWidth(80)
                self.table.setCellWidget(row, 10, wager_input)
                
                # Log Bet button
                log_bet_btn = QPushButton("💾 Log")
                log_bet_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
                log_bet_btn.clicked.connect(lambda checked, p=pred: self.log_bet_from_button(p))
                self.table.setCellWidget(row, 11, log_bet_btn)
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
                    
                    # Edge (negative or low)
                    edge_item = QTableWidgetItem(f"{highest_edge_bet['edge']:+.1%}")
                    edge_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 5, edge_item)
                    
                    # Prob
                    prob_item = QTableWidgetItem(f"{highest_edge_bet['model_prob']:.1%}")
                    prob_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 6, prob_item)
                    
                    # Stake - N/A
                    stake_item = QTableWidgetItem("N/A")
                    stake_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 7, stake_item)
                    
                    # Odds
                    kalshi_home_prob = pred.get('kalshi_home_prob')
                    kalshi_away_prob = pred.get('kalshi_away_prob')
                    if highest_edge_bet['pick'] == pred['home_team'] and kalshi_home_prob is not None:
                        odds_str = f"{kalshi_home_prob:.2f}"
                    elif highest_edge_bet['pick'] == pred['away_team'] and kalshi_away_prob is not None:
                        odds_str = f"{kalshi_away_prob:.2f}"
                    else:
                        odds_str = "TBD"
                    odds_item = QTableWidgetItem(odds_str)
                    odds_item.setForeground(QColor(200, 200, 200))
                    self.table.setItem(row, 8, odds_item)
                
                # Button
                view_btn = QPushButton("👁️ View")
                view_btn.setStyleSheet("background-color: #6c757d; color: white;")
                view_btn.clicked.connect(lambda checked, p=pred: self.show_game_details_from_button(p))
                self.table.setCellWidget(row, 9, view_btn)
                
                # Wager input
                wager_input = QLineEdit()
                wager_input.setPlaceholderText("$")
                wager_input.setMaximumWidth(80)
                self.table.setCellWidget(row, 10, wager_input)
                
                # Log Bet button
                log_bet_btn = QPushButton("💾 Log")
                log_bet_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
                log_bet_btn.clicked.connect(lambda checked, p=pred: self.log_bet_from_button(p))
                self.table.setCellWidget(row, 11, log_bet_btn)
        
        # Update summary
        bankroll_pct = (total_stake / self.predictor.bankroll) * 100 if self.predictor.bankroll > 0 else 0
        self.summary.setText(
            f"📊 Total Games: {len(filtered)} | "
            f"🎯 Recommended Bets: {bet_count} | "
            f"💰 Total Stake: ${total_stake:.2f} ({bankroll_pct:.1f}% of bankroll) | "
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
            # Get the actual prediction index from the item data (handles sorting)
            item = self.table.item(row, 2)  # Matchup column
            if item:
                pred_index = item.data(Qt.ItemDataRole.UserRole)
                if pred_index is not None and pred_index < len(self.current_predictions):
                    pred = self.current_predictions[pred_index]
                    dialog = GameDetailDialog(pred, self)
                    dialog.exec()
                    return
            
            # Fallback: try to find prediction by matchup text
            if item:
                matchup_text = item.text()
                for pred in self.current_predictions:
                    if f"{pred['away_team']} @ {pred['home_team']}" in matchup_text:
                        dialog = GameDetailDialog(pred, self)
                        dialog.exec()
                        return
            
            # Last resort: use row index if within bounds
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
<b>Model:</b> {model_name} (Dec 12, 2025 PRODUCTION)<br>
<b>Features:</b> {num_features} Features (Including {len(injury_features)} injury/shock features)<br>
<b>Feature Calculator:</b> FeatureCalculatorV5 (PRODUCTION)<br>
<b>Model Performance:</b> 36.7% ROI, 70.8% win rate<br>
<b>Database:</b> {DATABASE_PATH.name}<br>
<b>Last Updated:</b> {datetime.now().strftime('%Y-%m-%d')}<br>
<br>
<b>Bankroll File:</b> {BANKROLL_SETTINGS_FILE.name}<br>
<b>Status:</b> ✓ Production Model Loaded (All Injury Features Active)
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
        
        # Minimum Advantage %
        min_edge_layout = QHBoxLayout()
        min_edge_layout.addWidget(QLabel("<b>Minimum Edge %:</b>"))
        self.min_edge_spin = QDoubleSpinBox()
        self.min_edge_spin.setRange(0.0, 0.50)
        self.min_edge_spin.setSingleStep(0.01)
        self.min_edge_spin.setValue(MIN_EDGE)
        self.min_edge_spin.setDecimals(2)
        self.min_edge_spin.setSuffix("%")
        self.min_edge_spin.valueChanged.connect(self.update_risk_settings)
        min_edge_layout.addWidget(self.min_edge_spin)
        min_edge_layout.addWidget(QLabel("(Minimum edge to place bet)"))
        min_edge_layout.addStretch()
        risk_layout.addLayout(min_edge_layout)
        
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
        min_edge = self.min_edge_spin.value() if hasattr(self, 'min_edge_spin') else MIN_EDGE
        max_wager = self.max_wager_spin.value() if hasattr(self, 'max_wager_spin') else MAX_BET_PCT
        
        self.risk_display.setText(
            f"Current: Kelly {kelly:.0%} | Min Edge {min_edge:.1%} | Max Wager {max_wager:.1%} of bankroll"
        )
    
    def save_risk_settings(self):
        """Save risk management settings to file"""
        global KELLY_FRACTION, MIN_EDGE, MAX_BET_PCT
        
        new_kelly = self.kelly_spin.value()
        new_min_edge = self.min_edge_spin.value()
        new_max_wager = self.max_wager_spin.value()
        
        # Update global constants
        KELLY_FRACTION = new_kelly
        MIN_EDGE = new_min_edge
        MAX_BET_PCT = new_max_wager
        
        # Save to bankroll settings file
        try:
            settings = {}
            if BANKROLL_SETTINGS_FILE.exists():
                with open(BANKROLL_SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
            
            settings['kelly_fraction'] = new_kelly
            settings['min_edge'] = new_min_edge
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
                f"Minimum Edge: {new_min_edge:.1%}\n"
                f"Maximum Wager: {new_max_wager:.1%} of bankroll\n\n"
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
        self.performance_tab = PerformanceTab()
        self.settings_tab = SettingsTab(self.predictor, self)
        
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

