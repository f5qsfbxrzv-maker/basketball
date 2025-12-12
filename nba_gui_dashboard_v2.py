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
    QHeaderView, QMessageBox, QProgressBar, QDialog, QGridLayout, QCheckBox
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
    from feature_extractor_validated import ValidatedFeatureExtractor
    from V2.schedule_downloader import NBAScheduleDownloader
    from injury_service import InjuryService
    PREDICTION_ENGINE_AVAILABLE = True
except ImportError as e:
    PREDICTION_ENGINE_AVAILABLE = False
    print(f"[WARNING] Prediction engine not available: {e}")

try:
    from paper_trading_tracker import PaperTradingTracker
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False
    print("[WARNING] paper_trading_tracker not available")


class NBAPredictionEngine:
    """Prediction engine using ValidatedFeatureExtractor (97 features, +130.7% ROI)"""
    
    def __init__(self):
        self.bankroll = BANKROLL
        self.predictions_cache = {}
        
        if PREDICTION_ENGINE_AVAILABLE:
            # Load the validated model
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            self.model = joblib.load(MODEL_PATH)
            self.features = list(self.model.feature_names_in_)
            self.feature_extractor = ValidatedFeatureExtractor()
            self.injury_service = InjuryService()
            print(f"[OK] Loaded validated model: {len(self.features)} features")
            
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
            self.feature_extractor = None
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
        """Make prediction using ValidatedFeatureExtractor with odds quality filter"""
        try:
            if not self.model or not self.feature_extractor:
                return {'error': 'Model not available'}
            
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
            
            # Extract features
            features_dict = self.feature_extractor.extract_features(
                home_team=home_team,
                away_team=away_team,
                game_date=datetime.strptime(game_date, '%Y-%m-%d') if isinstance(game_date, str) else game_date
            )
            
            # Convert to DataFrame for model
            X = pd.DataFrame([features_dict])
            X = X[self.model.feature_names_in_]
            
            # Get model probability (home team win probability)
            home_prob = self.model.predict_proba(X)[0, 1]
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
            
            # Get injured players
            home_injuries = []
            away_injuries = []
            if self.injury_service:
                try:
                    home_injuries = self.injury_service.get_injured_players(home_team)
                    away_injuries = self.injury_service.get_injured_players(away_team)
                except Exception as e:
                    print(f"[WARNING] Could not get injured players: {e}")
            
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
                'odds_source': odds_source,
                'has_real_odds': has_real_odds,
                'home_injuries': home_injuries,
                'away_injuries': away_injuries,
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
        matchup = f"{self.prediction['away_team']} @ {self.prediction['home_team']}"
        self.setWindowTitle(f"Game Details - {matchup}")
        self.setGeometry(200, 100, 1000, 850)  # Larger window: wider and taller
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel(
            f"<h2 style='color: white; background-color: #2c3e50; padding: 10px;'>{matchup}</h2>"
            f"<p style='color: #ecf0f1; background-color: #34495e; padding: 5px;'>"
            f"{self.prediction['game_date']} at {self.prediction.get('game_time', 'TBD')}</p>"
        )
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Model Probabilities
        prob_group = QGroupBox("Model Predictions")
        prob_layout = QGridLayout()
        
        prob_layout.addWidget(QLabel("<b>Team</b>"), 0, 0)
        prob_layout.addWidget(QLabel("<b>Win Probability</b>"), 0, 1)
        
        prob_layout.addWidget(QLabel(f"<b>{self.prediction['home_team']}</b>"), 1, 0)
        prob_layout.addWidget(QLabel(f"<span style='color: #27ae60;'>{self.prediction['home_win_prob']:.1%}</span>"), 1, 1)
        
        prob_layout.addWidget(QLabel(f"<b>{self.prediction['away_team']}</b>"), 2, 0)
        prob_layout.addWidget(QLabel(f"<span style='color: #e74c3c;'>{self.prediction['away_win_prob']:.1%}</span>"), 2, 1)
        
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
            
            edge_item = QTableWidgetItem(f"{bet['edge']:+.1%}")
            if bet['edge'] >= 0.10:
                edge_item.setBackground(QColor(144, 238, 144))
            elif bet['edge'] >= 0.05:
                edge_item.setBackground(QColor(255, 255, 153))
            bets_table.setItem(row, 2, edge_item)
            
            bets_table.setItem(row, 3, QTableWidgetItem(f"{bet['model_prob']:.1%}"))
            bets_table.setItem(row, 4, QTableWidgetItem(f"{bet['market_prob']:.1%}"))
            
            # Format odds (handle int or float)
            odds_val = bet['odds']
            if isinstance(odds_val, int):
                odds_str = f"{odds_val:+d}"
            else:
                odds_str = f"{int(odds_val):+d}" if odds_val == int(odds_val) else f"{odds_val:+.0f}"
            bets_table.setItem(row, 5, QTableWidgetItem(odds_str))
            
            bets_table.setItem(row, 6, QTableWidgetItem(f"${bet['stake']:.2f}"))
        
        bets_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bets_layout.addWidget(bets_table)
        bets_group.setLayout(bets_layout)
        layout.addWidget(bets_group)
        
        # Matchup Breakdown - Comprehensive Stats
        breakdown_group = QGroupBox("🏀 Matchup Breakdown")
        breakdown_layout = QVBoxLayout()
        
        comparison_table = QTableWidget()
        comparison_table.setColumnCount(4)  # Added column for difference
        comparison_table.setHorizontalHeaderLabels(['Stat', self.prediction['home_team'], self.prediction['away_team'], 'Advantage'])
        
        features = self.prediction.get('features', {})
        stats_rows = []
        
        # Extract team names for database queries
        home_team = self.prediction['home_team']
        away_team = self.prediction['away_team']
        
        # Records - calculate from win percentage (direct features not available)
        home_win_pct = features.get('home_win_pct', 0.5)
        away_win_pct = features.get('away_win_pct', 0.5)
        
        # Fetch actual records from database
        import sqlite3
        conn = sqlite3.connect('V2/v2/data/nba_betting_data.db')
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
        home_pace = features.get('home_ewma_pace') or 100
        away_pace = features.get('away_ewma_pace') or 100
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
        
        # Injuries
        injury_diff = features.get('injury_impact_diff') or 0
        home_injuries = self.prediction.get('home_injuries', [])
        away_injuries = self.prediction.get('away_injuries', [])
        stats_rows.append((
            'Injuries', 
            len(home_injuries), 
            len(away_injuries), 
            'lower',
            f"{len(home_injuries)} out",
            f"{len(away_injuries)} out"
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
            stat_item.setForeground(QColor(255, 255, 255))  # White text
            stat_item.setBackground(QColor(40, 40, 40))  # Very dark gray
            stat_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            comparison_table.setItem(row, 0, stat_item)
            
            # Determine leader
            if comparison_type == 'record':
                # For records, compare wins
                # Handle both tuple (wins, losses) and string "W-L" formats
                if isinstance(home_val_raw, tuple):
                    home_wins = home_val_raw[0]
                    away_wins = away_val_raw[0]
                else:
                    # Parse "W-L" string
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
            home_item.setForeground(QColor(255, 255, 255))  # White text always
            if home_better:
                home_item.setBackground(QColor(0, 100, 0))  # Dark green background
                home_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            else:
                home_item.setBackground(QColor(60, 60, 60))  # Dark gray background
            comparison_table.setItem(row, 1, home_item)
            
            # Away value with highlighting
            away_item = QTableWidgetItem(away_val_str)
            away_item.setForeground(QColor(255, 255, 255))  # White text always
            if away_better:
                away_item.setBackground(QColor(139, 0, 0))  # Dark red background
                away_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            else:
                away_item.setBackground(QColor(60, 60, 60))  # Dark gray background
            comparison_table.setItem(row, 2, away_item)
            
            # Advantage/Difference column
            adv_item = QTableWidgetItem(diff_str)
            adv_item.setForeground(QColor(255, 255, 255))  # White text
            if home_better:
                adv_item.setBackground(QColor(0, 80, 0))  # Dark green
                adv_item.setToolTip(f"{self.prediction['home_team']} advantage")
            elif away_better:
                adv_item.setBackground(QColor(100, 0, 0))  # Dark red
                adv_item.setToolTip(f"{self.prediction['away_team']} advantage")
            else:
                adv_item.setBackground(QColor(60, 60, 60))  # Dark gray
            comparison_table.setItem(row, 3, adv_item)
        
        comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        comparison_table.verticalHeader().setVisible(False)
        comparison_table.setMinimumHeight(400)  # Ensure enough room to see all rows
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
                status_color = '#e74c3c' if inj['status'].lower() == 'out' else '#f39c12'
                inj_text = QLabel(f"<span style='color:{status_color};'>●</span> {inj['player']} - {inj['status']} ({inj['injury']})")
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
                status_color = '#e74c3c' if inj['status'].lower() == 'out' else '#f39c12'
                inj_text = QLabel(f"<span style='color:{status_color};'>●</span> {inj['player']} - {inj['status']} ({inj['injury']})")
                away_inj_layout.addWidget(inj_text)
        else:
            away_inj_layout.addWidget(QLabel("<i>No injuries reported</i>"))
        
        away_inj_widget.setLayout(away_inj_layout)
        injuries_layout.addWidget(away_inj_widget)
        
        injuries_group.setLayout(injuries_layout)
        layout.addWidget(injuries_group)
        
        # Predicted Total (if available)
        if self.prediction.get('predicted_total'):
            total_label = QLabel(f"<h3 style='color: #3498db;'>Predicted Total: {self.prediction['predicted_total']:.1f} points</h3>")
            total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(total_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)


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
        self.days_spin.setValue(1)  # Changed from 3 to 1 to avoid NBA API crashes
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
        self.kalshi_only_checkbox.setChecked(True)  # Default to ON
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
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            'Date', 'Time', 'Matchup', 'Best Bet', 'Type', 'Edge', 'Prob', 'Stake', 'Odds', 'Action'
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(self.show_game_details)
        
        # Fix selection colors for better readability - white text on dark blue instead of green
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
        
        # Initial refresh
        QTimer.singleShot(1000, self.refresh_predictions)
    
    def toggle_auto_refresh(self, state):
        """Toggle auto-refresh timer"""
        if state == Qt.CheckState.Checked.value:
            self.auto_refresh_timer.start(300000)  # 5 minutes
            self.status_label.setText("✅ Auto-refresh enabled (every 5 min)")
        else:
            self.auto_refresh_timer.stop()
            self.status_label.setText("")
    
    def refresh_predictions(self):
        """Load predictions using real schedule data"""
        self.status_label.setText("⏳ Loading predictions...")
        self.predictions = []
        
        # Get days_ahead from spinner
        days_ahead = self.days_spin.value()
        
        try:
            if not PREDICTION_ENGINE_AVAILABLE:
                self.status_label.setText("❌ Prediction engine not available")
                return
            
            schedule_downloader = NBAScheduleDownloader()
            
            # Fetch games for each day in the lookahead window
            all_games = []
            for day_offset in range(1, days_ahead + 1):  # Start from tomorrow
                target_date = (datetime.now() + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                day_games = schedule_downloader.get_games_for_date(target_date)
                
                # Add date to each game for reference
                for game in day_games:
                    game['prediction_date'] = target_date
                    all_games.append(game)
            
            if not all_games:
                self.status_label.setText(f"No games in next {days_ahead} days")
                return
            
            print(f"Found {len(all_games)} games in next {days_ahead} days")
            
            # Get predictions for each game
            for game in all_games:
                try:
                    pred = self.predictor.predict_game(
                        home_team=game['home_team'],
                        away_team=game['away_team'],
                        game_date=game['prediction_date'],
                        game_time=game.get('game_time', 'TBD'),
                        home_ml_odds=-110,
                        away_ml_odds=-110
                    )
                    if 'error' not in pred:
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
            self.update_table()
            
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
            elif best_bet and best_bet['edge'] >= min_edge:
                filtered.append(pred)
        
        # Sort by edge
        def sort_key(x):
            best_bet = x.get('best_bet')
            if best_bet:
                return best_bet['edge']
            # If no best_bet, use highest edge from all_bets
            all_bets = x.get('all_bets', [])
            return max([b['edge'] for b in all_bets], default=-999)
        
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
            matchup_item = QTableWidgetItem(matchup)
            matchup_item.setForeground(QColor(135, 206, 250))  # Light blue
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
                
                # Edge
                edge_item = QTableWidgetItem(f"{best_bet['edge']:+.1%}")
                edge_item.setForeground(QColor(255, 255, 255))
                if best_bet['edge'] >= 0.10:
                    edge_item.setBackground(QColor(0, 128, 0))  # Dark green
                elif best_bet['edge'] >= 0.05:
                    edge_item.setBackground(QColor(184, 134, 11))  # Dark goldenrod
                self.table.setItem(row, 5, edge_item)
                
                # Prob
                prob_item = QTableWidgetItem(f"{best_bet['model_prob']:.1%}")
                prob_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 6, prob_item)
                
                # Stake
                stake_item = QTableWidgetItem(f"${best_bet['stake']:.2f}")
                stake_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 7, stake_item)
                
                total_stake += best_bet['stake']
                
                # Odds with Kalshi implied probability
                kalshi_home_prob = pred.get('kalshi_home_prob')
                kalshi_away_prob = pred.get('kalshi_away_prob')
                odds_val = best_bet['odds']
                
                # Format odds (could be American odds integer, Kalshi price float, or None)
                if odds_val is None or odds_val == 0.5:
                    # Placeholder or missing odds
                    odds_str = "TBD"
                elif isinstance(odds_val, int) or (isinstance(odds_val, float) and abs(odds_val) >= 100):
                    # American odds
                    if kalshi_home_prob and best_bet['pick'] == pred['home_team']:
                        odds_str = f"{int(odds_val):+d} ({kalshi_home_prob:.1%})"
                    elif kalshi_away_prob and best_bet['pick'] == pred['away_team']:
                        odds_str = f"{int(odds_val):+d} ({kalshi_away_prob:.1%})"
                    else:
                        odds_str = f"{int(odds_val):+d}"
                else:
                    # Kalshi price (0-1 range)
                    odds_str = f"{odds_val:.0%}"
                
                odds_item = QTableWidgetItem(odds_str)
                odds_item.setForeground(QColor(255, 255, 255))
                self.table.setItem(row, 8, odds_item)
                
                # Button
                place_btn = QPushButton("📝 Details")
                place_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
                place_btn.clicked.connect(lambda checked, p=pred: self.show_game_details_from_button(p))
                self.table.setCellWidget(row, 9, place_btn)
            else:
                # No positive edge bet - show highest edge bet anyway for visibility
                all_bets = pred.get('all_bets', [])
                if all_bets:
                    highest_edge_bet = max(all_bets, key=lambda b: b['edge'])
                    
                    # Best Bet
                    bet_item = QTableWidgetItem(highest_edge_bet['pick'])
                    bet_item.setForeground(QColor(150, 150, 150))  # Gray
                    self.table.setItem(row, 3, bet_item)
                    
                    # Type
                    type_item = QTableWidgetItem(highest_edge_bet['type'])
                    type_item.setForeground(QColor(150, 150, 150))
                    self.table.setItem(row, 4, type_item)
                    
                    # Edge (negative)
                    edge_item = QTableWidgetItem(f"{highest_edge_bet['edge']:+.1%}")
                    edge_item.setForeground(QColor(255, 100, 100))  # Light red
                    self.table.setItem(row, 5, edge_item)
                    
                    # Prob
                    prob_item = QTableWidgetItem(f"{highest_edge_bet['model_prob']:.1%}")
                    prob_item.setForeground(QColor(150, 150, 150))
                    self.table.setItem(row, 6, prob_item)
                    
                    # Stake (show $0)
                    stake_item = QTableWidgetItem("$0.00")
                    stake_item.setForeground(QColor(150, 150, 150))
                    self.table.setItem(row, 7, stake_item)
                    
                    # Odds
                    odds_val = highest_edge_bet['odds']
                    if isinstance(odds_val, int):
                        odds_str = f"{odds_val:+d}"
                    else:
                        odds_str = f"{int(odds_val):+d}" if odds_val == int(odds_val) else f"{odds_val:+.0f}"
                    odds_item = QTableWidgetItem(odds_str)
                    odds_item.setForeground(QColor(150, 150, 150))
                    self.table.setItem(row, 8, odds_item)
                    
                    # Button
                    place_btn = QPushButton("📊 View")
                    place_btn.setStyleSheet("background-color: #6c757d; color: white;")
                    place_btn.clicked.connect(lambda checked, p=pred: self.show_game_details_from_button(p))
                    self.table.setCellWidget(row, 9, place_btn)
                else:
                    # No bets at all (edge case)
                    no_bet_item = QTableWidgetItem("No Bet")
                    no_bet_item.setForeground(QColor(128, 128, 128))
                    self.table.setItem(row, 3, no_bet_item)
                    
                    for col in [4, 5, 6, 7, 8]:
                        empty_item = QTableWidgetItem("-")
                        empty_item.setForeground(QColor(128, 128, 128))
                        self.table.setItem(row, col, empty_item)
                    
                    # Still show details button
                    details_btn = QPushButton("📊 View")
                    details_btn.setStyleSheet("background-color: #6c757d; color: white;")
                    details_btn.clicked.connect(lambda checked, p=pred: self.show_game_details_from_button(p))
                    self.table.setCellWidget(row, 9, details_btn)
        
        # Update summary
        bankroll_pct = (total_stake / self.predictor.bankroll) * 100 if self.predictor.bankroll > 0 else 0
        self.summary.setText(
            f"📊 Total Games: {len(filtered)} | "
            f"🎯 Recommended Bets: {bet_count} | "
            f"💰 Total Stake: ${total_stake:.2f} ({bankroll_pct:.1f}% of bankroll) | "
            f"💡 Double-click any game for full breakdown"
        )
    
    def show_game_details(self, row, col):
        """Show detailed game info on double-click"""
        # Get the actual prediction index from the item data (handles sorting)
        item = self.table.item(row, 2)  # Matchup column
        if item:
            pred_index = item.data(Qt.ItemDataRole.UserRole)
            if pred_index is not None and pred_index < len(self.current_predictions):
                pred = self.current_predictions[pred_index]
                dialog = GameDetailDialog(pred, self)
                dialog.exec()
        elif row < len(self.current_predictions):
            # Fallback if data not set
            pred = self.current_predictions[row]
            dialog = GameDetailDialog(pred, self)
            dialog.exec()
    
    def show_game_details_from_button(self, pred):
        """Show game details from button click"""
        dialog = GameDetailDialog(pred, self)
        dialog.exec()


class PerformanceTab(QWidget):
    """Performance tab with paper trading metrics"""
    
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
        
        # Results table
        results_group = QGroupBox("📋 Prediction Log")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            'Date', 'Game', 'Prediction', 'Stake', 'Odds', 'Result', 'Profit/Loss', 'ROI'
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
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
        
        # Quick adjustments
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Quick Adjust:"))
        
        for amount in [100, 500, 1000, -100, -500, -1000]:
            btn = QPushButton(f"{amount:+,d}")
            btn.clicked.connect(lambda checked, amt=amount: self.quick_adjust(amt))
            if amount > 0:
                btn.setStyleSheet("background-color: #d4edda; color: #155724;")
            else:
                btn.setStyleSheet("background-color: #f8d7da; color: #721c24;")
            quick_layout.addWidget(btn)
        
        quick_layout.addStretch()
        bankroll_layout.addLayout(quick_layout)
        
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
        
        if self.predictor.feature_extractor:
            info_text = f"""
<b>Model:</b> ValidatedFeatureExtractor<br>
<b>Features:</b> 97 Features (Validated Walk-Forward)<br>
<b>Validated Performance:</b> +130.7% ROI (Oct 2023 - Oct 2024)<br>
<b>Last Updated:</b> {datetime.now().strftime('%Y-%m-%d')}<br>
<br>
<b>Bankroll File:</b> {BANKROLL_SETTINGS_FILE.name}<br>
<b>Status:</b> Persistence Enabled ✓
"""
        else:
            info_text = "<b>Status:</b> Predictor not loaded"
        
        info_label = QLabel(info_text)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Risk settings
        risk_group = QGroupBox("⚙️ Risk Management")
        risk_layout = QVBoxLayout()
        
        risk_text = f"""
<b>Kelly Fraction:</b> {KELLY_FRACTION:.0%} (Half Kelly)<br>
<b>Max Bet Size:</b> {MAX_BET_PCT:.0%} of bankroll<br>
<b>Min Edge:</b> {MIN_EDGE:.0%}<br>
<b>Max Edge:</b> {MAX_EDGE:.0%} (No upper limit)
"""
        
        risk_label = QLabel(risk_text)
        risk_label.setTextFormat(Qt.TextFormat.RichText)
        risk_layout.addWidget(risk_label)
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
    
    def quick_adjust(self, amount: float):
        """Quick bankroll adjustment with save"""
        new_bankroll = self.predictor.bankroll + amount
        if new_bankroll < 0:
            QMessageBox.warning(self, "Invalid", "Bankroll cannot be negative")
            return
        
        self.predictor.bankroll = new_bankroll
        self.bankroll_input.setValue(new_bankroll)
        self.bankroll_display.setText(f"${new_bankroll:,.2f}")
        
        # Save to file
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
        self.setWindowTitle("NBA Betting System - Professional Dashboard V2")
        self.setGeometry(100, 100, 1600, 900)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("🏀 NBA BETTING SYSTEM - PROFESSIONAL DASHBOARD V2")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("padding: 15px; background-color: #2c3e50; color: white;")
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
