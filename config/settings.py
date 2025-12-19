"""
MASTER CONFIGURATION - NBA Betting System
All paths, settings, and constants in one place
"""

import os
from pathlib import Path

# === PATHS ===
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "output"

# Database
DB_PATH = DATA_DIR / "live" / "nba_betting_data.db"
BACKUP_DIR = DATA_DIR / "backups"

# Models
PRODUCTION_MODELS_DIR = MODELS_DIR / "production"
EXPERIMENTAL_MODELS_DIR = MODELS_DIR / "experimental"

# PRODUCTION MODEL - Trial 1306 (22 features, 49.7% ROI)
MONEYLINE_MODEL = MODELS_DIR / "xgboost_22features_trial1306_20251215_212306.json"
TOTALS_MODEL = PRODUCTION_MODELS_DIR / "totals_model_enhanced.pkl"
ISOTONIC_CALIBRATOR = MODELS_DIR / "isotonic_calibrator_final.pkl"  # Dec 12 calibrator
PLATT_CALIBRATOR = PRODUCTION_MODELS_DIR / "moneyline_calibrator_platt.pkl"

# Logs
PREDICTION_LOGS = LOGS_DIR / "predictions"
ERROR_LOGS = LOGS_DIR / "errors"

# Output
DAILY_PICKS_DIR = OUTPUT_DIR / "daily_picks"
VISUALS_DIR = OUTPUT_DIR / "visuals"

# === BETTING PARAMETERS ===
MIN_EDGE_FOR_BET = 0.03  # 3% minimum edge
MAX_BET_PCT_OF_BANKROLL = 0.05  # 5% maximum single bet
KELLY_FRACTION_MULTIPLIER = 0.25  # Quarter Kelly default
KALSHI_BUY_COMMISSION = 0.07  # 7% commission

# === CALIBRATION PARAMETERS ===
MIN_CALIBRATION_SAMPLES = 250  # Minimum games before refit
CALIBRATION_REFIT_DAYS = 7  # Days between refits

# === RECENCY WEIGHTING ===
RECENCY_STATS_BLEND_WEIGHT = 0.7  # Weight recent games

# === API KEYS (Load from environment or .env file) ===
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")
KALSHI_API_SECRET = os.getenv("KALSHI_API_SECRET", "")

# === SUPERSTAR OVERRIDE LIST ===
# Players whose absence MUST be heavily weighted (injury_impact multiplier: 5x)
DO_NOT_FLY_LIST = [
    "Stephen Curry",
    "LeBron James",  
    "Nikola Jokic",
    "Giannis Antetokounmpo",
    "Luka Doncic",
    "Joel Embiid",
    "Kevin Durant",
    "Damian Lillard",
    "Anthony Davis",
    "Jayson Tatum",
]

# === FEATURE ENGINEERING ===
# Four Factors Weights (Dean Oliver standard - for baseline display only, ML learns optimal)
FOUR_FACTORS_WEIGHTS = {
    'efg': 0.40,  # Effective Field Goal %
    'tov': 0.25,  # Turnover %
    'reb': 0.20,  # Rebounding %
    'ftr': 0.15   # Free Throw Rate
}

# ELO Parameters
INITIAL_ELO = 1500
ELO_K_FACTOR = 20
HOME_COURT_ADVANTAGE_ELO = 100

# === RISK MANAGEMENT ===
# Drawdown thresholds for Kelly scaling
DRAWDOWN_SEVERE = 0.20  # >20% DD → 25% Kelly
DRAWDOWN_HIGH = 0.10    # >10% DD → 50% Kelly  
DRAWDOWN_MODERATE = 0.05 # >5% DD → 75% Kelly

# === DATA VALIDATION ===
# Prevent data leakage - ensure all queries use these patterns
REQUIRE_DATE_FILTER = True  # Enforce game_date < as_of_date in all queries
MAX_LOOK_BACK_DAYS = 365 * 3  # Maximum historical data to consider

# === LOGGING ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def validate_paths():
    """Ensure all required directories exist"""
    for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR,
                     BACKUP_DIR, PRODUCTION_MODELS_DIR, EXPERIMENTAL_MODELS_DIR,
                     PREDICTION_LOGS, ERROR_LOGS, DAILY_PICKS_DIR, VISUALS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check critical files exist
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    if not MONEYLINE_MODEL.exists():
        raise FileNotFoundError(f"Moneyline model not found: {MONEYLINE_MODEL}")
    
    return True

if __name__ == "__main__":
    validate_paths()
    print("✓ All paths validated")
    print(f"Database: {DB_PATH}")
    print(f"Moneyline Model: {MONEYLINE_MODEL}")
    print(f"Totals Model: {TOTALS_MODEL}")
