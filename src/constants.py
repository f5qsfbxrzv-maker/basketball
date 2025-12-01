"""NBA Betting System Constants (V2)
Centralized configuration values to eliminate magic numbers and unify version metadata.
"""
# --- Version Identifiers ---
SYSTEM_VERSION: str = "2.0.0"
MODEL_VERSION: str = "v2"
FEATURE_VERSION: str = "v2"
CALIBRATION_VERSION: str = "v2"
DASHBOARD_VERSION: str = "v2.0"
LIVE_MODEL_VERSION: str = "v2.0"


from dataclasses import dataclass
from typing import Dict
from pathlib import Path

# Centralized database path (authoritative main DB)
CENTRAL_DB_PATH = Path(__file__).parent / "data" / "data/database/nba_betting_data.db"


# ==============================================================================
# PREDICTION & MODELING CONSTANTS
# ==============================================================================

# Blend weights for total prediction
HEURISTIC_BLEND_WEIGHT: float = 0.55  # Weight of raw model vs market line
MARKET_LINE_WEIGHT: float = 1.0 - HEURISTIC_BLEND_WEIGHT

# Recency blend weight for stats blending
RECENCY_STATS_BLEND_WEIGHT: float = 0.6  # Weight for recent vs season stats

# Injury impact scaling
INJURY_DIFF_TO_POINTS_SCALE: float = 2.0  # Convert injury differential to point impact
INJURY_DIFF_DIVISOR: float = 100.0

# Model probability bounds
MIN_PROBABILITY: float = 0.001
MAX_PROBABILITY: float = 0.999

# Total prediction sanity bounds
MIN_TOTAL_PREDICTION: float = 160.0
MAX_TOTAL_PREDICTION: float = 260.0

# Default offensive/defensive ratings
DEFAULT_OFFENSIVE_RATING: float = 112.0
DEFAULT_DEFENSIVE_RATING: float = 110.0

# ==============================================================================
# KELLY CRITERION & RISK MANAGEMENT
# ==============================================================================

KELLY_FRACTION_MULTIPLIER: float = 0.25  # Quarter Kelly
MAX_BET_PCT_OF_BANKROLL: float = 0.05  # 5% cap per bet
MIN_BET_PCT_OF_BANKROLL: float = 0.005  # 0.5% minimum threshold

# Edge thresholds
MIN_EDGE_FOR_BET: float = 0.03  # 3% minimum edge
STRONG_EDGE_THRESHOLD: float = 0.06  # 6% strong signal (tuned)

# ==============================================================================
# CALIBRATION QUALITY THRESHOLDS
# ==============================================================================

TARGET_BRIER_SCORE: float = 0.19  # Target Brier (post-v6 calibration realistic)
MAX_BRIER_FOR_BETTING: float = 0.25  # Refuse bets above this
MIN_CALIBRATION_SAMPLES: int = 250  # Minimum samples before trusting calibration
CALIBRATION_REFIT_INTERVAL_DAYS: int = 1  # Refit calibration nightly

# Edge computation heuristic coefficients
HEURISTIC_PACE_COEFFICIENT: float = 0.0016
HEURISTIC_COMPOSITE_ELO_COEFFICIENT: float = 0.0022
HEURISTIC_OFF_ELO_COEFFICIENT: float = 0.0014
HEURISTIC_DEF_ELO_COEFFICIENT: float = 0.0010
HEURISTIC_INJURY_COEFFICIENT: float = 0.01
HEURISTIC_TOTAL_LINE_COEFFICIENT: float = 0.001

# ==============================================================================
# KALSHI PRICING & COMMISSION
# ==============================================================================

KALSHI_BUY_COMMISSION: float = 0.02  # 2% on entry
KALSHI_SELL_COMMISSION: float = 0.02  # 2% on exit
KALSHI_EXPIRY_COMMISSION: float = 0.0  # 0% on expiry

# Kalshi price bounds (cents)
KALSHI_MIN_PRICE: int = 1
KALSHI_MAX_PRICE: int = 99

# ==============================================================================
# FEATURE ENGINEERING CONSTANTS
# ==============================================================================

# Recency decay
RECENCY_DECAY_DAYS: int = 14

# ELO adjustments
ELO_REST_BONUS_PER_DAY: float = 3.0
BACK_TO_BACK_PENALTY: float = -2.0

# Advanced ELO system constants
OFF_ELO_BASELINE: float = 1500.0
DEF_ELO_BASELINE: float = 1500.0
SEASON_REGRESSION_FACTOR: float = 0.75  # retain 75% of prior season delta
REGULAR_SEASON_BASE_K: float = 18.0
PLAYOFF_BASE_K: float = 24.0
ELO_MARGIN_SCALE: float = 20.0  # scale for margin influence
ELO_POINT_EXPECTATION_SCALE: float = 25.0  # scale converting point error to elo change
LEAGUE_AVG_POINTS: float = 111.0

# --- Injury ↔ Elo Interaction Scalars ---
# Portion of injury total (replacement-level point impact) attributed to offensive expectation reduction
INJURY_OFF_SHARE: float = 0.6  # 60% of measured impact assumed to depress scoring output
# Portion attributed to defensive deterioration (opponent scoring lift)
INJURY_DEF_SHARE: float = 0.4  # Remaining 40% increases opponent scoring environment
# Scaling factor converting injury point impact into Elo update context (smoothing)
INJURY_ELO_SCALER: float = 0.35  # Damp raw injury impact to avoid volatility in Elo adjustments

# Injury replacement
INJURY_REPLACEMENT_FACTOR: float = 0.6
CRITICAL_ABSENCE_THRESHOLD: float = 0.12  # PIE threshold

# Fallback PIE to use when player-level PIE lookup is unavailable (e.g., missing player_stats)
DEFAULT_PIE_FALLBACK: float = 0.11

# Injury pace impact (points reduction per critical injury)
INJURY_PACE_DECREMENT: float = 0.4

# Enhanced injury modeling constants
STATUS_PLAY_PROBABILITIES: Dict[str, float] = {
    'Out': 0.0,
    'Doubtful': 0.15,
    'Questionable': 0.55,
    'Probable': 0.85,
    'Active': 0.98,
}

# Chemistry / continuity lag effect for consecutive absences of star players
STAR_PLAYER_PIE_THRESHOLD: float = 0.16  # PIE above which player considered a star for lag multiplier
CHEMISTRY_LAG_FACTOR: float = 0.07       # Additional fractional impact per consecutive missed game
MAX_LAG_ABSENCES: int = 5                # Cap consecutive absence contribution

# Positional scarcity override multipliers (fine-tuning layer)
POSITION_SCARCITY_OVERRIDES: Dict[str, float] = {
    # Allows late-season tuning without editing core mapping in injury model
    'PG': 1.0,
    'C': 1.0,
}

# Pace blending
PACE_BLEND_WEIGHT: float = 0.5

# Travel distance threshold for fatigue
TRAVEL_FATIGUE_DISTANCE_MILES: float = 500.0

# ==============================================================================
# CALIBRATION SETTINGS
# ==============================================================================

MIN_SAMPLES_FOR_CALIBRATION: int = 200
ISOTONIC_MIN_SAMPLES: int = 250  # Minimum samples for isotonic regression
COMPONENT_VERSIONS: Dict[str, str] = {
    'feature_calculator': FEATURE_VERSION,
    'prediction_engine': MODEL_VERSION,
    'calibration': CALIBRATION_VERSION,
    'dashboard': DASHBOARD_VERSION,
    'live_model': LIVE_MODEL_VERSION,
}
RELIABILITY_CURVE_BINS: int = 10
MIN_BIN_COUNT: int = 15

# ==============================================================================
# UI & DISPLAY
# ==============================================================================

# Refresh intervals (milliseconds)
INJURY_REFRESH_INTERVAL_MS: int = 300000  # 5 minutes
ODDS_REFRESH_INTERVAL_MS: int = 60000     # 1 minute

# Table row limits
MAX_DISPLAY_ROWS: int = 100

# Colors (hex)
COLOR_POSITIVE_EDGE: str = "#00AA00"
COLOR_NEGATIVE_EDGE: str = "#AA0000"
COLOR_NEUTRAL: str = "#555555"

# ==============================================================================
# DATABASE LIMITS
# ==============================================================================

MAX_QUERY_RESULTS: int = 10000
DB_TIMEOUT_SECONDS: int = 30

# ==============================================================================
# TEAM MAPPINGS
# ==============================================================================

@dataclass
class TeamInfo:
    abbreviation: str
    full_name: str
    city: str
    altitude_ft: int = 0
    latitude: float = 0.0
    longitude: float = 0.0


TEAM_DATA: Dict[str, TeamInfo] = {
    'ATL': TeamInfo('ATL', 'Atlanta Hawks', 'Atlanta', 1050, 33.7573, -84.3963),
    'BOS': TeamInfo('BOS', 'Boston Celtics', 'Boston', 20, 42.3662, -71.0621),
    'BKN': TeamInfo('BKN', 'Brooklyn Nets', 'Brooklyn', 10, 40.6826, -73.9754),
    'CHA': TeamInfo('CHA', 'Charlotte Hornets', 'Charlotte', 750, 35.2251, -80.8392),
    'CHI': TeamInfo('CHI', 'Chicago Bulls', 'Chicago', 600, 41.8807, -87.6742),
    'CLE': TeamInfo('CLE', 'Cleveland Cavaliers', 'Cleveland', 650, 41.4964, -81.6882),
    'DAL': TeamInfo('DAL', 'Dallas Mavericks', 'Dallas', 430, 32.7905, -96.8103),
    'DEN': TeamInfo('DEN', 'Denver Nuggets', 'Denver', 5280, 39.7487, -105.0077),
    'DET': TeamInfo('DET', 'Detroit Pistons', 'Detroit', 600, 42.3410, -83.0552),
    'GSW': TeamInfo('GSW', 'Golden State Warriors', 'San Francisco', 0, 37.7680, -122.3878),
    'HOU': TeamInfo('HOU', 'Houston Rockets', 'Houston', 50, 29.7508, -95.3621),
    'IND': TeamInfo('IND', 'Indiana Pacers', 'Indianapolis', 715, 39.7640, -86.1555),
    'LAC': TeamInfo('LAC', 'LA Clippers', 'Los Angeles', 300, 34.0430, -118.2673),
    'LAL': TeamInfo('LAL', 'Los Angeles Lakers', 'Los Angeles', 300, 34.0430, -118.2673),
    'MEM': TeamInfo('MEM', 'Memphis Grizzlies', 'Memphis', 337, 35.1382, -90.0506),
    'MIA': TeamInfo('MIA', 'Miami Heat', 'Miami', 10, 25.7814, -80.1870),
    'MIL': TeamInfo('MIL', 'Milwaukee Bucks', 'Milwaukee', 635, 43.0436, -87.9170),
    'MIN': TeamInfo('MIN', 'Minnesota Timberwolves', 'Minneapolis', 830, 44.9795, -93.2760),
    'NOP': TeamInfo('NOP', 'New Orleans Pelicans', 'New Orleans', 10, 29.9490, -90.0821),
    'NYK': TeamInfo('NYK', 'New York Knicks', 'New York', 33, 40.7505, -73.9934),
    'OKC': TeamInfo('OKC', 'Oklahoma City Thunder', 'Oklahoma City', 1201, 35.4634, -97.5151),
    'ORL': TeamInfo('ORL', 'Orlando Magic', 'Orlando', 100, 28.5392, -81.3839),
    'PHI': TeamInfo('PHI', 'Philadelphia 76ers', 'Philadelphia', 40, 39.9012, -75.1720),
    'PHX': TeamInfo('PHX', 'Phoenix Suns', 'Phoenix', 1086, 33.4457, -112.0712),
    'POR': TeamInfo('POR', 'Portland Trail Blazers', 'Portland', 50, 45.5316, -122.6668),
    'SAC': TeamInfo('SAC', 'Sacramento Kings', 'Sacramento', 30, 38.5801, -121.4999),
    'SAS': TeamInfo('SAS', 'San Antonio Spurs', 'San Antonio', 650, 29.4270, -98.4375),
    'TOR': TeamInfo('TOR', 'Toronto Raptors', 'Toronto', 250, 43.6435, -79.3791),
    'UTA': TeamInfo('UTA', 'Utah Jazz', 'Salt Lake City', 4226, 40.7683, -111.9011),
    'WAS': TeamInfo('WAS', 'Washington Wizards', 'Washington', 410, 38.8981, -77.0209),
}


def get_team_altitude(team_abbr: str) -> int:
    """Get team's home arena altitude in feet."""
    return TEAM_DATA.get(team_abbr, TeamInfo('', '', '', 0)).altitude_ft


def get_team_coordinates(team_abbr: str) -> tuple[float, float]:
    """Get team's (latitude, longitude)."""
    info = TEAM_DATA.get(team_abbr, TeamInfo('', '', '', 0, 0.0, 0.0))
    return (info.latitude, info.longitude)
