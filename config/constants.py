"""
Core constants for the NBA betting system
"""

# Feature calculation constants
RECENCY_STATS_BLEND_WEIGHT = 0.6  # 60% weight to recent L10 games vs season average

# ELO system constants
INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE_ELO = 100

# Off/Def ELO system constants - Syndicate Level
OFF_ELO_BASELINE = 1500
DEF_ELO_BASELINE = 1500
SEASON_REGRESSION_FACTOR = 0.75  # Regress 25% toward mean between seasons (75/25 reversion)
SEASON_REGRESSION_TARGET = 1505  # Regress toward 1505, not 1500 (syndicate standard)

# Auto-Regressive K-Factors (OPTIMIZED via Grid Search - Log-Loss Minimization)
EARLY_SEASON_K = 15  # First 20 games (Gold Standard: stable learning rate)
REGULAR_SEASON_BASE_K = 15  # Games 21+ (Gold Standard: minimizes log-loss)
EARLY_SEASON_THRESHOLD = 20  # Switch from early to regular K after this many games
PLAYOFF_BASE_K = 20  # Playoff K-factor (slightly higher reactivity)

# Outcome Hammer (Win/Loss Weighting) - OPTIMIZED
WIN_WEIGHT = 30.0  # Outcome hammer (Gold Standard: forces respect for W/L records)

# Logarithmic Margin Dampening (fixes Brooklyn blowout issue) - OPTIMIZED
LOG_MARGIN_DAMPENER = 0.5  # MOV_BIAS cap (prevents margin over-weighting)
ELO_DIFF_SCALAR = 0.001  # Scales ELO difference impact on margin

ELO_POINT_EXPECTATION_SCALE = 40.0  # Points per 100 ELO difference (OPTIMIZED: prevents explosion)
LEAGUE_AVG_POINTS = 110  # League average offensive rating

# Rest-Adjusted Home Court Advantage (Dynamic HCA)
BASE_HCA = 55  # Base home court advantage
B2B_PENALTY = -15  # Back-to-back game penalty
THREE_IN_FOUR_PENALTY = -25  # 3 games in 4 nights penalty
ALTITUDE_BONUS = 10  # Denver/Utah altitude advantage

# Injury ELO adjustments
INJURY_OFF_SHARE = 0.15  # Injury impact on offensive ELO (15% max)
INJURY_DEF_SHARE = 0.10  # Injury impact on defensive ELO (10% max)
INJURY_ELO_SCALER = 100  # ELO points per unit of injury impact

# Injury impact constants
SUPERSTAR_INJURY_MULTIPLIER = 5.0  # 5x weight for DO_NOT_FLY_LIST players
REPLACEMENT_LEVEL_PIE = 0.05  # Baseline PIE score for replacement players
STAR_PLAYER_PIE_THRESHOLD = 0.15  # PIE threshold for "star" classification

# Injury status probabilities
STATUS_PLAY_PROBABILITIES = {
    'Out': 0.0,
    'Doubtful': 0.25,
    'Questionable': 0.50,
    'Probable': 0.75,
    'Available': 1.0,
    'GTD': 0.50,  # Game-Time Decision
}

# Chemistry lag (returning from injury)
CHEMISTRY_LAG_FACTOR = 0.85  # Returning players play at 85% effectiveness initially
MAX_LAG_ABSENCES = 10  # Lag applies if player missed ≥10 games

# Position scarcity overrides
POSITION_SCARCITY_OVERRIDES = {
    # Players who have outsized impact despite average PIE
    # Format: 'PLAYER_NAME': scarcity_multiplier
}

# Rest/fatigue constants
BACK_TO_BACK_PENALTY = 0.15  # 15% expected performance drop on B2B
THREE_IN_FOUR_PENALTY = 0.10  # 10% expected performance drop on 3-in-4

# Betting constants (also in config.settings but duplicated here for backwards compatibility)
MIN_EDGE_FOR_BET = 0.03
KELLY_FRACTION_MULTIPLIER = 0.25
MAX_BET_PCT_OF_BANKROLL = 0.05

# Prediction engine constants
MIN_PROBABILITY = 0.01  # Floor for probability estimates
MAX_PROBABILITY = 0.99  # Ceiling for probability estimates

# Heuristic fallback coefficients (when ML model unavailable)
HEURISTIC_PACE_COEFFICIENT = 0.0016  # Pace impact on total
HEURISTIC_COMPOSITE_ELO_COEFFICIENT = 0.0022  # ELO strength impact
HEURISTIC_OFF_ELO_COEFFICIENT = 0.0018  # Offensive ELO impact
HEURISTIC_DEF_ELO_COEFFICIENT = 0.0015  # Defensive ELO impact
HEURISTIC_INJURY_COEFFICIENT = 0.05  # Injury impact multiplier
HEURISTIC_TOTAL_LINE_COEFFICIENT = 1.0  # Market line weight

# Commission/vig constants
KALSHI_BUY_COMMISSION = 0.03  # 3% commission on buys
KALSHI_SELL_COMMISSION = 0.07  # 7% commission on sells
