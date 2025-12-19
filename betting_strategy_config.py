# NBA Betting Strategy Configuration
# Optimized thresholds based on Trial #215 split backtest
# Date: December 15, 2025

# Model Configuration
MODEL_TRIAL = 215
MODEL_STUDY = 'nba_matchup_optimized_2000trials'
MODEL_FEATURES = 24  # Matchup-optimized feature set

# Betting Thresholds (Optimized for Maximum Total Units)
FAVORITE_EDGE_THRESHOLD = 0.01    # 1.0% edge for favorites (odds < 2.00)
UNDERDOG_EDGE_THRESHOLD = 0.15    # 15.0% edge for underdogs (odds >= 2.00)
ODDS_SPLIT_THRESHOLD = 2.00       # Decimal odds threshold (2.00 = +100)

# Expected Performance (2024-25 Backtest)
EXPECTED_FAVORITES_WIN_RATE = 0.654  # 65.4%
EXPECTED_UNDERDOGS_WIN_RATE = 0.302  # 30.2%
EXPECTED_TOTAL_ROI = 0.078           # 7.8%

# Backtest Results Summary
BACKTEST_RESULTS = {
    'favorites': {
        'threshold': FAVORITE_EDGE_THRESHOLD,
        'bets': 257,
        'win_rate': EXPECTED_FAVORITES_WIN_RATE,
        'units': 9.81,
        'roi': 0.0382
    },
    'underdogs': {
        'threshold': UNDERDOG_EDGE_THRESHOLD,
        'bets': 461,
        'win_rate': EXPECTED_UNDERDOGS_WIN_RATE,
        'units': 46.18,
        'roi': 0.1002
    },
    'combined': {
        'total_bets': 718,
        'total_units': 55.99,
        'total_roi': EXPECTED_TOTAL_ROI
    }
}

# Risk Management
MIN_EDGE_FOR_ANY_BET = 0.01           # Absolute minimum (1%)
MAX_SINGLE_BET_SIZE = 0.05            # 5% of bankroll max
KELLY_FRACTION = 0.25                 # Quarter Kelly default

# Commission & Juice
KALSHI_COMMISSION = 0.048             # 4.8% on winnings
VIG_REMOVAL_REQUIRED = True           # Always remove vig before calculating edge

# Feature Set
FEATURE_SET = 'matchup_optimized_24'
FEATURE_VERSION = 'v4'

# Training Configuration
TRAIN_CUTOFF = '2024-10-01'          # Train on pre-2024, test on 2024-25
VALIDATION_METHOD = 'time_series_split'
CV_FOLDS = 5

# Model Performance Thresholds
MIN_ACCEPTABLE_LOGLOSS = 0.650
TARGET_LOGLOSS = 0.620               # "Monster" model threshold
MIN_ACCEPTABLE_AUC = 0.680

# Deployment Status
STRATEGY_STATUS = 'LOCKED'           # LOCKED, TESTING, or DEVELOPMENT
STRATEGY_VERSION = '1.0'
LAST_UPDATED = '2025-12-15'
