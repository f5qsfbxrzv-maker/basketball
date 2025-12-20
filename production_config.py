"""
PRODUCTION CONFIGURATION - Variant D (Trial #245 Optimized)
===========================================================
Ferrari Engine Specs + Calibrated Betting Thresholds
"""

# ==========================================
# 🏎️ FERRARI ENGINE: VARIANT D FEATURES
# ==========================================

ACTIVE_FEATURES = [
    'home_composite_elo', 
    'off_elo_diff', 
    'def_elo_diff',
    'projected_possession_margin', 
    'ewma_pace_diff', 
    'net_fatigue_score',
    'ewma_efg_diff', 
    'ewma_vol_3p_diff', 
    'three_point_matchup',
    'ewma_chaos_home', 
    'injury_impact_diff', 
    'injury_shock_diff',
    'star_power_leverage', 
    'season_progress', 
    'league_offensive_context',
    'total_foul_environment', 
    'net_free_throw_advantage',
    'pace_efficiency_interaction', 
    'offense_vs_defense_matchup'
]

# ==========================================
# ⚙️ TRIAL #245: OPTUNA-OPTIMIZED HYPERPARAMETERS
# ==========================================

XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'learning_rate': 0.066994,
    'max_depth': 2,
    'min_child_weight': 12,
    'gamma': 2.025432,
    'subsample': 0.630135,
    'colsample_bytree': 0.903401,
    'colsample_bylevel': 0.959686,
    'reg_alpha': 1.081072,
    'reg_lambda': 5.821363,
}

N_ESTIMATORS = 4529

# ==========================================
# 🎯 BETTING THRESHOLDS (CALIBRATED)
# ==========================================
# NOTE: These will be determined AFTER calibration is trained
# Initial conservative values - update after running train_isotonic_calibrator.py

MIN_EDGE_FAVORITE = 0.015   # 1.5% minimum edge for favorites (conservative)
MIN_EDGE_UNDERDOG = 0.005   # 0.5% minimum edge for underdogs (underdog hunter mode)

# ==========================================
# 📊 CONFIDENCE THRESHOLDS
# ==========================================

CONFIDENCE_THRESHOLD = 0.52  # Minimum model confidence to consider a bet

# ==========================================
# 💰 BANKROLL MANAGEMENT
# ==========================================

KELLY_FRACTION = 0.25        # Quarter Kelly (conservative)
MAX_BET_SIZE = 0.05          # Max 5% of bankroll per bet
MIN_BET_SIZE = 0.01          # Min 1% of bankroll per bet

# ==========================================
# 📁 FILE PATHS
# ==========================================

DATA_PATH = 'data/training_data_GOLD_ELO_22_features.csv'
CALIBRATOR_PATH = 'models/nba_isotonic_calibrator.joblib'
MODEL_PATH = 'models/variant_d_optimized.json'

# ==========================================
# 📝 METADATA
# ==========================================

MODEL_VERSION = "Variant_D_Trial245_Calibrated"
TRAINING_DATE = "2024-12-19"
FEATURES_COUNT = len(ACTIVE_FEATURES)
VIF_MAX = 2.34  # All features below 2.5 VIF threshold
