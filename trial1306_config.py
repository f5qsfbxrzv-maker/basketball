# Trial 1306 Model Configuration
# Production model with 22 features and optimized 2%/10% thresholds

# Model Paths
TRIAL1306_MODEL_PATH = "models/xgboost_22features_trial1306_20251215_212306.json"
TRIAL1306_PARAMS_PATH = "models/trial1306_params_20251215_212306.json"
TRIAL1306_CONFIG_PATH = "model_config.json"

# 22 Features (in correct order for Trial 1306)
TRIAL1306_FEATURES = [
    'home_composite_elo',
    'away_composite_elo',
    'off_elo_diff',
    'def_elo_diff',
    'net_fatigue_score',
    'ewma_efg_diff',
    'ewma_pace_diff',
    'ewma_tov_diff',
    'ewma_orb_diff',
    'ewma_vol_3p_diff',
    'injury_matchup_advantage',
    'ewma_chaos_home',
    'ewma_foul_synergy_home',
    'total_foul_environment',
    'league_offensive_context',
    'season_progress',
    'pace_efficiency_interaction',
    'projected_possession_margin',
    'three_point_matchup',
    'net_free_throw_advantage',
    'star_power_leverage',
    'offense_vs_defense_matchup'
]

# Optimized Betting Thresholds (from grid search)
TRIAL1306_FAVORITE_EDGE = 0.02  # 2%
TRIAL1306_UNDERDOG_EDGE = 0.10  # 10%
TRIAL1306_ODDS_SPLIT = 2.00     # Decimal odds threshold
TRIAL1306_KELLY_FRACTION = 0.25  # Quarter Kelly

# Model Performance Metrics
TRIAL1306_VALIDATION_LOG_LOSS = 0.6222
TRIAL1306_TRAINING_AUC = 0.7342
TRIAL1306_TRAINING_ACCURACY = 0.6769
TRIAL1306_BACKTEST_ROI = 0.497  # 49.7% (optimal strategy)
TRIAL1306_COMBINED_ROI = 0.1645  # 16.45% (combined backtest)

# Model Info
TRIAL1306_VERSION = "1.0.0"
TRIAL1306_TRAINING_GAMES = 12205
TRIAL1306_TRAINING_PERIOD = "2015-2024"
TRIAL1306_RELEASE_DATE = "2024-12-15"

# Feature Importance (Top 5)
TRIAL1306_TOP_FEATURES = [
    ("off_elo_diff", 61.3),
    ("away_composite_elo", 28.7),
    ("home_composite_elo", 27.6),
    ("ewma_efg_diff", 9.4),
    ("net_fatigue_score", 9.1)
]
