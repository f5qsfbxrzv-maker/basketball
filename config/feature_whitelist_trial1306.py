# ==============================================================================
# 🎯 FEATURE WHITELIST: TRIAL 1306 PRODUCTION (22 Features)
# ==============================================================================
# Updated for Trial 1306 model (49.7% ROI, 2% fav / 10% dog thresholds)
# This whitelist ensures only the 22 required features are calculated/passed
# ==============================================================================

def get_whitelist():
    """
    Returns the TRIAL 1306 feature list (22 features optimized via grid search).
    Used by src/features/feature_calculator_v5.py
    """
    return [
        # === ELO SYSTEM (4 features) ===
        'home_composite_elo',
        'away_composite_elo',
        'off_elo_diff',
        'def_elo_diff',
        
        # === FATIGUE & REST (1 feature) ===
        'net_fatigue_score',
        
        # === EWMA TRENDS (6 features) ===
        'ewma_efg_diff',
        'ewma_pace_diff',
        'ewma_tov_diff',
        'ewma_orb_diff',
        'ewma_vol_3p_diff',
        'ewma_chaos_home',
        
        # === INJURY (1 composite feature) ===
        'injury_matchup_advantage',
        
        # === ADVANCED METRICS (11 features) ===
        'ewma_foul_synergy_home',
        'total_foul_environment',
        'league_offensive_context',
        'season_progress',
        'pace_efficiency_interaction',
        'projected_possession_margin',
        'three_point_matchup',
        'net_free_throw_advantage',
        'star_power_leverage',
        'offense_vs_defense_matchup',
    ]

# Expose as FEATURE_WHITELIST for backwards compatibility
FEATURE_WHITELIST = get_whitelist()

# ==============================================================================
# DIAGNOSTIC PRINT (When run directly)
# ==============================================================================
if __name__ == "__main__":
    features = get_whitelist()
    print(f"✅ TRIAL 1306 FEATURE WHITELIST LOADED")
    print(f"📊 Total Count: {len(features)} features")
    print("-" * 50)
    print("\nTrial 1306 Features:")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")
    print(f"\n✅ All 22 features defined for Trial 1306")
