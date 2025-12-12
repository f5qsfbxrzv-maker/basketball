# ==============================================================================
# 🎯 FEATURE WHITELIST: THE "SNIPER" CONFIGURATION (v3 - ANTI-CANNIBALIZATION)
# ==============================================================================
# Reduction: ~107 Features -> 36 High-Signal Features
# 
# RECENT OPTIMIZATIONS (v3):
# - Added INJURY SHOCK features (new vs old news detection)
# - Added STAR BINARY flags (clear signal for superstars)
# - Prevents EWMA cannibalization (where rolling stats absorb injury signal)
#
# PRIOR OPTIMIZATIONS (v2):
# - Removed injury_elo_interaction (99.7% correlated with injury_impact_diff)
# - Fixed 3-in-4 logic (was flagging 95% of games, now ~15% as expected)
# ==============================================================================

def get_whitelist():
    """
    Returns the strict list of columns to keep during feature extraction.
    Used by src/features/feature_calculator_v5.py
    """
    return [
        # === 1. MANDATORY CONTEXT (The "Do Not Fly" List) ===
        
        # --- TRADITIONAL INJURY FEATURES (Baseline) ---
        'injury_impact_diff',       # (Home Inj Impact - Away Inj Impact)
        'injury_impact_abs',        # Total star power missing (Game quality)
        
        # --- ADVANCED INJURY FEATURES (Anti-Cannibalization) ---
        # Problem: After 2-3 games, EWMA stats "absorb" injury signal
        # Solution: Track injury SHOCK (new vs old news) + star binary flags
        'injury_shock_home',        # Today's injury - EWMA baseline (new news)
        'injury_shock_away',        # Away injury shock
        'injury_shock_diff',        # Net injury shock (home - away)
        'home_star_missing',        # BINARY: Elite player out (PIE >= 4.0)
        'away_star_missing',        # BINARY: Elite player out
        'star_mismatch',            # Net star advantage (home - away)
        
        # --- REST & FATIGUE ---
        'rest_advantage',           # (Home Rest - Away Rest)
        'fatigue_mismatch',         # Binary: Major rest difference?
        'home_rest_days',           # Absolute rest
        'away_rest_days',           # Absolute rest
        'home_back_to_back',        # Explicit fatigue flag
        'away_back_to_back',        # Explicit fatigue flag
        'home_3in4',                # Schedule compression (Deep fatigue)
        'away_3in4',                # Schedule compression
        'altitude_game',            # Denver/Utah advantage check

        # === 2. THE ELO ENGINE (Long-Term Strength) ===
        # We use ELO instead of Net Ratings because ELO accounts for opponent quality.
        'home_composite_elo',       # Sets the "Game Tier" (Clash of Titans vs Tank Bowl)
        'off_elo_diff',             # Offensive mismatch (Home Off - Away Off)
        'def_elo_diff',             # Defensive mismatch (Home Def - Away Def)

        # === 3. FOUL SYNERGY (Your #1 SHAP Predictor) ===
        # Captures the "Referee/Style" game within the game.
        'ewma_foul_synergy_home',   # Home team's recent foul tendency
        'ewma_foul_synergy_away',   # Away team's recent foul tendency
        'total_foul_environment',   # Predicted chaos/stoppage level

        # === 4. TECHNICAL MATCHUPS (EWMA Diffs) ===
        # "Who is playing better BASKETBALL right now?"
        'ewma_efg_diff',            # Shooting Efficiency Diff (The biggest factor)
        'ewma_tov_diff',            # Turnover Control Diff (Possession battle)
        'ewma_orb_diff',            # Rebounding Diff (Second chances)
        'ewma_pace_diff',           # Pace Preference Clash (Fast vs Slow)
        'ewma_vol_3p_diff',         # 3-Point Volume Diff (Math problem)

        # === 5. KEY ABSOLUTES (Baselines) ===
        # Sometimes raw values matter (e.g., specific 3P% thresholds)
        'home_ewma_3p_pct',         # Home shooting form
        'away_ewma_3p_pct',         # Away shooting form
        'away_ewma_tov_pct',        # Specific weakness check for away teams
        'home_orb',                 # Home rebounding baseline
        'away_orb',                 # Away rebounding baseline
        'away_ewma_fta_rate',       # Does away team get to the line?

        # === 6. CHAOS METRICS ===
        # Measures variance/volatility.
        'ewma_chaos_home',          # Is the home team consistent or erratic?
        'ewma_net_chaos',           # Matchup volatility
    ]

# Expose as FEATURE_WHITELIST for backwards compatibility
FEATURE_WHITELIST = get_whitelist()

# ==============================================================================
# DIAGNOSTIC PRINT (When run directly)
# ==============================================================================
if __name__ == "__main__":
    features = get_whitelist()
    print(f"✅ FEATURE WHITELIST LOADED (v3 - Anti-Cannibalization)")
    print(f"📊 Total Count: {len(features)}")
    print("-" * 50)
    
    # Group by category for readability
    injury_feats = [f for f in features if 'injury' in f or 'star' in f]
    rest_feats = [f for f in features if 'rest' in f or 'back_to_back' in f or '3in4' in f or 'fatigue' in f or 'altitude' in f]
    elo_feats = [f for f in features if 'elo' in f]
    foul_feats = [f for f in features if 'foul' in f]
    ewma_feats = [f for f in features if 'ewma' in f and 'diff' in f]
    
    print(f"\nInjury Features ({len(injury_feats)}):")
    for f in injury_feats:
        print(f"  • {f}")
    
    print(f"\nRest & Fatigue ({len(rest_feats)}):")
    for f in rest_feats:
        print(f"  • {f}")
    
    print(f"\nELO Features ({len(elo_feats)}):")
    for f in elo_feats:
        print(f"  • {f}")
    
    print(f"\nFoul Features ({len(foul_feats)}):")
    for f in foul_feats:
        print(f"  • {f}")
    
    print(f"\nEWMA Matchups ({len(ewma_feats)}):")
    for f in ewma_feats:
        print(f"  • {f}")
