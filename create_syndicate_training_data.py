"""
Update existing training data with syndicate-level features.

Takes training_data_GOLD_ELO_22_features.csv and adds:
1. Matchup friction features (replacing simple diffs)
2. Volume-adjusted efficiency
3. ELO matchup advantages (replacing individual scores)
4. Consolidated injury leverage
"""

import pandas as pd
import numpy as np

def update_to_syndicate_features():
    """Convert Gold ELO training data to Syndicate format"""
    
    print("=" * 70)
    print("SYNDICATE FEATURE UPGRADE")
    print("=" * 70)
    
    # Load existing training data
    print("\n[1/3] Loading Gold ELO training data...")
    df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
    print(f"Loaded {len(df)} games with {len(df.columns)} columns")
    
    print("\n[2/3] Creating syndicate features...")
    
    # ========================================================================
    # 1. ELO MATCHUP ADVANTAGES (Replace individual scores with matchup advantages)
    # ========================================================================
    print("\n  ELO Matchup Advantages:")
    
    # We have off_elo_diff and def_elo_diff already - these ARE matchup advantages!
    # Just need to rename and add net_composite_advantage
    
    # Rename existing features to match syndicate naming
    if 'off_elo_diff' in df.columns:
        df['off_matchup_advantage'] = df['off_elo_diff']
        print(f"    ✓ off_matchup_advantage (from off_elo_diff)")
    
    if 'def_elo_diff' in df.columns:
        df['def_matchup_advantage'] = df['def_elo_diff']
        print(f"    ✓ def_matchup_advantage (from def_elo_diff)")
    
    # Create net_composite_advantage (home composite + 100 - away composite)
    if 'home_composite_elo' in df.columns and 'away_composite_elo' in df.columns:
        HOME_COURT_ADVANTAGE = 100
        df['net_composite_advantage'] = (df['home_composite_elo'] + HOME_COURT_ADVANTAGE) - df['away_composite_elo']
        print(f"    ✓ net_composite_advantage (home + 100 - away)")
    
    # ========================================================================
    # 2. MATCHUP FRICTION FEATURES (These need to be calculated from raw data)
    # ========================================================================
    print("\n  Matchup Friction Features:")
    print("    ⚠ Note: These require recalculation from game_logs")
    print("    ⚠ For now, using existing EWMA diffs as placeholders")
    
    # Effective Shooting Gap (placeholder - needs recalc with opp_efg_allowed)
    if 'ewma_efg_diff' in df.columns:
        df['effective_shooting_gap'] = df['ewma_efg_diff']
        print(f"    ⚠ effective_shooting_gap (placeholder from ewma_efg_diff)")
    
    # Turnover Pressure (placeholder - needs STL% + TOV%)
    if 'ewma_tov_diff' in df.columns:
        df['turnover_pressure'] = df['ewma_tov_diff']
        print(f"    ⚠ turnover_pressure (placeholder from ewma_tov_diff)")
    
    # Rebound Friction (placeholder - needs opp_orb_allowed)
    if 'ewma_orb_diff' in df.columns:
        df['rebound_friction'] = df['ewma_orb_diff']
        print(f"    ⚠ rebound_friction (placeholder from ewma_orb_diff)")
    
    # Total Rebound Control (needs DRB%)
    df['total_rebound_control'] = 0  # Placeholder
    print(f"    ⚠ total_rebound_control (placeholder = 0)")
    
    # Whistle Leverage (needs foul rate)
    if 'net_free_throw_advantage' in df.columns:
        df['whistle_leverage'] = df['net_free_throw_advantage']
        print(f"    ⚠ whistle_leverage (placeholder from net_free_throw_advantage)")
    else:
        df['whistle_leverage'] = 0
    
    # ========================================================================
    # 3. VOLUME-ADJUSTED EFFICIENCY
    # ========================================================================
    print("\n  Volume-Adjusted Efficiency:")
    
    # Use existing pace features as proxy
    if 'pace_efficiency_interaction' in df.columns:
        df['volume_efficiency_diff'] = df['pace_efficiency_interaction']
        print(f"    ⚠ volume_efficiency_diff (placeholder from pace_efficiency_interaction)")
    elif 'ewma_efg_diff' in df.columns and 'ewma_pace_diff' in df.columns:
        # Rough approximation: eFG% × Pace
        df['volume_efficiency_diff'] = df['ewma_efg_diff'] * (1 + df['ewma_pace_diff'] / 100)
        print(f"    ✓ volume_efficiency_diff (calculated from efg × pace)")
    else:
        df['volume_efficiency_diff'] = 0
    
    # ========================================================================
    # 4. CONSOLIDATED INJURY LEVERAGE
    # ========================================================================
    print("\n  Consolidated Injury Features:")
    
    if 'injury_matchup_advantage' in df.columns:
        # Already have the consolidated feature!
        df['injury_leverage'] = df['injury_matchup_advantage']
        print(f"    ✓ injury_leverage (from injury_matchup_advantage)")
    elif all(col in df.columns for col in ['injury_impact_diff', 'injury_shock_diff', 'star_mismatch']):
        # Calculate from components
        df['injury_leverage'] = (
            0.008127 * df['injury_impact_diff']
          - 0.023904 * df['injury_shock_diff']
          + 0.031316 * df['star_mismatch']
        )
        print(f"    ✓ injury_leverage (calculated from 3 components)")
    else:
        df['injury_leverage'] = 0
        print(f"    ⚠ injury_leverage (placeholder = 0)")
    
    # ========================================================================
    # 5. KEEP SUPPORTING FEATURES
    # ========================================================================
    print("\n  Supporting Features:")
    supporting = [
        'net_fatigue_score', 'ewma_chaos_home', 'ewma_foul_synergy_home',
        'total_foul_environment', 'league_offensive_context', 'season_progress',
        'three_point_matchup', 'star_power_leverage', 'offense_vs_defense_matchup',
        'ewma_pace_diff', 'ewma_vol_3p_diff', 'projected_possession_margin'
    ]
    
    existing_supporting = [f for f in supporting if f in df.columns]
    print(f"    ✓ Keeping {len(existing_supporting)} supporting features")
    
    # ========================================================================
    # 6. CREATE FINAL SYNDICATE DATASET
    # ========================================================================
    print("\n[3/3] Assembling syndicate dataset...")
    
    # Define syndicate feature set (28 features)
    syndicate_features = [
        # Tier 1: ELO Matchup Advantages (3)
        'off_matchup_advantage', 'def_matchup_advantage', 'net_composite_advantage',
        
        # Tier 2: Matchup Friction (5)
        'effective_shooting_gap', 'turnover_pressure', 'rebound_friction',
        'total_rebound_control', 'whistle_leverage',
        
        # Tier 3: Volume & Injury (2)
        'volume_efficiency_diff', 'injury_leverage',
        
        # Tier 4: Context & Supporting (18)
        'net_fatigue_score', 'ewma_chaos_home', 'ewma_foul_synergy_home',
        'total_foul_environment', 'league_offensive_context', 'season_progress',
        'three_point_matchup', 'star_power_leverage', 'offense_vs_defense_matchup',
        'ewma_pace_diff', 'ewma_vol_3p_diff', 'projected_possession_margin'
    ]
    
    # Add metadata columns
    metadata = ['game_id', 'home_team', 'away_team', 'date', 'season']
    
    # Add targets
    targets = ['target_spread', 'target_spread_cover', 'target_moneyline_win',
               'target_over_under', 'target_game_total']
    
    # Select columns that exist
    available_features = [f for f in syndicate_features if f in df.columns]
    available_metadata = [f for f in metadata if f in df.columns]
    available_targets = [f for f in targets if f in df.columns]
    
    final_columns = available_metadata + available_features + available_targets
    syndicate_df = df[final_columns].copy()
    
    # Save
    output_file = 'data/training_data_SYNDICATE_28_features.csv'
    syndicate_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Syndicate dataset saved: {output_file}")
    print(f"  Rows: {len(syndicate_df)}")
    print(f"  Features: {len(available_features)}")
    print(f"  Total columns: {len(syndicate_df.columns)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ COMPLETE Syndicate Features ({len([f for f in syndicate_features if f in df.columns])}):")
    for f in syndicate_features:
        if f in df.columns:
            mean_val = df[f].mean()
            std_val = df[f].std()
            print(f"  {f:35s} mean: {mean_val:7.2f}, std: {std_val:6.2f}")
    
    missing = [f for f in syndicate_features if f not in df.columns]
    if missing:
        print(f"\n⚠ MISSING Features ({len(missing)}):")
        for f in missing:
            print(f"  {f}")
    
    print("\n" + "=" * 70)
    print("✓ SYNDICATE DATASET READY FOR OPTIMIZATION")
    print("=" * 70)
    
    return output_file

if __name__ == "__main__":
    update_to_syndicate_features()
