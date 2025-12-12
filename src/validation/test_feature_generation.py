"""
Test Feature Generation with Whitelist
Verify that all 31 whitelisted features are generated correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST
from datetime import datetime

def test_feature_generation():
    """Test that feature calculator generates all 31 whitelisted features"""
    
    print("=" * 80)
    print("FEATURE GENERATION TEST")
    print("=" * 80)
    
    # Initialize feature calculator
    print("\n1. Initializing feature calculator...")
    calc = FeatureCalculatorV5()
    
    # Test game: GSW @ CLE (Nov 15, 2024)
    test_game = {
        'home_team': 'CLE',
        'away_team': 'GSW',
        'game_date': '2024-11-15'
    }
    
    print(f"\n2. Test game: {test_game['away_team']} @ {test_game['home_team']} on {test_game['game_date']}")
    
    # Calculate features
    print("\n3. Calculating features...")
    features = calc.calculate_game_features(
        home_team=test_game['home_team'],
        away_team=test_game['away_team'],
        game_date=test_game['game_date']
    )
    
    print(f"\n4. Features generated: {len(features)}")
    print(f"   Whitelist target: {len(FEATURE_WHITELIST)}")
    
    # Check which whitelisted features are present
    print("\n5. Whitelist coverage:")
    missing = []
    present = []
    
    for feature in FEATURE_WHITELIST:
        if feature in features:
            present.append(feature)
        else:
            missing.append(feature)
    
    print(f"   ✅ Present: {len(present)}/{len(FEATURE_WHITELIST)}")
    print(f"   ❌ Missing: {len(missing)}/{len(FEATURE_WHITELIST)}")
    
    if missing:
        print("\n   Missing features:")
        for feat in missing:
            print(f"      - {feat}")
    
    # Show generated features by category
    print("\n6. Generated features by category:")
    
    categories = {
        'Mandatory (12)': ['injury_impact_diff', 'injury_impact_abs', 'injury_elo_interaction',
                          'rest_advantage', 'fatigue_mismatch', 'home_rest_days', 'away_rest_days',
                          'home_back_to_back', 'away_back_to_back', 'home_3in4', 'away_3in4',
                          'altitude_game'],
        'ELO Engine (3)': ['home_composite_elo', 'off_elo_diff', 'def_elo_diff'],
        'Foul Synergy (3)': ['ewma_foul_synergy_home', 'ewma_foul_synergy_away', 'total_foul_environment'],
        'EWMA Diffs (5)': ['ewma_efg_diff', 'ewma_tov_diff', 'ewma_orb_diff', 'ewma_pace_diff', 'ewma_vol_3p_diff'],
        'Baselines (6)': ['home_ewma_3p_pct', 'away_ewma_3p_pct', 'away_ewma_tov_pct',
                         'home_orb', 'away_orb', 'away_ewma_fta_rate'],
        'Chaos Metrics (2)': ['ewma_chaos_home', 'ewma_net_chaos']
    }
    
    for category, feature_list in categories.items():
        count = sum(1 for f in feature_list if f in features)
        print(f"\n   {category}: {count}/{len(feature_list)}")
        for feat in feature_list:
            if feat in features:
                print(f"      ✅ {feat}: {features[feat]:.4f}")
            else:
                print(f"      ❌ {feat}: MISSING")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if len(present) == len(FEATURE_WHITELIST):
        print(f"✅ SUCCESS: All {len(FEATURE_WHITELIST)} whitelisted features generated!")
    else:
        print(f"⚠️  INCOMPLETE: {len(present)}/{len(FEATURE_WHITELIST)} features generated")
        print(f"   Missing {len(missing)} features - see list above")
    
    return features, present, missing


if __name__ == "__main__":
    features, present, missing = test_feature_generation()
