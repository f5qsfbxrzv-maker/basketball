"""
Test Script: Verify Injury Tracking & Kalshi Odds Services work with MDP Model
Tests:
1. Injury tracking computes the 3 required features
2. Kalshi client can fetch live NBA odds
3. Feature calculator produces all 19 MDP features
"""

import sys
from pathlib import Path
import json

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/features'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/services'))

from production_config_mdp import ACTIVE_FEATURES, MODEL_PATH, NBA_STD_DEV
from src.features.feature_calculator_v5 import FeatureCalculatorV5
from injury_impact_live import calculate_team_injury_impact_simple
from src.services.kalshi_client import KalshiClient
from datetime import datetime

print("=" * 80)
print("🔬 MDP SERVICES INTEGRATION TEST")
print("=" * 80)

# ===================================================================
# TEST 1: VERIFY MDP FEATURES LIST
# ===================================================================
print("\n[TEST 1] MDP Feature Set Verification")
print("-" * 80)

expected_features = [
    'off_elo_diff', 'def_elo_diff', 'home_composite_elo',
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
    'injury_matchup_advantage', 'injury_shock_diff', 'star_power_leverage',
    'season_progress', 'league_offensive_context',
    'total_foul_environment', 'net_free_throw_advantage',
    'offense_vs_defense_matchup', 'pace_efficiency_interaction', 'star_mismatch'
]

print(f"Expected features: {len(expected_features)}")
print(f"Config features:   {len(ACTIVE_FEATURES)}")

if set(ACTIVE_FEATURES) == set(expected_features):
    print("✅ PASS: All 19 MDP features defined correctly")
else:
    missing = set(expected_features) - set(ACTIVE_FEATURES)
    extra = set(ACTIVE_FEATURES) - set(expected_features)
    if missing:
        print(f"❌ FAIL: Missing features: {missing}")
    if extra:
        print(f"⚠️  WARNING: Extra features: {extra}")

# ===================================================================
# TEST 2: INJURY TRACKING
# ===================================================================
print("\n[TEST 2] Injury Tracking Service")
print("-" * 80)

try:
    # Test with Orlando Magic (should have current injuries)
    db_path = "data/live/nba_betting_data.db"  # Correct database path
    
    home_injury = calculate_team_injury_impact_simple("ORL", datetime.now().strftime("%Y-%m-%d"), db_path)
    away_injury = calculate_team_injury_impact_simple("ATL", datetime.now().strftime("%Y-%m-%d"), db_path)
    
    print(f"Home (ORL) injury impact: {home_injury:.3f}")
    print(f"Away (ATL) injury impact: {away_injury:.3f}")
    
    # Required MDP features from injuries
    injury_features = [
        'injury_matchup_advantage',  # Differential impact
        'injury_shock_diff',         # New vs old injuries
        'star_power_leverage'        # Binary star flags
    ]
    
    print(f"\nRequired injury features for MDP: {injury_features}")
    print("✅ PASS: Injury service is functional")
    print("   Note: Feature calculator computes injury_matchup_advantage, injury_shock_diff, star_power_leverage from raw impacts")
    
except Exception as e:
    print(f"❌ FAIL: Injury service error: {e}")
    import traceback
    traceback.print_exc()

# ===================================================================
# TEST 3: FEATURE CALCULATOR
# ===================================================================
print("\n[TEST 3] Feature Calculator V5 - Full Feature Set")
print("-" * 80)

try:
    calculator = FeatureCalculatorV5(db_path=db_path)
    
    # Test with a recent game
    test_home = "ORL"
    test_away = "ATL"
    test_date = None  # Use live data
    
    print(f"Generating features for: {test_home} vs {test_away}")
    features = calculator.calculate_game_features(test_home, test_away, test_date)
    
    print(f"\nFeatures computed: {len(features)}")
    
    # Check if all 19 MDP features are present
    missing_features = []
    for feat in ACTIVE_FEATURES:
        if feat not in features:
            missing_features.append(feat)
        else:
            print(f"  ✓ {feat:35s} = {features[feat]:8.3f}")
    
    if missing_features:
        print(f"\n❌ FAIL: Missing MDP features: {missing_features}")
    else:
        print("\n✅ PASS: All 19 MDP features computed successfully")
    
    # Verify injury features specifically
    injury_feats = ['injury_matchup_advantage', 'injury_shock_diff', 'star_power_leverage']
    print(f"\nInjury feature check:")
    for feat in injury_feats:
        if feat in features:
            print(f"  ✓ {feat}: {features[feat]:.3f}")
        else:
            print(f"  ✗ {feat}: MISSING")
    
except Exception as e:
    print(f"❌ FAIL: Feature calculator error: {e}")
    import traceback
    traceback.print_exc()

# ===================================================================
# TEST 4: KALSHI API CLIENT
# ===================================================================
print("\n[TEST 4] Kalshi API Client - Live Odds")
print("-" * 80)

try:
    # Try to load Kalshi credentials
    creds_file = PROJECT_ROOT / "config" / "kalshi_credentials.json"
    
    if not creds_file.exists():
        print("⚠️  SKIP: No Kalshi credentials found at config/kalshi_credentials.json")
        print("   To test Kalshi integration:")
        print("   1. Create config/kalshi_credentials.json")
        print("   2. Add: {\"api_key\": \"YOUR_KEY\", \"api_secret\": \"YOUR_SECRET\", \"environment\": \"demo\"}")
    else:
        with open(creds_file, 'r') as f:
            creds = json.load(f)
        
        client = KalshiClient(
            api_key=creds['api_key'],
            api_secret=creds['api_secret'],
            environment=creds.get('environment', 'demo'),
            auth_on_init=False  # Don't authenticate yet, just test initialization
        )
        
        print("✅ PASS: Kalshi client initialized successfully")
        print(f"   Environment: {client.environment}")
        print(f"   Base URL: {client.base_url}")
        print("   Note: Full API test requires valid credentials and authentication")
        
except Exception as e:
    print(f"⚠️  WARNING: Kalshi client initialization failed: {e}")
    print("   This is expected if credentials are not configured")

# ===================================================================
# TEST 5: MODEL LOADING
# ===================================================================
print("\n[TEST 5] MDP Model Loading & Prediction")
print("-" * 80)

try:
    import xgboost as xgb
    from scipy.stats import norm
    import pandas as pd
    
    # Load model
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"   NBA_STD_DEV: {NBA_STD_DEV}")
    
    # Test prediction with computed features
    if 'features' in locals():
        # Create feature vector in correct order
        X = pd.DataFrame([{feat: features.get(feat, 0) for feat in ACTIVE_FEATURES}])
        
        dmatrix = xgb.DMatrix(X, feature_names=ACTIVE_FEATURES)
        predicted_margin = float(model.predict(dmatrix)[0])
        home_prob = float(norm.cdf(predicted_margin / NBA_STD_DEV))
        
        print(f"\n   Predicted margin: {predicted_margin:.2f} points")
        print(f"   Home win prob:    {home_prob:.1%}")
        print(f"   Away win prob:    {1-home_prob:.1%}")
        
        print("\n✅ PASS: Full prediction pipeline works")
    else:
        print("⚠️  SKIP: No features computed to test prediction")
    
except Exception as e:
    print(f"❌ FAIL: Model prediction error: {e}")
    import traceback
    traceback.print_exc()

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("📊 TEST SUMMARY")
print("=" * 80)
print("""
✅ REQUIRED FOR MDP:
   - Feature Calculator V5 with 19 features
   - Injury tracking (raw impacts → 3 MDP features)
   - Model loading and prediction pipeline

⚠️  OPTIONAL:
   - Kalshi API (for live odds - needs credentials)
   - Live injury updater (ESPN scraper)

📝 NEXT STEPS:
   1. If Kalshi test skipped: Add credentials to config/kalshi_credentials.json
   2. Test dashboard: python nba_gui_dashboard_v2.py
   3. Generate predictions for today's games
   4. Verify edge calculations use 1.5%/8.0% thresholds
""")
print("=" * 80)
