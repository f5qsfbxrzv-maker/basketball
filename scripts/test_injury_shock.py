"""
Test the new advanced injury features (shock + star binary)
"""
import sys
sys.path.append('src')

from features.feature_calculator_v5 import FeatureCalculatorV5
import pandas as pd

# Initialize calculator
print("Initializing FeatureCalculatorV5...")
calc = FeatureCalculatorV5(db_path='data/nba_betting_data.db')

# Test on a recent game (need actual game from your database)
print("\nCalculating features for test game...")
try:
    features = calc.calculate_game_features(
        home_team='LAL',
        away_team='GSW',
        current_date='2024-12-01',
        game_date='2024-12-01'  # For historical mode
    )
    
    print("\n" + "="*60)
    print("INJURY FEATURES COMPARISON")
    print("="*60)
    
    print("\n📊 TRADITIONAL FEATURES:")
    print(f"  injury_impact_diff:  {features.get('injury_impact_diff', 'MISSING'):.3f}")
    print(f"  injury_impact_abs:   {features.get('injury_impact_abs', 'MISSING'):.3f}")
    
    print("\n🔥 SHOCK FEATURES (New News vs EWMA):")
    print(f"  injury_shock_home:   {features.get('injury_shock_home', 'MISSING'):.3f}")
    print(f"  injury_shock_away:   {features.get('injury_shock_away', 'MISSING'):.3f}")
    print(f"  injury_shock_diff:   {features.get('injury_shock_diff', 'MISSING'):.3f}")
    
    print("\n⭐ STAR BINARY FLAGS:")
    print(f"  home_star_missing:   {features.get('home_star_missing', 'MISSING')}")
    print(f"  away_star_missing:   {features.get('away_star_missing', 'MISSING')}")
    print(f"  star_mismatch:       {features.get('star_mismatch', 'MISSING')}")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    
    shock_diff = features.get('injury_shock_diff', 0)
    if shock_diff > 2.0:
        print("🚨 HOME team has NEW injury crisis (worse than rolling average)")
    elif shock_diff < -2.0:
        print("🚨 AWAY team has NEW injury crisis")
    else:
        print("✅ No new injury shocks (injuries already priced into EWMA)")
    
    if features.get('home_star_missing', 0) == 1:
        print("⭐ HOME team missing elite player (PIE >= 4.0)")
    if features.get('away_star_missing', 0) == 1:
        print("⭐ AWAY team missing elite player (PIE >= 4.0)")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete!")
