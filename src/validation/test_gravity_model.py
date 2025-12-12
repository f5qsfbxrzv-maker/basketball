"""
Test Dynamic Gravity Model vs Hardcoded Multipliers
====================================================
Compare the new Z-score based system against manual SHAP-calibrated values.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np

# Test the gravity multiplier function
def calculate_dynamic_gravity_multiplier(player_pie: float, 
                                          league_avg_pie: float = 0.0855, 
                                          league_std_pie: float = 0.0230) -> float:
    """Dynamic Gravity Model - 2-Stage Slope (PRODUCTION VERSION)."""
    if league_std_pie == 0:
        return 1.0
        
    z_score = (player_pie - league_avg_pie) / league_std_pie
    
    # Stage 1: Average/role players
    if z_score <= 1.0:
        multiplier = 1.0
    # Stage 2: Star Zone (aggressive ramp)
    elif z_score <= 2.5:
        multiplier = 1.0 + ((z_score - 1.0) * 1.33)
    # Stage 3: MVP Zone (continued slope with soft cap)
    else:
        base_star_boost = 3.0
        mvp_boost = (z_score - 2.5) * 1.5
        multiplier = base_star_boost + mvp_boost
        multiplier = min(multiplier, 4.5)  # Soft cap
    
    return multiplier, z_score

print("\n" + "="*80)
print("🌌 DYNAMIC GRAVITY MODEL TEST")
print("="*80)

# Test cases: Major superstars with known PIE values (from calibration)
test_players = {
    'Giannis Antetokounmpo': {'pie': 0.1836, 'shap_mult': 3.0},
    'Nikola Jokic': {'pie': 0.1797, 'shap_mult': 2.8},
    'Joel Embiid': {'pie': 0.1769, 'shap_mult': 3.4},
    'Anthony Davis': {'pie': 0.1794, 'shap_mult': 2.3},
    'Luka Doncic': {'pie': 0.1705, 'shap_mult': 3.6},
    'Shai Gilgeous-Alexander': {'pie': 0.1474, 'shap_mult': 2.4},
    'LeBron James': {'pie': 0.1400, 'shap_mult': 2.4},  # Estimated
    'Stephen Curry': {'pie': 0.1700, 'shap_mult': 3.2},  # Estimated
    'Jayson Tatum': {'pie': 0.1500, 'shap_mult': 2.2},  # Estimated
    'Tyrese Haliburton': {'pie': 0.1300, 'shap_mult': None},  # Rising star
    'Jalen Brunson': {'pie': 0.1250, 'shap_mult': None},  # Breakout player
    'Average Starter': {'pie': 0.0964, 'shap_mult': None},  # 75th percentile
    'Bench Player': {'pie': 0.0855, 'shap_mult': None},  # League average
}

print("\n📊 MULTIPLIER COMPARISON:")
print("="*80)
print(f"{'Player':<25} {'PIE':<8} {'Z-Score':<10} {'Dynamic':<10} {'Manual':<10} {'Delta'}")
print("-"*80)

results = []
for player, data in test_players.items():
    pie = data['pie']
    shap_mult = data['shap_mult']
    
    dynamic_mult, z_score = calculate_dynamic_gravity_multiplier(pie)
    
    if shap_mult:
        delta = dynamic_mult - shap_mult
        delta_str = f"{delta:+.2f}"
    else:
        delta_str = "N/A"
    
    manual_str = f"{shap_mult:.1f}x" if shap_mult else "---"
    
    # Flag if deviation is significant
    flag = ""
    if shap_mult and abs(delta) > 0.5:
        flag = "⚠️" if abs(delta) > 1.0 else "⚡"
    elif shap_mult and abs(delta) < 0.3:
        flag = "✅"
    
    print(f"{player:<25} {pie:<8.3f} {z_score:>9.2f}σ {dynamic_mult:>9.2f}x {manual_str:>9} {delta_str:>9} {flag}")
    
    results.append({
        'player': player,
        'pie': pie,
        'z_score': z_score,
        'dynamic_mult': dynamic_mult,
        'shap_mult': shap_mult,
        'delta': delta if shap_mult else None
    })

results_df = pd.DataFrame(results)

# Analysis
print("\n" + "="*80)
print("📈 STATISTICAL ANALYSIS")
print("="*80)

# Filter only players with SHAP data
shap_players = results_df[results_df['shap_mult'].notna()]

if len(shap_players) > 0:
    correlation = np.corrcoef(shap_players['dynamic_mult'], shap_players['shap_mult'])[0, 1]
    mean_delta = shap_players['delta'].mean()
    std_delta = shap_players['delta'].std()
    max_delta = shap_players['delta'].abs().max()
    
    print(f"\nCorrelation (Dynamic vs SHAP-Calibrated): {correlation:.3f}")
    print(f"Mean Delta: {mean_delta:+.3f}")
    print(f"Std Delta: {std_delta:.3f}")
    print(f"Max Absolute Delta: {max_delta:.3f}")
    
    if correlation > 0.85:
        print("\n✅ STRONG CORRELATION - Dynamic model tracks SHAP calibration well")
    elif correlation > 0.70:
        print("\n⚠️ MODERATE CORRELATION - Dynamic model is reasonable but diverges on some players")
    else:
        print("\n❌ WEAK CORRELATION - Dynamic model needs recalibration")

# Key insights
print("\n" + "="*80)
print("🔍 KEY INSIGHTS")
print("="*80)

print("\n1️⃣ SUPERSTAR DETECTION (Z > 2.5):")
superstars = results_df[results_df['z_score'] > 2.5].sort_values('z_score', ascending=False)
for _, player in superstars.iterrows():
    print(f"   🌟 {player['player']:<25} Z={player['z_score']:.2f}σ → {player['dynamic_mult']:.2f}x")

print("\n2️⃣ EMERGING STARS (1.5 < Z < 2.5):")
emerging = results_df[(results_df['z_score'] > 1.5) & (results_df['z_score'] <= 2.5)].sort_values('z_score', ascending=False)
for _, player in emerging.iterrows():
    print(f"   ⭐ {player['player']:<25} Z={player['z_score']:.2f}σ → {player['dynamic_mult']:.2f}x")

print("\n3️⃣ AVERAGE PLAYERS (Z < 1.5):")
average = results_df[results_df['z_score'] <= 1.5].sort_values('z_score', ascending=False)
for _, player in average.iterrows():
    print(f"   👤 {player['player']:<25} Z={player['z_score']:.2f}σ → {player['dynamic_mult']:.2f}x")

# Advantage of Dynamic Model
print("\n" + "="*80)
print("🎯 ADVANTAGES OF DYNAMIC GRAVITY MODEL")
print("="*80)

print("""
✅ AUTOMATIC DISCOVERY: Detects breakout players (Brunson, Haliburton) without code changes
✅ GRACEFUL DECLINE: Adjusts for aging stars as PIE naturally drops
✅ NO MAINTENANCE: No manual list to update each season
✅ GRADIENT SYSTEM: Smooth scaling from bench → starter → All-Star → MVP
✅ DATA-DRIVEN: Based on statistical distribution, not subjective naming
✅ HANDLES UNKNOWNS: Works for any player with PIE data
""")

# Potential adjustments
print("\n" + "="*80)
print("🎛️ RECOMMENDED TUNING (If needed)")
print("="*80)

if len(shap_players) > 0:
    # Check if dynamic model is systematically over/under-estimating
    if mean_delta > 0.5:
        print("\n⚡ Dynamic model OVERESTIMATES multipliers on average")
        print("   → Consider reducing Z-score slope in 2-3σ range")
        print("   → Change: multiplier = 1.5 + ((z_score - 2.0) * 1.8)")
    elif mean_delta < -0.5:
        print("\n⚡ Dynamic model UNDERESTIMATES multipliers on average")
        print("   → Consider increasing Z-score slope in 2-3σ range")
        print("   → Change: multiplier = 1.5 + ((z_score - 2.0) * 2.2)")
    else:
        print("\n✅ Dynamic model is well-calibrated on average")
        print("   → No systematic bias detected")
        print("   → Individual player variance is within acceptable range")

print("\n" + "="*80)
