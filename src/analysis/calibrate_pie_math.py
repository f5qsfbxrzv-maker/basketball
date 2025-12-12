"""
Calibrate PIE Distribution for Dynamic Gravity Model
=====================================================
Determines the actual Mean and Std Dev of PIE in your dataset
to ensure Z-score calculations are accurate.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import sqlite3

print("="*80)
print("📏 CALIBRATING PIE GRAVITY CONSTANTS")
print("="*80)

# Load player stats from database
db_path = 'data/live/nba_betting_data.db'

try:
    with sqlite3.connect(db_path) as conn:
        # Load all player stats with PIE values
        df = pd.read_sql_query(
            """
            SELECT player_id, player_name, team_abbreviation, season, pie, 
                   usg_pct, net_rating
            FROM player_stats
            WHERE pie IS NOT NULL AND pie > 0
            """,
            conn
        )
    print(f"✅ Loaded {len(df)} player records from database")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

print(f"\n📊 RAW DATA STATS:")
print(f"   Total players: {len(df)}")
print(f"   PIE range: {df['pie'].min():.4f} to {df['pie'].max():.4f}")
print(f"   Mean (all): {df['pie'].mean():.4f}")
print(f"   Std (all): {df['pie'].std():.4f}")

# Filter: Only significant rotation players
# Use PIE itself as a filter - anyone below 0.05 is basically garbage time
print(f"\n🔍 FILTERING TO ROTATION PLAYERS (PIE > 0.05):")
rotation_players = df[df['pie'] >= 0.05].copy()
print(f"   Players with PIE >= 0.05: {len(rotation_players)}")

# Calculate calibrated distribution
pie_values = rotation_players['pie']

mean_pie = pie_values.mean()
std_pie = pie_values.std()
median_pie = pie_values.median()
p25_pie = pie_values.quantile(0.25)
p75_pie = pie_values.quantile(0.75)
p90_pie = pie_values.quantile(0.90)
p99_pie = pie_values.quantile(0.99)

print(f"\n📈 CALIBRATED DISTRIBUTION (Rotation Players):")
print(f"   Count:      {len(rotation_players)}")
print(f"   Mean:       {mean_pie:.4f} ⭐ USE THIS")
print(f"   Std Dev:    {std_pie:.4f} ⭐ USE THIS")
print(f"   Median:     {median_pie:.4f}")
print(f"   25th %ile:  {p25_pie:.4f} (Below avg starters)")
print(f"   75th %ile:  {p75_pie:.4f} (Good starters)")
print(f"   90th %ile:  {p90_pie:.4f} (All-Stars)")
print(f"   99th %ile:  {p99_pie:.4f} (MVP-tier)")

# Test the formula on top players
print("\n" + "="*80)
print("🧪 TESTING MULTIPLIERS ON TOP 20 PLAYERS")
print("="*80)

# Sort by PIE and get top 20
top_players = rotation_players.nlargest(20, 'pie')

print(f"{'Rank':<6} {'Player':<25} {'PIE':<8} {'Z-Score':<10} {'Multiplier':<12} {'Tier'}")
print("-"*80)

def calculate_multiplier(pie, mean, std):
    """Calculate multiplier using the tuned gravity model."""
    z_score = (pie - mean) / std
    
    if z_score <= 1.0:
        multiplier = 1.0
        tier = "Avg/Below"
    elif z_score <= 2.0:
        multiplier = 1.0 + (z_score - 1.0) * 1.0
        tier = "Starter"
    elif z_score <= 3.0:
        multiplier = 2.0 + ((z_score - 2.0) * 1.5)
        tier = "All-Star"
    else:
        excess_sigma = z_score - 3.0
        multiplier = 3.5 + (excess_sigma * 0.8)
        multiplier = min(multiplier, 4.5)
        tier = "MVP 🌟"
    
    return z_score, multiplier, tier

for idx, (_, row) in enumerate(top_players.iterrows(), 1):
    name = row['player_name']
    pie = row['pie']
    
    z_score, mult, tier = calculate_multiplier(pie, mean_pie, std_pie)
    
    print(f"{idx:<6} {name:<25} {pie:.4f}   {z_score:>8.2f}σ  {mult:>10.2f}x   {tier}")

# Distribution by tier
print("\n" + "="*80)
print("📊 PLAYER DISTRIBUTION BY TIER")
print("="*80)

tiers = []
for _, row in rotation_players.iterrows():
    z_score, mult, tier = calculate_multiplier(row['pie'], mean_pie, std_pie)
    tiers.append(tier)

tier_counts = pd.Series(tiers).value_counts()
print(f"\n{tier_counts}")

# Validate against expected superstars
print("\n" + "="*80)
print("🔍 VALIDATING KNOWN SUPERSTARS")
print("="*80)

known_superstars = [
    'Giannis Antetokounmpo',
    'Nikola Jokic', 
    'Luka Doncic',
    'Joel Embiid',
    'Stephen Curry',
    'Shai Gilgeous-Alexander',
    'LeBron James',
    'Anthony Davis',
]

print(f"{'Player':<30} {'Found':<8} {'PIE':<8} {'Z-Score':<10} {'Multiplier'}")
print("-"*80)

for superstar in known_superstars:
    # Try to find player (case-insensitive partial match)
    matches = rotation_players[
        rotation_players['player_name'].str.contains(superstar.split()[-1], case=False, na=False)
    ]
    
    if not matches.empty:
        player = matches.iloc[0]
        pie = player['pie']
        z_score, mult, tier = calculate_multiplier(pie, mean_pie, std_pie)
        print(f"{player['player_name']:<30} {'✅':<8} {pie:.4f}   {z_score:>8.2f}σ  {mult:.2f}x")
    else:
        print(f"{superstar:<30} {'❌':<8} {'---':<8} {'---':<10} {'---'}")

# Output recommended constants
print("\n" + "="*80)
print("🎯 RECOMMENDED CONSTANTS")
print("="*80)

print(f"""
Update in src/features/feature_calculator_v5.py:

    LEAGUE_AVG_PIE = {mean_pie:.4f}  # Current: 0.095
    LEAGUE_STD_PIE = {std_pie:.4f}  # Current: 0.035

Expected Impact Examples:
    • PIE {mean_pie:.3f} (Average)       → 1.0x multiplier
    • PIE {p75_pie:.3f} (75th %ile)     → {calculate_multiplier(p75_pie, mean_pie, std_pie)[1]:.2f}x multiplier
    • PIE {p90_pie:.3f} (90th %ile)     → {calculate_multiplier(p90_pie, mean_pie, std_pie)[1]:.2f}x multiplier
    • PIE {p99_pie:.3f} (99th %ile)     → {calculate_multiplier(p99_pie, mean_pie, std_pie)[1]:.2f}x multiplier (MVP tier)
""")

# Sanity checks
print("="*80)
print("🛡️ SANITY CHECKS")
print("="*80)

if mean_pie < 0.08:
    print("⚠️ WARNING: Mean PIE is lower than expected (< 0.08)")
    print("   This will cause inflated multipliers. Consider:")
    print("   1. Using stricter MPG filter (20+ instead of 15+)")
    print("   2. Dampening the Z-score slope (reduce from 2.0 to 1.5)")
elif mean_pie > 0.12:
    print("⚠️ WARNING: Mean PIE is higher than expected (> 0.12)")
    print("   This will cause deflated multipliers for superstars.")
    print("   Consider increasing the Z-score slope.")
else:
    print("✅ Mean PIE is in expected range (0.08-0.12)")

if std_pie < 0.025:
    print("⚠️ WARNING: Std Dev is low (< 0.025)")
    print("   Z-scores will be inflated. May need to dampen curve.")
elif std_pie > 0.045:
    print("⚠️ WARNING: Std Dev is high (> 0.045)")
    print("   Z-scores will be deflated. May need to steepen curve.")
else:
    print("✅ Std Dev is in expected range (0.025-0.045)")

# Check if top player gets reasonable multiplier
top_pie = top_players.iloc[0]['pie']
top_z, top_mult, top_tier = calculate_multiplier(top_pie, mean_pie, std_pie)

if top_mult > 5.0:
    print(f"\n⚠️ WARNING: Top player multiplier is very high ({top_mult:.2f}x)")
    print("   Consider adding a stricter cap or dampening the curve.")
elif top_mult < 3.0:
    print(f"\n⚠️ WARNING: Top player multiplier is low ({top_mult:.2f}x)")
    print("   MVP-tier players should be in 3.5-4.5x range.")
else:
    print(f"\n✅ Top player multiplier is reasonable ({top_mult:.2f}x)")

print("\n" + "="*80)
print("✅ CALIBRATION COMPLETE")
print("="*80)
