"""
CRITICAL LEAK DETECTION
Check if away_composite_elo contains future information
"""

import pandas as pd
import numpy as np

print("\n" + "="*90)
print("DATA LEAK DETECTOR - ELO TIME TRAVEL CHECK")
print("="*90)

# Load the backtest data
df = pd.read_csv('data/training_data_36features.csv')
df['date'] = pd.to_datetime(df['date'])

# Filter to test period (2024-25 & 2025-26)
test_cutoff = pd.to_datetime('2024-10-01')
df_test = df[df['date'] >= test_cutoff].copy()

print(f"\nTest period: {df_test['date'].min().date()} to {df_test['date'].max().date()}")
print(f"Total test games: {len(df_test)}")

# Check Celtics away games
print("\n" + "="*90)
print("CELTICS AWAY GAMES - ELO PROGRESSION")
print("="*90)

celtics_away = df_test[df_test['away_team'] == 'BOS'].sort_values('date').head(10)

print(f"\n{'Date':<12} {'Opp':<5} {'Away_ELO':<12} {'Home_ELO':<12} {'Cover?':<8} {'ELO_Δ':<10}")
print("-"*90)

prev_elo = None
prev_result = None
for idx, row in celtics_away.iterrows():
    elo_change = row['away_composite_elo'] - prev_elo if prev_elo else 0
    
    # Check if ELO increased BEFORE a win (LEAK) or AFTER a win (CLEAN)
    leak_flag = ""
    if prev_elo and prev_result is not None:
        if prev_result == 1 and elo_change < 0:
            leak_flag = " ⚠️ SUSPICIOUS: ELO dropped after WIN"
        elif prev_result == 0 and elo_change > 10:
            leak_flag = " ⚠️ SUSPICIOUS: ELO jumped after LOSS"
    
    print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['home_team']:<5} {row['away_composite_elo']:<12.2f} {row['home_composite_elo']:<12.2f} {int(row['target_spread_cover']):<8} {elo_change:+10.2f}{leak_flag}")
    
    prev_elo = row['away_composite_elo']
    prev_result = row['target_spread_cover']

# Now check ALL teams for systematic pattern
print("\n" + "="*90)
print("SYSTEMATIC LEAK CHECK: Do ELO values correlate with CURRENT game results?")
print("="*90)

# Calculate correlation between away_composite_elo and target_spread_cover
corr_away_elo = df_test['away_composite_elo'].corr(df_test['target_spread_cover'])
corr_home_elo = df_test['home_composite_elo'].corr(df_test['target_spread_cover'])

print(f"\nCorrelation Analysis:")
print(f"  away_composite_elo vs target_spread_cover: {corr_away_elo:+.4f}")
print(f"  home_composite_elo vs target_spread_cover: {corr_home_elo:+.4f}")

if abs(corr_away_elo) > 0.3:
    print("\n  ⚠️ WARNING: Strong correlation between away_composite_elo and current game result!")
    print("  This suggests the ELO rating CONTAINS information about the game outcome.")
    print("  DIAGNOSIS: DATA LEAK - ELO is being updated BEFORE being saved to the row")
else:
    print("\n  ✓ Correlation is weak - ELO does not strongly predict current game outcome")
    print("  This is expected for a clean feature (ELO changes reflect PAST results)")

# Check win rate by ELO quartile
print("\n" + "="*90)
print("ELO QUARTILE ANALYSIS")
print("="*90)

df_test['away_elo_quartile'] = pd.qcut(df_test['away_composite_elo'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High'])

quartile_stats = df_test.groupby('away_elo_quartile').agg({
    'target_spread_cover': ['count', 'sum', 'mean']
}).round(4)

print("\nAway Team Cover Rate by ELO Quartile:")
print("(If Q4 has 80%+ cover rate, ELO is leaking)")
print(f"\n{'Quartile':<12} {'Games':<10} {'Covers':<10} {'Cover%':<10}")
print("-"*90)

for quartile in ['Q1_Low', 'Q2', 'Q3', 'Q4_High']:
    subset = df_test[df_test['away_elo_quartile'] == quartile]
    games = len(subset)
    covers = subset['target_spread_cover'].sum()
    cover_pct = covers / games * 100
    
    leak_flag = " ⚠️ LEAK!" if cover_pct > 70 else ""
    print(f"{quartile:<12} {games:<10} {int(covers):<10} {cover_pct:<10.1f}{leak_flag}")

print("\n" + "="*90)
print("ODDS ASSUMPTION CHECK")
print("="*90)

# Check the backtest assumptions
print("\nBacktest used:")
print("  Implied odds: 50% fair price (vig-removed)")
print("  Actual odds: -110 both sides (1.91 decimal)")
print("  Commission: 4.8% on winnings")
print("\n⚠️ CRITICAL ISSUE: Backtest assumed 50/50 fair odds on ALL games!")
print("  Reality: Favorites have implied prob ~60-70%, dogs ~30-40%")
print("  Your model betting 'cover' on high ELO teams = betting favorites")
print("  With 67% win rate, you're likely betting heavy favorites at EVEN MONEY odds")
print("  This is THE LEAK - you cannot get even money on Lakers -7")

print("\n" + "="*90)
print("VERDICT")
print("="*90)
print("\nRUN THESE CHECKS:")
print("1. If away_composite_elo correlation > 0.3 → DATA LEAK in ELO calculation")
print("2. If Q4 ELO cover rate > 70% → ELO contains future information")  
print("3. Odds assumption is BROKEN - need actual market odds for each game")
print("\nDO NOT BET REAL MONEY until these are fixed!")
print("="*90)
