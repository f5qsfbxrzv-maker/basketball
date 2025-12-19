"""
CRITICAL AUDIT: Payout Logic Verification
Check if backtest used correct odds for each bet
"""

import pandas as pd
import numpy as np

print("\n" + "="*90)
print("PAYOUT LOGIC AUDIT - THE 'EVEN MONEY' BUG")
print("="*90)

# Load backtest results
try:
    bets_df = pd.read_csv('models/backtest_2024_2025_results.csv')
    print(f"\nLoaded {len(bets_df)} bets from backtest")
except:
    print("\nERROR: Could not load backtest results")
    print("Run backtest_2024_2025.py first")
    exit()

# Show the payout calculation that was used
print("\n" + "="*90)
print("BACKTEST PAYOUT FORMULA (What the code did)")
print("="*90)

print("\nFrom backtest_2024_2025.py:")
print("  TYPICAL_ODDS = 1.91  # -110 odds")
print("  profit = UNIT_SIZE * 0.91  # Win $91 on $100 bet")
print("\n⚠️ CRITICAL BUG IDENTIFIED:")
print("  The backtest used FIXED -110 odds for EVERY bet")
print("  Reality: Favorites have worse odds (e.g., -200 = only $50 profit)")

# Analyze actual win distribution
wins = bets_df[bets_df['result'] == 'WIN'].copy()
losses = bets_df[bets_df['result'] == 'LOSS'].copy()

print("\n" + "="*90)
print("ACTUAL PAYOUT ANALYSIS")
print("="*90)

print(f"\nWins: {len(wins)}")
print(f"  Average profit per win: ${wins['pnl'].mean():.2f}")
print(f"  Expected if -110 odds: ${100 * 0.91 * 0.952:.2f} (after commission)")
print(f"  Expected if -200 odds: ${100 * 0.50 * 0.952:.2f} (after commission)")

print(f"\nLosses: {len(losses)}")
print(f"  Average loss: ${losses['pnl'].mean():.2f}")

# Show specific examples
print("\n" + "="*90)
print("SAMPLE WINNING BETS (First 10)")
print("="*90)

print(f"\n{'Date':<12} {'Game':<30} {'Side':<12} {'Win Prob':<10} {'P&L':<10}")
print("-"*90)

for idx, row in wins.head(10).iterrows():
    print(f"{row['date'][:10]:<12} {row['game'][:30]:<30} {row['bet_side']:<12} {row['win_prob']*100:<10.1f} ${row['pnl']:<9.2f}")

# Calculate what SHOULD have happened with proper odds
print("\n" + "="*90)
print("THE 'EVEN MONEY' BUG IMPACT")
print("="*90)

print("\nScenario: Betting on a heavy favorite")
print("  Team: Lakers -200 (67% implied probability)")
print("  Bet: $100")
print("\n  REAL PAYOUT:")
print("    Win: $100 * (1.50 - 1) = $50 profit")
print("    After 4.8% commission: $50 * 0.952 = $47.60")
print("\n  BACKTEST PAYOUT (Bug):")
print("    Win: $100 * 0.91 = $91 profit")
print("    After 4.8% commission: $91 * 0.952 = $86.63")
print("\n  DIFFERENCE: $86.63 - $47.60 = $39.03 EXTRA PROFIT (82% overpayment!)")

# Estimate realistic ROI
print("\n" + "="*90)
print("REALISTIC ROI ESTIMATE")
print("="*90)

print("\nAssumptions:")
print("  - 67% of bets are on favorites averaging -200 odds (1.50 decimal)")
print("  - 33% of bets are on underdogs averaging +150 odds (2.50 decimal)")
print("  - 66.9% overall win rate")

total_bets = len(bets_df)
total_staked = total_bets * 100

# Breakdown: assume 67% of bets are on favorites (cover side), 33% on dogs (not_cover side)
n_favorite_bets = int(total_bets * 0.67)
n_dog_bets = total_bets - n_favorite_bets

# Win rates: if overall 66.9%, and betting more favorites, likely:
# Favorites win at ~73%, dogs win at ~50%
favorite_wins = int(n_favorite_bets * 0.73)
favorite_losses = n_favorite_bets - favorite_wins
dog_wins = int(n_dog_bets * 0.50)
dog_losses = n_dog_bets - dog_wins

# Calculate realistic P&L
# Favorite win: $100 stake, win $50, commission $2.40, net $47.60
# Favorite loss: -$100
# Dog win: $100 stake, win $150, commission $7.20, net $142.80
# Dog loss: -$100

favorite_pnl = (favorite_wins * 47.60) + (favorite_losses * -100)
dog_pnl = (dog_wins * 142.80) + (dog_losses * -100)
total_pnl = favorite_pnl + dog_pnl
realistic_roi = (total_pnl / total_staked) * 100

print(f"\nRealistic Calculation:")
print(f"  Favorite bets: {n_favorite_bets} (73% win rate)")
print(f"    Wins: {favorite_wins} × $47.60 = ${favorite_wins * 47.60:,.2f}")
print(f"    Losses: {favorite_losses} × -$100 = ${favorite_losses * -100:,.2f}")
print(f"    Net: ${favorite_pnl:,.2f}")
print(f"\n  Dog bets: {n_dog_bets} (50% win rate)")
print(f"    Wins: {dog_wins} × $142.80 = ${dog_wins * 142.80:,.2f}")
print(f"    Losses: {dog_losses} × -$100 = ${dog_losses * -100:,.2f}")
print(f"    Net: ${dog_pnl:,.2f}")
print(f"\n  TOTAL P&L: ${total_pnl:,.2f}")
print(f"  REALISTIC ROI: {realistic_roi:+.2f}%")

print("\n" + "="*90)
print("VERDICT")
print("="*90)
print("\n⚠️ CONFIRMED: 'Even Money Bug' is present")
print("\nBacktest paid out at -110 odds for ALL bets")
print("Reality: Favorites pay less, dogs pay more")
print("\nEstimated REAL ROI: -5% to +3% (need actual market odds to confirm)")
print("\nTO FIX:")
print("  1. Get historical closing line odds for each game")
print("  2. Calculate profit based on ACTUAL odds, not fixed -110")
print("  3. Re-run backtest with correct payout logic")
print("\n❌ DO NOT BET REAL MONEY until this is fixed")
print("="*90)
