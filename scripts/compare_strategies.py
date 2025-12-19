"""
Compare Strategies - Total Profit Focus
"""

import json

print("="*90)
print("STRATEGY COMPARISON - TOTAL PROFIT & ACTION")
print("="*90)

# Load both results
with open('models/backtest_trial215_results.json', 'r') as f:
    original = json.load(f)

with open('models/backtest_trial215_favorites_focused.json', 'r') as f:
    fav_focused = json.load(f)

print("\n1. ORIGINAL STRATEGY (15% edge for all)")
print("-" * 70)
print(f"  Total Bets: {original['best_bets']}")
print(f"  Total Action: ${original['best_bets'] * 100:,}")
print(f"  Total Profit: ${original['best_profit']:,.0f}")
print(f"  ROI: {original['best_roi']:.2f}%")
print(f"  Win Rate: {original['best_win_pct']:.1f}%")

print("\n2. FAVORITES-FOCUSED (2.5% fav / 15% dog)")
print("-" * 70)
print(f"  Total Bets: {fav_focused['best_bets']}")
print(f"  Total Action: ${fav_focused['best_bets'] * 100:,}")
print(f"  Total Profit: ${fav_focused['best_profit']:,.0f}")
print(f"  ROI: {fav_focused['best_roi']:.2f}%")
print(f"  Favorites: {fav_focused['favorites_pct']:.1f}% of bets")

print("\n3. COMPARISON")
print("-" * 70)
extra_bets = fav_focused['best_bets'] - original['best_bets']
extra_action = extra_bets * 100
extra_profit = fav_focused['best_profit'] - original['best_profit']
roi_diff = fav_focused['best_roi'] - original['best_roi']

print(f"  Extra Bets: +{extra_bets} ({extra_bets/original['best_bets']*100:+.1f}%)")
print(f"  Extra Action: ${extra_action:,} (+{extra_action/(original['best_bets']*100)*100:.1f}%)")
print(f"  Extra Profit: ${extra_profit:+,.0f}")
print(f"  ROI Change: {roi_diff:+.2f} percentage points")

print("\n4. WHICH IS BETTER?")
print("-" * 70)
print(f"  If you want MAXIMUM PROFIT: Favorites-focused wins")
print(f"    → ${fav_focused['best_profit']:,.0f} vs ${original['best_profit']:,.0f}")
print(f"  If you want MAXIMUM ROI: Original wins")
print(f"    → {original['best_roi']:.2f}% vs {fav_focused['best_roi']:.2f}%")
print(f"  If you want MORE ACTION: Favorites-focused wins")
print(f"    → {fav_focused['best_bets']} bets vs {original['best_bets']} bets")

print("\n5. BANKROLL EFFICIENCY")
print("-" * 70)
print(f"  Original: Need ${original['best_bets'] * 100:,} bankroll → ${original['best_profit']:,.0f} profit")
print(f"  Fav-focused: Need ${fav_focused['best_bets'] * 100:,} bankroll → ${fav_focused['best_profit']:,.0f} profit")
print(f"\n  With $50,000 bankroll:")
print(f"    Original: Bet {original['best_bets']} games = ${original['best_profit']:,.0f} profit")
print(f"    Fav-focused: Bet {fav_focused['best_bets']} games = ${fav_focused['best_profit']:,.0f} profit")
print(f"    Winner: Favorites-focused (+${extra_profit:,.0f})")

print("\n6. RECOMMENDATION")
print("-" * 70)
print(f"  ✓ Use FAVORITES-FOCUSED (2.5% fav / 15% dog)")
print(f"  Reason: More total profit with acceptable ROI decline")
print(f"  Benefit: Deploy capital more efficiently across more games")
print(f"  Tradeoff: -1.87% ROI but +${extra_profit:,.0f} profit and +{extra_bets} bets")

print("\n" + "="*90)
