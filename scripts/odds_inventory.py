"""
SUMMARY: Available Odds Data
"""

import pandas as pd
import sqlite3

print("="*90)
print("ODDS DATA INVENTORY")
print("="*90)

print("\n1. MONEYLINE ODDS (Available)")
print("-"*90)

# CSV file
df_csv = pd.read_csv('data/live/closing_odds_2024_25.csv')
print(f"\nSource: closing_odds_2024_25.csv")
print(f"  Games: {len(df_csv):,}")
print(f"  Date range: {df_csv['game_date'].min()} to {df_csv['game_date'].max()}")
print(f"  Data: Moneyline odds only (home_ml_odds, away_ml_odds)")
print(f"  Bookmaker: {df_csv['bookmaker'].unique()}")
print(f"\nSample:")
print(df_csv[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']].head(10).to_string(index=False))

print("\n\n2. SPREAD ODDS (Not Available for 2024-25)")
print("-"*90)
print("\n✗ No historical spread odds found for 2024-25 season")
print("✗ Database contains only -110 placeholder values")
print("✗ Only 1 real spread odds entry found (future game)")

print("\n\n3. CONCLUSION")
print("-"*90)
print("\n✓ We HAVE: Moneyline odds for 1,141 games (2024-25 season)")
print("✗ We DON'T HAVE: Spread odds for historical games")
print("\nOptions:")
print("  1. Backtest MONEYLINE predictions (not spread)")
print("  2. Use moneyline odds to estimate spread odds (approximation)")
print("  3. Download historical spread odds from The Odds API")
print("  4. Accept we cannot calculate realistic spread ROI")

print("\n" + "="*90)
