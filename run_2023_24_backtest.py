"""Quick script to run backtest on 2023-24 season"""
import sys
import subprocess

# Update the config in backtest script
with open('backtest_walk_forward.py', 'r', encoding='utf-8') as f:
    script = f.read()

# Replace the values
script = script.replace(
    "ODDS_PATH = 'data/live/closing_odds_2024_25.csv'  # Will be updated per season",
    "ODDS_PATH = 'data/closing_odds_2023_24.csv'  # 2023-24 season"
)
script = script.replace(
    "TEST_SEASON = '2024-25'  # Will be updated per season",
    "TEST_SEASON = '2023-24'  # 2023-24 season"
)

# Execute
exec(script)
