"""
Test injury PIE calculation to verify it works correctly
"""
import sys
import sqlite3
sys.path.insert(0, 'src/features')

# Import directly from the file
from injury_replacement_model import calculate_team_injury_impact_advanced
from datetime import datetime

# Test with today's teams
today = datetime.now().strftime('%Y-%m-%d')
db_path = 'data/live/nba_betting_data.db'

# Team full names (as stored in active_injuries table)
test_teams = [
    'New York Knicks',
    'Orlando Magic',
    'San Antonio Spurs',
    'Oklahoma City Thunder'
]

print("\n=== TESTING PIE-BASED INJURY CALCULATIONS ===\n")
print(f"Date: {today}\n")

for team in test_teams:
    try:
        impact = calculate_team_injury_impact_advanced(team, today, db_path)
        print(f"{team:25s} - PIE Impact: {impact:6.2f}")
    except Exception as e:
        print(f"{team:25s} - ERROR: {e}")

print("\n✅ If all teams show numeric PIE Impact scores (0.0 = no injuries, higher = more impact), the system is working correctly!")
print("NOTE: Impact scores represent expected points lost due to injuries (PIE-weighted with position scarcity)")
