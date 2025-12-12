import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

print("=" * 60)
print("INJURY DATA COVERAGE ANALYSIS")
print("=" * 60)

# Historical inactives
print("\n1. Historical Inactives Table:")
df = pd.read_sql("""
    SELECT 
        MIN(game_date) as min_date,
        MAX(game_date) as max_date,
        COUNT(*) as total_records,
        COUNT(DISTINCT game_date) as unique_dates,
        COUNT(DISTINCT player_name) as unique_players
    FROM historical_inactives
""", conn)
print(df.to_string(index=False))

# Coverage by year
print("\n2. Coverage by Year:")
yearly = pd.read_sql("""
    SELECT 
        substr(game_date, 1, 4) as year,
        COUNT(*) as inactive_records,
        COUNT(DISTINCT game_date) as game_dates,
        COUNT(DISTINCT player_name) as players
    FROM historical_inactives
    GROUP BY substr(game_date, 1, 4)
    ORDER BY year DESC
""", conn)
print(yearly.to_string(index=False))

# Recent sample
print("\n3. Sample of Recent Injury Data (last 10 game dates):")
recent = pd.read_sql("""
    SELECT 
        game_date,
        COUNT(*) as inactive_count,
        GROUP_CONCAT(DISTINCT player_name, ', ') as players
    FROM historical_inactives
    WHERE game_date >= '2024-11-01'
    GROUP BY game_date
    ORDER BY game_date DESC
    LIMIT 10
""", conn)
print(recent.to_string(index=False))

# Active injuries
print("\n4. Active Injuries Table:")
active = pd.read_sql("SELECT COUNT(*) as count FROM active_injuries", conn)
print(f"Active injuries: {active['count'][0]}")

# Check if injury data exists for training period
print("\n5. Training Period Coverage (2023-01-01 to 2025-11-01):")
training = pd.read_sql("""
    SELECT 
        COUNT(DISTINCT game_date) as dates_with_injury_data,
        COUNT(*) as total_injury_records
    FROM historical_inactives
    WHERE game_date >= '2023-01-01' AND game_date < '2025-11-01'
""", conn)
print(training.to_string(index=False))

# Compare to game results
games = pd.read_sql("""
    SELECT COUNT(DISTINCT game_date) as total_game_dates
    FROM game_results
    WHERE game_date >= '2023-01-01' AND game_date < '2025-11-01'
""", conn)
print(f"\nTotal game dates in training period: {games['total_game_dates'][0]}")

coverage_pct = (training['dates_with_injury_data'][0] / games['total_game_dates'][0]) * 100 if games['total_game_dates'][0] > 0 else 0
print(f"Injury data coverage: {coverage_pct:.1f}%")

conn.close()
