import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check what we actually stored in historical_odds
print('=== WHAT WE STORED IN HISTORICAL_ODDS ===')
sample = pd.read_sql("""
SELECT game_date, home_team, away_team, spread_line, total_line, source
FROM historical_odds
WHERE game_date >= '2023-01-01'
ORDER BY game_date
LIMIT 10
""", conn)
print(sample)
print()

# Check the original master_features data
print('=== ORIGINAL MASTER_FEATURES DATA ===')
original = pd.read_sql("""
SELECT 
    game_date, home_team, away_team,
    opening_spread_home,
    closing_spread_home,
    "opening_total.1" as opening_total,
    "closing_total.1" as closing_total,
    home_score,
    away_score
FROM master_features
WHERE game_date >= '2023-01-01'
ORDER BY game_date
LIMIT 10
""", conn)
print(original)
print()

# Check if totals are being stored as spreads
print('=== SPREAD VALUES DISTRIBUTION ===')
spread_stats = pd.read_sql("""
SELECT 
    MIN(spread_line) as min_spread,
    MAX(spread_line) as max_spread,
    AVG(spread_line) as avg_spread,
    COUNT(CASE WHEN spread_line > 100 THEN 1 END) as spreads_over_100,
    COUNT(*) as total
FROM historical_odds
WHERE game_date >= '2023-01-01'
""", conn)
print(spread_stats)

conn.close()
