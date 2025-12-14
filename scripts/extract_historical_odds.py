"""
Extract historical odds from master_features and consolidate into historical_odds table
"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check master_features odds coverage
print("="*80)
print("MASTER_FEATURES ODDS COVERAGE (2023+)")
print("="*80)

coverage = pd.read_sql("""
SELECT 
    COUNT(*) as total_games,
    SUM(CASE WHEN opening_spread_home IS NOT NULL AND opening_spread_home != 0 THEN 1 ELSE 0 END) as has_opening_spread,
    SUM(CASE WHEN closing_spread_home IS NOT NULL AND closing_spread_home != 0 THEN 1 ELSE 0 END) as has_closing_spread,
    SUM(CASE WHEN "opening_total.1" IS NOT NULL AND "opening_total.1" != 0 THEN 1 ELSE 0 END) as has_opening_total,
    SUM(CASE WHEN "closing_total.1" IS NOT NULL AND "closing_total.1" != 0 THEN 1 ELSE 0 END) as has_closing_total,
    MIN(game_date) as earliest,
    MAX(game_date) as latest
FROM master_features
WHERE game_date >= '2023-01-01'
""", conn)

print(coverage)
print()

# Sample of data
sample = pd.read_sql("""
SELECT game_date, home_team, away_team, 
       opening_spread_home, closing_spread_home,
       "opening_total.1" as opening_total, "closing_total.1" as closing_total,
       home_score, away_score
FROM master_features
WHERE game_date >= '2023-01-01' 
  AND opening_spread_home != 0
ORDER BY game_date DESC
LIMIT 10
""", conn)

print("="*80)
print("SAMPLE HISTORICAL ODDS FROM MASTER_FEATURES")
print("="*80)
print(sample)
print()

# Now extract and transform all historical odds
print("="*80)
print("EXTRACTING HISTORICAL ODDS FROM MASTER_FEATURES")
print("="*80)
print("NOTE: Using OPENING spreads/totals (closing data has column swap issues)")
print()

# Get all games with odds - USE OPENING VALUES ONLY
historical_df = pd.read_sql("""
SELECT 
    game_id,
    game_date,
    home_team,
    away_team,
    opening_spread_home,
    opening_spread_visitor,
    "opening_total.1" as opening_total,
    home_score,
    away_score
FROM master_features
WHERE game_date >= '2020-01-01'
  AND opening_spread_home IS NOT NULL
  AND opening_spread_home != 0
ORDER BY game_date
""", conn)

print(f"Found {len(historical_df)} games with opening odds data")
print(f"Date range: {historical_df['game_date'].min()} to {historical_df['game_date'].max()}")
print()

# Create historical_odds table if needed
conn.execute("""
CREATE TABLE IF NOT EXISTS historical_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    game_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    source TEXT,
    home_ml_odds REAL,
    away_ml_odds REAL,
    spread_line REAL,
    spread_home_odds REAL,
    spread_away_odds REAL,
    total_line REAL,
    over_odds REAL,
    under_odds REAL,
    home_win_prob REAL,
    away_win_prob REAL,
    raw_data TEXT,
    UNIQUE(game_date, home_team, away_team, timestamp)
)
""")
conn.commit()

# Clear existing data and insert fresh
conn.execute("DELETE FROM historical_odds")
conn.commit()

# Insert with smart spread/total detection
print("Inserting historical odds with smart column detection...")
inserted = 0
skipped = 0
swapped = 0

for idx, row in historical_df.iterrows():
    if idx % 500 == 0:
        print(f"  Progress: {idx}/{len(historical_df)}")
    
    # Use opening spread and total
    spread_line = row['opening_spread_home']
    total_line = row['opening_total'] if row['opening_total'] != 0 else None
    
    # CRITICAL FIX: Detect if spread/total are swapped
    # NBA spreads are typically -20 to +20, totals are 200-250
    if spread_line > 100:  # This is actually a total
        if total_line is not None and total_line < 100:  # And total looks like a spread
            spread_line, total_line = total_line, spread_line
            swapped += 1
    
    try:
        conn.execute("""
            INSERT OR IGNORE INTO historical_odds 
            (timestamp, game_date, home_team, away_team, source, 
             spread_line, spread_home_odds, spread_away_odds,
             total_line, over_odds, under_odds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['game_date'],  # Use game_date as timestamp for historical
            row['game_date'],
            row['home_team'],
            row['away_team'],
            'master_features_fixed',  # Indicate columns were intelligently swapped
            spread_line,
            -110,  # Standard vig
            -110,  # Standard vig  
            total_line,
            -110,  # Standard vig
            -110   # Standard vig
        ))
        inserted += 1
    except Exception as e:
        print(f"Error inserting {row['game_date']} {row['home_team']} vs {row['away_team']}: {e}")
        skipped += 1

conn.commit()

print(f"\nInserted {inserted} historical odds records")
print(f"Swapped spread/total on {swapped} corrupted games")
if skipped > 0:
    print(f"Skipped {skipped} games (errors)")

# Verify final state
final_count = pd.read_sql("SELECT COUNT(*) as count FROM historical_odds", conn).iloc[0]['count']
date_range = pd.read_sql("SELECT MIN(game_date) as min_date, MAX(game_date) as max_date FROM historical_odds", conn)

print(f"\n" + "="*80)
print("HISTORICAL ODDS TABLE UPDATED")
print("="*80)
print(f"Total rows: {final_count:,}")
print(f"Date range: {date_range.iloc[0]['min_date']} to {date_range.iloc[0]['max_date']}")

# Sample
sample_final = pd.read_sql("""
    SELECT game_date, home_team, away_team, spread_line, total_line, source
    FROM historical_odds
    ORDER BY game_date DESC
    LIMIT 10
""", conn)

print(f"\nRecent odds:")
print(sample_final)

conn.close()

print(f"\n{'='*80}")
print(f"Historical odds ready for backtesting!")
print(f"   Table: historical_odds in data/live/nba_betting_data.db")
print(f"={'='*80}")
