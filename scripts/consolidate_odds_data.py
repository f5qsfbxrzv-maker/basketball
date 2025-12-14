"""
Consolidate historical odds data from backup databases into main database
"""
import sqlite3
import pandas as pd
from pathlib import Path

print("="*80)
print("ODDS DATA CONSOLIDATION")
print("="*80)

# Check what's in the odds history databases
odds_dbs = [
    'data/backups/nba_ODDS_history.db',
    'data/backups/odds_history.db'
]

for db_path in odds_dbs:
    print(f"\n{'='*80}")
    print(f"DATABASE: {db_path}")
    print("="*80)
    
    if not Path(db_path).exists():
        print(f"  ❌ File not found")
        continue
    
    conn = sqlite3.connect(db_path)
    
    # Get tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"\nTables: {tables['name'].tolist()}")
    
    # Check each table
    for table in tables['name']:
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
            print(f"\n--- {table}: {count:,} rows ---")
            
            if count > 0:
                # Get schema
                schema = pd.read_sql(f"PRAGMA table_info({table})", conn)
                cols = schema['name'].tolist()
                print(f"Columns: {cols}")
                
                # Get date range if available
                if 'game_date' in cols:
                    date_range = pd.read_sql(f"SELECT MIN(game_date) as min_date, MAX(game_date) as max_date FROM {table}", conn)
                    print(f"Date range: {date_range.iloc[0]['min_date']} to {date_range.iloc[0]['max_date']}")
                
                # Sample data
                sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
                print("\nSample:")
                print(sample)
                
        except Exception as e:
            print(f"  Error: {e}")
    
    conn.close()

print("\n" + "="*80)
print("CURRENT MAIN DATABASE: data/live/nba_betting_data.db")
print("="*80)

main_conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check current odds tables
for table in ['game_odds', 'odds_snapshots']:
    try:
        count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", main_conn).iloc[0]['count']
        print(f"\n{table}: {count:,} rows")
        
        if count > 0:
            date_check = pd.read_sql(f"SELECT MIN(game_date) as min_date, MAX(game_date) as max_date FROM {table} WHERE game_date >= '2023-01-01'", main_conn)
            print(f"  Date range (2023+): {date_check.iloc[0]['min_date']} to {date_check.iloc[0]['max_date']}")
    except Exception as e:
        print(f"  Error checking {table}: {e}")

main_conn.close()

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)

# Now consolidate the data
print("\n" + "="*80)
print("CONSOLIDATING ODDS DATA INTO MAIN DATABASE")
print("="*80)

main_conn = sqlite3.connect('data/live/nba_betting_data.db')

# Create historical_odds table if it doesn't exist
create_table_sql = """
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
"""

main_conn.execute(create_table_sql)
main_conn.commit()
print("\n✅ Created historical_odds table")

# Import from both backup databases
total_imported = 0
duplicates = 0

for db_path in odds_dbs:
    if not Path(db_path).exists():
        continue
    
    print(f"\nImporting from {db_path}...")
    backup_conn = sqlite3.connect(db_path)
    
    # Read all odds history
    odds_df = pd.read_sql("SELECT * FROM odds_history", backup_conn)
    print(f"  Found {len(odds_df)} rows")
    
    # Insert into main database (ignore duplicates)
    for idx, row in odds_df.iterrows():
        try:
            main_conn.execute("""
                INSERT OR IGNORE INTO historical_odds 
                (timestamp, game_date, home_team, away_team, source, 
                 home_ml_odds, away_ml_odds, spread_line, spread_home_odds, spread_away_odds,
                 total_line, over_odds, under_odds, home_win_prob, away_win_prob, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['timestamp'], row['game_date'], row['home_team'], row['away_team'], row['source'],
                row['home_ml_odds'], row['away_ml_odds'], row['spread_line'], row['spread_home_odds'], row['spread_away_odds'],
                row['total_line'], row['over_odds'], row['under_odds'], row['home_win_prob'], row['away_win_prob'], row['raw_data']
            ))
            total_imported += 1
        except Exception as e:
            duplicates += 1
    
    backup_conn.close()
    main_conn.commit()
    print(f"  ✅ Imported {len(odds_df)} rows")

# Verify
final_count = pd.read_sql("SELECT COUNT(*) as count FROM historical_odds", main_conn).iloc[0]['count']
date_range = pd.read_sql("SELECT MIN(game_date) as min_date, MAX(game_date) as max_date FROM historical_odds", main_conn)

print(f"\n" + "="*80)
print("✅ CONSOLIDATION COMPLETE")
print("="*80)
print(f"\nTotal rows in historical_odds: {final_count:,}")
print(f"Date range: {date_range.iloc[0]['min_date']} to {date_range.iloc[0]['max_date']}")
print(f"\nUnique games:")
unique_games = pd.read_sql("SELECT COUNT(DISTINCT game_date || home_team || away_team) as count FROM historical_odds", main_conn).iloc[0]['count']
print(f"  {unique_games:,} unique matchups")

# Sample of what was added
print(f"\nSample of historical odds:")
sample = pd.read_sql("""
    SELECT game_date, home_team, away_team, home_ml_odds, away_ml_odds, spread_line, total_line
    FROM historical_odds
    ORDER BY game_date DESC
    LIMIT 5
""", main_conn)
print(sample)

main_conn.close()

print(f"\n{'='*80}")
print(f"✅ Historical odds now available in data/live/nba_betting_data.db")
print(f"   Table: historical_odds")
print("="*80)

