"""
Find all historical odds data in the workspace
"""
import sqlite3
import pandas as pd
from pathlib import Path

# Check the main database for existing odds-related tables
conn = sqlite3.connect('data/live/nba_betting_data.db')

# Get all table names
tables_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
all_tables = pd.read_sql(tables_query, conn)

print('='*80)
print('ALL TABLES IN MAIN DATABASE')
print('='*80)
for t in all_tables['name']:
    print(f"  {t}")

# Check for any table that might have historical odds
print('\n' + '='*80)
print('CHECKING TABLES FOR HISTORICAL ODDS DATA')
print('='*80)

odds_keywords = ['odds', 'spread', 'moneyline', 'total', 'line', 'betting', 'market', 'wager']

for table in all_tables['name']:
    # Check if table name contains odds keywords or check structure
    try:
        schema = pd.read_sql(f"PRAGMA table_info({table})", conn)
        cols = schema['name'].tolist()
        
        # Check if has odds-related columns
        has_odds_cols = any(keyword in ' '.join(cols).lower() for keyword in odds_keywords)
        has_odds_name = any(keyword in table.lower() for keyword in odds_keywords)
        
        if has_odds_cols or has_odds_name:
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
            
            print(f'\n{table}: {count:,} rows')
            print(f'  Columns: {cols[:10]}...' if len(cols) > 10 else f'  Columns: {cols}')
            
            # Check date range
            if count > 0:
                if 'game_date' in cols:
                    date_range = pd.read_sql(f"SELECT MIN(game_date) as min_date, MAX(game_date) as max_date FROM {table}", conn)
                    print(f'  Date range: {date_range.iloc[0]["min_date"]} to {date_range.iloc[0]["max_date"]}')
                    
                    # Sample
                    sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn)
                    print(f'  Sample row:')
                    for col in sample.columns:
                        print(f'    {col}: {sample[col].iloc[0]}')
                elif 'timestamp' in cols:
                    sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn)
                    print(f'  Sample timestamp: {sample["timestamp"].iloc[0] if "timestamp" in sample else "N/A"}')
    except Exception as e:
        pass

conn.close()

# Also check CSV files
print('\n' + '='*80)
print('CHECKING FOR ODDS CSV FILES')
print('='*80)

data_dir = Path('data')
for csv_file in data_dir.rglob('*.csv'):
    file_name = csv_file.name.lower()
    if any(keyword in file_name for keyword in odds_keywords):
        print(f'\n{csv_file}')
        print(f'  Size: {csv_file.stat().st_size:,} bytes')
        print(f'  Modified: {csv_file.stat().st_mtime}')
        
        # Try to read first few rows
        try:
            sample = pd.read_csv(csv_file, nrows=2)
            print(f'  Columns: {list(sample.columns)}')
            print(f'  Rows: {len(pd.read_csv(csv_file))}')
        except Exception as e:
            print(f'  Error reading: {e}')
