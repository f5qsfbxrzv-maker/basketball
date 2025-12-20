import sqlite3
import pandas as pd
import os

# Check all .db files in data directory
db_files = [f for f in os.listdir('data') if f.endswith('.db')]

print(f"Found {len(db_files)} database files:")
for db_file in db_files:
    print(f"\n{'='*60}")
    print(f"Database: {db_file}")
    print('='*60)
    
    try:
        conn = sqlite3.connect(f'data/{db_file}')
        
        # Get all tables
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        print(f"\nTables: {tables['name'].tolist()}")
        
        for table in tables['name']:
            # Get table info
            cols = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
            print(f"\n{table} columns:")
            print(cols[['name', 'type']].to_string())
            
            # Check for score-related columns
            score_cols = [c for c in cols['name'].values if 'score' in c.lower() or 'point' in c.lower()]
            if score_cols:
                print(f"  ⭐ Score columns found: {score_cols}")
                
                # Sample data
                sample = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3", conn)
                print(f"\nSample data from {table}:")
                print(sample.head())
        
        conn.close()
    except Exception as e:
        print(f"Error reading {db_file}: {e}")
