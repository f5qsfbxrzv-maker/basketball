import sqlite3
import pandas as pd

databases = [
    'data/backups/nba_ODDS_history.db',
    'data/backups/nba_MAIN_database.db',
    'data/backups/odds_history.db',
    'data/backups/nba_betting_data.db'
]

for db_path in databases:
    print(f"\n{'='*70}")
    print(f"Checking: {db_path}")
    print('='*70)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"Tables: {tables}")
        
        for table in tables:
            if 'odds' in table.lower() or 'money' in table.lower() or 'line' in table.lower():
                print(f"\n{table}:")
                df = pd.read_sql(f"SELECT * FROM {table} LIMIT 2", conn)
                print(f"Columns: {df.columns.tolist()}")
                print(df)
                
                # Check for 2024-25 data
                date_cols = [c for c in df.columns if 'date' in c.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    count_2024 = pd.read_sql(
                        f"SELECT COUNT(*) as cnt FROM {table} WHERE {date_col} >= '2024-10-01'", 
                        conn
                    )['cnt'].iloc[0]
                    print(f"2024-25 games: {count_2024}")
                    
                    if count_2024 > 0:
                        sample = pd.read_sql(
                            f"SELECT * FROM {table} WHERE {date_col} >= '2024-10-01' LIMIT 3",
                            conn
                        )
                        print(f"2024-25 sample:\n{sample}")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")
