import sqlite3

paths = [
    'data/nba_betting_data.db',
    'data/live/nba_betting_data.db'
]

for db_path in paths:
    print(f"\n{db_path}:")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = cursor.fetchall()
        
        if tables:
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table[0]}')
                count = cursor.fetchone()[0]
                print(f"  {table[0]}: {count} rows")
        else:
            print("  (empty database)")
        
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")
