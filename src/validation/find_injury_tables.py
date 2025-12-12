import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

print("All injury-related tables:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%injur%'")
tables = cursor.fetchall()

for table in tables:
    print(f"\n{table[0]}:")
    
    # Get schema first
    cursor.execute(f"PRAGMA table_info({table[0]})")
    cols = cursor.fetchall()
    col_names = [c[1] for c in cols]
    print(f"  Columns: {', '.join(col_names)}")
    
    # Get count
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    print(f"  Total records: {count}")
    
    # Try to get date range if date column exists
    date_cols = [c for c in col_names if 'date' in c.lower()]
    if date_cols and count > 0:
        date_col = date_cols[0]
        cursor.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM {table[0]}")
        result = cursor.fetchone()
        print(f"  Date range: {result[0]} to {result[1]}")

conn.close()
