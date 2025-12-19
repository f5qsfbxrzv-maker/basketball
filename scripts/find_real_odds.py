"""
Check all odds sources
"""

import pandas as pd
import sqlite3

print("="*90)
print("CHECKING ALL ODDS DATA SOURCES")
print("="*90)

# Check CSV file
print("\n1. CSV FILE: closing_odds_2024_25.csv")
print("-"*90)
try:
    df_csv = pd.read_csv('data/live/closing_odds_2024_25.csv')
    print(f"Rows: {len(df_csv):,}")
    print(f"Columns: {df_csv.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df_csv.head(3))
    
    if 'spread_home_odds' in df_csv.columns:
        print(f"\nSpread odds range:")
        print(f"  home: {df_csv['spread_home_odds'].min():.0f} to {df_csv['spread_home_odds'].max():.0f}")
        print(f"  away: {df_csv['spread_away_odds'].min():.0f} to {df_csv['spread_away_odds'].max():.0f}")
        print(f"\nUnique odds values (home): {df_csv['spread_home_odds'].nunique()}")
        print(f"Sample odds: {df_csv['spread_home_odds'].value_counts().head(10).to_dict()}")
except Exception as e:
    print(f"Error: {e}")

# Check historical_closing_odds.db
print("\n\n2. DATABASE: historical_closing_odds.db")
print("-"*90)
try:
    conn = sqlite3.connect('data/live/historical_closing_odds.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"Tables: {tables}")
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count:,} rows")
        
        # Sample data
        df_sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
        print(f"  Columns: {df_sample.columns.tolist()}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")

# Check nba_betting_data.db historical_odds table more carefully
print("\n\n3. DATABASE: nba_betting_data.db (historical_odds table)")
print("-"*90)
try:
    conn = sqlite3.connect('data/live/nba_betting_data.db')
    
    # Check for varied odds
    df_check = pd.read_sql("""
        SELECT 
            spread_home_odds,
            spread_away_odds,
            COUNT(*) as count
        FROM historical_odds
        WHERE game_date >= '2024-10-01'
        AND spread_home_odds IS NOT NULL
        GROUP BY spread_home_odds, spread_away_odds
        ORDER BY count DESC
        LIMIT 20
    """, conn)
    
    print(f"Unique odds combinations:")
    print(df_check.to_string(index=False))
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")

# Check backups
print("\n\n4. BACKUP DATABASES")
print("-"*90)
backup_dbs = [
    'data/backups/nba_ODDS_history.db',
    'data/backups/odds_history.db'
]

for db_path in backup_dbs:
    print(f"\n{db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"  {table}: {count:,} rows")
                
                # Check for odds columns
                df_sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn)
                odds_cols = [c for c in df_sample.columns if 'odds' in c.lower() or 'spread' in c.lower()]
                if odds_cols:
                    print(f"    Odds columns: {odds_cols}")
        
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*90)
