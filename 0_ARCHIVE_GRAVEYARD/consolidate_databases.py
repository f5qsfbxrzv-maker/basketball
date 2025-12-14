"""Compare both databases and consolidate into one"""
import sqlite3
import pandas as pd
import shutil
from datetime import datetime

print("="*80)
print("DATABASE CONSOLIDATION AUDIT")
print("="*80)

# Check both databases
db_root = 'nba_betting_data.db'
db_live = 'data/live/nba_betting_data.db'

def get_table_info(db_path):
    """Get info about all tables in a database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    info = {}
    for table in tables:
        try:
            df = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", conn)
            info[table] = df['cnt'][0]
        except:
            info[table] = 'ERROR'
    
    conn.close()
    return info

print(f"\n[1] ROOT DATABASE: {db_root}")
print("-"*80)
root_tables = get_table_info(db_root)
for table, count in sorted(root_tables.items()):
    print(f"  {table:30s}: {count:>10} rows")

print(f"\n[2] LIVE DATABASE: {db_live}")
print("-"*80)
live_tables = get_table_info(db_live)
for table, count in sorted(live_tables.items()):
    print(f"  {table:30s}: {count:>10} rows")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# Tables only in root
only_root = set(root_tables.keys()) - set(live_tables.keys())
if only_root:
    print(f"\nTables ONLY in root: {', '.join(only_root)}")
else:
    print("\nNo unique tables in root")

# Tables only in live
only_live = set(live_tables.keys()) - set(root_tables.keys())
if only_live:
    print(f"\nTables ONLY in live: {', '.join(only_live)}")
else:
    print("\nNo unique tables in live")

# Tables in both but with different counts
both = set(root_tables.keys()) & set(live_tables.keys())
print(f"\nTables in BOTH:")
for table in sorted(both):
    root_count = root_tables[table]
    live_count = live_tables[table]
    if root_count != live_count:
        print(f"  {table:30s}: ROOT={root_count:>10} | LIVE={live_count:>10} {'← LIVE HAS MORE' if live_count > root_count else '← ROOT HAS MORE'}")
    else:
        print(f"  {table:30s}: {root_count:>10} rows (same)")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print("""
The code uses: data/live/nba_betting_data.db (via config/settings.py)

ACTION PLAN:
1. Copy any missing tables FROM root TO live
2. For tables where root has MORE data, update live
3. Archive root database to data/backups/
4. Keep data/live/nba_betting_data.db as the ONLY active database
""")

# Check game_logs dates specifically
conn_root = sqlite3.connect(db_root)
conn_live = sqlite3.connect(db_live)

print("\nGAME_LOGS date comparison:")
try:
    root_gl = pd.read_sql("SELECT MIN(GAME_DATE) as min, MAX(GAME_DATE) as max, COUNT(*) as cnt FROM game_logs", conn_root)
    print(f"  ROOT: {root_gl['min'][0]} to {root_gl['max'][0]} ({root_gl['cnt'][0]} rows)")
except:
    print("  ROOT: No game_logs")

try:
    live_gl = pd.read_sql("SELECT MIN(GAME_DATE) as min, MAX(GAME_DATE) as max, COUNT(*) as cnt FROM game_logs", conn_live)
    print(f"  LIVE: {live_gl['min'][0]} to {live_gl['max'][0]} ({live_gl['cnt'][0]} rows)")
except:
    print("  LIVE: No game_logs")

conn_root.close()
conn_live.close()
