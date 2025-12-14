"""Copy essential tables from main database to live database"""
import sqlite3
import shutil
from datetime import datetime

print("Copying essential tables from nba_MAIN_database.db to live database...")

# Backup first
backup_name = f'data/live/nba_betting_data_BEFORE_TABLE_COPY_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
shutil.copy2('data/live/nba_betting_data.db', backup_name)
print(f'✅ Backed up to: {backup_name}')

# Connect to both
source = sqlite3.connect('data/backups/nba_MAIN_database.db')
dest = sqlite3.connect('data/live/nba_betting_data.db')

# Tables to copy with data
tables_to_copy = [
    'game_logs',
    'active_injuries',
    'game_advanced_stats',
    'team_stats',
    'player_stats'
]

for table in tables_to_copy:
    try:
        print(f'\nCopying {table}...')
        
        # Get table schema
        cursor = source.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
        create_stmt = cursor.fetchone()
        
        if not create_stmt:
            print(f'  ⚠️  Table {table} not found in source')
            continue
            
        # Create table in dest
        dest.execute(create_stmt[0])
        
        # Copy data
        source_data = source.execute(f"SELECT * FROM {table}").fetchall()
        
        if source_data:
            # Get column count
            col_count = len(source_data[0])
            placeholders = ','.join(['?' for _ in range(col_count)])
            
            dest.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})", source_data)
            dest.commit()
            
            print(f'  ✅ Copied {len(source_data)} rows')
        else:
            print(f'  ⚠️  Table {table} is empty in source')
            
    except Exception as e:
        print(f'  ❌ Error copying {table}: {e}')

source.close()
dest.close()

print('\n✅ Table copy complete!')
print('\nVerifying data...')

# Verify
conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [t[0] for t in cursor.fetchall()]
print(f"Tables in live database: {', '.join(tables)}")

for table in ['game_logs', 'elo_ratings', 'active_injuries']:
    if table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")

conn.close()
