"""Setup active_injuries table with correct schema"""
import sqlite3
from src.services.live_injury_updater import LiveInjuryUpdater

DB_PATH = 'nba_betting_data.db'

# Recreate table with EXACT schema the updater expects
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('DROP TABLE IF EXISTS active_injuries')
cursor.execute('''
    CREATE TABLE active_injuries (
        player_name TEXT,
        team_name TEXT,
        position TEXT,
        status TEXT,
        injury_desc TEXT,
        source TEXT DEFAULT 'ESPN',
        last_updated TEXT
    )
''')
conn.commit()
conn.close()
print('✅ Recreated active_injuries table with correct schema')

# Populate with live data
updater = LiveInjuryUpdater(db_path=DB_PATH)
count = updater.update_active_injuries()
print(f'✅ Updated {count} injuries')
