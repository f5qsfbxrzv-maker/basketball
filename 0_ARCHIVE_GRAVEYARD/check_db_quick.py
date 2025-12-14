import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
print("📊 Database Tables:")
for t in tables:
    print(f"  • {t[0]}")

conn.close()
