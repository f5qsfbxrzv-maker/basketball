import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

# Check injury tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%injur%'")
tables = cursor.fetchall()
print("Injury tables:", [t[0] for t in tables])

# Check active_injuries
if ('active_injuries',) in tables:
    cursor.execute("SELECT COUNT(*) FROM active_injuries")
    count = cursor.fetchone()[0]
    print(f"active_injuries count: {count}")
    
    if count > 0:
        cursor.execute("SELECT * FROM active_injuries LIMIT 3")
        print("\nSample active_injuries:")
        for row in cursor.fetchall():
            print(row)
else:
    print("active_injuries table does NOT exist")

# Check historical_inactives
if ('historical_inactives',) in tables:
    cursor.execute("SELECT COUNT(*) FROM historical_inactives")
    count = cursor.fetchone()[0]
    print(f"\nhistorical_inactives count: {count}")

conn.close()
