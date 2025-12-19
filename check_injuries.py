import sqlite3
conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()
cursor.execute('SELECT team_name, COUNT(*) FROM active_injuries GROUP BY team_name ORDER BY COUNT(*) DESC')
teams = cursor.fetchall()
print(f'Teams with injuries: {len(teams)}\n')
for t in teams[:15]:
    print(f'  {t[0]}: {t[1]} players')
conn.close()
