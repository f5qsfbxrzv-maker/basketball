import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cur = conn.cursor()

# Check 2024-25 season
cur.execute('''SELECT COUNT(*), MIN(game_date), MAX(game_date), COUNT(DISTINCT team) 
               FROM elo_ratings WHERE season = "2024-25"''')
result = cur.fetchone()
print('2024-25 ELO Ratings:', result)
print(f'  Total records: {result[0]}')
print(f'  Date range: {result[1]} to {result[2]}')
print(f'  Teams: {result[3]}')

# Check latest records
cur.execute('''SELECT team, game_date, off_elo, def_elo, composite_elo 
               FROM elo_ratings 
               WHERE season = "2024-25" 
               ORDER BY game_date DESC LIMIT 10''')
print('\nLatest 10 records:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]} | Off={row[2]:.0f} Def={row[3]:.0f} Comp={row[4]:.0f}')

# Check for LAL and LAC specifically
print('\n' + '='*70)
print('LAL & LAC ELO ratings before 2025-12-20:')
for team in ['LAL', 'LAC']:
    cur.execute('''SELECT team, game_date, off_elo, def_elo, composite_elo 
                   FROM elo_ratings 
                   WHERE season = "2024-25" AND team = ? AND game_date < "2025-12-20"
                   ORDER BY game_date DESC LIMIT 1''', (team,))
    row = cur.fetchone()
    if row:
        print(f'  {row[0]}: {row[1]} | Off={row[2]:.0f} Def={row[3]:.0f} Comp={row[4]:.0f}')
    else:
        print(f'  {team}: NO DATA FOUND')

conn.close()
