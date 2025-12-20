import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cur = conn.cursor()

# Check historical_inactives table
cur.execute('''SELECT COUNT(*), MIN(game_date), MAX(game_date) 
               FROM historical_inactives 
               WHERE team_abbreviation IN ('LAL', 'LAC')''')
result = cur.fetchone()
print('historical_inactives for LAL/LAC:', result)
print(f'  Records: {result[0]}')
print(f'  Date range: {result[1]} to {result[2]}')

# Check sample data
cur.execute('''SELECT game_date, team_abbreviation, player_name 
               FROM historical_inactives 
               WHERE team_abbreviation IN ('LAL', 'LAC')
               ORDER BY game_date DESC LIMIT 10''')
print('\nSample records:')
for row in cur.fetchall():
    print(f'  {row}')

conn.close()
