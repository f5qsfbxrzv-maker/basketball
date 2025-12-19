from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta
import json

# Test tomorrow's games
dt = datetime.now() + timedelta(days=1)
print(f'Querying for: {dt.strftime("%m/%d/%Y")} ({dt.strftime("%Y-%m-%d")})')

board = scoreboardv2.ScoreboardV2(game_date=dt.strftime('%m/%d/%Y'))
data = board.get_dict()

print('\nResultSets:', [rs['name'] for rs in data.get('resultSets', [])])

# Get GameHeader
gh = [rs for rs in data['resultSets'] if rs['name'] == 'GameHeader'][0]
print(f'\nGameHeader columns: {gh["headers"]}')

if gh['rowSet']:
    print(f'\n=== FIRST GAME DATA ===')
    game = dict(zip(gh['headers'], gh['rowSet'][0]))
    print(f"GAME_ID: {game.get('GAME_ID')}")
    print(f"GAME_DATE_EST: {game.get('GAME_DATE_EST')}")
    print(f"GAME_STATUS_TEXT: {game.get('GAME_STATUS_TEXT')}")
    print(f"HOME_TEAM_ID: {game.get('HOME_TEAM_ID')}")
    print(f"VISITOR_TEAM_ID: {game.get('VISITOR_TEAM_ID')}")
    
    # Check LineScore for team abbrs
    ls = [rs for rs in data['resultSets'] if rs['name'] == 'LineScore'][0]
    print(f'\nLineScore columns: {ls["headers"]}')
    if ls['rowSet']:
        for row in ls['rowSet'][:2]:
            team_data = dict(zip(ls['headers'], row))
            print(f"  TEAM_ID={team_data.get('TEAM_ID')}, TEAM_ABBREVIATION={team_data.get('TEAM_ABBREVIATION')}")

print(f'\n=== Total games found: {len(gh["rowSet"])} ===')
