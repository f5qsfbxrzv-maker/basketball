from nba_api.live.nba.endpoints import scoreboard

board = scoreboard.ScoreBoard()
games = board.get_dict().get('scoreboard', {}).get('games', [])

print(f'Games today: {len(games)}')
for g in games:
    away = g.get('awayTeam', {}).get('teamTricode')
    home = g.get('homeTeam', {}).get('teamTricode')
    print(f'  {away} @ {home}')
