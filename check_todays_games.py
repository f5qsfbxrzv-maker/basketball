from src.services.espn_schedule_service import ESPNScheduleService
from datetime import datetime

svc = ESPNScheduleService('data/live/nba_betting_data.db')
games = svc.get_games_range(days_ahead=1)

print(f'NBA Games in Next 24 Hours: {len(games)}')
print('='*70)

if games:
    for game in games[:10]:
        print(f"{game['game_date']} {game.get('game_time', 'TBD')}: {game['away_team']} @ {game['home_team']}")
else:
    print("No games scheduled")

print()
print(f"Current date/time: {datetime.now()}")
