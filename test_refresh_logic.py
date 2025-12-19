"""Test the dashboard's refresh logic to understand why it's not working"""
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2

print("=" * 60)
print("TESTING MULTI-DAY GAME FETCHING")
print("=" * 60)

days_ahead = 3
print(f"\nTesting with days_ahead = {days_ahead}")
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

all_games = []

for day_offset in range(days_ahead):
    target_date = datetime.now() + timedelta(days=day_offset)
    target_date_str = target_date.strftime('%Y-%m-%d')
    date_param = target_date.strftime('%m/%d/%Y')
    
    print(f"\n--- Day Offset {day_offset} ---")
    print(f"Target date: {target_date_str}")
    print(f"API parameter: {date_param}")
    
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_param)
        games_data = board.get_dict()
        
        day_game_count = 0
        
        if 'resultSets' in games_data:
            for result_set in games_data['resultSets']:
                if result_set['name'] == 'GameHeader':
                    headers = result_set['headers']
                    for row in result_set['rowSet']:
                        game_dict = dict(zip(headers, row))
                        
                        # Get actual game date from API
                        api_game_date = game_dict.get('GAME_DATE_EST')
                        if api_game_date:
                            actual_game_date = api_game_date.split('T')[0]
                        else:
                            actual_game_date = target_date_str
                        
                        # Get team IDs
                        home_team_id = game_dict.get('HOME_TEAM_ID')
                        away_team_id = game_dict.get('VISITOR_TEAM_ID')
                        
                        # Get time
                        game_status = game_dict.get('GAME_STATUS_TEXT', 'TBD')
                        game_time = game_status.split()[0] if game_status else 'TBD'
                        
                        # Get team abbreviations
                        home_abbr = None
                        away_abbr = None
                        for rs in games_data['resultSets']:
                            if rs['name'] == 'LineScore':
                                ls_headers = rs['headers']
                                for ls_row in rs['rowSet']:
                                    ls_dict = dict(zip(ls_headers, ls_row))
                                    if ls_dict.get('TEAM_ID') == home_team_id:
                                        home_abbr = ls_dict.get('TEAM_ABBREVIATION')
                                    elif ls_dict.get('TEAM_ID') == away_team_id:
                                        away_abbr = ls_dict.get('TEAM_ABBREVIATION')
                        
                        if home_abbr and away_abbr:
                            game_info = {
                                'home_team': home_abbr,
                                'away_team': away_abbr,
                                'game_date': actual_game_date,
                                'game_time': game_time
                            }
                            all_games.append(game_info)
                            day_game_count += 1
                            print(f"  ✓ {away_abbr} @ {home_abbr} - {actual_game_date} {game_time}")
        
        print(f"  Found {day_game_count} games for this day")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 60)
print(f"TOTAL GAMES FOUND: {len(all_games)}")
print("=" * 60)

if all_games:
    print("\nAll games by date:")
    from collections import defaultdict
    games_by_date = defaultdict(list)
    for game in all_games:
        games_by_date[game['game_date']].append(f"{game['away_team']} @ {game['home_team']}")
    
    for date in sorted(games_by_date.keys()):
        print(f"\n{date}:")
        for matchup in games_by_date[date]:
            print(f"  - {matchup}")
