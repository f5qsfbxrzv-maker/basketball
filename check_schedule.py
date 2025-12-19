"""Check NBA schedule for next several days"""
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta

print("Checking NBA schedule...")
print(f"Today: {datetime.now().strftime('%Y-%m-%d')}\n")

for offset in range(7):  # Check next 7 days
    date = datetime.now() + timedelta(days=offset)
    date_str = date.strftime('%m/%d/%Y')
    
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str)
        data = board.get_dict()
        
        game_count = 0
        games = []
        
        if 'resultSets' in data:
            for rs in data['resultSets']:
                if rs['name'] == 'GameHeader':
                    game_count = len(rs['rowSet'])
                    
                    # Get team abbreviations
                    for row in rs['rowSet']:
                        game_dict = dict(zip(rs['headers'], row))
                        home_id = game_dict.get('HOME_TEAM_ID')
                        away_id = game_dict.get('VISITOR_TEAM_ID')
                        
                        # Find team names
                        for ls_rs in data['resultSets']:
                            if ls_rs['name'] == 'LineScore':
                                for ls_row in ls_rs['rowSet']:
                                    ls_dict = dict(zip(ls_rs['headers'], ls_row))
                                    if ls_dict.get('TEAM_ID') == home_id:
                                        home_abbr = ls_dict.get('TEAM_ABBREVIATION')
                                    if ls_dict.get('TEAM_ID') == away_id:
                                        away_abbr = ls_dict.get('TEAM_ABBREVIATION')
                        
                        if 'home_abbr' in locals() and 'away_abbr' in locals():
                            games.append(f"{away_abbr}@{home_abbr}")
        
        label = "TODAY" if offset == 0 else f"+{offset}"
        if game_count > 0:
            print(f"{date.strftime('%Y-%m-%d')} ({label:>6}): {game_count} games - {', '.join(games)}")
        else:
            print(f"{date.strftime('%Y-%m-%d')} ({label:>6}): NO GAMES")
            
    except Exception as e:
        print(f"{date_str}: Error - {e}")
