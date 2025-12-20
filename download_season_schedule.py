"""
Download full 2024-25 NBA season schedule and save to CSV
Uses ESPN API to get all games for the season
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ESPN team name mapping
ESPN_TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

def download_season_schedule():
    """Download full season schedule from ESPN"""
    all_games = []
    
    # NBA 2025-26 season: Oct 22, 2025 to Apr 13, 2026 (regular season)
    start_date = datetime(2025, 10, 22)
    end_date = datetime(2026, 4, 13)
    
    current_date = start_date
    
    print(f"Downloading NBA 2025-26 schedule from {start_date.date()} to {end_date.date()}...")
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            events = data.get('events', [])
            if events:
                print(f"  {current_date.strftime('%Y-%m-%d')}: {len(events)} games")
            
            for event in events:
                try:
                    competitions = event.get('competitions', [{}])[0]
                    competitors = competitions.get('competitors', [])
                    
                    home_competitor = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                    away_competitor = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                    
                    if not home_competitor or not away_competitor:
                        continue
                    
                    home_team_name = home_competitor.get('team', {}).get('displayName', '')
                    away_team_name = away_competitor.get('team', {}).get('displayName', '')
                    
                    home_team = ESPN_TEAM_MAP.get(home_team_name, home_team_name)
                    away_team = ESPN_TEAM_MAP.get(away_team_name, away_team_name)
                    
                    # Get game time
                    game_date_str = event.get('date', '')
                    if game_date_str:
                        game_dt = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                        # Convert to Eastern Time
                        from zoneinfo import ZoneInfo
                        et_dt = game_dt.astimezone(ZoneInfo('America/New_York'))
                        game_date = et_dt.strftime('%Y-%m-%d')
                        game_time = et_dt.strftime('%H:%M')
                    else:
                        game_date = current_date.strftime('%Y-%m-%d')
                        game_time = 'TBD'
                    
                    all_games.append({
                        'game_date': game_date,
                        'game_time': game_time,
                        'away_team': away_team,
                        'home_team': home_team,
                        'game_id': event.get('id', ''),
                        'status': event.get('status', {}).get('type', {}).get('description', 'Scheduled')
                    })
                    
                except Exception as e:
                    print(f"    Warning: Failed to parse game: {e}")
                    continue
        
        except Exception as e:
            print(f"  Error fetching {current_date.strftime('%Y-%m-%d')}: {e}")
        
        current_date += timedelta(days=1)
    
    # Save to CSV
    df = pd.DataFrame(all_games)
    output_file = Path('data/nba_schedule_2025_26.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Downloaded {len(all_games)} games")
    print(f"✅ Saved to {output_file}")
    print(f"\nGames by status:")
    print(df['status'].value_counts())
    
    return df

if __name__ == '__main__':
    schedule_df = download_season_schedule()
