"""
NBA Schedule Service - Fetches upcoming games from The Odds API
Provides 7-day lookahead for game scheduling
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sqlite3

API_KEY = '683a25aa5f99df0fd4aa3a70acf279be'  # Your API key
SPORT = 'basketball_nba'
REGIONS = 'us'
MARKETS = 'h2h'
ODDS_FORMAT = 'american'

# Team name mapping (API names to our abbreviations)
TEAM_MAP = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'LA Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}


class ScheduleService:
    """Fetches NBA schedule from The Odds API"""
    
    def __init__(self, api_key: str = API_KEY, db_path: str = 'data/live/nba_betting_data.db'):
        self.api_key = api_key
        self.db_path = db_path
        self.cache = {}
        self._init_database()
    
    def _init_database(self):
        """Create schedule table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_schedule (
                game_id TEXT PRIMARY KEY,
                game_date TEXT NOT NULL,
                game_time TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                commence_time TEXT,
                fetched_at TEXT,
                UNIQUE(game_date, home_team, away_team)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def fetch_upcoming_games(self, save_to_db: bool = True) -> List[Dict]:
        """
        Fetch upcoming games from The Odds API
        Returns list of games with schedule info
        """
        url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/'
        
        params = {
            'apiKey': self.api_key,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': 'iso'
        }
        
        print(f"[ODDS API] Fetching upcoming games...")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"[ERROR] API returned {response.status_code}: {response.text}")
                return []
            
            # Check API usage
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')
            print(f"[ODDS API] Usage: {used} used, {remaining} remaining")
            
            data = response.json()
            games = self._parse_games(data)
            
            print(f"[ODDS API] Found {len(games)} upcoming games")
            
            if save_to_db and games:
                self._save_to_database(games)
            
            return games
            
        except Exception as e:
            print(f"[ERROR] Fetching from The Odds API: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_games(self, api_data: List[Dict]) -> List[Dict]:
        """Parse API response into game dicts"""
        games = []
        
        for game in api_data:
            try:
                # Extract basic info
                game_id = game.get('id', '')
                commence_time_str = game.get('commence_time', '')
                
                if not commence_time_str:
                    continue
                
                # Parse time
                commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                game_date = commence_time.strftime('%Y-%m-%d')
                game_time = commence_time.strftime('%H:%M:%S')
                
                # Get teams
                home_team_full = game.get('home_team', '')
                away_team_full = game.get('away_team', '')
                
                # Map to abbreviations
                home_team = TEAM_MAP.get(home_team_full, home_team_full)
                away_team = TEAM_MAP.get(away_team_full, away_team_full)
                
                games.append({
                    'game_id': game_id,
                    'game_date': game_date,
                    'game_time': game_time,
                    'home_team': home_team,
                    'away_team': away_team,
                    'commence_time': commence_time_str,
                    'game_status': 'Scheduled'
                })
                
            except Exception as e:
                print(f"[WARNING] Failed to parse game: {e}")
                continue
        
        return games
    
    def _save_to_database(self, games: List[Dict]):
        """Save games to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        fetched_at = datetime.now().isoformat()
        
        for game in games:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO game_schedule 
                    (game_id, game_date, game_time, home_team, away_team, commence_time, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    game['game_id'],
                    game['game_date'],
                    game['game_time'],
                    game['home_team'],
                    game['away_team'],
                    game['commence_time'],
                    fetched_at
                ))
            except Exception as e:
                print(f"[WARNING] Failed to save game {game['home_team']} vs {game['away_team']}: {e}")
        
        conn.commit()
        conn.close()
        print(f"[SCHEDULE] Saved {len(games)} games to database")
    
    def get_games_for_date(self, target_date: str) -> List[Dict]:
        """
        Get games for a specific date (YYYY-MM-DD)
        First tries database, falls back to API if needed
        """
        # Check cache
        if target_date in self.cache:
            return self.cache[target_date]
        
        # Try database first
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT game_id, game_date, game_time, home_team, away_team, commence_time
            FROM game_schedule
            WHERE game_date = ?
        """, (target_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        games = []
        for row in rows:
            games.append({
                'game_id': row[0],
                'game_date': row[1],
                'game_time': row[2],
                'home_team': row[3],
                'away_team': row[4],
                'commence_time': row[5],
                'game_status': 'Scheduled'
            })
        
        if games:
            print(f"[SCHEDULE] Found {len(games)} games in database for {target_date}")
            self.cache[target_date] = games
            return games
        
        # If no games in database, check if date is in future
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        days_ahead = (target_dt - datetime.now()).days
        
        if 0 <= days_ahead <= 7:
            # Within API range - fetch fresh data
            print(f"[SCHEDULE] Fetching from API for {target_date}...")
            all_games = self.fetch_upcoming_games(save_to_db=True)
            
            # Filter for target date
            games = [g for g in all_games if g['game_date'] == target_date]
            self.cache[target_date] = games
            return games
        else:
            print(f"[SCHEDULE] No games found for {target_date}")
            return []
    
    def get_games_range(self, days_ahead: int = 7) -> List[Dict]:
        """Get all games for the next N days"""
        all_games = []
        
        for offset in range(days_ahead):
            target_date = (datetime.now() + timedelta(days=offset)).strftime('%Y-%m-%d')
            games = self.get_games_for_date(target_date)
            all_games.extend(games)
        
        return all_games


if __name__ == '__main__':
    # Test the schedule service
    service = ScheduleService()
    
    print("\n=== TESTING SCHEDULE SERVICE ===\n")
    
    # Fetch upcoming games
    games = service.fetch_upcoming_games()
    
    if games:
        print(f"\nUpcoming games:")
        for game in games[:10]:  # Show first 10
            print(f"  {game['game_date']} - {game['away_team']} @ {game['home_team']}")
        
        # Test getting games for today
        today = datetime.now().strftime('%Y-%m-%d')
        today_games = service.get_games_for_date(today)
        print(f"\nGames today ({today}): {len(today_games)}")
        
        # Test getting games for next 3 days
        future_games = service.get_games_range(days_ahead=3)
        print(f"\nGames in next 3 days: {len(future_games)}")
    else:
        print("No games found - check API key or NBA season schedule")
