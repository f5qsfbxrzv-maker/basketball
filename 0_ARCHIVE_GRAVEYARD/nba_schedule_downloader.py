"""
NBA Schedule Downloader - Fetches games for prediction
Uses nba_api to get today's games with CSV caching
"""
from nba_api.live.nba.endpoints import scoreboard
from datetime import datetime, timedelta
from typing import List, Dict
import time
import csv
from pathlib import Path

# Team abbreviation mapping (nba_api uses different format)
TEAM_ABB_MAP = {
    'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BRK', 'CHA': 'CHO', 'CHI': 'CHI',
    'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GSW': 'GSW',
    'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC', 'LAL': 'LAL', 'MEM': 'MEM',
    'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN', 'NOP': 'NOP', 'NYK': 'NYK',
    'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI', 'PHX': 'PHO', 'POR': 'POR',
    'SAC': 'SAC', 'SAS': 'SAS', 'TOR': 'TOR', 'UTA': 'UTA', 'WAS': 'WAS'
}

# Reverse mapping for nba_api abbreviations
REVERSE_MAP = {v: k for k, v in TEAM_ABB_MAP.items()}
REVERSE_MAP['BRK'] = 'BKN'
REVERSE_MAP['CHO'] = 'CHA'
REVERSE_MAP['PHO'] = 'PHX'


class NBAScheduleDownloader:
    """Fetch NBA games using nba_api scoreboard endpoint with CSV caching"""
    
    def __init__(self, cache_dir: str = 'data/live/schedule_cache'):
        self.cached_games = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_games_for_date(self, date_str: str) -> List[Dict]:
        """Fetch games for a specific date (YYYY-MM-DD format)"""
        # Check memory cache first
        if date_str in self.cached_games:
            print(f"[MEMORY CACHE] Using cached games for {date_str}")
            return self.cached_games[date_str]
        
        # Check CSV cache
        csv_file = self.cache_dir / f"schedule_{date_str}.csv"
        if csv_file.exists():
            cache_age = datetime.now().timestamp() - csv_file.stat().st_mtime
            if cache_age < 6 * 3600:  # 6 hours
                print(f"[CSV CACHE] Loading games from {csv_file.name}")
                games = self._load_from_csv(csv_file)
                self.cached_games[date_str] = games
                return games
            else:
                print(f"[CSV CACHE] Cache expired, fetching fresh data")
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            if date_str == today:
                print(f"[NBA API] Fetching today's games ({date_str})...")
                board = scoreboard.ScoreBoard()
                games_data = board.get_dict()
                
                games = []
                for game in games_data.get('scoreboard', {}).get('games', []):
                    home_team_raw = game.get('homeTeam', {}).get('teamTricode', '')
                    away_team_raw = game.get('awayTeam', {}).get('teamTricode', '')
                    
                    home_team = REVERSE_MAP.get(home_team_raw, home_team_raw)
                    away_team = REVERSE_MAP.get(away_team_raw, away_team_raw)
                    
                    games.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_time': game.get('gameTimeUTC', 'TBD'),
                        'game_date': date_str,
                        'game_id': game.get('gameId', ''),
                        'game_status': game.get('gameStatusText', 'Scheduled')
                    })
                
                print(f"[NBA API] Found {len(games)} games for {date_str}")
                
                if games:
                    self._save_to_csv(games, csv_file)
                
                self.cached_games[date_str] = games
                return games
            else:
                print(f"[INFO] Cannot fetch future games from nba_api (only provides today)")
                return []
                
        except Exception as e:
            print(f"[ERROR] Fetching games: {e}")
            return []
    
    def _save_to_csv(self, games: List[Dict], csv_file: Path):
        """Save games to CSV file"""
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                if games:
                    writer = csv.DictWriter(f, fieldnames=['game_date', 'home_team', 'away_team', 'game_time', 'game_id', 'game_status'])
                    writer.writeheader()
                    writer.writerows(games)
            print(f"[CSV CACHE] Saved {len(games)} games to {csv_file.name}")
        except Exception as e:
            print(f"[WARNING] Could not save to CSV: {e}")
    
    def _load_from_csv(self, csv_file: Path) -> List[Dict]:
        """Load games from CSV file"""
        games = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                games = list(reader)
            print(f"[CSV CACHE] Loaded {len(games)} games")
        except Exception as e:
            print(f"[WARNING] Could not load from CSV: {e}")
        return games
