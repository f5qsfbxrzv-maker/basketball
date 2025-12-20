"""
ESPN Schedule Service - Fetches NBA schedule from ESPN API
More reliable than The Odds API for schedule data
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict
import sqlite3
from pathlib import Path

# Team name mapping (ESPN to our abbreviations)
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


class ESPNScheduleService:
    """Fetch NBA schedule from ESPN API"""
    
    def __init__(self, db_path: str = 'data/live/nba_betting_data.db', schedule_csv: str = 'data/nba_schedule_2025_26.csv'):
        self.db_path = db_path
        self.schedule_csv = schedule_csv
        self.cache = {}
        self.schedule_df = None
        self._init_database()
        self._load_schedule_csv()
    
    def _load_schedule_csv(self):
        """Load full season schedule from CSV if available"""
        from pathlib import Path
        import pandas as pd
        
        csv_path = Path(self.schedule_csv)
        if csv_path.exists():
            try:
                self.schedule_df = pd.read_csv(csv_path)
                print(f"[SCHEDULE CSV] Loaded {len(self.schedule_df)} games from {csv_path.name}")
            except Exception as e:
                print(f"[SCHEDULE CSV] Failed to load: {e}")
                self.schedule_df = None
        else:
            print(f"[SCHEDULE CSV] Not found: {csv_path}")
            print(f"[SCHEDULE CSV] Run 'python download_season_schedule.py' to create it")
    
    def clear_cache(self, target_date: str = None):
        """Clear cache for a specific date or all dates"""
        if target_date:
            if target_date in self.cache:
                del self.cache[target_date]
                print(f"[ESPN CACHE] Cleared cache for {target_date}")
        else:
            self.cache.clear()
            print("[ESPN CACHE] Cleared all cached games")
    
    def _init_database(self):
        """Create schedule table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS espn_schedule (
                game_id TEXT PRIMARY KEY,
                game_date TEXT NOT NULL,
                game_time TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                season TEXT DEFAULT '2024-25',
                game_status TEXT,
                fetched_at TEXT,
                UNIQUE(game_date, home_team, away_team)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_espn_game_date ON espn_schedule(game_date)
        """)
        
        conn.commit()
        conn.close()
    
    def fetch_games_for_date(self, target_date: str, save_to_db: bool = True) -> List[Dict]:
        """
        Fetch games from ESPN for a specific date
        
        Args:
            target_date: Date in YYYY-MM-DD format
            save_to_db: Whether to save to database
            
        Returns:
            List of game dicts
        """
        # Check cache
        if target_date in self.cache:
            print(f"[ESPN CACHE] Using cached games for {target_date}")
            return self.cache[target_date]
        
        # Check CSV schedule first (fastest, no API needed)
        if self.schedule_df is not None:
            csv_games = self._get_from_csv(target_date)
            if csv_games:
                self.cache[target_date] = csv_games
                if save_to_db:
                    self._save_to_database(csv_games)
                return csv_games
        
        # Check database
        games = self._get_from_database(target_date)
        if games:
            self.cache[target_date] = games
            return games
        
        # Fetch from ESPN API as last resort
        try:
            # ESPN uses YYYYMMDD format
            espn_date = target_date.replace('-', '')
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={espn_date}"
            
            print(f"[ESPN API] Fetching games for {target_date}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Debug: print raw event count
            raw_events = data.get('events', [])
            print(f"[ESPN API] Raw response has {len(raw_events)} events for {target_date}")
            
            games = self._parse_espn_response(data, target_date)
            
            if games and save_to_db:
                self._save_to_database(games)
            
            print(f"[ESPN API] Parsed {len(games)} games for {target_date}")
            self.cache[target_date] = games
            return games
            
        except Exception as e:
            print(f"[ESPN ERROR] {e}")
            return []
    
    def _parse_espn_response(self, data: Dict, game_date: str) -> List[Dict]:
        """Parse ESPN API response"""
        games = []
        
        for event in data.get('events', []):
            try:
                # Get teams
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
                
                # Get game time and extract actual game date from ESPN (not the query date)
                game_date_str = event.get('date', '')
                actual_game_date = game_date  # Default to query date
                if game_date_str:
                    game_dt = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                    # Convert to local time (EST/EDT for NBA)
                    from zoneinfo import ZoneInfo
                    local_dt = game_dt.astimezone(ZoneInfo('America/New_York'))
                    actual_game_date = local_dt.strftime('%Y-%m-%d')
                    game_time = local_dt.strftime('%H:%M:%S')
                else:
                    game_time = 'TBD'
                
                # Get status
                status = event.get('status', {}).get('type', {}).get('description', 'Scheduled')
                
                games.append({
                    'game_id': event.get('id', f"{away_team}@{home_team}_{actual_game_date}"),
                    'game_date': actual_game_date,
                    'game_time': game_time,
                    'home_team': home_team,
                    'away_team': away_team,
                    'game_status': status
                })
                
            except Exception as e:
                print(f"[ESPN WARNING] Failed to parse game: {e}")
                continue
        
        return games
    
    def _get_from_csv(self, target_date: str) -> List[Dict]:
        """Get games from CSV schedule"""
        if self.schedule_df is None:
            return []
        
        try:
            date_games = self.schedule_df[self.schedule_df['game_date'] == target_date]
            
            if len(date_games) == 0:
                return []
            
            games = []
            for _, row in date_games.iterrows():
                games.append({
                    'game_id': row.get('game_id', f"{row['away_team']}@{row['home_team']}_{target_date}"),
                    'game_date': row['game_date'],
                    'game_time': row.get('game_time', 'TBD'),
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'game_status': row.get('status', 'Scheduled')
                })
            
            print(f"[SCHEDULE CSV] Found {len(games)} games for {target_date}")
            return games
            
        except Exception as e:
            print(f"[SCHEDULE CSV] Error reading CSV: {e}")
            return []
    
    def _save_to_database(self, games: List[Dict]):
        """Save games to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        fetched_at = datetime.now().isoformat()
        
        for game in games:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO espn_schedule 
                    (game_id, game_date, game_time, home_team, away_team, season, game_status, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game['game_id'],
                    game['game_date'],
                    game['game_time'],
                    game['home_team'],
                    game['away_team'],
                    '2024-25',
                    game['game_status'],
                    fetched_at
                ))
            except Exception as e:
                print(f"[ESPN WARNING] Failed to save game: {e}")
        
        conn.commit()
        conn.close()
        print(f"[ESPN DB] Saved {len(games)} games")
    
    def _get_from_database(self, target_date: str) -> List[Dict]:
        """Get games from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if data is fresh (less than 6 hours old)
        cursor.execute("""
            SELECT game_id, game_date, game_time, home_team, away_team, game_status, fetched_at
            FROM espn_schedule
            WHERE game_date = ?
        """, (target_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Check freshness
        fetched_at_str = rows[0][6] if rows else None
        if fetched_at_str:
            fetched_at = datetime.fromisoformat(fetched_at_str)
            age = (datetime.now() - fetched_at).total_seconds()
            if age > 6 * 3600:  # 6 hours
                print(f"[ESPN DB] Cache expired for {target_date}")
                return []
        
        games = []
        for row in rows:
            games.append({
                'game_id': row[0],
                'game_date': row[1],
                'game_time': row[2],
                'home_team': row[3],
                'away_team': row[4],
                'game_status': row[5]
            })
        
        print(f"[ESPN DB] Loaded {len(games)} games from database for {target_date}")
        return games
    
    def get_games_range(self, days_ahead: int = 7) -> List[Dict]:
        """Get games for the next N days"""
        all_games = []
        
        for offset in range(days_ahead):
            target_date = (datetime.now() + timedelta(days=offset)).strftime('%Y-%m-%d')
            games = self.fetch_games_for_date(target_date)
            all_games.extend(games)
        
        return all_games


if __name__ == '__main__':
    # Test ESPN schedule service
    service = ESPNScheduleService()
    
    print("\n=== TESTING ESPN SCHEDULE SERVICE ===\n")
    
    # Test today
    today = datetime.now().strftime('%Y-%m-%d')
    games = service.fetch_games_for_date(today)
    
    if games:
        print(f"\nGames today ({today}):")
        for game in games:
            print(f"  {game['away_team']} @ {game['home_team']} - {game['game_time']} ({game['game_status']})")
    
    # Test next 3 days
    future = service.get_games_range(days_ahead=3)
    print(f"\nTotal games in next 3 days: {len(future)}")
