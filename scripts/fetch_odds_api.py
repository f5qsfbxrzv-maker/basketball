"""
Fetch NBA moneyline odds from The Odds API
Builds clean odds database for 2024-25 season validation
"""

import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import json

API_KEY = '683a25aa5f99df0fd4aa3a70acf279be'
SPORT = 'basketball_nba'
REGIONS = 'us'
MARKETS = 'h2h'  # head-to-head (moneyline)
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

def fetch_odds(api_key: str, date_from: str = None):
    """Fetch odds from The Odds API"""
    
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/'
    
    params = {
        'apiKey': api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso'
    }
    
    if date_from:
        params['commenceTimeFrom'] = date_from
    
    print(f"Fetching odds from The Odds API...")
    print(f"URL: {url}")
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    # Check remaining requests
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    print(f"API Usage: {used} used, {remaining} remaining")
    
    data = response.json()
    print(f"Fetched {len(data)} games")
    
    return data

def parse_odds(games_data):
    """Parse API response into clean DataFrame"""
    
    records = []
    
    for game in games_data:
        commence_time = game['commence_time']
        game_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        
        home_team_full = game['home_team']
        away_team_full = game['away_team']
        
        # Map to abbreviations
        home_team = TEAM_MAP.get(home_team_full, home_team_full)
        away_team = TEAM_MAP.get(away_team_full, away_team_full)
        
        # Get bookmaker odds (prefer consensus/average)
        bookmakers = game.get('bookmakers', [])
        
        if not bookmakers:
            continue
        
        # Use first available bookmaker (usually DraftKings or FanDuel)
        bookmaker = bookmakers[0]
        markets = bookmaker.get('markets', [])
        
        if not markets:
            continue
        
        # Find h2h market
        h2h_market = None
        for market in markets:
            if market['key'] == 'h2h':
                h2h_market = market
                break
        
        if not h2h_market:
            continue
        
        outcomes = h2h_market['outcomes']
        
        # Map outcomes to home/away
        home_odds = None
        away_odds = None
        
        for outcome in outcomes:
            team_full = outcome['name']
            odds = outcome['price']
            
            if team_full == home_team_full:
                home_odds = odds
            elif team_full == away_team_full:
                away_odds = odds
        
        if home_odds is None or away_odds is None:
            continue
        
        records.append({
            'game_date': game_date.strftime('%Y-%m-%d'),
            'game_time': game_date.strftime('%H:%M:%S'),
            'home_team': home_team,
            'away_team': away_team,
            'home_ml_odds': int(home_odds),
            'away_ml_odds': int(away_odds),
            'bookmaker': bookmaker['title'],
            'timestamp': datetime.now().isoformat(),
            'api_game_id': game['id']
        })
    
    return pd.DataFrame(records)

def save_to_database(df, db_path='data/live/clean_odds.db'):
    """Save odds to SQLite database"""
    
    conn = sqlite3.connect(db_path)
    
    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS moneyline_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            game_time TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_ml_odds INTEGER NOT NULL,
            away_ml_odds INTEGER NOT NULL,
            bookmaker TEXT,
            timestamp TEXT,
            api_game_id TEXT UNIQUE,
            UNIQUE(game_date, home_team, away_team)
        )
    """)
    
    # Insert or replace
    df.to_sql('moneyline_odds', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    
    print(f"✓ Saved {len(df)} games to {db_path}")

def validate_odds(df):
    """Validate odds data quality"""
    
    print("\n" + "="*70)
    print("ODDS VALIDATION")
    print("="*70)
    
    # Check for impossible odds
    both_favorites = df[(df['home_ml_odds'] < 0) & (df['away_ml_odds'] < 0)]
    both_underdogs = df[(df['home_ml_odds'] > 0) & (df['away_ml_odds'] > 0)]
    
    print(f"\nGames where both teams are favorites: {len(both_favorites)}")
    print(f"Games where both teams are underdogs: {len(both_underdogs)}")
    
    if len(both_favorites) > 0:
        print("⚠️  WARNING: Found impossible odds (both favorites)")
        print(both_favorites[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']].head())
    
    # Check vig
    def american_to_implied(odds):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)
    
    df['home_implied'] = df['home_ml_odds'].apply(american_to_implied)
    df['away_implied'] = df['away_ml_odds'].apply(american_to_implied)
    df['total_implied'] = df['home_implied'] + df['away_implied']
    
    print(f"\nVig (Total Implied Probability):")
    print(f"  Mean: {df['total_implied'].mean():.4f} (should be ~1.04-1.06)")
    print(f"  Min:  {df['total_implied'].min():.4f}")
    print(f"  Max:  {df['total_implied'].max():.4f}")
    
    suspicious = df[(df['total_implied'] < 1.00) | (df['total_implied'] > 1.15)]
    if len(suspicious) > 0:
        print(f"\n⚠️  {len(suspicious)} games with suspicious vig:")
        print(suspicious[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds', 'total_implied']].head())
    else:
        print("\n✅ All odds look clean (realistic vig)")
    
    # Check for duplicates
    duplicates = df.groupby(['game_date', 'home_team', 'away_team']).size()
    dupe_games = duplicates[duplicates > 1]
    
    print(f"\nDuplicate games: {len(dupe_games)}")
    if len(dupe_games) > 0:
        print("⚠️  Found duplicates:")
        print(dupe_games.head())
    
    print("\n" + "="*70)

def main():
    """Main execution"""
    
    print("="*70)
    print("NBA ODDS SCRAPER - The Odds API")
    print("="*70)
    
    # Fetch current and upcoming games
    games_data = fetch_odds(API_KEY)
    
    if not games_data:
        print("Failed to fetch odds")
        return
    
    # Parse into DataFrame
    df = parse_odds(games_data)
    
    print(f"\nParsed {len(df)} games")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Show sample
    print("\nSample odds:")
    print(df[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds', 'bookmaker']].head(10))
    
    # Validate
    validate_odds(df)
    
    # Save to database
    save_to_database(df)
    
    # Save to CSV backup
    csv_path = 'data/live/moneyline_odds_current.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. API returns UPCOMING games only (next ~7 days)")
    print("2. For historical 2024-25 season, need to:")
    print("   - Check if you have saved CSV from previous scrape")
    print("   - Or fetch historical odds (may require paid plan)")
    print("3. Run this script daily to build odds history")
    print("4. After 50-100 games, re-run backtest with clean odds")

if __name__ == '__main__':
    main()
