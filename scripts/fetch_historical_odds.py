"""
Fetch historical NBA moneyline odds from The Odds API
Builds complete odds database for 2024-25 season validation

This script fetches historical closing odds at 10-minute intervals before each game.
Cost: 10 credits per request (1 market, 1 region)
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

def fetch_historical_odds(api_key: str, date: str):
    """Fetch historical odds snapshot at a specific date"""
    
    url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds'
    
    params = {
        'apiKey': api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso',
        'date': date
    }
    
    print(f"Fetching odds for {date}...", end=' ')
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None, None, None
    
    # Check remaining requests
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    last_cost = response.headers.get('x-requests-last')
    
    print(f"✓ (Cost: {last_cost}, Remaining: {remaining})")
    
    result = response.json()
    
    timestamp = result.get('timestamp')
    data = result.get('data', [])
    
    return data, timestamp, remaining

def parse_odds(games_data, snapshot_timestamp):
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
        
        # Get bookmaker odds (use DraftKings or FanDuel if available)
        bookmakers = game.get('bookmakers', [])
        
        if not bookmakers:
            continue
        
        # Prefer DraftKings, then FanDuel, then first available
        preferred_books = ['draftkings', 'fanduel', 'betmgm']
        bookmaker = None
        
        for book_key in preferred_books:
            for bm in bookmakers:
                if bm['key'] == book_key:
                    bookmaker = bm
                    break
            if bookmaker:
                break
        
        if not bookmaker:
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
            'snapshot_timestamp': snapshot_timestamp,
            'api_game_id': game['id']
        })
    
    return pd.DataFrame(records)

def get_2024_25_season_dates():
    """Generate list of dates for 2024-25 NBA season"""
    
    # 2024-25 season: October 22, 2024 to April 13, 2025
    start_date = datetime(2024, 10, 22)
    end_date = datetime(2024, 12, 12)  # Today (we'll fetch up to current)
    
    dates = []
    current = start_date
    
    while current <= end_date:
        # For each game day, fetch odds 2 hours before typical game times
        # Most NBA games are 19:00-22:30 ET (00:00-03:30 UTC next day)
        # Fetch at 22:00 UTC (5:00 PM ET) to get closing lines
        snapshot_time = current.replace(hour=22, minute=0, second=0)
        dates.append(snapshot_time.isoformat() + 'Z')
        
        current += timedelta(days=1)
    
    return dates

def save_to_database(df, db_path='data/live/historical_odds_2024_25.db'):
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
            snapshot_timestamp TEXT,
            api_game_id TEXT,
            UNIQUE(game_date, home_team, away_team, snapshot_timestamp)
        )
    """)
    
    # Insert or replace
    df.to_sql('moneyline_odds', conn, if_exists='append', index=False)
    
    conn.commit()
    
    # Get total count
    total = conn.execute("SELECT COUNT(*) FROM moneyline_odds").fetchone()[0]
    unique_games = conn.execute("""
        SELECT COUNT(DISTINCT game_date || home_team || away_team) 
        FROM moneyline_odds
    """).fetchone()[0]
    
    conn.close()
    
    return total, unique_games

def main():
    """Main execution"""
    
    print("="*70)
    print("NBA HISTORICAL ODDS SCRAPER - The Odds API")
    print("="*70)
    print(f"Season: 2024-25 (Oct 22, 2024 - Dec 12, 2024)")
    print(f"Cost: 10 credits per day snapshot")
    print("="*70)
    
    # Get dates to fetch
    dates = get_2024_25_season_dates()
    print(f"\nTotal snapshots to fetch: {len(dates)}")
    print(f"Estimated cost: {len(dates) * 10} credits")
    
    # Confirm before proceeding
    response = input("\nProceed with fetch? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted")
        return
    
    all_odds = []
    fetch_count = 0
    error_count = 0
    
    for date in dates:
        try:
            games_data, timestamp, remaining = fetch_historical_odds(API_KEY, date)
            
            if games_data is None:
                error_count += 1
                continue
            
            if len(games_data) == 0:
                print(f"  No games on {date[:10]}")
                continue
            
            df = parse_odds(games_data, timestamp)
            
            if len(df) > 0:
                all_odds.append(df)
                fetch_count += 1
                print(f"  Parsed {len(df)} games")
            
            # Rate limiting - be respectful
            time.sleep(0.5)
            
            # Stop if running low on credits
            if remaining and int(remaining) < 100:
                print(f"\n⚠️  Low on credits ({remaining} remaining). Stopping.")
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            error_count += 1
            continue
    
    print("\n" + "="*70)
    print("FETCH COMPLETE")
    print("="*70)
    print(f"Successful fetches: {fetch_count}")
    print(f"Errors: {error_count}")
    
    if not all_odds:
        print("No odds data collected")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_odds, ignore_index=True)
    
    print(f"\nTotal odds records: {len(combined_df)}")
    print(f"Date range: {combined_df['game_date'].min()} to {combined_df['game_date'].max()}")
    print(f"Unique games: {combined_df.groupby(['game_date', 'home_team', 'away_team']).ngroups}")
    
    # Save to database
    db_path = 'data/live/historical_odds_2024_25.db'
    total_records, unique_games = save_to_database(combined_df, db_path)
    
    print(f"\n✓ Saved to {db_path}")
    print(f"  Total records: {total_records}")
    print(f"  Unique games: {unique_games}")
    
    # Save to CSV backup
    csv_path = 'data/live/historical_moneyline_2024_25.csv'
    combined_df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")
    
    # Validate data quality
    print("\n" + "="*70)
    print("DATA QUALITY CHECK")
    print("="*70)
    
    # Check for impossible odds
    both_favorites = combined_df[(combined_df['home_ml_odds'] < 0) & (combined_df['away_ml_odds'] < 0)]
    both_underdogs = combined_df[(combined_df['home_ml_odds'] > 0) & (combined_df['away_ml_odds'] > 0)]
    
    print(f"Both teams favorites: {len(both_favorites)}")
    print(f"Both teams underdogs: {len(both_underdogs)}")
    
    if len(both_favorites) > 0 or len(both_underdogs) > 0:
        print("⚠️  WARNING: Found impossible odds")
    else:
        print("✅ No impossible odds combinations")
    
    # Check vig
    def american_to_implied(odds):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)
    
    combined_df['home_implied'] = combined_df['home_ml_odds'].apply(american_to_implied)
    combined_df['away_implied'] = combined_df['away_ml_odds'].apply(american_to_implied)
    combined_df['total_implied'] = combined_df['home_implied'] + combined_df['away_implied']
    
    print(f"\nVig (Total Implied Probability):")
    print(f"  Mean: {combined_df['total_implied'].mean():.4f} (should be ~1.04-1.06)")
    print(f"  Std:  {combined_df['total_implied'].std():.4f}")
    
    suspicious = combined_df[(combined_df['total_implied'] < 1.00) | (combined_df['total_implied'] > 1.15)]
    if len(suspicious) > 0:
        print(f"\n⚠️  {len(suspicious)} games with suspicious vig")
    else:
        print(f"\n✅ All odds have realistic vig")
    
    print("\n" + "="*70)
    print("READY FOR BACKTEST")
    print("="*70)
    print(f"Use {db_path} for clean moneyline validation")
    print(f"Expected sample size: {unique_games} unique games")
    print(f"This is {unique_games / 12.3:.1f}% of full 2024-25 season (~1230 games)")

if __name__ == '__main__':
    main()
