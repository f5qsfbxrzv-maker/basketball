"""
Fetch CLOSING LINE odds for 2024-25 season games
More efficient: fetch odds 30 minutes before each game (closing line)
Requires knowing game times from our training data
"""

import pandas as pd
import requests
import sqlite3
from datetime import datetime, timedelta
import time
import json

API_KEY = '683a25aa5f99df0fd4aa3a70acf279be'
SPORT = 'basketball_nba'
REGIONS = 'us'
MARKETS = 'h2h'
ODDS_FORMAT = 'american'

# Team name mapping
TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 
    'Los Angeles Lakers': 'LAL', 'LA Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

def load_training_data():
    """Load games from training data to get game schedule"""
    
    print("Loading training data for game schedule...")
    df = pd.read_csv('data/training_data_with_temporal_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to FULL COMPLETED 2024-25 season (Oct 2024 - Apr 2025)
    season_df = df[df['season'] == '2024-25'].copy()
    
    # Most NBA games start at 19:00, 19:30, 20:00, 22:00, 22:30 ET
    # Fetch closing odds 30 min before midnight UTC for all games on that day
    season_df['fetch_timestamp'] = season_df['date'].apply(
        lambda d: (d + timedelta(days=1)).replace(hour=0, minute=30, second=0)  # 00:30 UTC next day = ~7:30 PM ET
    )
    
    print(f"Found {len(season_df)} games in COMPLETED 2024-25 season")
    print(f"Date range: {season_df['date'].min().date()} to {season_df['date'].max().date()}")
    
    return season_df[['date', 'home_team', 'away_team', 'fetch_timestamp']].drop_duplicates()

def fetch_historical_odds_for_date(api_key: str, date: str):
    """Fetch historical odds snapshot for a specific timestamp"""
    
    url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds'
    
    params = {
        'apiKey': api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso',
        'date': date
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        return None, None, None
    
    remaining = response.headers.get('x-requests-remaining')
    last_cost = response.headers.get('x-requests-last')
    
    result = response.json()
    timestamp = result.get('timestamp')
    data = result.get('data', [])
    
    return data, timestamp, remaining, last_cost

def parse_and_match_odds(games_data, expected_games, snapshot_timestamp):
    """Parse odds and match to expected games"""
    
    records = []
    
    # Convert expected games to lookup dict
    expected = {}
    for _, row in expected_games.iterrows():
        key = (row['home_team'], row['away_team'])
        expected[key] = row['date']
    
    for game in games_data:
        home_team_full = game['home_team']
        away_team_full = game['away_team']
        
        home_team = TEAM_MAP.get(home_team_full, home_team_full)
        away_team = TEAM_MAP.get(away_team_full, away_team_full)
        
        # Check if this is an expected game
        if (home_team, away_team) not in expected:
            continue
        
        game_date = expected[(home_team, away_team)]
        
        bookmakers = game.get('bookmakers', [])
        if not bookmakers:
            continue
        
        # Prefer DraftKings/FanDuel
        bookmaker = None
        for book_key in ['draftkings', 'fanduel', 'betmgm']:
            for bm in bookmakers:
                if bm['key'] == book_key:
                    bookmaker = bm
                    break
            if bookmaker:
                break
        
        if not bookmaker:
            bookmaker = bookmakers[0]
        
        # Get h2h market
        h2h_market = None
        for market in bookmaker.get('markets', []):
            if market['key'] == 'h2h':
                h2h_market = market
                break
        
        if not h2h_market:
            continue
        
        outcomes = h2h_market['outcomes']
        home_odds = None
        away_odds = None
        
        for outcome in outcomes:
            if outcome['name'] == home_team_full:
                home_odds = outcome['price']
            elif outcome['name'] == away_team_full:
                away_odds = outcome['price']
        
        if home_odds is None or away_odds is None:
            continue
        
        records.append({
            'game_date': game_date.strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'home_ml_odds': int(home_odds),
            'away_ml_odds': int(away_odds),
            'bookmaker': bookmaker['title'],
            'snapshot_timestamp': snapshot_timestamp,
            'api_game_id': game['id']
        })
    
    return pd.DataFrame(records)

def main():
    """Main execution - fetch closing odds for all 2024-25 games"""
    
    print("="*70)
    print("FETCH CLOSING LINE ODDS FOR 2024-25 SEASON")
    print("="*70)
    
    # Load expected games
    games_df = load_training_data()
    
    # Group by date to minimize API calls
    dates = games_df['date'].unique()
    print(f"\nUnique game dates: {len(dates)}")
    print(f"Estimated API cost: {len(dates) * 10} credits")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        return
    
    all_odds = []
    matched = 0
    unmatched = 0
    
    for game_date in sorted(dates):
        games_on_date = games_df[games_df['date'] == game_date]
        fetch_time = games_on_date['fetch_timestamp'].iloc[0]
        
        date_str = game_date.strftime('%Y-%m-%d')
        fetch_str = fetch_time.isoformat() + 'Z'
        
        print(f"\n{date_str}: {len(games_on_date)} games", end=' ')
        
        try:
            games_data, timestamp, remaining, cost = fetch_historical_odds_for_date(API_KEY, fetch_str)
            
            if games_data is None:
                print("❌ Error fetching")
                unmatched += len(games_on_date)
                continue
            
            print(f"(API: {len(games_data)} games, cost: {cost}, remaining: {remaining})")
            
            df = parse_and_match_odds(games_data, games_on_date, timestamp)
            
            if len(df) > 0:
                all_odds.append(df)
                matched += len(df)
                print(f"  ✓ Matched {len(df)}/{len(games_on_date)} games")
                
                if len(df) < len(games_on_date):
                    unmatched += (len(games_on_date) - len(df))
            else:
                print(f"  ⚠️  No matches found")
                unmatched += len(games_on_date)
            
            time.sleep(0.5)  # Rate limiting
            
            if remaining and int(remaining) < 50:
                print(f"\n⚠️  Low credits ({remaining}). Stopping.")
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            unmatched += len(games_on_date)
            continue
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Games matched: {matched}")
    print(f"Games unmatched: {unmatched}")
    print(f"Success rate: {matched/(matched+unmatched)*100:.1f}%")
    
    if not all_odds:
        print("\nNo odds collected")
        return
    
    # Combine and save
    combined_df = pd.concat(all_odds, ignore_index=True)
    
    # Remove duplicates (keep first)
    combined_df = combined_df.drop_duplicates(subset=['game_date', 'home_team', 'away_team'], keep='first')
    
    print(f"\nUnique games with odds: {len(combined_df)}")
    
    # Save to database
    db_path = 'data/live/historical_closing_odds.db'
    conn = sqlite3.connect(db_path)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS moneyline_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_ml_odds INTEGER NOT NULL,
            away_ml_odds INTEGER NOT NULL,
            bookmaker TEXT,
            snapshot_timestamp TEXT,
            api_game_id TEXT,
            UNIQUE(game_date, home_team, away_team)
        )
    """)
    
    combined_df.to_sql('moneyline_odds', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✓ Saved to {db_path}")
    
    # Save CSV
    csv_path = 'data/live/closing_odds_2024_25.csv'
    combined_df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")
    
    # Quick validation
    def american_to_implied(odds):
        return -odds/(-odds+100) if odds < 0 else 100/(odds+100)
    
    combined_df['total_implied'] = (
        combined_df['home_ml_odds'].apply(american_to_implied) +
        combined_df['away_ml_odds'].apply(american_to_implied)
    )
    
    print(f"\nData Quality:")
    print(f"  Mean vig: {combined_df['total_implied'].mean():.4f}")
    print(f"  Bookmakers: {combined_df['bookmaker'].unique()}")
    
    both_fav = combined_df[(combined_df['home_ml_odds'] < 0) & (combined_df['away_ml_odds'] < 0)]
    print(f"  Both favorites: {len(both_fav)} ({'✅ Good' if len(both_fav) == 0 else '⚠️  Check'})")

if __name__ == '__main__':
    main()
