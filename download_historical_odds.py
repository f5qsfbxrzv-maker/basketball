"""
Download Historical NBA Moneyline Odds - 2023-24 Season
Uses The Odds API (https://the-odds-api.com/)
Fetches historical moneyline odds and saves to CSV for backtesting
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = '683a25aa5f99df0fd4aa3a70acf279be'
SPORT = 'basketball_nba'
MARKETS = 'h2h'  # h2h = moneyline (head-to-head)
REGIONS = 'us'   # US bookmakers
BOOKMAKER = 'draftkings'  # Or 'fanduel', 'betmgm', etc.

# 2023-24 NBA Season dates
SEASON_START = datetime(2023, 10, 24)
SEASON_END = datetime(2024, 4, 14)  # Regular season end

OUTPUT_FILE = 'data/closing_odds_2023_24.csv'

# ==============================================================================
# API FUNCTIONS
# ==============================================================================
def fetch_historical_odds(api_key, date, sport='basketball_nba', regions='us', markets='h2h'):
    """
    Fetch historical odds for a specific date
    
    Note: The Odds API's historical endpoint requires a premium subscription
    If you don't have access to historical data, you'll need to use a different approach
    """
    url = f'https://api.the-odds-api.com/v4/historical/sports/{sport}/odds'
    
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
        'date': date.strftime('%Y-%m-%dT12:00:00Z'),
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Check remaining requests
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        print(f"  API requests used: {used}, remaining: {remaining}")
        
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(f"  ❌ Authentication failed. Check your API key.")
        elif e.response.status_code == 422:
            print(f"  ⚠️  No data available for {date.date()}")
        else:
            print(f"  ❌ HTTP Error {e.response.status_code}: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error fetching data: {e}")
        return None

def parse_odds_data(games_data, target_bookmaker='draftkings'):
    """Parse odds data into a clean format"""
    parsed_games = []
    
    for game in games_data.get('data', []):
        game_id = game.get('id')
        commence_time = game.get('commence_time')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Find the target bookmaker's odds
        bookmaker_data = None
        for bookmaker in game.get('bookmakers', []):
            if bookmaker.get('key') == target_bookmaker:
                bookmaker_data = bookmaker
                break
        
        if not bookmaker_data:
            continue
        
        # Extract h2h (moneyline) odds
        for market in bookmaker_data.get('markets', []):
            if market.get('key') == 'h2h':
                outcomes = market.get('outcomes', [])
                
                home_odds = None
                away_odds = None
                
                for outcome in outcomes:
                    if outcome.get('name') == home_team:
                        home_odds = outcome.get('price')
                    elif outcome.get('name') == away_team:
                        away_odds = outcome.get('price')
                
                if home_odds and away_odds:
                    # Convert decimal odds to American odds
                    home_ml = decimal_to_american(home_odds)
                    away_ml = decimal_to_american(away_odds)
                    
                    parsed_games.append({
                        'game_date': datetime.fromisoformat(commence_time.replace('Z', '+00:00')).date(),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_ml_odds': home_ml,
                        'away_ml_odds': away_ml,
                        'bookmaker': target_bookmaker,
                        'snapshot_timestamp': datetime.utcnow().isoformat(),
                        'api_game_id': game_id
                    })
    
    return parsed_games

def decimal_to_american(decimal_odds):
    """Convert decimal odds to American odds"""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))

# ==============================================================================
# MAIN DOWNLOAD PROCESS
# ==============================================================================
def download_season_odds():
    print("="*90)
    print("DOWNLOADING 2023-24 NBA SEASON HISTORICAL ODDS")
    print("="*90)
    
    if API_KEY == 'YOUR_API_KEY_HERE':
        print("\n❌ ERROR: Please set your Odds API key in the script")
        print("   Get your API key from: https://the-odds-api.com/")
        return
    
    print(f"\n[1/3] Configuration:")
    print(f"  Season: 2023-24 ({SEASON_START.date()} to {SEASON_END.date()})")
    print(f"  Sport: {SPORT}")
    print(f"  Market: {MARKETS} (moneyline)")
    print(f"  Bookmaker: {BOOKMAKER}")
    
    # Test API connection
    print(f"\n[2/3] Testing API connection...")
    test_response = requests.get(
        f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds',
        params={'apiKey': API_KEY, 'regions': REGIONS, 'markets': MARKETS}
    )
    
    if test_response.status_code == 401:
        print("  ❌ Authentication failed. Check your API key.")
        return
    elif test_response.status_code != 200:
        print(f"  ❌ API test failed with status {test_response.status_code}")
        return
    
    print("  ✅ API connection successful")
    
    # Download historical data
    print(f"\n[3/3] Downloading historical odds...")
    print(f"  Note: Historical data requires a premium API subscription")
    print(f"  Fetching {(SEASON_END - SEASON_START).days} days of data...")
    
    all_games = []
    current_date = SEASON_START
    
    while current_date <= SEASON_END:
        print(f"\n  Fetching {current_date.date()}...", end=" ")
        
        odds_data = fetch_historical_odds(API_KEY, current_date, SPORT, REGIONS, MARKETS)
        
        if odds_data:
            games = parse_odds_data(odds_data, BOOKMAKER)
            all_games.extend(games)
            print(f"✅ {len(games)} games")
        else:
            print("⚠️ No data")
        
        current_date += timedelta(days=1)
        time.sleep(1)  # Rate limiting - 1 second between requests
    
    # Save to CSV
    if all_games:
        df = pd.DataFrame(all_games)
        df = df.sort_values('game_date')
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n{'='*90}")
        print(f"✅ SUCCESS")
        print(f"{'='*90}")
        print(f"  Total games downloaded: {len(df)}")
        print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        print(f"  Saved to: {OUTPUT_FILE}")
        print(f"{'='*90}\n")
    else:
        print(f"\n❌ No games downloaded. Check your API subscription level.")
        print(f"   Historical data requires a premium plan.")

# ==============================================================================
# ALTERNATIVE: MANUAL CSV FORMAT
# ==============================================================================
def create_template_csv():
    """
    If historical API is not available, create a template CSV
    You can manually fill this from other sources (odds aggregators, web scraping, etc.)
    """
    print("\n" + "="*90)
    print("CREATING TEMPLATE CSV")
    print("="*90)
    print("\nIf you don't have access to historical API data, you can:")
    print("  1. Use this template to manually enter odds")
    print("  2. Export from a paid odds database service")
    print("  3. Web scrape historical odds from sites like oddsportal.com")
    print("  4. Use archived odds data from betting databases")
    
    template_data = {
        'game_date': ['2023-10-24', '2023-10-24'],
        'home_team': ['Los Angeles Lakers', 'Golden State Warriors'],
        'away_team': ['Denver Nuggets', 'Phoenix Suns'],
        'home_ml_odds': [-150, 120],
        'away_ml_odds': [130, -140],
        'bookmaker': ['draftkings', 'draftkings'],
        'snapshot_timestamp': [datetime.utcnow().isoformat(), datetime.utcnow().isoformat()],
        'api_game_id': ['example_1', 'example_2']
    }
    
    df = pd.DataFrame(template_data)
    template_file = 'data/closing_odds_2023_24_TEMPLATE.csv'
    df.to_csv(template_file, index=False)
    
    print(f"\n  Template created: {template_file}")
    print(f"  Replace with real data before backtesting")
    print("="*90 + "\n")

if __name__ == '__main__':
    # Try to download from API
    download_season_odds()
    
    # Uncomment if you need the template instead
    # create_template_csv()
