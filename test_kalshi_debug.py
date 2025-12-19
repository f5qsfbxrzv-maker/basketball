"""
Debug Kalshi NBA Market Discovery
Test different API parameters to find NBA markets
"""

import json
import sys
from pathlib import Path

# Direct import
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'services'))
from kalshi_client import KalshiClient

def load_credentials():
    """Load API credentials from .kalshi_credentials file"""
    creds_file = Path('.kalshi_credentials')
    content = creds_file.read_text()
    
    # Parse API_KEY_ID
    api_key = None
    for line in content.split('\n'):
        if line.startswith('API_KEY_ID='):
            api_key = line.split('=', 1)[1].strip()
            break
    
    # Extract private key
    start_marker = '-----BEGIN RSA PRIVATE KEY-----'
    end_marker = '-----END RSA PRIVATE KEY-----'
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        private_key = content[start_idx:end_idx + len(end_marker)]
    else:
        return None, None
    
    return api_key, private_key


def main():
    print("=" * 80)
    print("KALSHI NBA MARKET DISCOVERY DEBUG")
    print("=" * 80)
    
    # Load and authenticate
    api_key, private_key = load_credentials()
    client = KalshiClient(api_key, private_key, environment='prod')
    
    print("\n[1] Authentication Test...")
    account = client.get_account_info()
    print(f"    Balance: ${account.get('balance', 0) / 100:.2f}")
    print(f"    Exchange Active: {account.get('exchange_active', False)}")
    
    # Test different market queries
    print("\n[2] Testing Different Market Queries...")
    print("-" * 80)
    
    # Query 1: Original method
    print("\n  Query 1: Original get_nba_markets()")
    try:
        markets = client.get_nba_markets(status='open')
        print(f"    Result: {len(markets)} markets")
        if markets:
            print(f"    Sample: {markets[0].get('title', 'N/A')[:60]}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Query 2: Series ticker BBALLNBA
    print("\n  Query 2: /markets with series_ticker=BBALLNBA")
    try:
        response = client._make_request('GET', '/markets', params={
            'series_ticker': 'BBALLNBA',
            'status': 'open',
            'limit': 100
        })
        markets = response.get('markets', [])
        print(f"    Result: {len(markets)} markets")
        if markets:
            print(f"    Sample: {markets[0].get('ticker', 'N/A')}")
            print(f"    Title: {markets[0].get('title', 'N/A')[:60]}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Query 3: Just NBA
    print("\n  Query 3: /markets with series_ticker=NBA")
    try:
        response = client._make_request('GET', '/markets', params={
            'series_ticker': 'NBA',
            'status': 'open',
            'limit': 100
        })
        markets = response.get('markets', [])
        print(f"    Result: {len(markets)} markets")
        if markets:
            print(f"    Sample: {markets[0].get('ticker', 'N/A')}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Query 4: Events endpoint
    print("\n  Query 4: /events with event_ticker=NBA")
    try:
        response = client._make_request('GET', '/events', params={
            'event_ticker': 'NBA',
            'status': 'open',
            'limit': 100
        })
        events = response.get('events', [])
        print(f"    Result: {len(events)} events")
        if events:
            print(f"    Sample: {events[0].get('title', 'N/A')[:60]}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Query 5: All open markets (then filter)
    print("\n  Query 5: All open markets (filter for NBA/Basketball)")
    try:
        response = client._make_request('GET', '/markets', params={
            'status': 'open',
            'limit': 500
        })
        all_markets = response.get('markets', [])
        
        # Filter for NBA
        nba_keywords = ['NBA', 'BASKETBALL', 'LAKERS', 'WARRIORS', 'CELTICS', 
                        'BUCKS', 'HEAT', 'NUGGETS', 'SUNS', 'MAVERICKS']
        nba_markets = []
        for m in all_markets:
            title = m.get('title', '').upper()
            ticker = m.get('ticker', '').upper()
            if any(keyword in title or keyword in ticker for keyword in nba_keywords):
                nba_markets.append(m)
        
        print(f"    Result: {len(all_markets)} total, {len(nba_markets)} NBA markets")
        if nba_markets:
            print(f"\n    NBA Markets Found:")
            for i, m in enumerate(nba_markets[:5]):
                print(f"      {i+1}. {m.get('ticker', 'N/A')}")
                print(f"         {m.get('title', 'N/A')[:70]}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Query 6: Get series list
    print("\n  Query 6: Get all available series")
    try:
        response = client._make_request('GET', '/series', params={'limit': 100})
        series = response.get('series', [])
        
        # Look for NBA-related series
        nba_series = [s for s in series if 'NBA' in s.get('ticker', '').upper() 
                      or 'BASKETBALL' in s.get('title', '').upper()]
        
        print(f"    Result: {len(series)} total series, {len(nba_series)} NBA-related")
        if nba_series:
            print(f"\n    NBA Series:")
            for s in nba_series:
                print(f"      Ticker: {s.get('ticker', 'N/A')}")
                print(f"      Title: {s.get('title', 'N/A')}")
                print(f"      Category: {s.get('category', 'N/A')}")
    except Exception as e:
        print(f"    Error: {e}")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE - Check results above to find working query")
    print("=" * 80)


if __name__ == "__main__":
    main()
