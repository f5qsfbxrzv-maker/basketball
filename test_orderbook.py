"""
Debug Kalshi Orderbook Response
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'services'))
from kalshi_client import KalshiClient

def load_credentials():
    creds_file = Path('.kalshi_credentials')
    content = creds_file.read_text()
    
    api_key = None
    for line in content.split('\n'):
        if line.startswith('API_KEY_ID='):
            api_key = line.split('=', 1)[1].strip()
            break
    
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
    api_key, private_key = load_credentials()
    client = KalshiClient(api_key, private_key, environment='prod')
    
    # Get one moneyline market
    markets = client.get_nba_markets(status='open', market_type='moneyline')
    
    if markets:
        market = markets[0]
        ticker = market.get('ticker')
        
        print(f"Market: {ticker}")
        print(f"Title: {market.get('title')}")
        print("\n" + "=" * 80)
        print("FULL MARKET OBJECT:")
        print("=" * 80)
        print(json.dumps(market, indent=2))
        
        print("\n" + "=" * 80)
        print("ORDERBOOK REQUEST:")
        print("=" * 80)
        
        try:
            orderbook = client._make_request('GET', f'/markets/{ticker}/orderbook')
            print(json.dumps(orderbook, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    
    # Also check a spread market
    print("\n" + "=" * 80)
    print("SPREAD MARKET:")
    print("=" * 80)
    
    spread_markets = client.get_nba_markets(status='open', market_type='spread')
    if spread_markets:
        market = spread_markets[0]
        ticker = market.get('ticker')
        
        print(f"Market: {ticker}")
        print(f"Title: {market.get('title')}")
        
        try:
            orderbook = client._make_request('GET', f'/markets/{ticker}/orderbook')
            print("\nOrderbook:")
            print(json.dumps(orderbook, indent=2))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
