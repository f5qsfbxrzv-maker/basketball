"""
Test specific NBA game series
"""

import json
import sys
from pathlib import Path
from datetime import datetime

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
    print("=" * 80)
    print("KALSHI NBA GAME MARKETS TEST")
    print("=" * 80)
    
    api_key, private_key = load_credentials()
    client = KalshiClient(api_key, private_key, environment='prod')
    
    # Test the specific series we found
    nba_series_tickers = [
        'KXNBAGAME',  # Professional Basketball Game (THIS IS THE ONE!)
        'KXNBASPREAD',  # Pro Basketball Spread
        'KXNBASERIES',  # Professional Basketball Series
        'KXNBATOTAL',  # Pro Basketball Total Points
    ]
    
    for series_ticker in nba_series_tickers:
        print(f"\n{'-' * 80}")
        print(f"Series: {series_ticker}")
        print(f"{'-' * 80}")
        
        try:
            # Get markets for this series
            response = client._make_request('GET', '/markets', params={
                'series_ticker': series_ticker,
                'status': 'open',
                'limit': 100
            })
            
            markets = response.get('markets', [])
            print(f"Found {len(markets)} open markets")
            
            if markets:
                print(f"\nSample markets (first 5):")
                for i, market in enumerate(markets[:5]):
                    ticker = market.get('ticker', 'N/A')
                    title = market.get('title', 'N/A')
                    close_time = market.get('close_time', 'N/A')
                    
                    print(f"\n  {i+1}. {ticker}")
                    print(f"     Title: {title}")
                    print(f"     Close: {close_time}")
                    
                    # Try to get prices
                    try:
                        orderbook = client._make_request('GET', f'/markets/{ticker}/orderbook')
                        yes_price = orderbook.get('yes_price')
                        no_price = orderbook.get('no_price')
                        
                        if yes_price and no_price:
                            print(f"     YES: {yes_price}c, NO: {no_price}c")
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
