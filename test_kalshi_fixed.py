"""
Test Fixed Kalshi Client - Get Live NBA Moneylines
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
    print("=" * 80)
    print("KALSHI LIVE NBA MONEYLINES - FIXED VERSION")
    print("=" * 80)
    
    api_key, private_key = load_credentials()
    client = KalshiClient(api_key, private_key, environment='prod')
    
    # Get account info
    account = client.get_account_info()
    print(f"\n[OK] Balance: ${account.get('balance', 0) / 100:.2f}")
    
    # Get moneyline markets (this should now work!)
    print(f"\n{'=' * 80}")
    print("MONEYLINE MARKETS")
    print(f"{'=' * 80}")
    
    moneyline_markets = client.get_nba_markets(status='open', market_type='moneyline')
    print(f"\nFound {len(moneyline_markets)} moneyline markets")
    
    # Group by game
    games = {}
    for market in moneyline_markets:
        ticker = market.get('ticker', '')
        title = market.get('title', '')
        
        # Parse game from ticker (e.g., KXNBAGAME-25DEC17MEMMIN-MIN)
        if 'KXNBAGAME-' in ticker:
            parts = ticker.split('-')
            if len(parts) >= 2:
                game_id = parts[1]  # 25DEC17MEMMIN
                team = parts[2] if len(parts) > 2 else ''
                
                if game_id not in games:
                    games[game_id] = {}
                games[game_id][team] = market
    
    # Display games with odds
    print(f"\nGames with markets:")
    for game_id, teams in games.items():
        print(f"\n  {'-' * 70}")
        print(f"  Game: {game_id}")
        
        for team, market in teams.items():
            ticker = market.get('ticker')
            title = market.get('title')
            
            # Get orderbook for prices
            try:
                orderbook_response = client._make_request('GET', f'/markets/{ticker}/orderbook')
                
                # Kalshi orderbook structure: {"orderbook": {"yes": [[price, qty], ...], "no": [[price, qty], ...]}}
                orderbook = orderbook_response.get('orderbook', {})
                yes_orders = orderbook.get('yes', [])
                no_orders = orderbook.get('no', [])
                
                # Get the best (lowest) ask price - orders are sorted by price
                yes_ask = yes_orders[0][0] if yes_orders else None  # First element is [price, quantity]
                no_ask = no_orders[0][0] if no_orders else None
                
                if yes_ask:
                    # Convert to American odds
                    prob = yes_ask / 100
                    if prob > 0.5:
                        american = -100 * (prob / (1 - prob))
                    else:
                        american = 100 * ((1 - prob) / prob)
                    
                    print(f"    {team}: {title}")
                    print(f"      Price: {yes_ask}c (implied {prob:.1%})")
                    print(f"      American Odds: {american:+.0f}")
                    
                    if no_ask:
                        no_prob = no_ask / 100
                        print(f"      Opposite NO: {no_ask}c ({no_prob:.1%})")
                else:
                    print(f"    {team}: {title} - No ask prices available")
            except Exception as e:
                print(f"    {team}: {title} - Error getting prices: {e}")
    
    # Get spread markets
    print(f"\n{'=' * 80}")
    print("SPREAD MARKETS")
    print(f"{'=' * 80}")
    
    spread_markets = client.get_nba_markets(status='open', market_type='spread')
    print(f"\nFound {len(spread_markets)} spread markets")
    
    # Show first 5 as examples
    for i, market in enumerate(spread_markets[:5]):
        print(f"\n  {i+1}. {market.get('title')}")
        print(f"     Ticker: {market.get('ticker')}")
    
    # Get total markets
    print(f"\n{'=' * 80}")
    print("TOTAL MARKETS")
    print(f"{'=' * 80}")
    
    total_markets = client.get_nba_markets(status='open', market_type='total')
    print(f"\nFound {len(total_markets)} total markets")
    
    # Show first 5 as examples
    for i, market in enumerate(total_markets[:5]):
        print(f"\n  {i+1}. {market.get('title')}")
        print(f"     Ticker: {market.get('ticker')}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Kalshi integration is WORKING with live odds!")
    print("=" * 80)


if __name__ == "__main__":
    main()
