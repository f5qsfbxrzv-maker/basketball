"""
Test Kalshi Orderbook Structure for Cleveland vs Chicago
"""

import sys
sys.path.insert(0, 'src/services')
from kalshi_client import KalshiClient

# Load credentials
with open('.kalshi_credentials', 'r') as f:
    lines = f.readlines()
    api_key = lines[0].replace('API_KEY=', '').strip()
    private_key = ''.join(lines[1:]).replace('PRIVATE_KEY=', '').strip()

# Initialize client
client = KalshiClient(
    api_key=api_key,
    api_secret=private_key,
    environment='prod'
)

print("=" * 70)
print("TESTING CLEVELAND VS CHICAGO ODDS")
print("=" * 70)

# Get NBA markets
markets = client.get_nba_markets(status='open', market_type='moneyline')
print(f"\nFound {len(markets)} open moneyline markets")

# Find Cleveland vs Chicago market
cle_chi_market = None
for market in markets:
    ticker = market.get('ticker', '')
    if 'CLE' in ticker and 'CHI' in ticker:
        cle_chi_market = market
        print(f"\nFound market: {ticker}")
        print(f"Market subtitle: {market.get('subtitle', 'N/A')}")
        print(f"Close time: {market.get('close_time')}")
        break

if not cle_chi_market:
    print("\n❌ Could not find Cleveland vs Chicago market")
    sys.exit(1)

ticker = cle_chi_market['ticker']

# Print FULL market structure first
print(f"\n{'='*70}")
print("FULL MARKET DATA STRUCTURE")
print("=" * 70)
import json
print(json.dumps(cle_chi_market, indent=2))

# Check for price fields in market data
print(f"\n{'='*70}")
print("EXTRACTING PRICES FROM MARKET DATA")
print("=" * 70)

# Check common price fields
price_fields = ['last_price', 'yes_bid', 'yes_ask', 'no_bid', 'no_ask', 
                'previous_yes_bid', 'previous_yes_ask', 'previous_no_bid', 'previous_no_ask']
for field in price_fields:
    if field in cle_chi_market:
        print(f"{field}: {cle_chi_market[field]}")

# Now try orderbook
print(f"\n{'='*70}")
print("ORDERBOOK DATA")
print("=" * 70)

try:
    orderbook_response = client._make_request('GET', f'/markets/{ticker}/orderbook')
    print("Orderbook response:")
    print(json.dumps(orderbook_response, indent=2))
except Exception as e:
    print(f"Error getting orderbook: {e}")
print("EXTRACTING ACTUAL PRICES")
print("=" * 70)

# Method 1: Use last_price
if 'market' in orderbook and orderbook['market'].get('last_price') is not None:
    last_price = orderbook['market']['last_price']
    print(f"\n✅ Last traded price: {last_price}c ({last_price}%)")
    cleveland_prob = last_price / 100.0
    chicago_prob = 1 - cleveland_prob
    print(f"Cleveland probability: {cleveland_prob:.1%}")
    print(f"Chicago probability: {chicago_prob:.1%}")

# Method 2: Use previous_yes_bid
if 'market' in orderbook:
    market_data = orderbook['market']
    if market_data.get('previous_yes_bid'):
        yes_bid = market_data['previous_yes_bid']
        print(f"\n✅ Previous YES bid: {yes_bid}c ({yes_bid}%)")
    if market_data.get('previous_yes_ask'):
        yes_ask = market_data['previous_yes_ask']
        print(f"✅ Previous YES ask: {yes_ask}c ({yes_ask}%)")

# Method 3: Use orderbook arrays
yes_orders = orderbook.get('yes', [])
no_orders = orderbook.get('no', [])

if yes_orders:
    print(f"\n✅ YES orderbook (first 5 levels):")
    for i, order in enumerate(yes_orders[:5]):
        print(f"  Level {i+1}: Price={order[0]}c, Quantity={order[1]}")
        
if no_orders:
    print(f"\n✅ NO orderbook (first 5 levels):")
    for i, order in enumerate(no_orders[:5]):
        print(f"  Level {i+1}: Price={order[0]}c, Quantity={order[1]}")

# Best available prices
if yes_orders and no_orders:
    best_yes_ask = min([order[0] for order in yes_orders])  # Lowest ask to buy YES
    best_no_ask = min([order[0] for order in no_orders])    # Lowest ask to buy NO
    
    print(f"\n{'='*70}")
    print("BEST AVAILABLE PRICES")
    print("=" * 70)
    print(f"Best YES ask (to buy Cleveland wins): {best_yes_ask}c")
    print(f"Best NO ask (to buy Cleveland loses): {best_no_ask}c")
    
    # Calculate fair price (midpoint)
    implied_yes = 100 - best_no_ask  # If NO costs X, YES is worth 100-X
    fair_yes = (best_yes_ask + implied_yes) / 2
    
    print(f"\nImplied YES price from NO: {implied_yes}c")
    print(f"Fair YES price (midpoint): {fair_yes:.1f}c")
    print(f"\nCleveland win probability: {fair_yes:.1f}%")
    print(f"Chicago win probability: {100 - fair_yes:.1f}%")
