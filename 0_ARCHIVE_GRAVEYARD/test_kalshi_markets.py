"""Test what Kalshi NBA markets exist today"""
from multi_source_odds_service import MultiSourceOddsService
from datetime import datetime

service = MultiSourceOddsService()

if service.kalshi_client:
    print("✅ Kalshi connected")
    markets = service.kalshi_client.get_nba_markets(status='open')
    
    print(f"\n📊 Found {len(markets)} NBA markets\n")
    
    # Show first 10 market titles
    for i, market in enumerate(markets[:10]):
        title = market.get('title', 'No title')
        ticker = market.get('ticker', 'No ticker')
        close_time = market.get('close_time', 'Unknown')
        print(f"{i+1}. {title}")
        print(f"   Ticker: {ticker}")
        print(f"   Closes: {close_time}\n")
else:
    print("❌ Kalshi not connected")
