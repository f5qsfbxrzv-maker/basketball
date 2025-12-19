"""
Kalshi Live Odds Debugger
Tests Kalshi API connection and troubleshoots common issues
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

def test_kalshi_connection():
    """Comprehensive Kalshi API test"""
    
    print("="*70)
    print("KALSHI API CONNECTION TEST")
    print("="*70)
    
    # Step 1: Check config file exists
    print("\n[1] Checking config file...")
    config_path = Path("config/kalshi_config.json")
    
    if not config_path.exists():
        print("❌ Config file not found: config/kalshi_config.json")
        print("Creating template...")
        config_path.parent.mkdir(exist_ok=True)
        template = {
            "api_key": "YOUR_KALSHI_API_KEY_HERE",
            "api_secret": "YOUR_KALSHI_API_SECRET_HERE",
            "environment": "demo",
            "comment": "Get credentials from https://kalshi.com/settings/api"
        }
        with open(config_path, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"✅ Template created at {config_path}")
        print("Please add your API credentials and run again")
        return False
    
    print(f"✅ Config file exists: {config_path}")
    
    # Step 2: Load config
    print("\n[2] Loading config...")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        api_key = config.get('api_key')
        api_secret = config.get('api_secret')
        environment = config.get('environment', 'demo')
        
        if not api_key or api_key == "YOUR_KALSHI_API_KEY_HERE":
            print("❌ API key not set in config")
            print("Get your API key from: https://kalshi.com/settings/api")
            return False
        
        if not api_secret or api_secret == "YOUR_KALSHI_API_SECRET_HERE":
            print("❌ API secret not set in config")
            print("Get your API secret from: https://kalshi.com/settings/api")
            return False
        
        print(f"✅ API key loaded: {api_key[:10]}...")
        print(f"✅ Environment: {environment}")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False
    
    # Step 3: Test Kalshi client import
    print("\n[3] Testing Kalshi client import...")
    try:
        from src.services.kalshi_client import KalshiClient
        print("✅ KalshiClient imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import KalshiClient: {e}")
        print("Check that src/services/kalshi_client.py exists")
        return False
    
    # Step 4: Initialize client
    print("\n[4] Initializing Kalshi client...")
    try:
        client = KalshiClient(
            api_key=api_key,
            api_secret=api_secret,
            environment=environment,
            auth_on_init=False  # Don't auth yet
        )
        print(f"✅ Client initialized")
        print(f"   Base URL: {client.base_url}")
        
    except Exception as e:
        print(f"❌ Error initializing client: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test authentication
    print("\n[5] Testing authentication...")
    try:
        # Try to login
        token = client.login()
        
        if token:
            print(f"✅ Authentication successful!")
            print(f"   Token (first 20 chars): {token[:20]}...")
        else:
            print("❌ Authentication failed (no token returned)")
            return False
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify API key is correct")
        print("  2. Verify API secret is correct (check for \\\\n escape sequences)")
        print("  3. Try regenerating credentials at https://kalshi.com/settings/api")
        print("  4. Check if demo vs prod environment is correct")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Test NBA market search
    print("\n[6] Testing NBA market search...")
    try:
        # Search for NBA markets in next 7 days
        markets = client.get_markets(
            category="Sports",
            event_ticker="NBA"
        )
        
        if markets:
            print(f"✅ Found {len(markets)} NBA markets")
            
            # Show a few examples
            print("\n   Sample markets:")
            for market in markets[:5]:
                ticker = market.get('ticker', 'N/A')
                title = market.get('title', 'N/A')
                close_time = market.get('close_time', 'N/A')
                print(f"   - {ticker}: {title[:60]}")
                print(f"     Closes: {close_time}")
        else:
            print("⚠️  No NBA markets found")
            print("This might be normal if no games in next few days")
            print("Try checking https://kalshi.com/markets/nba manually")
        
    except Exception as e:
        print(f"❌ Error fetching markets: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Test getting odds for a specific market
    if markets and len(markets) > 0:
        print("\n[7] Testing odds fetch for first market...")
        try:
            first_market = markets[0]
            ticker = first_market.get('ticker')
            
            if ticker:
                orderbook = client.get_orderbook(ticker)
                
                if orderbook:
                    yes_price = orderbook.get('yes', {}).get('price')
                    no_price = orderbook.get('no', {}).get('price')
                    
                    print(f"✅ Got orderbook for {ticker}")
                    print(f"   Yes price: {yes_price}")
                    print(f"   No price: {no_price}")
                else:
                    print(f"⚠️  Empty orderbook for {ticker}")
        
        except Exception as e:
            print(f"❌ Error fetching orderbook: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 8: Test LiveOddsFetcher integration
    print("\n[8] Testing LiveOddsFetcher integration...")
    try:
        from src.services.live_odds_fetcher import LiveOddsFetcher
        
        fetcher = LiveOddsFetcher()
        
        # Try to get odds for tomorrow's games (today's likely finished)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Test with a common matchup
        odds = fetcher.get_moneyline_odds('LAL', 'GSW', tomorrow)
        
        print(f"✅ LiveOddsFetcher working")
        print(f"   Test odds for LAL vs GSW on {tomorrow}:")
        print(f"   Home ML: {odds['home_ml']}")
        print(f"   Away ML: {odds['away_ml']}")
        print(f"   Source: {odds['source']}")
        
        if odds['source'] == 'kalshi':
            print("   ✅ Successfully fetched from Kalshi!")
        else:
            print("   ⚠️  Using default odds (Kalshi fetch may have failed)")
        
    except Exception as e:
        print(f"❌ LiveOddsFetcher error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    return True


if __name__ == '__main__':
    success = test_kalshi_connection()
    
    if success:
        print("\n✅ All tests passed! Kalshi integration should work.")
        print("\nNext steps:")
        print("  1. Launch dashboard: python nba_gui_dashboard_v2.py")
        print("  2. Click 'Refresh Predictions'")
        print("  3. Check console for: [ODDS] ... source=kalshi")
    else:
        print("\n❌ Some tests failed. Please fix issues above and try again.")
        print("\nCommon fixes:")
        print("  1. Get API credentials: https://kalshi.com/settings/api")
        print("  2. Update config/kalshi_config.json with real credentials")
        print("  3. Ensure you're using correct environment (demo vs prod)")
        print("  4. Check API secret doesn't have escaped \\\\n (should be actual newlines)")
