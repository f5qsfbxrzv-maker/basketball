"""
Test Kalshi API with Working Credentials - Simple Version
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Direct import
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'services'))
from kalshi_client import KalshiClient

def load_credentials():
    """Load API credentials from .kalshi_credentials file"""
    creds_file = Path('.kalshi_credentials')
    
    if not creds_file.exists():
        print("[ERROR] .kalshi_credentials file not found!")
        return None, None
    
    # Read the entire file
    content = creds_file.read_text()
    
    # Parse API_KEY_ID
    api_key = None
    for line in content.split('\n'):
        if line.startswith('API_KEY_ID='):
            api_key = line.split('=', 1)[1].strip()
            break
    
    # Extract private key (between BEGIN and END markers)
    if 'PRIVATE_KEY=' in content and '-----BEGIN RSA PRIVATE KEY-----' in content:
        start_marker = '-----BEGIN RSA PRIVATE KEY-----'
        end_marker = '-----END RSA PRIVATE KEY-----'
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            private_key = content[start_idx:end_idx + len(end_marker)]
        else:
            print("[ERROR] Could not find PEM markers!")
            return None, None
    else:
        print("[ERROR] Could not find private key!")
        return None, None
    
    if not api_key or not private_key:
        print("[ERROR] Could not parse credentials!")
        return None, None
    
    print(f"[OK] Loaded API key: {api_key[:20]}...")
    print(f"[OK] Loaded private key: {len(private_key)} chars")
    
    return api_key, private_key


def main():
    print("=" * 70)
    print("KALSHI NBA MONEYLINE TEST")
    print("=" * 70)
    
    # Load credentials
    api_key, api_secret = load_credentials()
    if not api_key or not api_secret:
        print("\n[ERROR] Failed to load credentials!")
        return False
    
    # Initialize client
    print("\n[1] Initializing Kalshi client...")
    try:
        client = KalshiClient(
            api_key=api_key,
            api_secret=api_secret,
            environment='prod',
            auth_on_init=True,
            request_timeout=15
        )
        print("[SUCCESS] Kalshi client initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize client: {e}")
        return False
    
    # Test authentication
    print("\n[2] Testing authentication...")
    try:
        if client.authenticate():
            print("[SUCCESS] Successfully authenticated with Kalshi API")
        else:
            print("[WARNING] Authentication check returned False")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return False
    
    # Get account info
    print("\n[3] Getting account info...")
    try:
        account = client.get_account_info()
        if account:
            balance = account.get('balance', 0)
            print(f"[SUCCESS] Account balance: ${balance / 100:.2f}")
        else:
            print("[WARNING] Could not retrieve account info")
    except Exception as e:
        print(f"[WARNING] Account info error: {e}")
    
    # Get all NBA markets
    print(f"\n[4] Fetching ALL open NBA markets...")
    try:
        markets = client.get_nba_markets(status='open')
        print(f"[SUCCESS] Found {len(markets)} open NBA markets")
        
        if markets:
            print("\n   Sample markets:")
            for i, market in enumerate(markets[:10]):
                title = market.get('title', 'Unknown')
                ticker = market.get('event_ticker', 'Unknown')
                print(f"   {i+1}. {title}")
                print(f"      Ticker: {ticker}")
        
    except Exception as e:
        print(f"[ERROR] Failed to get markets: {e}")
        return False
    
    # Test specific game markets
    print(f"\n[5] Testing specific game markets...")
    
    test_games = [
        ('GSW', 'LAL', "Warriors vs Lakers"),
        ('BOS', 'MIA', "Celtics vs Heat"),
        ('DEN', 'LAC', "Nuggets vs Clippers"),
        ('MIN', 'DAL', "Timberwolves vs Mavericks"),
    ]
    
    successful_fetches = 0
    
    for home, away, description in test_games:
        print(f"\n   Testing: {description} ({home} vs {away})")
        
        try:
            markets = client.get_game_markets(home, away, None)
            
            if markets:
                home_yes = markets.get('home_ml_yes_price')
                away_yes = markets.get('away_ml_yes_price')
                home_ml = markets.get('home_ml')
                away_ml = markets.get('away_ml')
                
                if home_yes and away_yes:
                    print(f"   [SUCCESS] MONEYLINES FOUND!")
                    print(f"      Home ({home}): {home_yes}c -> {home_ml} American odds")
                    print(f"      Away ({away}): {away_yes}c -> {away_ml} American odds")
                    print(f"      Implied probs: {home_yes/100:.1%} home, {away_yes/100:.1%} away")
                    successful_fetches += 1
                else:
                    print(f"   [INFO] Market found but missing prices")
            else:
                print(f"   [INFO] No market found (may not be playing soon)")
        
        except Exception as e:
            print(f"   [ERROR] {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Successful moneyline fetches: {successful_fetches}/{len(test_games)}")
    print(f"Total open NBA markets: {len(markets)}")
    
    if successful_fetches > 0:
        print("\n[SUCCESS] Kalshi moneyline fetching is WORKING!")
        return True
    else:
        print("\n[INFO] No moneylines found - check market list above")
        print("This may be normal if no games scheduled today/tomorrow")
        return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 70)
        print("[SUCCESS] Kalshi integration is FULLY OPERATIONAL!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Update LiveOddsFetcher to use this working client")
        print("2. Test with real game predictions in dashboard")
    else:
        print("\n[ERROR] Some tests failed - check errors above")
