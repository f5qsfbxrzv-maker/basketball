"""Test Kalshi API connection with real credentials"""
import sys
import os

# Add path and import directly
kalshi_client_path = r'c:\Users\d76do\OneDrive\Documents\New Basketball Model\src\services\kalshi_client.py'

# Read credentials
creds_path = r'c:\Users\d76do\OneDrive\Documents\New Basketball Model\.kalshi_credentials'

def read_credentials():
    """Read API key and private key from credentials file"""
    with open(creds_path, 'r') as f:
        content = f.read()
    
    # Extract API key
    api_key = None
    for line in content.split('\n'):
        if line.startswith('API_KEY_ID='):
            api_key = line.split('=', 1)[1].strip()
            break
    
    # Extract private key (between BEGIN and END)
    private_key_lines = []
    in_key = False
    for line in content.split('\n'):
        if 'BEGIN RSA PRIVATE KEY' in line:
            in_key = True
        if in_key:
            private_key_lines.append(line)
        if 'END RSA PRIVATE KEY' in line:
            break
    
    private_key = '\n'.join(private_key_lines)
    
    return api_key, private_key

# Import the KalshiClient directly
import importlib.util
spec = importlib.util.spec_from_file_location("kalshi_client", kalshi_client_path)
kalshi_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kalshi_module)
KalshiClient = kalshi_module.KalshiClient

from datetime import datetime, timedelta

def test_connection():
    print("=" * 60)
    print("KALSHI API CONNECTION TEST")
    print("=" * 60)
    
    try:
        # Read credentials
        print("\n1. Reading credentials from .kalshi_credentials...")
        api_key, private_key = read_credentials()
        print(f"   API Key: {api_key[:20]}...")
        print(f"   Private Key: {len(private_key)} chars")
        
        # Initialize client
        print("\n2. Initializing KalshiClient (auto-authenticates)...")
        client = KalshiClient(
            api_key=api_key,
            api_secret=private_key,
            environment='prod'
        )
        print("   ✅ Authentication successful!")
        
        # Get account balance
        print("\n3. Fetching account info...")
        account_info = client.get_account_info()
        balance = account_info.get('balance', 0) / 100  # Convert cents to dollars
        print(f"   💰 Account Balance: ${balance:,.2f}")
        
        # Get available NBA markets
        print("\n4. Fetching NBA markets...")
        markets = client.get_nba_markets(status='open')
        
        nba_markets = [m for m in markets if 'NBA' in m.get('title', '').upper()]
        print(f"   📊 Found {len(nba_markets)} NBA markets")
        
        # Show next few games
        print("\n5. Upcoming NBA Games:")
        game_count = 0
        for market in nba_markets[:10]:  # Show first 10
            title = market.get('title', 'Unknown')
            close_time = market.get('close_time', '')
            yes_price = market.get('yes_bid', 0)
            no_price = market.get('no_bid', 0)
            
            if close_time:
                try:
                    close_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                    if close_dt > datetime.now().astimezone():
                        game_count += 1
                        print(f"\n   Game {game_count}: {title}")
                        print(f"   Close: {close_dt.strftime('%Y-%m-%d %H:%M')}")
                        print(f"   Yes: {yes_price}¢ | No: {no_price}¢")
                except Exception as e:
                    pass  # Skip malformed dates
        
        if game_count == 0:
            print("   ⚠️ No future games found (markets may be closed for today)")
            print("   Note: Markets close ~10 minutes before game time")
        
        print("\n" + "=" * 60)
        print("✅ KALSHI API TEST COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()
