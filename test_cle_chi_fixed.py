"""
Test Cleveland vs Chicago Odds After Fix
"""

import sys
sys.path.insert(0, 'src/services')
from kalshi_client import KalshiClient

# Load credentials
with open('.kalshi_credentials', 'r') as f:
    lines = f.readlines()
    api_key = lines[0].replace('API_KEY=', '').replace('API_KEY_ID=', '').strip()
    private_key = ''.join(lines[1:]).replace('PRIVATE_KEY=', '').strip()

# Initialize client
client = KalshiClient(
    api_key=api_key,
    api_secret=private_key,
    environment='prod'
)

print("=" * 70)
print("TESTING FIXED ODDS FOR CLEVELAND VS CHICAGO")
print("=" * 70)

# Use get_game_markets which is what LiveOddsFetcher calls
markets = client.get_game_markets('CLE', 'CHI', '2025-12-17')

if markets:
    print("\n✅ SUCCESS - Found market data:")
    print(f"  Home (Cleveland):")
    print(f"    Probability: {markets.get('home_ml_yes_price', 'N/A')}%")
    print(f"    American Odds: {markets.get('home_ml', 'N/A')}")
    
    print(f"\n  Away (Chicago):")
    print(f"    Probability: {markets.get('away_ml_yes_price', 'N/A')}%")
    print(f"    American Odds: {markets.get('away_ml', 'N/A')}")
    
    # Check if they match expected values
    cle_prob = markets.get('home_ml_yes_price', 0)
    chi_prob = markets.get('away_ml_yes_price', 0)
    
    print(f"\n{'='*70}")
    print("VALIDATION")
    print("=" * 70)
    
    if cle_prob == 66:
        print("✅ Cleveland probability CORRECT (66%)")
    else:
        print(f"❌ Cleveland probability WRONG (got {cle_prob}%, expected 66%)")
    
    if chi_prob == 34:
        print("✅ Chicago probability CORRECT (34%)")
    else:
        print(f"❌ Chicago probability WRONG (got {chi_prob}%, expected 34%)")
    
    # Note: Kalshi probabilities don't always sum to 100 due to vig
    total = cle_prob + chi_prob
    print(f"\nTotal implied probability: {total}% (includes vig)")
    
else:
    print("\n❌ FAILED - No market data found")
    print("This might mean:")
    print("  1. Game hasn't started yet / market not open")
    print("  2. API returned no data")
    print("  3. Team name matching failed")
