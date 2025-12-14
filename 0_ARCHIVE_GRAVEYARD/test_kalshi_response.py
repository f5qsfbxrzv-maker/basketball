"""Debug Kalshi response"""
from src.services._OLD_kalshi_client import KalshiClient
import json

# Load credentials
with open('config/api_credentials.json', 'r') as f:
    creds = json.load(f)

client = KalshiClient(
    api_key=creds['kalshi_api_key'],
    api_secret=creds['kalshi_api_secret'],
    auth_on_init=True
)

print("🔍 Getting game markets for GSW vs MIN\n")
market_data = client.get_game_markets('GSW', 'MIN', '2025-12-12')

print("Raw response:")
print(json.dumps(market_data, indent=2))
