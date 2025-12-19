"""Test Kalshi Sports Filters Endpoint"""
import sys
sys.path.insert(0, 'src/services')
from kalshi_client import KalshiClient
import json

# Load credentials
with open('.kalshi_credentials', 'r') as f:
    lines = f.readlines()
    api_key = lines[0].replace('API_KEY_ID=', '').strip()
    private_key = ''.join(lines[1:]).replace('PRIVATE_KEY=', '').strip()

# Initialize client
client = KalshiClient(
    api_key=api_key,
    api_secret=private_key,
    environment='prod'
)

print("=" * 70)
print("KALSHI SPORTS FILTERS")
print("=" * 70)

# Call the filters endpoint
response = client._make_request('GET', '/search/filters_by_sport')

print("\nSports Ordering:")
for sport in response.get('sport_ordering', []):
    print(f"  - {sport}")

print("\nFilters by Sport:")
filters = response.get('filters_by_sports', {})

# Look for basketball/NBA specifically
for sport_name, sport_data in filters.items():
    if 'basketball' in sport_name.lower() or 'nba' in sport_name.lower():
        print(f"\n{sport_name}:")
        print(json.dumps(sport_data, indent=2))

# If not found in filtered list, show all sports
if not any('basketball' in s.lower() or 'nba' in s.lower() for s in filters.keys()):
    print("\nAll available sports:")
    for sport_name in filters.keys():
        print(f"  - {sport_name}")
