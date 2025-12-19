"""
Create Dummy Scenario Data for Logic Testing
Tests edge cases to verify betting logic is correct
"""

import pandas as pd

# Define the Test Scenarios
data = [
    # CASE 1: The "Safe" Favorite
    # Odds: -167 (1.60) | Implied: 62.5%
    # Model: 64.5% (Edge: +2.0%) -> SHOULD BET (Fav Logic > 1%)
    {'date': '2025-12-16', 'team': 'BOS', 'opponent': 'DET', 'moneyline_decimal': 1.60, 'mock_model_prob': 0.645},

    # CASE 2: The "Trap" Favorite
    # Odds: -200 (1.50) | Implied: 66.7%
    # Model: 67.0% (Edge: +0.3%) -> SHOULD PASS (Fav Logic requires > 1%)
    {'date': '2025-12-16', 'team': 'LAL', 'opponent': 'GSW', 'moneyline_decimal': 1.50, 'mock_model_prob': 0.670},

    # CASE 3: The "Jackpot" Underdog
    # Odds: +550 (6.50) | Implied: 15.4%
    # Model: 32.0% (Edge: +16.6%) -> SHOULD BET (Dog Logic > 15%)
    {'date': '2025-12-16', 'team': 'WAS', 'opponent': 'MIA', 'moneyline_decimal': 6.50, 'mock_model_prob': 0.320},

    # CASE 4: The "Noise" Underdog
    # Odds: +300 (4.00) | Implied: 25.0%
    # Model: 30.0% (Edge: +5.0%) -> SHOULD PASS (Dog Logic requires > 15%)
    {'date': '2025-12-16', 'team': 'CHA', 'opponent': 'NYK', 'moneyline_decimal': 4.00, 'mock_model_prob': 0.300},

    # CASE 5: The "Deep" Longshot (High Variance Test)
    # Odds: +800 (9.00) | Implied: 11.1%
    # Model: 27.0% (Edge: +15.9%) -> SHOULD BET (Dog Logic > 15%)
    {'date': '2025-12-16', 'team': 'POR', 'opponent': 'DEN', 'moneyline_decimal': 9.00, 'mock_model_prob': 0.270},

    # CASE 6: The "Mega" Favorite
    # Odds: -900 (1.11) | Implied: 90.1%
    # Model: 92.0% (Edge: +1.9%) -> SHOULD BET (Fav Logic > 1%)
    {'date': '2025-12-16', 'team': 'OKC', 'opponent': 'UTA', 'moneyline_decimal': 1.11, 'mock_model_prob': 0.920}
]

# Create DataFrame
df = pd.DataFrame(data)

# Add dummy feature columns so prediction script doesn't crash
for i in range(1, 25):
    df[f'feature_{i}'] = 0 

# Save
df.to_csv('data/blind_test_games.csv', index=False)
print("✓ SUCCESS: Created 'data/blind_test_games.csv' with 6 logic test cases.")
print(f"\nTest Cases:")
print(f"  1. BOS -167 (Safe Fav, 2.0% edge) → Should BET")
print(f"  2. LAL -200 (Trap Fav, 0.3% edge) → Should REJECT")
print(f"  3. WAS +550 (Jackpot Dog, 16.6% edge) → Should BET")
print(f"  4. CHA +300 (Noise Dog, 5.0% edge) → Should REJECT")
print(f"  5. POR +800 (Deep Dog, 15.9% edge) → Should BET")
print(f"  6. OKC -900 (Mega Fav, 1.9% edge) → Should BET")
