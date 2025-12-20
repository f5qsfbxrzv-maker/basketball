from src.services.injury_scraper import InjuryScraper

scraper = InjuryScraper()

print("=== TESTING INJURY SCRAPER ===\n")

# Test LAL
lal_data = scraper.get_team_injury_impact('LAL')
print(f"Lakers (LAL) Injury Impact:")
print(f"  Total Impact: {lal_data['total_impact']}")
print(f"  Star Injuries: {lal_data['star_injuries']}")
print(f"  Details: {lal_data.get('breakdown', 'N/A')}\n")

# Test LAC  
lac_data = scraper.get_team_injury_impact('LAC')
print(f"Clippers (LAC) Injury Impact:")
print(f"  Total Impact: {lac_data['total_impact']}")
print(f"  Star Injuries: {lac_data['star_injuries']}")
print(f"  Details: {lac_data.get('breakdown', 'N/A')}\n")

# Test differential
differential = scraper.get_game_injury_differential('LAL', 'LAC')
print(f"Game Differential (LAL home vs LAC):")
print(f"  Value: {differential}")
print(f"  Interpretation: {'LAL more injured' if differential < 0 else 'LAC more injured' if differential > 0 else 'Equal'}")
