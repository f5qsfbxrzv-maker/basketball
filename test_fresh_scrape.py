from src.services.injury_scraper import InjuryScraper

scraper = InjuryScraper()

print("=== FRESH SCRAPE (NO CACHE) ===\n")
df = scraper.scrape_injuries(use_cache=False)

print(f"Total injuries scraped: {len(df)}\n")

lal_data = scraper.get_team_injury_impact('LAL', df)
print(f"LAL Impact: {lal_data['total_penalty']}")
print(f"  Injuries: {len(lal_data['injuries'])}")

lac_data = scraper.get_team_injury_impact('LAC', df)
print(f"\nLAC Impact: {lac_data['total_penalty']}")
print(f"  Injuries: {len(lac_data['injuries'])}")

differential = scraper.get_game_injury_differential('LAL', 'LAC', df)
print(f"\nDifferential: {differential}")
