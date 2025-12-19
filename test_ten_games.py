"""Process games one at a time to find where explosion happens"""
import sqlite3
from src.features.off_def_elo_system import OffDefEloSystem

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

# Clear all ELO
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DELETE FROM elo_ratings WHERE season = '2025-26'")
conn.commit()

# Get first 10 games
cursor.execute("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_results
    WHERE season = '2025-26'
    ORDER BY game_date, ROWID
    LIMIT 10
""")
games = cursor.fetchall()
conn.close()

# Initialize ELO system
elo_system = OffDefEloSystem(db_path=DB_PATH)

print("=" * 80)
print("PROCESSING FIRST 10 GAMES")
print("=" * 80)

for i, (game_date, home_team, away_team, home_score, away_score) in enumerate(games):
    print(f"\nGame {i+1}: {home_team} {home_score} vs {away_team} {away_score} ({game_date})")
    
    result = elo_system.update_game(
        season='2025-26',
        game_date=game_date,
        home_team=home_team,
        away_team=away_team,
        home_points=home_score,
        away_points=away_score,
        is_playoffs=False
    )
    
    # Check for explosion
    home_composite = (result['home_off_after'] + result['home_def_after']) / 2
    away_composite = (result['away_off_after'] + result['away_def_after']) / 2
    
    print(f"   {home_team}: {home_composite:.1f}   {away_team}: {away_composite:.1f}")
    
    if abs(home_composite) > 10000 or abs(away_composite) > 10000:
        print(f"\n❌ EXPLOSION DETECTED AT GAME {i+1}!")
        print(f"   {home_team} composite: {home_composite:.1f}")
        print(f"   {away_team} composite: {away_composite:.1f}")
        print(f"   Home off: {result['home_off_after']:.1f}, def: {result['home_def_after']:.1f}")
        print(f"   Away off: {result['away_off_after']:.1f}, def: {result['away_def_after']:.1f}")
        break

print("\n" + "=" * 80)
print("✅ All 10 games processed successfully" if i == 9 else f"⚠️ Stopped at game {i+1}")
