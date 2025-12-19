"""Test ELO update for ONE game to debug explosion"""
import sqlite3
from src.features.off_def_elo_system import OffDefEloSystem

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

# Clear all ELO
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DELETE FROM elo_ratings WHERE season = '2025-26'")
conn.commit()
conn.close()

# Initialize ELO system
elo_system = OffDefEloSystem(db_path=DB_PATH)

# Process FIRST game: BOS vs NYK on 2025-10-22, BOS won 132-109
print("=" * 80)
print("TESTING FIRST GAME: BOS 132 vs NYK 109 (Oct 22, 2025)")
print("=" * 80)

print("\n📊 Initial State:")
print("   BOS: off_elo=1500, def_elo=1500, composite=1500")
print("   NYK: off_elo=1500, def_elo=1500, composite=1500")

result = elo_system.update_game(
    season='2025-26',
    game_date='2025-10-22',
    home_team='BOS',
    away_team='NYK',
    home_points=132,
    away_points=109,
    is_playoffs=False
)

print("\n📈 Game Result:")
print(f"   Home: BOS scored {result['home_points']} (expected {result['exp_home_pts']:.1f})")
print(f"   Away: NYK scored {result['away_points']} (expected {result['exp_away_pts']:.1f})")
print(f"   Margin: {result['margin']} points (BOS won)")

print("\n⚙️ K-Factors:")
print(f"   k_off_home: {result['k_off_home']:.2f}")
print(f"   k_off_away: {result['k_off_away']:.2f}")

print("\n🎯 NEW ELO VALUES:")
print(f"   BOS Offense: {result['home_off_before']:.1f} → {result['home_off_after']:.1f} (Δ {result['home_off_after'] - result['home_off_before']:.1f})")
print(f"   BOS Defense: {result['home_def_before']:.1f} → {result['home_def_after']:.1f} (Δ {result['home_def_after'] - result['home_def_before']:.1f})")
print(f"   BOS Composite: {(result['home_off_before'] + result['home_def_before'])/2:.1f} → {(result['home_off_after'] + result['home_def_after'])/2:.1f}")
print()
print(f"   NYK Offense: {result['away_off_before']:.1f} → {result['away_off_after']:.1f} (Δ {result['away_off_after'] - result['away_off_before']:.1f})")
print(f"   NYK Defense: {result['away_def_before']:.1f} → {result['away_def_after']:.1f} (Δ {result['away_def_after'] - result['away_def_before']:.1f})")
print(f"   NYK Composite: {(result['away_off_before'] + result['away_def_before'])/2:.1f} → {(result['away_off_after'] + result['away_def_after'])/2:.1f}")

print("\n" + "=" * 80)
if abs(result['home_off_after']) > 10000:
    print("❌ EXPLOSION DETECTED!")
    print(f"   BOS off_elo went to {result['home_off_after']}")
else:
    print("✅ Values look reasonable")
