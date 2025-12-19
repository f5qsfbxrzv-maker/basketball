"""
Debug Brooklyn's defensive ELO - why is it so high when they're 7-18?
"""
import sqlite3

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 100)
print("BROOKLYN DEFENSIVE ELO DEBUG")
print("=" * 100)

# Get all Brooklyn games with ELO changes
cursor.execute("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_results
    WHERE season = '2025-26' AND (home_team = 'BKN' OR away_team = 'BKN')
    ORDER BY game_date
""")
games = cursor.fetchall()

cursor.execute("""
    SELECT game_date, off_elo, def_elo, composite_elo
    FROM elo_ratings
    WHERE season = '2025-26' AND team = 'BKN'
    ORDER BY game_date
""")
elos = cursor.fetchall()

print(f"\nBrooklyn: 7 wins, 18 losses")
print(f"Question: Why is their DEFENSIVE ELO so high (1594)?")
print(f"\nLet's trace each game and see how defense changes:\n")

print(f"{'Date':<12} {'Opponent':<8} {'BKN Pts':<8} {'Opp Pts':<8} {'Result':<6} {'Margin':<7} {'Def ELO':<10} {'Change':<10}")
print("=" * 100)

prev_def = 1500  # Starting value

for i, (game_date, home, away, home_score, away_score) in enumerate(games):
    is_home = (home == 'BKN')
    opponent = away if is_home else home
    bkn_pts = home_score if is_home else away_score
    opp_pts = away_score if is_home else home_score
    
    result = 'WIN' if bkn_pts > opp_pts else 'LOSS'
    margin = bkn_pts - opp_pts
    
    # Get ELO after this game
    if i < len(elos):
        _, off, def_elo, comp = elos[i]
        change = def_elo - prev_def
        
        # Flag suspicious changes
        flag = ""
        if result == 'LOSS' and change > 0:
            flag = "🚨 LOSS but DEF UP!"
        elif result == 'WIN' and change < 0:
            flag = "⚠️  WIN but DEF DOWN"
        elif abs(change) > 50:
            flag = "📈 HUGE CHANGE"
        
        print(f"{game_date:<12} {opponent:<8} {bkn_pts:<8} {opp_pts:<8} {result:<6} {margin:<+7} {def_elo:<10.1f} {change:<+10.1f} {flag}")
        prev_def = def_elo
    else:
        print(f"{game_date:<12} {opponent:<8} {bkn_pts:<8} {opp_pts:<8} {result:<6} {margin:<+7} {'N/A':<10} {'N/A':<10}")

print("=" * 100)

# Analysis
print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)

# Count suspicious patterns
losses_def_up = 0
wins_def_down = 0

prev_def = 1500
for i, (game_date, home, away, home_score, away_score) in enumerate(games):
    is_home = (home == 'BKN')
    bkn_pts = home_score if is_home else away_score
    opp_pts = away_score if is_home else home_score
    result = 'WIN' if bkn_pts > opp_pts else 'LOSS'
    
    if i < len(elos):
        _, off, def_elo, comp = elos[i]
        change = def_elo - prev_def
        
        if result == 'LOSS' and change > 0:
            losses_def_up += 1
        if result == 'WIN' and change < 0:
            wins_def_down += 1
        
        prev_def = def_elo

print(f"\n🚨 SUSPICIOUS PATTERNS:")
print(f"   Losses where DEF ELO went UP: {losses_def_up} (should be going DOWN!)")
print(f"   Wins where DEF ELO went DOWN: {wins_def_down}")

print(f"\n💡 THE PROBLEM:")
print(f"   If Brooklyn LOSES games but their defensive ELO goes UP,")
print(f"   the defensive ELO calculation has the WRONG SIGN.")
print(f"   ")
print(f"   When Brooklyn loses 100-113 (allowing 113 points), their defense")
print(f"   should get WORSE (lower rating), not BETTER (higher rating).")
print(f"   ")
print(f"   Current system seems to be updating defense BACKWARDS:")
print(f"   - Opponent scores a lot → Brooklyn defense 'surprised' → ELO goes UP ❌")
print(f"   - Should be: Opponent scores a lot → Brooklyn defense failed → ELO goes DOWN ✓")

# Check the math
print("\n" + "=" * 100)
print("ELO UPDATE LOGIC CHECK")
print("=" * 100)

print("""
From off_def_elo_system.py:

# Defensive error is negative of opponent offensive error
home_def_error = -away_off_error
away_def_error = -home_off_error

# Update defense
new_home_def = home_latest.def_elo - k_def * (away_off_error / ELO_POINT_EXPECTATION_SCALE)
new_away_def = away_latest.def_elo - k_def * (home_off_error / ELO_POINT_EXPECTATION_SCALE)

The issue: Higher defensive ELO should mean BETTER defense (fewer points allowed).
But the current formula might have the sign flipped.

If away team scores MORE than expected (positive error), then:
- home defense FAILED → should go DOWN
- Current: new_home_def = old - k * (positive) = old - positive = LOWER ✓

If away team scores LESS than expected (negative error), then:
- home defense SUCCEEDED → should go UP
- Current: new_home_def = old - k * (negative) = old + positive = HIGHER ✓

Wait... the logic looks correct. Let me check if it's the INTERPRETATION that's wrong.
Maybe HIGHER def_elo = WORSE defense (more points allowed)?
But that contradicts the composite formula which treats both off/def equally.
""")

conn.close()
