"""
ELO System Diagnostic - Analyzing the Brooklyn Nets "Ghost" Rating

Current Configuration:
"""
import sqlite3

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

print("=" * 80)
print("ELO SYSTEM CONFIGURATION")
print("=" * 80)

print("""
From config/constants.py:
├── REGULAR_SEASON_BASE_K = 20
├── PLAYOFF_BASE_K = 30
├── ELO_MARGIN_SCALE = 0.25
├── ELO_POINT_EXPECTATION_SCALE = 10 (points per 100 ELO diff)
├── LEAGUE_AVG_POINTS = 110
├── OFF_ELO_BASELINE = 1500
└── DEF_ELO_BASELINE = 1500

From off_def_elo_system.py update_game():
├── Base K-factor: 20 (regular season)
├── Margin Factor: min(effective_margin / 0.25, 1.5)
│   └── Caps at 1.5x for large margins
├── K_off = base_k * (1 + 0.5 * margin_factor)
│   └── Maximum: 20 * (1 + 0.5 * 1.5) = 35
├── K_def = base_k * (1 + 0.5 * margin_factor)
│   └── Maximum: 20 * (1 + 0.5 * 1.5) = 35
│
├── BLOWOUT DAMPENING:
│   ├── Threshold: 20 points
│   └── Dampening: 0.5 (reduce excess by 50%)
│
├── Expected Points Calculation:
│   ├── exp_home_pts = 110 + (home_off_elo - away_def_elo) / 10
│   └── exp_away_pts = 110 + (away_off_elo - home_def_elo) / 10
│
└── ELO Update:
    ├── new_off_elo = old_off_elo + k_off * (actual_pts - expected_pts) / 10
    └── new_def_elo = old_def_elo - k_def * (opponent_error) / 10

Composite ELO = (off_elo + def_elo) / 2
""")

print("=" * 80)
print("BROOKLYN NETS ELO HISTORY (2025-26 Season)")
print("=" * 80)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get all Brooklyn games
cursor.execute("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_results
    WHERE season = '2025-26' AND (home_team = 'BKN' OR away_team = 'BKN')
    ORDER BY game_date
""")
bkn_games = cursor.fetchall()

# Get Brooklyn's ELO progression
cursor.execute("""
    SELECT game_date, off_elo, def_elo, composite_elo
    FROM elo_ratings
    WHERE season = '2025-26' AND team = 'BKN'
    ORDER BY game_date
""")
bkn_elo_history = cursor.fetchall()

print(f"\nBrooklyn played {len(bkn_games)} games")
print(f"Brooklyn has {len(bkn_elo_history)} ELO records")

# Match them up
print("\n" + "=" * 100)
print(f"{'Date':<12} {'Matchup':<20} {'Score':<10} {'Result':<6} {'Off ELO':<10} {'Def ELO':<10} {'Composite':<10}")
print("=" * 100)

record_wins = 0
record_losses = 0

for i, (game_date, home, away, home_score, away_score) in enumerate(bkn_games):
    is_home = (home == 'BKN')
    opponent = away if is_home else home
    bkn_score = home_score if is_home else away_score
    opp_score = away_score if is_home else home_score
    
    result = 'WIN' if bkn_score > opp_score else 'LOSS'
    margin = abs(bkn_score - opp_score)
    
    if result == 'WIN':
        record_wins += 1
    else:
        record_losses += 1
    
    matchup = f"{'BKN' if is_home else opponent} @ {opponent if is_home else 'BKN'}"
    score = f"{bkn_score}-{opp_score}"
    
    # Find corresponding ELO
    elo_record = None
    for elo_date, off, def_e, comp in bkn_elo_history:
        if elo_date == game_date:
            elo_record = (off, def_e, comp)
            break
    
    if elo_record:
        off, def_e, comp = elo_record
        flag = "🔴" if margin > 20 else ""
        print(f"{game_date:<12} {matchup:<20} {score:<10} {result:<6} {off:<10.1f} {def_e:<10.1f} {comp:<10.1f} {flag}")
    else:
        print(f"{game_date:<12} {matchup:<20} {score:<10} {result:<6} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

print("=" * 100)
print(f"\nBrooklyn Record: {record_wins}-{record_losses}")

# Analysis
if bkn_elo_history:
    first_elo = bkn_elo_history[0]
    last_elo = bkn_elo_history[-1]
    
    print("\n" + "=" * 80)
    print("BROOKLYN ELO ANALYSIS")
    print("=" * 80)
    
    print(f"\nStarting ELO (after game 1):")
    print(f"   Off: {first_elo[1]:.1f}")
    print(f"   Def: {first_elo[2]:.1f}")
    print(f"   Composite: {first_elo[3]:.1f}")
    
    print(f"\nCurrent ELO (after game {len(bkn_elo_history)}):")
    print(f"   Off: {last_elo[1]:.1f}")
    print(f"   Def: {last_elo[2]:.1f}")
    print(f"   Composite: {last_elo[3]:.1f}")
    
    print(f"\nELO Change:")
    print(f"   Off: {last_elo[1] - first_elo[1]:+.1f}")
    print(f"   Def: {last_elo[2] - first_elo[2]:+.1f}")
    print(f"   Composite: {last_elo[3] - first_elo[3]:+.1f}")
    
    print(f"\n🚨 THE PROBLEM:")
    print(f"   Brooklyn is 7-18 but has composite ELO of {last_elo[3]:.1f}")
    print(f"   Their defensive ELO ({last_elo[2]:.1f}) is HIGHER than OKC (1606)!")
    print(f"   This is likely due to:")
    print(f"   1. Small blowout wins being over-weighted")
    print(f"   2. K-factor too small (20) - not reactive enough")
    print(f"   3. Starting all teams at 1500 (no prior knowledge)")

# Check for big Brooklyn wins
print("\n" + "=" * 80)
print("BROOKLYN'S BIGGEST WINS")
print("=" * 80)

big_wins = []
for game_date, home, away, home_score, away_score in bkn_games:
    is_home = (home == 'BKN')
    bkn_score = home_score if is_home else away_score
    opp_score = away_score if is_home else home_score
    
    if bkn_score > opp_score:
        margin = bkn_score - opp_score
        opponent = away if is_home else home
        big_wins.append((game_date, opponent, margin, bkn_score, opp_score))

big_wins.sort(key=lambda x: x[2], reverse=True)

for game_date, opponent, margin, bkn_score, opp_score in big_wins[:5]:
    print(f"   {game_date}: BKN {bkn_score}, {opponent} {opp_score} (Margin: {margin})")

conn.close()

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
The ELO system is TOO CONSERVATIVE. Recommendations:

1. INCREASE K-FACTOR:
   - Change REGULAR_SEASON_BASE_K from 20 → 32 or 40
   - This makes ratings react faster to wins/losses
   - Early season needs higher K to separate teams quickly

2. ADD RECORD-BASED INITIALIZATION:
   - Instead of starting all teams at 1500
   - Use previous season final ELO with regression
   - Or use current win% to set starting ELO

3. STRENGTHEN BLOWOUT DAMPENING:
   - Current: 50% reduction for margin > 20
   - Consider: 70% reduction or cap at 15 effective points

4. ADD WIN PROBABILITY COMPONENT:
   - Current system only uses scoring margin
   - Should also heavily weight W/L outcome
   - A 1-point win should still boost ELO significantly
""")
