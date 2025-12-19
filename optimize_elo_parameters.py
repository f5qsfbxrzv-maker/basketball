"""
Grid Search Optimization for ELO Hyperparameters
Finds the optimal K-Factor, ELO_SCALE, WIN_WEIGHT, and MOV_BIAS
using Log-Loss minimization across all 2025-26 games.
"""
import sqlite3
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import itertools

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

@dataclass
class TeamRating:
    off_elo: float
    def_elo: float
    
    @property
    def composite(self) -> float:
        return (self.off_elo + self.def_elo) / 2.0

def calculate_log_loss(actual_outcome: int, predicted_prob: float) -> float:
    """Calculate log-loss for a single prediction.
    
    Args:
        actual_outcome: 1 if home team won, 0 if away team won
        predicted_prob: Probability home team wins (0-1)
    
    Returns:
        Log-loss value (lower is better)
    """
    # Clip probability to avoid log(0)
    p = max(0.001, min(0.999, predicted_prob))
    
    if actual_outcome == 1:
        return -math.log(p)
    else:
        return -math.log(1 - p)

def simulate_season(k_factor: float, elo_scale: float, win_weight: float, mov_bias: float) -> Tuple[float, Dict]:
    """Simulate entire season with given parameters and return total log-loss.
    
    Args:
        k_factor: Base K-factor for updates
        elo_scale: Divider for point expectations (higher = more stable)
        win_weight: Bonus/penalty for wins/losses applied to composite
        mov_bias: Multiplier for margin of victory impact (max cap)
    
    Returns:
        (total_log_loss, final_ratings_dict)
    """
    # Get all games chronologically
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT game_date, home_team, away_team, home_score, away_score
        FROM game_results
        WHERE season = '2025-26'
        ORDER BY game_date, ROWID
    """)
    games = cursor.fetchall()
    conn.close()
    
    # Initialize all teams at baseline
    ratings = {}
    for team in ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
                 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
                 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']:
        ratings[team] = TeamRating(1500.0, 1500.0)
    
    total_log_loss = 0.0
    
    for game_date, home_team, away_team, home_score, away_score in games:
        home_rating = ratings[home_team]
        away_rating = ratings[away_team]
        
        # Calculate win probability BEFORE updating (this is the prediction)
        composite_diff = home_rating.composite - away_rating.composite
        composite_diff = max(-800, min(800, composite_diff))  # Prevent overflow
        win_prob_home = 1.0 / (1.0 + 10 ** (-composite_diff / 400))
        
        # Calculate log-loss for this prediction
        home_won = 1 if home_score > away_score else 0
        log_loss = calculate_log_loss(home_won, win_prob_home)
        total_log_loss += log_loss
        
        # Now update ratings for next game
        # Expected points
        exp_home_pts = 110 + (home_rating.off_elo - away_rating.def_elo) / elo_scale
        exp_away_pts = 110 + (away_rating.off_elo - home_rating.def_elo) / elo_scale
        
        # Errors
        home_off_error = home_score - exp_home_pts
        away_off_error = away_score - exp_away_pts
        
        # Margin of victory dampening
        margin = abs(home_score - away_score)
        log_margin = math.log(margin + 1)
        margin_mult = min(log_margin / 3.0, mov_bias)  # mov_bias is the cap
        
        # Point-based updates
        k = k_factor * margin_mult
        new_home_off = home_rating.off_elo + k * (home_off_error / elo_scale)
        new_away_def = away_rating.def_elo + k * (home_off_error / elo_scale)
        new_away_off = away_rating.off_elo + k * (away_off_error / elo_scale)
        new_home_def = home_rating.def_elo + k * (away_off_error / elo_scale)
        
        # Outcome hammer (win/loss adjustment)
        outcome_surprise_home = home_won - win_prob_home
        outcome_surprise_away = (1 - home_won) - (1 - win_prob_home)
        
        # Apply to both offense and defense (50/50 split)
        outcome_adj = win_weight * 0.5
        new_home_off += outcome_surprise_home * outcome_adj
        new_home_def += outcome_surprise_home * outcome_adj
        new_away_off += outcome_surprise_away * outcome_adj
        new_away_def += outcome_surprise_away * outcome_adj
        
        # Update ratings
        ratings[home_team] = TeamRating(new_home_off, new_home_def)
        ratings[away_team] = TeamRating(new_away_off, new_away_def)
    
    return total_log_loss, ratings

def main():
    print("=" * 80)
    print("GRID SEARCH OPTIMIZATION FOR ELO HYPERPARAMETERS")
    print("=" * 80)
    print("\nObjective: Minimize Log-Loss across all 386 games (2025-26 season)")
    print("Log-Loss rewards confident + correct, penalizes confident + wrong\n")
    
    # Define search space
    k_factors = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    elo_scales = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    win_weights = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    mov_biases = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    total_combinations = len(k_factors) * len(elo_scales) * len(win_weights) * len(mov_biases)
    print(f"🔍 Testing {total_combinations} parameter combinations...")
    print(f"   K-Factor: {k_factors}")
    print(f"   ELO_SCALE: {elo_scales}")
    print(f"   WIN_WEIGHT: {win_weights}")
    print(f"   MOV_BIAS: {mov_biases}\n")
    
    best_params = None
    best_log_loss = float('inf')
    best_ratings = None
    
    tested = 0
    for k, scale, win_w, mov in itertools.product(k_factors, elo_scales, win_weights, mov_biases):
        tested += 1
        
        if tested % 100 == 0:
            print(f"   Progress: {tested}/{total_combinations} ({100*tested/total_combinations:.1f}%) | Best Log-Loss: {best_log_loss:.2f}")
        
        try:
            log_loss, ratings = simulate_season(k, scale, win_w, mov)
            
            # Check for explosion (any rating > 5000 or < -1000)
            exploded = any(abs(r.off_elo) > 5000 or abs(r.def_elo) > 5000 for r in ratings.values())
            
            if not exploded and log_loss < best_log_loss:
                best_log_loss = log_loss
                best_params = (k, scale, win_w, mov)
                best_ratings = ratings
        except (OverflowError, ValueError):
            # Parameters caused explosion, skip
            continue
    
    print(f"\n✅ Optimization Complete! Tested {tested} combinations\n")
    
    print("=" * 80)
    print("OPTIMAL PARAMETERS (GOLD STANDARD)")
    print("=" * 80)
    print(f"K-Factor: {best_params[0]:.1f}")
    print(f"ELO_SCALE: {best_params[1]:.1f}")
    print(f"WIN_WEIGHT: {best_params[2]:.1f}")
    print(f"MOV_BIAS: {best_params[3]:.1f}")
    print(f"\nTotal Log-Loss: {best_log_loss:.2f}")
    print(f"Average Log-Loss per game: {best_log_loss/386:.4f}")
    
    # Get team records
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            team,
            SUM(CASE WHEN (team = home_team AND home_score > away_score) OR 
                         (team = away_team AND away_score > home_score) THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN (team = home_team AND home_score < away_score) OR 
                         (team = away_team AND away_score < home_score) THEN 1 ELSE 0 END) as losses
        FROM (
            SELECT home_team as team, home_team, away_team, home_score, away_score FROM game_results WHERE season = '2025-26'
            UNION ALL
            SELECT away_team as team, home_team, away_team, home_score, away_score FROM game_results WHERE season = '2025-26'
        )
        GROUP BY team
        ORDER BY wins DESC, losses ASC
    """)
    records = {row[0]: f"{row[1]}-{row[2]}" for row in cursor.fetchall()}
    conn.close()
    
    # Show top 10 and bottom 10 rankings
    sorted_teams = sorted(best_ratings.items(), key=lambda x: x[1].composite, reverse=True)
    
    print("\n" + "=" * 80)
    print("TOP 10 TEAMS (Optimal Parameters)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Team':<5} {'Composite':<10} {'Off':<10} {'Def':<10} {'Record':<8}")
    print("-" * 80)
    for i, (team, rating) in enumerate(sorted_teams[:10], 1):
        print(f"{i:<6} {team:<5} {rating.composite:<10.1f} {rating.off_elo:<10.1f} {rating.def_elo:<10.1f} {records.get(team, 'N/A'):<8}")
    
    print("\n" + "=" * 80)
    print("BOTTOM 10 TEAMS (Optimal Parameters)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Team':<5} {'Composite':<10} {'Off':<10} {'Def':<10} {'Record':<8}")
    print("-" * 80)
    for i, (team, rating) in enumerate(sorted_teams[-10:], len(sorted_teams)-9):
        print(f"{i:<6} {team:<5} {rating.composite:<10.1f} {rating.off_elo:<10.1f} {rating.def_elo:<10.1f} {records.get(team, 'N/A'):<8}")
    
    # Brooklyn verification
    brooklyn_rank = next(i for i, (t, _) in enumerate(sorted_teams, 1) if t == 'BKN')
    brooklyn_rating = best_ratings['BKN']
    
    print("\n" + "=" * 80)
    print("BROOKLYN VERIFICATION")
    print("=" * 80)
    print(f"Rank: #{brooklyn_rank}/30")
    print(f"Record: {records.get('BKN', 'N/A')}")
    print(f"Composite ELO: {brooklyn_rating.composite:.1f}")
    print(f"Off ELO: {brooklyn_rating.off_elo:.1f}")
    print(f"Def ELO: {brooklyn_rating.def_elo:.1f}")
    print(f"\n{'✅ FIXED' if brooklyn_rank > 15 else '❌ STILL BROKEN'}: Brooklyn should rank 20-25, currently #{brooklyn_rank}")
    
    # ELO range
    highest = sorted_teams[0][1].composite
    lowest = sorted_teams[-1][1].composite
    elo_range = highest - lowest
    
    print("\n" + "=" * 80)
    print("ELO RANGE ANALYSIS")
    print("=" * 80)
    print(f"Highest: {highest:.1f} ({sorted_teams[0][0]})")
    print(f"Lowest: {lowest:.1f} ({sorted_teams[-1][0]})")
    print(f"Range: {elo_range:.1f} points")
    print(f"\n{'✅ GOOD' if 200 <= elo_range <= 300 else '⚠️ ADJUST'}: Ideal range is 200-250 points")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Update config/constants.py with optimal parameters:")
    print(f"   REGULAR_SEASON_BASE_K = {best_params[0]:.1f}")
    print(f"   ELO_POINT_EXPECTATION_SCALE = {best_params[1]:.1f}")
    print(f"   WIN_WEIGHT = {best_params[2]:.1f}")
    print(f"   LOG_MARGIN_DAMPENER = {best_params[3]:.1f}  # (as MOV_BIAS cap)")
    print("\n2. Run recalculate_syndicate_elo.py with new parameters")
    print("3. Verify Brooklyn drops to rank 20-25")
    print("4. Test dashboard predictions for Detroit vs Dallas on 12/18/2025")

if __name__ == "__main__":
    main()
