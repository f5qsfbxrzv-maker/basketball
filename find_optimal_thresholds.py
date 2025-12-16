import pandas as pd
import numpy as np
import xgboost as xgb
import itertools

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = 'models/xgboost_22features_trial1306_20251215_212306.json'
FEATURES_PATH = 'data/training_data_matchup_with_injury_advantage_FIXED.csv'
ODDS_2023_PATH = 'data/closing_odds_2023_24.csv'
ODDS_2024_PATH = 'data/live/closing_odds_2024_25.csv'

# Grid Search Ranges (The "Knobs" we are turning)
FAV_EDGES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]  # 0.5% to 3.0%
DOG_EDGES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20] # 5% to 20%

KELLY_FRACTION = 0.25
BANKROLL = 10000

# ==============================================================================
# DATA PREPARATION
# ==============================================================================
def prepare_dataset():
    """Merge features with historical odds data"""
    print("[PREP] Loading features dataset...")
    features = pd.read_csv(FEATURES_PATH)
    features['date'] = pd.to_datetime(features['date'])
    
    # Load odds from both seasons
    print("[PREP] Loading 2023-24 odds...")
    odds_2023 = pd.read_csv(ODDS_2023_PATH)
    odds_2023['game_date'] = pd.to_datetime(odds_2023['game_date'])
    
    print("[PREP] Loading 2024-25 odds...")
    odds_2024 = pd.read_csv(ODDS_2024_PATH)
    odds_2024['game_date'] = pd.to_datetime(odds_2024['game_date'])
    
    # Combine odds
    all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
    
    # Team name mapping (abbreviations to full names)
    team_map = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
        'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }
    
    # Create merge keys
    features['home_full'] = features['home_team'].map(team_map)
    features['away_full'] = features['away_team'].map(team_map)
    
    # Merge on date + teams
    print("[PREP] Merging features with odds...")
    merged = features.merge(
        all_odds,
        left_on=['date', 'home_full', 'away_full'],
        right_on=['game_date', 'home_team', 'away_team'],
        how='inner'
    )
    
    print(f"[PREP] Successfully merged {len(merged)} games with odds data")
    
    # Filter outliers
    merged = merged[
        (merged['home_ml_odds'].between(-2000, 2000)) &
        (merged['away_ml_odds'].between(-2000, 2000))
    ].copy()
    
    print(f"[PREP] After outlier filter: {len(merged)} games remain")
    
    # Convert American odds to decimal
    def american_to_decimal(odds):
        if odds > 0:
            return 1 + (odds / 100)
        else:
            return 1 + (100 / abs(odds))
    
    merged['home_decimal'] = merged['home_ml_odds'].apply(american_to_decimal)
    merged['away_decimal'] = merged['away_ml_odds'].apply(american_to_decimal)
    
    # Rename outcome column for clarity
    merged['home_won'] = merged['target_moneyline_win']
    
    return merged

# ==============================================================================
# OPTIMIZATION ENGINE
# ==============================================================================
def optimize_thresholds():
    print(f"===================================================================")
    print(f"STRATEGY OPTIMIZER: FINDING THE SWEET SPOT")
    print(f"===================================================================\n")
    
    # 1. Prepare merged dataset
    df = prepare_dataset()
    
    if len(df) == 0:
        print("ERROR: No data after merge. Check team names and date ranges.")
        return
    
    # Load Model
    print("[MODEL] Loading Trial 1306 model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Define the 22 features used in Trial 1306
    feature_cols = [
        'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
        'net_fatigue_score', 'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff',
        'ewma_orb_diff', 'ewma_vol_3p_diff', 'injury_matchup_advantage',
        'ewma_chaos_home', 'ewma_foul_synergy_home', 'total_foul_environment',
        'league_offensive_context', 'season_progress', 'pace_efficiency_interaction',
        'projected_possession_margin', 'three_point_matchup', 'net_free_throw_advantage',
        'star_power_leverage', 'offense_vs_defense_matchup'
    ]
    
    # Verify all features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}")
        return
    
    print(f"[PREDICT] Generating home win probabilities for {len(df)} games...")
    X = df[feature_cols]
    df['home_win_prob'] = model.predict_proba(X)[:, 1]
    
    # Calculate implied probabilities from odds
    df['home_implied'] = 1 / df['home_decimal']
    df['away_implied'] = 1 / df['away_decimal']
    
    # Calculate edges for both sides
    df['home_edge'] = df['home_win_prob'] - df['home_implied']
    df['away_edge'] = (1 - df['home_win_prob']) - df['away_implied']
    
    # Create long format: each row becomes two potential bets (home and away)
    print("[TRANSFORM] Creating bet opportunities...")
    bets = []
    
    for _, row in df.iterrows():
        # Home bet opportunity
        bets.append({
            'date': row['date'],
            'team': row['home_team_y'] if 'home_team_y' in row else row['home_full'],
            'opponent': row['away_team_y'] if 'away_team_y' in row else row['away_full'],
            'odds_decimal': row['home_decimal'],
            'win_prob': row['home_win_prob'],
            'edge': row['home_edge'],
            'outcome': row['home_won'],
            'is_favorite': row['home_decimal'] < 2.0
        })
        
        # Away bet opportunity
        bets.append({
            'date': row['date'],
            'team': row['away_team_y'] if 'away_team_y' in row else row['away_full'],
            'opponent': row['home_team_y'] if 'home_team_y' in row else row['home_full'],
            'odds_decimal': row['away_decimal'],
            'win_prob': 1 - row['home_win_prob'],
            'edge': row['away_edge'],
            'outcome': 1 - row['home_won'],
            'is_favorite': row['away_decimal'] < 2.0
        })
    
    bets_df = pd.DataFrame(bets)
    print(f"[TRANSFORM] Created {len(bets_df)} bet opportunities from {len(df)} games")
    
    results = []
    
    # 2. Grid Search Loop
    combinations = list(itertools.product(FAV_EDGES, DOG_EDGES))
    print(f"\n[SEARCH] Testing {len(combinations)} strategy combinations...")
    print(f"{'FAV EDGE':<10} | {'DOG EDGE':<10} | {'BETS':<6} | {'PROFIT ($)':<12} | {'ROI %':<8} | {'WIN %':<6}")
    print("-" * 75)
    
    for fav_cut, dog_cut in combinations:
        
        # Apply Filters
        # Favs (< 2.00) need > fav_cut edge
        # Dogs (>= 2.00) need > dog_cut edge
        
        mask_fav = (bets_df['is_favorite']) & (bets_df['edge'] > fav_cut)
        mask_dog = (~bets_df['is_favorite']) & (bets_df['edge'] > dog_cut)
        
        active_bets = bets_df[mask_fav | mask_dog].copy()
        
        if len(active_bets) < 50:
            continue # Ignore strategies with too little volume
            
        # Simulate PnL with Kelly sizing
        # Kelly formula: f* = (bp - q) / b, where b = decimal_odds - 1
        b = active_bets['odds_decimal'] - 1
        q = 1 - active_bets['win_prob']
        f_star = (b * active_bets['win_prob'] - q) / b
        active_bets['stake_pct'] = f_star.clip(lower=0) * KELLY_FRACTION
        active_bets['stake'] = BANKROLL * active_bets['stake_pct']
        
        # Calculate Profit
        active_bets['profit'] = np.where(
            active_bets['outcome'] == 1,
            active_bets['stake'] * (active_bets['odds_decimal'] - 1),
            -active_bets['stake']
        )
        
        net_profit = active_bets['profit'].sum()
        count = len(active_bets)
        total_staked = active_bets['stake'].sum()
        roi = (net_profit / total_staked) * 100 if total_staked > 0 else 0
        win_rate = active_bets['outcome'].mean() * 100
        
        results.append({
            'fav': fav_cut,
            'dog': dog_cut,
            'bets': count,
            'profit': net_profit,
            'roi': roi,
            'win_rate': win_rate
        })
        
        # Print live if it looks promising
        if roi > 5.0 and count > 200:
             print(f"{fav_cut*100:>4.1f}%     | {dog_cut*100:>4.1f}%     | {count:<6} | ${net_profit:,.0f}      | {roi:>5.1f}%   | {win_rate:>5.1f}%")

    # 3. Best Results
    results_df = pd.DataFrame(results)
    best_roi = results_df.sort_values('roi', ascending=False).head(5)
    best_profit = results_df.sort_values('profit', ascending=False).head(5)
    
    print("\n" + "="*75)
    print("TOP 5 STRATEGIES BY ROI (Efficiency)")
    print("="*75)
    print(best_roi.to_string(index=False))
    
    print("\n" + "="*75)
    print("TOP 5 STRATEGIES BY TOTAL PROFIT (Volume)")
    print("="*75)
    print(best_profit.to_string(index=False))

if __name__ == "__main__":
    optimize_thresholds()
