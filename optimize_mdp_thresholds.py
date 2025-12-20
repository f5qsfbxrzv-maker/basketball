import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
import sys

# CONFIG
DATA_PATH = 'data/training_data_MDP_with_margins.csv'
ODDS_PATH = 'data/closing_odds_2024_25_CLEANED.csv'
NBA_STD_DEV = 13.42
START_DATE_24 = '2024-10-22'  # Focus on current calibrated season

# Load your finalized params
try:
    from production_config_mdp import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS, FILTER_MAX_FAV_ODDS, FILTER_MIN_OFF_ELO
except ImportError:
    print("❌ Config missing. Using defaults.")
    FILTER_MAX_FAV_ODDS = -150.0
    FILTER_MIN_OFF_ELO = -90.0
    N_ESTIMATORS = 500
    
    ACTIVE_FEATURES = [
        'off_elo_diff', 'def_elo_diff', 'home_composite_elo',
        'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
        'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
        'injury_matchup_advantage', 'injury_shock_diff', 'star_power_leverage',
        'season_progress', 'league_offensive_context',
        'total_foul_environment', 'net_free_throw_advantage',
        'offense_vs_defense_matchup', 'pace_efficiency_interaction', 'star_mismatch'
    ]
    
    XGB_PARAMS = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_jobs': -1,
        'random_state': 42,
        'max_depth': 2,
        'min_child_weight': 50,
        'learning_rate': 0.012367,
        'gamma': 3.4291,
        'subsample': 0.6001,
        'colsample_bytree': 0.8955,
        'reg_alpha': 0.0505,
        'reg_lambda': 0.0112
    }

def run_optimization():
    print("🧪 LAUNCHING MDP THRESHOLD GRID SEARCH...")
    print(f"   Target: Maximize Total Units Won (2024-25 Season)")
    print(f"   Filters: Fav Odds > {FILTER_MAX_FAV_ODDS}, Off ELO > {FILTER_MIN_OFF_ELO}")
    print("-" * 80)
    
    # 1. Load & Prep
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load odds
    odds = pd.read_csv(ODDS_PATH)
    odds['game_date'] = pd.to_datetime(odds['game_date'])
    
    # Take consensus odds
    odds_agg = odds.groupby(['game_date', 'home_team', 'away_team']).agg({
        'home_ml_odds': 'mean',
        'away_ml_odds': 'mean'
    }).reset_index()
    
    season_df = df[df['date'] >= START_DATE_24].copy()
    print(f"✓ Loaded {len(season_df)} games from 2024-25 season")
    
    # Generate MDP Probabilities
    # We retrain on pre-2024 data to be fair
    train_df = df[df['date'] < START_DATE_24].copy()
    
    print(f"✓ Training on {len(train_df)} games before {START_DATE_24}")
    dtrain = xgb.DMatrix(train_df[ACTIVE_FEATURES], label=train_df['margin_target'])
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    
    # Predict
    dtest = xgb.DMatrix(season_df[ACTIVE_FEATURES])
    pred_margin = model.predict(dtest)
    season_df['model_prob'] = norm.cdf(pred_margin / NBA_STD_DEV)
    
    # Merge with odds
    season_df = season_df.merge(
        odds_agg,
        left_on=['date', 'home_team', 'away_team'],
        right_on=['game_date', 'home_team', 'away_team'],
        how='left'
    )
    season_df = season_df.dropna(subset=['home_ml_odds', 'away_ml_odds'])
    print(f"✓ Matched {len(season_df)} games with odds")
    
    # Odds Prep
    def get_implied(ml): return (-ml)/(-ml+100) if ml<0 else 100/(ml+100)
    def get_payout(ml): return (ml/100)+1 if ml>0 else (100/abs(ml))+1
    
    season_df['home_implied'] = season_df['home_ml_odds'].apply(get_implied)
    season_df['away_implied'] = season_df['away_ml_odds'].apply(get_implied)
    season_df['home_payout'] = season_df['home_ml_odds'].apply(get_payout)
    season_df['away_payout'] = season_df['away_ml_odds'].apply(get_payout)
    season_df['target'] = (season_df['margin_target'] > 0).astype(int)
    
    # 2. THE GRID SEARCH
    # We test edges from 0.0% to 10.0%
    edges = np.arange(0.0, 0.105, 0.005)  # 0%, 0.5%, 1.0% ... 10%
    
    fav_results = []
    dog_results = []
    
    print("\n📊 OPTIMIZING FAVORITES (With Odds > -150 Filter)...")
    
    for edge_min in edges:
        # Vectorized Logic for Speed
        # 1. Determine Model Pick
        pick_home = season_df['model_prob'] > 0.5
        model_conf = np.where(pick_home, season_df['model_prob'], 1 - season_df['model_prob'])
        
        # 2. Determine if Pick is Vegas Fav
        home_ml = season_df['home_ml_odds']
        away_ml = season_df['away_ml_odds']
        is_vegas_fav = np.where(pick_home, home_ml < away_ml, away_ml < home_ml)
        
        # 3. Calculate Edge
        implied = np.where(pick_home, season_df['home_implied'], season_df['away_implied'])
        edge = model_conf - implied
        
        # 4. Filter: Must be Vegas Fav, Must meet Edge, Must match 'Cheap Fav' Rule
        pick_odds = np.where(pick_home, home_ml, away_ml)
        
        mask = (is_vegas_fav) & (edge >= edge_min) & (pick_odds >= FILTER_MAX_FAV_ODDS)
        
        subset = season_df[mask].copy()
        
        if len(subset) > 10:
            # Calculate Profit
            won = np.where(pick_home[mask], subset['target']==1, subset['target']==0)
            payout = np.where(pick_home[mask], subset['home_payout'], subset['away_payout'])
            profit = np.where(won, payout - 1, -1).sum()
            roi = profit / len(subset)
            win_rate = won.mean()
            
            fav_results.append({
                'edge': edge_min,
                'bets': len(subset),
                'profit': profit,
                'roi': roi,
                'win_rate': win_rate
            })

    print("\n📊 OPTIMIZING UNDERDOGS (With Anti-Tank Filter)...")
    
    for edge_min in edges:
        # Same logic, but for Underdogs
        pick_home = season_df['model_prob'] > 0.5
        is_vegas_fav = np.where(pick_home, season_df['home_ml_odds'] < season_df['away_ml_odds'], 
                                season_df['away_ml_odds'] < season_df['home_ml_odds'])
        
        # Determine Off ELO Diff (for Anti-Tank filter)
        off_diff = np.where(pick_home, season_df['off_elo_diff'], -1 * season_df['off_elo_diff'])
        
        model_conf = np.where(pick_home, season_df['model_prob'], 1 - season_df['model_prob'])
        implied = np.where(pick_home, season_df['home_implied'], season_df['away_implied'])
        edge = model_conf - implied
        
        # Filter: Must be Underdog, Meet Edge, Pass Anti-Tank
        mask = (~is_vegas_fav) & (edge >= edge_min) & (off_diff >= FILTER_MIN_OFF_ELO)
        
        subset = season_df[mask].copy()
        
        if len(subset) > 10:
            won = np.where(pick_home[mask], subset['target']==1, subset['target']==0)
            payout = np.where(pick_home[mask], subset['home_payout'], subset['away_payout'])
            profit = np.where(won, payout - 1, -1).sum()
            roi = profit / len(subset)
            win_rate = won.mean()
            
            dog_results.append({
                'edge': edge_min,
                'bets': len(subset),
                'profit': profit,
                'roi': roi,
                'win_rate': win_rate
            })

    # OUTPUT
    print("\n" + "="*80)
    print("🦁 FAVORITE OPTIMIZATION (Sorted by Profit)")
    print("="*80)
    print(f"{'MIN EDGE':<10} | {'BETS':<6} | {'WIN %':<8} | {'PROFIT':<10} | {'ROI':<8}")
    print("-" * 65)
    for r in sorted(fav_results, key=lambda x: x['profit'], reverse=True)[:10]:
        print(f"+{r['edge']:.1%}       | {r['bets']:<6} | {r['win_rate']:.1%}  | {r['profit']:>+6.2f}u    | {r['roi']:+.1%}")

    print("\n" + "="*80)
    print("🐶 UNDERDOG OPTIMIZATION (Sorted by Profit)")
    print("="*80)
    print(f"{'MIN EDGE':<10} | {'BETS':<6} | {'WIN %':<8} | {'PROFIT':<10} | {'ROI':<8}")
    print("-" * 65)
    for r in sorted(dog_results, key=lambda x: x['profit'], reverse=True)[:10]:
        print(f"+{r['edge']:.1%}       | {r['bets']:<6} | {r['win_rate']:.1%}  | {r['profit']:>+6.2f}u    | {r['roi']:+.1%}")
    
    # Find optimal
    best_fav = max(fav_results, key=lambda x: x['profit'])
    best_dog = max(dog_results, key=lambda x: x['profit'])
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDED THRESHOLDS")
    print("="*80)
    print(f"FAVORITES: MIN_EDGE = {best_fav['edge']:.1%} ({best_fav['bets']} bets, {best_fav['profit']:+.2f}u, {best_fav['roi']:+.1%} ROI)")
    print(f"UNDERDOGS: MIN_EDGE = {best_dog['edge']:.1%} ({best_dog['bets']} bets, {best_dog['profit']:+.2f}u, {best_dog['roi']:+.1%} ROI)")
    print(f"\nCOMBINED PROFIT: {best_fav['profit'] + best_dog['profit']:+.2f}u")

if __name__ == "__main__":
    run_optimization()
