import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
import sys

DATA_PATH = 'data/training_data_MDP_with_margins.csv'
ODDS_PATH = 'data/closing_odds_2024_25_CLEANED.csv'
NBA_STD_DEV = 13.42
START_DATE_24 = '2024-10-22'

try:
    from production_config_mdp import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS
except ImportError:
    print("❌ Config missing. Using defaults.")
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

def analyze_misses():
    print("🕵️ AUTOPSY: ANALYZING MDP LOSSES (2024-25)...")
    print("-" * 80)
    
    # Load & Train
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load odds
    odds = pd.read_csv(ODDS_PATH)
    odds['game_date'] = pd.to_datetime(odds['game_date'])
    odds_agg = odds.groupby(['game_date', 'home_team', 'away_team']).agg({
        'home_ml_odds': 'mean',
        'away_ml_odds': 'mean'
    }).reset_index()
    
    season_df = df[df['date'] >= START_DATE_24].copy()
    
    # Train on history
    train_df = df[df['date'] < START_DATE_24].copy()
    
    print(f"✓ Training on {len(train_df)} historical games")
    dtrain = xgb.DMatrix(train_df[ACTIVE_FEATURES], label=train_df['margin_target'])
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    
    # Predict
    dtest = xgb.DMatrix(season_df[ACTIVE_FEATURES])
    season_df['model_prob'] = norm.cdf(model.predict(dtest) / NBA_STD_DEV)
    
    # Merge with odds
    season_df = season_df.merge(
        odds_agg,
        left_on=['date', 'home_team', 'away_team'],
        right_on=['game_date', 'home_team', 'away_team'],
        how='left'
    )
    season_df = season_df.dropna(subset=['home_ml_odds', 'away_ml_odds'])
    
    print(f"✓ Analyzing {len(season_df)} games with odds")
    
    # Identify Bets Made (Using edge > 1% to cast wide net)
    def get_implied(ml): return (-ml)/(-ml+100) if ml<0 else 100/(ml+100)
    season_df['home_implied'] = season_df['home_ml_odds'].apply(get_implied)
    season_df['away_implied'] = season_df['away_ml_odds'].apply(get_implied)
    
    season_df['pick_home'] = season_df['model_prob'] > 0.5
    season_df['pick_conf'] = np.where(season_df['pick_home'], season_df['model_prob'], 1-season_df['model_prob'])
    season_df['pick_implied'] = np.where(season_df['pick_home'], season_df['home_implied'], season_df['away_implied'])
    season_df['edge'] = season_df['pick_conf'] - season_df['pick_implied']
    
    # Add target
    season_df['target'] = (season_df['margin_target'] > 0).astype(int)
    
    # Did we win?
    season_df['won'] = np.where(season_df['pick_home'], season_df['target']==1, season_df['target']==0)
    
    # ISOLATE LOSERS (Edge > 2.5% to capture relevant losses)
    losses = season_df[(season_df['won'] == False) & (season_df['edge'] > 0.025)].copy()
    wins = season_df[(season_df['won'] == True) & (season_df['edge'] > 0.025)].copy()
    
    print(f"\n📊 DATASET: {len(wins)} Wins vs {len(losses)} Losses (Edge > 2.5%)")
    
    # ------------------------------------------------------------------
    # 1. EDGE DISTRIBUTION OF LOSSES
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("1. EDGE BUCKET PERFORMANCE")
    print("="*80)
    print(f"{'EDGE RANGE':<15} | {'BETS':<6} | {'WIN %':<8} | {'AVG EDGE':<10} | {'STATUS'}")
    print("-" * 65)
    
    bins = [0.025, 0.05, 0.10, 0.15, 0.50]
    labels = ['2.5-5%', '5-10%', '10-15%', '15%+']
    
    # Combine wins and losses for rate calculation
    all_bets = pd.concat([wins, losses])
    all_bets['edge_bin'] = pd.cut(all_bets['edge'], bins=bins, labels=labels)
    
    for label in labels:
        subset = all_bets[all_bets['edge_bin'] == label]
        if len(subset) > 0:
            win_rate = subset['won'].mean()
            avg_edge = subset['edge'].mean()
            status = "✅ GOOD" if win_rate > 0.5 else "⚠️ LOW"
            if win_rate < 0.4: status = "🚨 TOXIC"
            print(f"{label:<15} | {len(subset):<6} | {win_rate:.1%}  | {avg_edge:.1%}      | {status}")

    # ------------------------------------------------------------------
    # 2. FEATURE DIVERGENCE
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("2. FORENSIC FEATURE AUDIT (Wins vs Losses)")
    print("="*80)
    check_feats = [
        'off_elo_diff', 'def_elo_diff', 'net_fatigue_score', 
        'injury_matchup_advantage', 'season_progress', 
        'projected_possession_margin', 'ewma_pace_diff'
    ]
    
    print(f"{'FEATURE':<30} | {'WIN AVG':<10} | {'LOSS AVG':<10} | {'DIFF':<10} | {'SIGNAL'}")
    print("-" * 80)
    
    for f in check_feats:
        if f not in season_df.columns:
            continue
            
        # Orient feature to the PICK (for diff features)
        if 'diff' in f or 'advantage' in f or 'margin' in f:
            wins_oriented = np.where(wins['pick_home'], wins[f], -1*wins[f])
            losses_oriented = np.where(losses['pick_home'], losses[f], -1*losses[f])
        else:
            wins_oriented = wins[f]
            losses_oriented = losses[f]
        
        w_avg = wins_oriented.mean()
        l_avg = losses_oriented.mean()
        diff = l_avg - w_avg
        diff_pct = (diff / w_avg * 100) if w_avg != 0 else 0
        
        # Determine signal
        if abs(diff_pct) > 20:
            signal = "🚨 BIG GAP"
        elif abs(diff_pct) > 10:
            signal = "⚠️ Notable"
        else:
            signal = "✓ Similar"
            
        print(f"{f:<30} | {w_avg:10.2f} | {l_avg:10.2f} | {diff:+10.2f} | {signal}")

    # ------------------------------------------------------------------
    # 3. CONFIDENCE BUCKET ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("3. MODEL CONFIDENCE BUCKETS")
    print("="*80)
    print(f"{'CONFIDENCE':<15} | {'BETS':<6} | {'WIN %':<8} | {'EXPECTED %':<12} | {'CALIBRATION'}")
    print("-" * 75)
    
    conf_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    conf_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    all_bets['conf_bin'] = pd.cut(all_bets['pick_conf'], bins=conf_bins, labels=conf_labels)
    
    for label in conf_labels:
        subset = all_bets[all_bets['conf_bin'] == label]
        if len(subset) > 0:
            actual_wr = subset['won'].mean()
            expected_wr = subset['pick_conf'].mean()
            calib_error = actual_wr - expected_wr
            
            if abs(calib_error) > 0.10:
                calib_status = "🚨 OFF"
            elif abs(calib_error) > 0.05:
                calib_status = "⚠️ Drift"
            else:
                calib_status = "✅ Good"
                
            print(f"{label:<15} | {len(subset):<6} | {actual_wr:.1%}  | {expected_wr:.1%}      | {calib_error:+.1%} {calib_status}")

    # ------------------------------------------------------------------
    # 4. FAVORITES VS UNDERDOGS
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("4. FAVORITE VS UNDERDOG PERFORMANCE")
    print("="*80)
    
    # Determine if pick is favorite
    all_bets['is_fav'] = np.where(
        all_bets['pick_home'], 
        all_bets['home_ml_odds'] < all_bets['away_ml_odds'],
        all_bets['away_ml_odds'] < all_bets['home_ml_odds']
    )
    
    print(f"{'TYPE':<15} | {'BETS':<6} | {'WIN %':<8} | {'AVG EDGE':<10} | {'PROFIT'}")
    print("-" * 60)
    
    for bet_type, is_fav in [('Favorites', True), ('Underdogs', False)]:
        subset = all_bets[all_bets['is_fav'] == is_fav]
        if len(subset) > 0:
            wr = subset['won'].mean()
            avg_edge = subset['edge'].mean()
            
            # Calculate profit
            pick_odds = np.where(subset['pick_home'], subset['home_ml_odds'], subset['away_ml_odds'])
            payouts = np.where(pick_odds > 0, (pick_odds/100)+1, (100/abs(pick_odds))+1)
            profit = np.where(subset['won'], payouts-1, -1).sum()
            
            print(f"{bet_type:<15} | {len(subset):<6} | {wr:.1%}  | {avg_edge:.1%}      | {profit:+.2f}u")

    print("\n" + "="*80)
    print("💡 INSIGHTS")
    print("="*80)
    print("• If high-edge bets have low win rates → Overconfidence leak")
    print("• If injury_matchup_advantage is higher in losses → Overrating injuries")
    print("• If fatigue difference is negative → Model too aggressive on tired teams")
    print("• If high confidence buckets underperform → Calibration drift")

if __name__ == "__main__":
    analyze_misses()
