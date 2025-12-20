import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.metrics import log_loss, mean_squared_error
import sys

# ==========================================
# ⚙️ BACKTEST CONFIGURATION
# ==========================================
DATA_PATH = 'data/training_data_MDP_with_margins.csv'
ODDS_2023_PATH = 'data/closing_odds_2023_24_CLEANED.csv'
ODDS_2024_PATH = 'data/closing_odds_2024_25_CLEANED.csv'
NBA_STD_DEV = 13.42  # Calibrated to Model RMSE

# TUNED HYPERPARAMETERS (Trial #21)
XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_jobs': -1,
    'random_state': 42,
    'max_depth': 2,
    'min_child_weight': 50,    # Anti-Blowout
    'learning_rate': 0.012367,
    'n_estimators': 500,
    'gamma': 3.4291,
    'subsample': 0.6001,
    'colsample_bytree': 0.8955,
    'reg_alpha': 0.0505,
    'reg_lambda': 0.0112
}

# BETTING RULES
MIN_EDGE_FAVORITE = 0.04
MIN_EDGE_UNDERDOG = 0.025
FILTER_MAX_FAV_ODDS = -150.0  # We only bet "Cheap" Favorites
FILTER_MIN_OFF_ELO = -90.0    # Anti-Tank Rule

def run_backtest():
    print("⏳ LAUNCHING 2-SEASON WALK-FORWARD BACKTEST (MDP ENGINE)...")
    print(f"   Strategy: Flat Betting (1 Unit)")
    print(f"   Filters: Fav Edge > {MIN_EDGE_FAVORITE:.1%} (Odds > {FILTER_MAX_FAV_ODDS}) | Dog Edge > {MIN_EDGE_UNDERDOG:.1%}")
    print("-" * 80)

    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        if 'margin_target' not in df.columns:
            print("❌ Data missing 'margin_target'.")
            return
            
        # Load odds
        odds_23 = pd.read_csv(ODDS_2023_PATH)
        odds_24 = pd.read_csv(ODDS_2024_PATH)
        odds = pd.concat([odds_23, odds_24])
        odds['game_date'] = pd.to_datetime(odds['game_date'])
        
        # Take consensus (mean) odds per game
        odds_agg = odds.groupby(['game_date', 'home_team', 'away_team']).agg({
            'home_ml_odds': 'mean',
            'away_ml_odds': 'mean'
        }).reset_index()
        
        print(f"📊 Loaded {len(odds_agg)} games with odds")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Helper Functions
    def get_implied(ml): return (-ml)/(-ml+100) if ml<0 else 100/(ml+100)
    def get_payout(ml): return (ml/100)+1 if ml>0 else (100/abs(ml))+1
    
    # Features - use the 19 features from MDP training data
    FEATURES = [
        'off_elo_diff', 'def_elo_diff', 'home_composite_elo',
        'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
        'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
        'injury_matchup_advantage', 'injury_shock_diff', 'star_power_leverage',
        'season_progress', 'league_offensive_context',
        'total_foul_environment', 'net_free_throw_advantage',
        'offense_vs_defense_matchup', 'pace_efficiency_interaction', 'star_mismatch'
    ]
    
    # Validation
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"⚠️ Warning: Features missing from CSV: {missing}")
        FEATURES = [f for f in FEATURES if f in df.columns]
    
    print(f"✓ Using {len(FEATURES)} features")

    # Add target (home team won)
    df['target'] = (df['margin_target'] > 0).astype(int)

    # ------------------------------------------------------------------
    # SEASON 1: 2023-24
    # ------------------------------------------------------------------
    print("\n🏀 SEASON 2023-24 VALIDATION")
    START_DATE_23 = '2023-10-24'
    END_DATE_23 = '2024-06-20'
    
    train_23 = df[df['date'] < START_DATE_23].copy()
    test_23 = df[(df['date'] >= START_DATE_23) & (df['date'] <= END_DATE_23)].copy()
    
    print(f"   Training: {len(train_23)} games (before {START_DATE_23})")
    print(f"   Testing:  {len(test_23)} games")
    
    if len(test_23) == 0:
        print("   ⚠️ No 2023-24 games found in dataset")
        test_23 = pd.DataFrame()
    else:
        dtrain_23 = xgb.DMatrix(train_23[FEATURES], label=train_23['margin_target'])
        dtest_23 = xgb.DMatrix(test_23[FEATURES])
        
        model_23 = xgb.train(XGB_PARAMS, dtrain_23, num_boost_round=500, verbose_eval=False)
        
        # Predict Margin -> Convert to Prob
        margins_23 = model_23.predict(dtest_23)
        probs_23 = norm.cdf(margins_23 / NBA_STD_DEV)
        test_23['model_prob'] = probs_23
        test_23['pred_margin'] = margins_23
        
        # Calculate metrics
        y_true = test_23['target']
        rmse_23 = np.sqrt(mean_squared_error(test_23['margin_target'], margins_23))
        logloss_23 = log_loss(y_true, probs_23)
        acc_23 = (((probs_23 > 0.5).astype(int)) == y_true).mean()
        
        print(f"   RMSE:     {rmse_23:.2f} points")
        print(f"   Log Loss: {logloss_23:.4f}")
        print(f"   Accuracy: {acc_23:.1%}")
        
        # Merge with odds
        test_23 = test_23.merge(
            odds_agg, 
            left_on=['date', 'home_team', 'away_team'],
            right_on=['game_date', 'home_team', 'away_team'],
            how='left'
        )
        test_23 = test_23.dropna(subset=['home_ml_odds', 'away_ml_odds'])
        print(f"   Matched {len(test_23)} games with odds")
    
    # ------------------------------------------------------------------
    # SEASON 2: 2024-25
    # ------------------------------------------------------------------
    print("\n🏀 SEASON 2024-25 VALIDATION")
    START_DATE_24 = '2024-10-22'
    
    train_24 = df[df['date'] < START_DATE_24].copy()
    test_24 = df[df['date'] >= START_DATE_24].copy()
    
    print(f"   Training: {len(train_24)} games (before {START_DATE_24})")
    print(f"   Testing:  {len(test_24)} games")
    
    if len(test_24) == 0:
        print("   ⚠️ No 2024-25 games found in dataset")
        test_24 = pd.DataFrame()
    else:
        dtrain_24 = xgb.DMatrix(train_24[FEATURES], label=train_24['margin_target'])
        dtest_24 = xgb.DMatrix(test_24[FEATURES])
        
        model_24 = xgb.train(XGB_PARAMS, dtrain_24, num_boost_round=500, verbose_eval=False)
        
        margins_24 = model_24.predict(dtest_24)
        probs_24 = norm.cdf(margins_24 / NBA_STD_DEV)
        test_24['model_prob'] = probs_24
        test_24['pred_margin'] = margins_24
        
        # Calculate metrics
        y_true = test_24['target']
        rmse_24 = np.sqrt(mean_squared_error(test_24['margin_target'], margins_24))
        logloss_24 = log_loss(y_true, probs_24)
        acc_24 = (((probs_24 > 0.5).astype(int)) == y_true).mean()
        
        print(f"   RMSE:     {rmse_24:.2f} points")
        print(f"   Log Loss: {logloss_24:.4f}")
        print(f"   Accuracy: {acc_24:.1%}")
        
        # Merge with odds
        test_24 = test_24.merge(
            odds_agg,
            left_on=['date', 'home_team', 'away_team'],
            right_on=['game_date', 'home_team', 'away_team'],
            how='left'
        )
        test_24 = test_24.dropna(subset=['home_ml_odds', 'away_ml_odds'])
        print(f"   Matched {len(test_24)} games with odds")
    
    # Combine Results
    full_test = pd.concat([test_23, test_24])
    
    if len(full_test) == 0:
        print("\n❌ No test data available for either season")
        return
    
    # ------------------------------------------------------------------
    # APPLYING THE STRATEGY
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("💰 CALCULATING PROFITABILITY (FLAT BETTING)")
    print("="*80)
    
    bets = []
    
    for _, row in full_test.iterrows():
        # Skip if no odds
        if pd.isna(row.get('home_ml_odds')) or pd.isna(row.get('away_ml_odds')):
            continue
            
        prob = row['model_prob']
        
        # Determine Model Pick
        if prob > 0.5:
            pick_home = True
            conf = prob
            off_diff = row['off_elo_diff']
        else:
            pick_home = False
            conf = 1 - prob
            off_diff = -1 * row['off_elo_diff']
            
        # Get Odds & Implied
        ml = row['home_ml_odds'] if pick_home else row['away_ml_odds']
        implied = get_implied(ml)
        payout = get_payout(ml)
        
        # Calculate Edge
        edge = conf - implied
        
        # Determine Vegas Status
        home_ml = row['home_ml_odds']
        away_ml = row['away_ml_odds']
        
        is_vegas_fav = False
        if pick_home:
            if home_ml < away_ml: is_vegas_fav = True
        else:
            if away_ml < home_ml: is_vegas_fav = True
            
        # --- THE DECISION LOGIC ---
        place_bet = False
        
        # 1. Edge Threshold
        if is_vegas_fav:
            if edge >= MIN_EDGE_FAVORITE: place_bet = True
        else:
            if edge >= MIN_EDGE_UNDERDOG: place_bet = True
            
        # 2. Filters (If Edge Met)
        if place_bet:
            # Filter A: Expensive Favorites
            if is_vegas_fav and ml < FILTER_MAX_FAV_ODDS:
                place_bet = False # Skip expensive chalk
            
            # Filter B: Anti-Tank (Bad Offense)
            if off_diff < FILTER_MIN_OFF_ELO:
                place_bet = False
                
        # Record Result
        if place_bet:
            won = (row['target'] == 1) if pick_home else (row['target'] == 0)
            profit = (payout - 1) if won else -1
            
            season = "2023-24" if row['date'] < pd.Timestamp('2024-08-01') else "2024-25"
            bet_type = "Favorite" if is_vegas_fav else "Underdog"
            
            bets.append({
                'date': row['date'],
                'season': season,
                'type': bet_type,
                'odds': ml,
                'edge': edge,
                'won': won,
                'profit': profit
            })
            
    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------
    results = pd.DataFrame(bets)
    
    if len(results) == 0:
        print("❌ No bets placed. Check filters.")
        return

    print(f"\n🏆 COMBINED RESULTS (2 SEASONS)")
    print(f"   Total Bets:   {len(results)}")
    print(f"   Win Rate:     {results['won'].mean():.1%}")
    print(f"   Total Units:  {results['profit'].sum():+.2f}u")
    print(f"   ROI:          {results['profit'].mean():.2%}")
    
    print("\n📅 SEASON BREAKDOWN")
    print(f"{'SEASON':<10} | {'BETS':<6} | {'WIN %':<8} | {'PROFIT':<10} | {'ROI'}")
    print("-" * 60)
    for season in ['2023-24', '2024-25']:
        s_data = results[results['season'] == season]
        if len(s_data) > 0:
            print(f"{season:<10} | {len(s_data):<6} | {s_data['won'].mean():.1%}  | {s_data['profit'].sum():>+7.2f}u  | {s_data['profit'].mean():+.2%}")
            
    print("\n🦁 TYPE BREAKDOWN")
    print(f"{'TYPE':<10} | {'BETS':<6} | {'WIN %':<8} | {'PROFIT':<10} | {'ROI'}")
    print("-" * 60)
    for btype in ['Favorite', 'Underdog']:
        t_data = results[results['type'] == btype]
        if len(t_data) > 0:
            print(f"{btype:<10} | {len(t_data):<6} | {t_data['won'].mean():.1%}  | {t_data['profit'].sum():>+7.2f}u  | {t_data['profit'].mean():+.2%}")
    
    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_backtest()
