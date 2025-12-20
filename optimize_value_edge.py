"""
VALUE EDGE OPTIMIZER: Find Optimal Betting Thresholds
======================================================
Tests different minimum edge thresholds for Favorites vs Underdogs
to identify the true profitability "cliff" for each category.

Key Questions:
1. Do favorites need a higher edge buffer to beat the vig?
2. Can we bet underdogs with smaller edges profitably?
3. Is the model well-calibrated or systematically biased?
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import json

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

# Variant D Features (19 features)
FEATURES = [
    'home_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
    'ewma_chaos_home', 'injury_impact_diff', 'injury_shock_diff',
    'star_power_leverage', 'season_progress', 'league_offensive_context',
    'total_foul_environment', 'net_free_throw_advantage',
    'pace_efficiency_interaction', 'offense_vs_defense_matchup'
]

# Trial #245 Optimized Parameters
OPTIMIZED_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'learning_rate': 0.066994,
    'max_depth': 2,
    'min_child_weight': 12,
    'gamma': 2.025432,
    'subsample': 0.630135,
    'colsample_bytree': 0.903401,
    'colsample_bylevel': 0.959686,
    'reg_alpha': 1.081072,
    'reg_lambda': 5.821363,
}

N_ESTIMATORS = 4529

# Data files
TRAINING_DATA = 'data/training_data_GOLD_ELO_22_features.csv'
ODDS_2023_24 = 'data/closing_odds_2023_24_CLEANED.csv'
ODDS_2024_25 = 'data/closing_odds_2024_25_CLEANED.csv'

# Test seasons
TEST_START = '2023-10-01'
TRAIN_END = '2023-10-01'

# ==========================================
# 📊 HELPER FUNCTIONS
# ==========================================

def american_to_implied_prob(odds):
    """Convert American odds to implied probability"""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def american_to_payout(odds):
    """Convert American odds to payout multiplier (includes stake)"""
    if odds < 0:
        return 1 + (100 / abs(odds))
    else:
        return 1 + (odds / 100)

def calculate_profit(bet_on_home, home_won, home_odds, away_odds):
    """Calculate profit for a single bet"""
    if bet_on_home:
        payout = american_to_payout(home_odds)
        return payout - 1 if home_won else -1
    else:
        payout = american_to_payout(away_odds)
        return payout - 1 if not home_won else -1

# ==========================================
# 🧪 MAIN OPTIMIZER
# ==========================================

def run_edge_optimization():
    print("="*80)
    print("💰 VALUE EDGE OPTIMIZER: FAVORITES vs UNDERDOGS")
    print("="*80)
    print("Testing minimum edge thresholds from 0% to 10%")
    print("Analyzing: ROI and Total Units Won")
    print("="*80)
    
    # 1. Load data
    try:
        df = pd.read_csv(TRAINING_DATA)
        
        if 'date' in df.columns:
            df['game_date'] = pd.to_datetime(df['date'])
        elif 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        
        df = df.sort_values('game_date').reset_index(drop=True)
        print(f"\n✓ Loaded {len(df):,} games")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 2. Load historical closing odds
    print("\n📥 Loading historical closing odds...")
    
    try:
        odds_2023 = pd.read_csv(ODDS_2023_24)
        odds_2024 = pd.read_csv(ODDS_2024_25)
        
        # Combine both seasons
        all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
        all_odds['game_date'] = pd.to_datetime(all_odds['game_date'])
        
        print(f"✓ Loaded {len(odds_2023):,} games from 2023-24 season")
        print(f"✓ Loaded {len(odds_2024):,} games from 2024-25 season")
        print(f"✓ Total: {len(all_odds):,} games with odds")
        
        # Standardize team names for merging
        all_odds['home_team'] = all_odds['home_team'].str.strip()
        all_odds['away_team'] = all_odds['away_team'].str.strip()
        df['home_team'] = df['home_team'].str.strip()
        df['away_team'] = df['away_team'].str.strip()
        
        use_simulated = False
        
    except Exception as e:
        print(f"⚠️  WARNING: Could not load odds files: {e}")
        print("   Falling back to simulated odds")
        use_simulated = True
        all_odds = None
    
    # 3. Filter to test period (2023-24 and 2024-25 seasons)
    test_df = df[df['game_date'] >= TEST_START].copy()
    train_df = df[df['game_date'] < TRAIN_END].copy()
    
    print(f"\n📚 Training: {len(train_df):,} games (before {TRAIN_END})")
    print(f"🔮 Testing:  {len(test_df):,} games (from {TEST_START} onward)")
    
    # 4. Train model
    print("\n⚙️  Training model...")
    
    X_train = train_df[FEATURES].copy()
    y_train = train_df['target_moneyline_win'].copy()
    
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train, y_train = X_train[mask], y_train[mask]
    
    model = xgb.XGBClassifier(**OPTIMIZED_PARAMS, n_estimators=N_ESTIMATORS)
    model.fit(X_train, y_train, verbose=False)
    
    print("✓ Model trained")
    
    # 5. Generate predictions on test set
    X_test = test_df[FEATURES].copy()
    y_test = test_df['target_moneyline_win'].copy()
    
    mask = ~(X_test.isna().any(axis=1) | y_test.isna())
    test_df_clean = test_df[mask].copy()
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    test_df_clean['model_prob'] = model.predict_proba(X_test)[:, 1]
    
    print(f"✓ Generated predictions for {len(test_df_clean):,} games")
    
    # 6. Merge with odds data
    if not use_simulated:
        # Merge odds with predictions
        print("\n🔗 Merging odds with predictions...")
        
        test_df_clean = test_df_clean.merge(
            all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
            on=['game_date', 'home_team', 'away_team'],
            how='left'
        )
        
        # Check merge success
        missing_odds = test_df_clean['home_ml_odds'].isna().sum()
        if missing_odds > 0:
            print(f"⚠️  {missing_odds} games missing odds data ({missing_odds/len(test_df_clean)*100:.1f}%)")
            print("   These games will be excluded from analysis")
            test_df_clean = test_df_clean.dropna(subset=['home_ml_odds', 'away_ml_odds'])
        
        print(f"✓ Merged {len(test_df_clean):,} games with odds")
        
        # Calculate implied probabilities from real odds
        test_df_clean['home_odds'] = test_df_clean['home_ml_odds']
        test_df_clean['away_odds'] = test_df_clean['away_ml_odds']
        test_df_clean['home_implied'] = test_df_clean['home_odds'].apply(american_to_implied_prob)
        test_df_clean['away_implied'] = test_df_clean['away_odds'].apply(american_to_implied_prob)
        test_df_clean['home_is_fav'] = test_df_clean['home_implied'] > test_df_clean['away_implied']
        
        print(f"✓ Calculated implied probabilities from REAL MARKET ODDS")
    
    else:
        # This shouldn't happen since we loaded odds successfully
        print("⚠️  WARNING: Using simulated odds as fallback")
        test_df_clean['home_is_fav'] = test_df_clean['model_prob'] > 0.5
        test_df_clean['home_implied'] = np.where(test_df_clean['home_is_fav'], 0.52, 0.48)
        test_df_clean['away_implied'] = 1 - test_df_clean['home_implied']
        test_df_clean['home_odds'] = test_df_clean['home_implied'].apply(
            lambda p: int(-100 * p / (1-p)) if p > 0.5 else int(100 * (1-p) / p)
        )
        test_df_clean['away_odds'] = test_df_clean['away_implied'].apply(
            lambda p: int(-100 * p / (1-p)) if p > 0.5 else int(100 * (1-p) / p)
        )
    
    # Calculate edges
    test_df_clean['home_edge'] = test_df_clean['model_prob'] - test_df_clean['home_implied']
    test_df_clean['away_edge'] = (1 - test_df_clean['model_prob']) - test_df_clean['away_implied']
    
    # 7. Run grid search across edge thresholds
    print("\n" + "="*80)
    print("🔍 TESTING EDGE THRESHOLDS (WITH REAL MARKET ODDS)")
    print("="*80)
    
    edge_thresholds = np.arange(0.0, 0.105, 0.005)  # 0% to 10% in 0.5% steps
    
    fav_results = []
    dog_results = []
    
    for min_edge in edge_thresholds:
        # ========== FAVORITES ==========
        # Home favorites
        home_fav_mask = (test_df_clean['home_is_fav']) & (test_df_clean['home_edge'] > min_edge)
        home_fav_bets = test_df_clean[home_fav_mask].copy()
        home_fav_bets['profit'] = home_fav_bets.apply(
            lambda row: calculate_profit(True, row['target_moneyline_win'] == 1, 
                                        row['home_odds'], row['away_odds']),
            axis=1
        )
        
        # Away favorites
        away_fav_mask = (~test_df_clean['home_is_fav']) & (test_df_clean['away_edge'] > min_edge)
        away_fav_bets = test_df_clean[away_fav_mask].copy()
        away_fav_bets['profit'] = away_fav_bets.apply(
            lambda row: calculate_profit(False, row['target_moneyline_win'] == 1,
                                        row['home_odds'], row['away_odds']),
            axis=1
        )
        
        all_fav_bets = pd.concat([home_fav_bets, away_fav_bets])
        
        if len(all_fav_bets) > 0:
            total_profit = all_fav_bets['profit'].sum()
            roi = (total_profit / len(all_fav_bets)) * 100
            win_rate = (all_fav_bets['profit'] > 0).sum() / len(all_fav_bets)
            
            fav_results.append({
                'edge': min_edge,
                'bets': len(all_fav_bets),
                'total_units': total_profit,
                'roi': roi,
                'win_rate': win_rate
            })
        
        # ========== UNDERDOGS ==========
        # Home underdogs
        home_dog_mask = (~test_df_clean['home_is_fav']) & (test_df_clean['home_edge'] > min_edge)
        home_dog_bets = test_df_clean[home_dog_mask].copy()
        home_dog_bets['profit'] = home_dog_bets.apply(
            lambda row: calculate_profit(True, row['target_moneyline_win'] == 1,
                                        row['home_odds'], row['away_odds']),
            axis=1
        )
        
        # Away underdogs
        away_dog_mask = (test_df_clean['home_is_fav']) & (test_df_clean['away_edge'] > min_edge)
        away_dog_bets = test_df_clean[away_dog_mask].copy()
        away_dog_bets['profit'] = away_dog_bets.apply(
            lambda row: calculate_profit(False, row['target_moneyline_win'] == 1,
                                        row['home_odds'], row['away_odds']),
            axis=1
        )
        
        all_dog_bets = pd.concat([home_dog_bets, away_dog_bets])
        
        if len(all_dog_bets) > 0:
            total_profit = all_dog_bets['profit'].sum()
            roi = (total_profit / len(all_dog_bets)) * 100
            win_rate = (all_dog_bets['profit'] > 0).sum() / len(all_dog_bets)
            
            dog_results.append({
                'edge': min_edge,
                'bets': len(all_dog_bets),
                'total_units': total_profit,
                'roi': roi,
                'win_rate': win_rate
            })
    
    # 8. Display results
    print("\n" + "="*80)
    print("🦁 FAVORITES: OPTIMAL EDGE THRESHOLDS")
    print("="*80)
    print(f"{'MIN EDGE':<12} | {'BETS':<8} | {'TOTAL UNITS':<15} | {'ROI':<10} | {'WIN RATE':<10}")
    print("-"*80)
    
    fav_sorted = sorted(fav_results, key=lambda x: x['total_units'], reverse=True)
    for r in fav_sorted[:10]:
        print(f"+{r['edge']*100:5.1f}%      | {r['bets']:6d}   | {r['total_units']:+13.2f}   | {r['roi']:+7.2f}%  | {r['win_rate']:7.2%}")
    
    print("\n" + "="*80)
    print("🐶 UNDERDOGS: OPTIMAL EDGE THRESHOLDS")
    print("="*80)
    print(f"{'MIN EDGE':<12} | {'BETS':<8} | {'TOTAL UNITS':<15} | {'ROI':<10} | {'WIN RATE':<10}")
    print("-"*80)
    
    dog_sorted = sorted(dog_results, key=lambda x: x['total_units'], reverse=True)
    for r in dog_sorted[:10]:
        print(f"+{r['edge']*100:5.1f}%      | {r['bets']:6d}   | {r['total_units']:+13.2f}   | {r['roi']:+7.2f}%  | {r['win_rate']:7.2%}")
    
    # 9. Key insights
    print("\n" + "="*80)
    print("🎯 KEY INSIGHTS")
    print("="*80)
    
    best_fav = fav_sorted[0]
    best_dog = dog_sorted[0]
    
    print(f"\n🦁 FAVORITES:")
    print(f"   Optimal Threshold: +{best_fav['edge']*100:.1f}% edge")
    print(f"   Performance: {best_fav['bets']} bets, {best_fav['total_units']:+.2f} units, {best_fav['roi']:+.2f}% ROI")
    
    print(f"\n🐶 UNDERDOGS:")
    print(f"   Optimal Threshold: +{best_dog['edge']*100:.1f}% edge")
    print(f"   Performance: {best_dog['bets']} bets, {best_dog['total_units']:+.2f} units, {best_dog['roi']:+.2f}% ROI")
    
    # Check if model is well-calibrated
    if best_fav['edge'] < 0.01:
        print("\n⚠️  FAVORITES: Model may be OVERCONFIDENT (profitable at <1% edge)")
    elif best_fav['edge'] > 0.04:
        print("\n⚠️  FAVORITES: Model may be UNDERCONFIDENT (needs >4% edge buffer)")
    else:
        print("\n✅ FAVORITES: Model appears well-calibrated (2-4% edge sweet spot)")
    
    if best_dog['edge'] < 0.01:
        print("✅ UNDERDOGS: Market undervalues them (bet with minimal edge)")
    elif best_dog['edge'] > 0.04:
        print("⚠️  UNDERDOGS: Model may be overconfident (needs large edge buffer)")
    else:
        print("✅ UNDERDOGS: Model appears well-calibrated")
    
    # 10. Save results
    results = {
        'favorites': fav_results,
        'underdogs': dog_results,
        'best_fav_threshold': best_fav['edge'],
        'best_dog_threshold': best_dog['edge'],
        'test_period': f"{TEST_START} onward",
        'test_games': len(test_df_clean),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('models/experimental/edge_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: models/experimental/edge_optimization_results.json")
    
    # 11. Recommendation
    print("\n" + "="*80)
    print("📋 RECOMMENDED THRESHOLDS")
    print("="*80)
    print(f"Favorites: Bet when edge > +{best_fav['edge']*100:.1f}%")
    print(f"Underdogs: Bet when edge > +{best_dog['edge']*100:.1f}%")
    print(f"\nExpected Performance:")
    print(f"  Favorites: {best_fav['total_units']:+.2f} units on {best_fav['bets']} bets ({best_fav['roi']:+.2f}% ROI)")
    print(f"  Underdogs: {best_dog['total_units']:+.2f} units on {best_dog['bets']} bets ({best_dog['roi']:+.2f}% ROI)")
    total = best_fav['total_units'] + best_dog['total_units']
    print(f"  COMBINED:  {total:+.2f} units on {best_fav['bets'] + best_dog['bets']} bets")

if __name__ == "__main__":
    run_edge_optimization()
