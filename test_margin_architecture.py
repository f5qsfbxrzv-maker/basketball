import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

# CONFIGURATION
DATA_PATH = 'data/training_data_MDP_with_margins.csv'  # ← REAL SCORES!
ODDS_PATH = 'data/closing_odds_2024_25_CLEANED.csv'
SPLIT_DATE = '2024-10-22'

# NBA Standard Deviation (points)
NBA_STDEV = 13.5  # Historical average margin standard deviation

try:
    from production_config import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS
except ImportError:
    print("⚠️ Using fallback config")
    ACTIVE_FEATURES = [
        'off_elo_diff', 'def_elo_diff', 'home_composite_elo',           
        'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
        'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',          
        'injury_impact_diff', 'injury_shock_diff', 'star_power_leverage',
        'season_progress', 'league_offensive_context',     
        'total_foul_environment', 'net_free_throw_advantage', 'whistle_leverage',             
        'offense_vs_defense_matchup', 'pace_efficiency_interaction'
    ]
    XGB_PARAMS = {
        'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'learning_rate': 0.067, 'max_depth': 2, 'min_child_weight': 12,
        'reg_alpha': 1.08, 'reg_lambda': 5.82, 'subsample': 0.63, 'colsample_bytree': 0.90,
        'n_jobs': -1, 'random_state': 42, 'base_score': 0.5
    }
    N_ESTIMATORS = 4529

def margin_to_win_prob(margin, stdev=NBA_STDEV):
    """
    Convert predicted point margin to win probability using normal CDF.
    
    margin: Home Score - Away Score (positive = home favored)
    stdev: Standard deviation of NBA game margins
    
    Returns: Probability that home team wins
    """
    return norm.cdf(margin / stdev)

def test_architectures():
    print("="*85)
    print("🏗️  ARCHITECTURE A/B TEST: CLASSIFIER vs MARGIN-DERIVED PROBABILITY")
    print("="*85)
    print("Hypothesis: Regression on Margin → Better Favorite Performance")
    print("-"*85)
    
    # 1. LOAD DATA
    print("\n📂 Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Handle date column
    if 'date' in df.columns:
        df['game_date'] = pd.to_datetime(df['date'])
    else:
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Handle target column
    if 'target' not in df.columns and 'target_moneyline_win' in df.columns:
        df['target'] = df['target_moneyline_win']
    
    # Check for actual score margin (should exist in MDP data)
    if 'margin_target' in df.columns:
        df['score_margin'] = df['margin_target']
        print(f"   ✓ Using REAL game margins from margin_target column")
    elif 'home_score' in df.columns and 'away_score' in df.columns:
        df['score_margin'] = df['home_score'] - df['away_score']
        print("   ✓ Using REAL game margins calculated from scores")
    else:
        print("   ⚠️  WARNING: No actual scores found. Using synthetic margins...")
        # For testing purposes, create synthetic margin from win/loss + noise
        # In production, you'd need actual scores
        df['score_margin'] = np.where(
            df['target'] == 1,
            np.random.normal(10, 8, len(df)),  # Home wins by ~10±8
            np.random.normal(-10, 8, len(df))  # Home loses by ~10±8
        )
    
    print(f"   ✓ Loaded {len(df):,} games")
    print(f"   ✓ Margin range: {df['score_margin'].min():.1f} to {df['score_margin'].max():.1f}")
    print(f"   ✓ Margin std dev: {df['score_margin'].std():.2f} (NBA typical: ~13.5)")
    
    # Split data
    train_df = df[df['game_date'] < SPLIT_DATE].copy()
    test_df = df[df['game_date'] >= SPLIT_DATE].copy()
    
    print(f"   ✓ Training: {len(train_df):,} games")
    print(f"   ✓ Testing: {len(test_df):,} games (2024-25 season)")
    
    # 2. LOAD ODDS
    print("\n📂 Loading odds data...")
    try:
        odds_df = pd.read_csv(ODDS_PATH)
        odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])
        
        test_df = test_df.merge(
            odds_df[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
            on=['game_date', 'home_team', 'away_team'],
            how='left'
        )
        
        test_df = test_df[test_df['home_ml_odds'].notna()].copy()
        print(f"   ✓ Odds matched: {len(test_df):,} games")
    except Exception as e:
        print(f"   ⚠️ Could not load odds: {e}")
        print("   Continuing with probability evaluation only...")
        test_df['home_ml_odds'] = -110
        test_df['away_ml_odds'] = -110
    
    # Calculate implied probabilities
    def get_implied(ml):
        if pd.isna(ml):
            return np.nan
        return (-ml)/(-ml+100) if ml<0 else 100/(ml+100)
    
    def get_payout(ml):
        if pd.isna(ml):
            return np.nan
        return (ml/100)+1 if ml>0 else (100/abs(ml))+1
    
    test_df['home_implied'] = test_df['home_ml_odds'].apply(get_implied)
    test_df['away_implied'] = test_df['away_ml_odds'].apply(get_implied)
    test_df['home_payout'] = test_df['home_ml_odds'].apply(get_payout)
    test_df['away_payout'] = test_df['away_ml_odds'].apply(get_payout)
    test_df['home_is_vegas_fav'] = test_df['home_implied'] > test_df['away_implied']
    
    # =========================================================================
    # MODEL A: BINARY CLASSIFIER (Current Architecture)
    # =========================================================================
    print("\n" + "="*85)
    print("🅰️  MODEL A: BINARY CLASSIFIER (Win/Loss)")
    print("="*85)
    
    X_train = train_df[ACTIVE_FEATURES]
    y_train = train_df['target']
    X_test = test_df[ACTIVE_FEATURES]
    y_test = test_df['target']
    
    print("⚙️  Training XGBoost Classifier...")
    dtrain_clf = xgb.DMatrix(X_train, label=y_train)
    dtest_clf = xgb.DMatrix(X_test)
    
    model_clf = xgb.train(XGB_PARAMS, dtrain_clf, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    
    prob_clf = model_clf.predict(dtest_clf)
    test_df['prob_classifier'] = prob_clf
    
    # Metrics
    logloss_clf = log_loss(y_test, prob_clf)
    brier_clf = brier_score_loss(y_test, prob_clf)
    
    print(f"📊 Probability Metrics:")
    print(f"   Log Loss: {logloss_clf:.5f}")
    print(f"   Brier Score: {brier_clf:.5f}")
    print(f"   Prob Range: {prob_clf.min():.3f} to {prob_clf.max():.3f}")
    print(f"   Prob Std Dev: {prob_clf.std():.3f}")
    
    # =========================================================================
    # MODEL B: MARGIN REGRESSOR → Win Probability
    # =========================================================================
    print("\n" + "="*85)
    print("🅱️  MODEL B: MARGIN REGRESSOR (Margin-Derived Probability)")
    print("="*85)
    
    y_train_margin = train_df['score_margin']
    y_test_margin = test_df['score_margin']
    
    # XGBoost params for regression
    reg_params = XGB_PARAMS.copy()
    reg_params['objective'] = 'reg:squarederror'
    reg_params['eval_metric'] = 'mae'
    
    print("⚙️  Training XGBoost Regressor...")
    dtrain_reg = xgb.DMatrix(X_train, label=y_train_margin)
    dtest_reg = xgb.DMatrix(X_test)
    
    model_reg = xgb.train(reg_params, dtrain_reg, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    
    pred_margin = model_reg.predict(dtest_reg)
    test_df['pred_margin'] = pred_margin
    
    # Convert margin to win probability
    prob_mdp = margin_to_win_prob(pred_margin, NBA_STDEV)
    test_df['prob_mdp'] = prob_mdp
    
    # Metrics
    mae_margin = mean_absolute_error(y_test_margin, pred_margin)
    logloss_mdp = log_loss(y_test, prob_mdp)
    brier_mdp = brier_score_loss(y_test, prob_mdp)
    
    print(f"📊 Margin Prediction:")
    print(f"   MAE: {mae_margin:.2f} points")
    print(f"   Margin Range: {pred_margin.min():.1f} to {pred_margin.max():.1f}")
    print(f"   Margin Std Dev: {pred_margin.std():.2f}")
    
    print(f"\n📊 Converted Win Probability:")
    print(f"   Log Loss: {logloss_mdp:.5f}")
    print(f"   Brier Score: {brier_mdp:.5f}")
    print(f"   Prob Range: {prob_mdp.min():.3f} to {prob_mdp.max():.3f}")
    print(f"   Prob Std Dev: {prob_mdp.std():.3f}")
    
    # =========================================================================
    # COMPARISON 1: PROBABILITY CALIBRATION
    # =========================================================================
    print("\n" + "="*85)
    print("📊 COMPARISON 1: PROBABILITY CALIBRATION")
    print("="*85)
    
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ['0-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%']
    
    test_df['bin_clf'] = pd.cut(test_df['prob_classifier'], bins=bins, labels=labels)
    test_df['bin_mdp'] = pd.cut(test_df['prob_mdp'], bins=bins, labels=labels)
    
    print(f"\n{'BUCKET':<12} | {'CLASSIFIER':<25} | {'MARGIN-DERIVED':<25}")
    print(f"{'':12} | {'Pred%':<8} {'Act%':<8} {'Diff':<8} | {'Pred%':<8} {'Act%':<8} {'Diff':<8}")
    print("-"*85)
    
    for label in labels:
        # Classifier
        subset_clf = test_df[test_df['bin_clf'] == label]
        if len(subset_clf) > 0:
            pred_clf_avg = subset_clf['prob_classifier'].mean()
            act_clf = subset_clf['target'].mean()
            diff_clf = act_clf - pred_clf_avg
        else:
            pred_clf_avg = act_clf = diff_clf = np.nan
        
        # MDP
        subset_mdp = test_df[test_df['bin_mdp'] == label]
        if len(subset_mdp) > 0:
            pred_mdp_avg = subset_mdp['prob_mdp'].mean()
            act_mdp = subset_mdp['target'].mean()
            diff_mdp = act_mdp - pred_mdp_avg
        else:
            pred_mdp_avg = act_mdp = diff_mdp = np.nan
        
        print(f"{label:<12} | {pred_clf_avg:.1%}  {act_clf:.1%}  {diff_clf:+.1%} | {pred_mdp_avg:.1%}  {act_mdp:.1%}  {diff_mdp:+.1%}")
    
    # =========================================================================
    # COMPARISON 2: FAVORITE PERFORMANCE (THE CRITICAL TEST)
    # =========================================================================
    print("\n" + "="*85)
    print("🎯 COMPARISON 2: FAVORITE PERFORMANCE (The Critical Test)")
    print("="*85)
    
    # For each model, bet on favorites when model agrees (prob > 0.5 for home fav)
    def evaluate_favorites(prob_col, model_name):
        df_eval = test_df.copy()
        df_eval['model_prob'] = df_eval[prob_col]
        
        # Model agrees with Vegas favorite
        df_eval['bet_on_fav'] = np.where(
            df_eval['home_is_vegas_fav'],
            df_eval['model_prob'] > 0.5,
            df_eval['model_prob'] < 0.5
        )
        
        fav_bets = df_eval[df_eval['bet_on_fav']].copy()
        
        if len(fav_bets) == 0:
            return None
        
        # Did favorite win?
        fav_bets['fav_won'] = np.where(
            fav_bets['home_is_vegas_fav'],
            fav_bets['target'] == 1,
            fav_bets['target'] == 0
        )
        
        # Payout
        fav_bets['payout'] = np.where(
            fav_bets['home_is_vegas_fav'],
            fav_bets['home_payout'],
            fav_bets['away_payout']
        )
        
        fav_bets['profit'] = np.where(fav_bets['fav_won'], fav_bets['payout'] - 1, -1)
        
        # By odds bucket
        fav_bets['fav_ml'] = np.where(
            fav_bets['home_is_vegas_fav'],
            fav_bets['home_ml_odds'],
            fav_bets['away_ml_odds']
        )
        
        results = {
            'total_bets': len(fav_bets),
            'win_rate': fav_bets['fav_won'].mean(),
            'total_profit': fav_bets['profit'].sum(),
            'roi': fav_bets['profit'].mean()
        }
        
        # Bucket analysis
        bins_odds = [-10000, -500, -250, -150, -110]
        labels_odds = ['Locks (-500+)', 'Heavy (-250 to -500)', 'Moderate (-150 to -250)', 'Cheap (-110 to -150)']
        fav_bets['odds_bin'] = pd.cut(fav_bets['fav_ml'], bins=bins_odds, labels=labels_odds)
        
        bucket_results = {}
        for label in labels_odds:
            subset = fav_bets[fav_bets['odds_bin'] == label]
            if len(subset) > 0:
                bucket_results[label] = {
                    'bets': len(subset),
                    'win_rate': subset['fav_won'].mean(),
                    'roi': subset['profit'].mean(),
                    'profit': subset['profit'].sum()
                }
        
        results['buckets'] = bucket_results
        
        return results
    
    results_clf = evaluate_favorites('prob_classifier', 'Classifier')
    results_mdp = evaluate_favorites('prob_mdp', 'Margin-Derived')
    
    print(f"\n{'MODEL':<20} | {'BETS':<6} | {'WIN %':<8} | {'ROI':<8} | {'PROFIT'}")
    print("-"*85)
    print(f"{'Classifier':<20} | {results_clf['total_bets']:<6} | {results_clf['win_rate']:.1%}  | {results_clf['roi']:+.1%}  | {results_clf['total_profit']:+.2f}u")
    print(f"{'Margin-Derived':<20} | {results_mdp['total_bets']:<6} | {results_mdp['win_rate']:.1%}  | {results_mdp['roi']:+.1%}  | {results_mdp['total_profit']:+.2f}u")
    
    # Detailed bucket comparison
    print(f"\n📊 BUCKET-BY-BUCKET COMPARISON:")
    print(f"\n{'ODDS BUCKET':<20} | {'CLASSIFIER ROI':<15} | {'MDP ROI':<15} | {'WINNER'}")
    print("-"*85)
    
    for bucket in ['Locks (-500+)', 'Heavy (-250 to -500)', 'Moderate (-150 to -250)', 'Cheap (-110 to -150)']:
        clf_bucket = results_clf['buckets'].get(bucket, {'roi': np.nan, 'bets': 0})
        mdp_bucket = results_mdp['buckets'].get(bucket, {'roi': np.nan, 'bets': 0})
        
        if clf_bucket['bets'] > 0 or mdp_bucket['bets'] > 0:
            winner = ""
            if not np.isnan(clf_bucket['roi']) and not np.isnan(mdp_bucket['roi']):
                if mdp_bucket['roi'] > clf_bucket['roi']:
                    winner = "✅ MDP Better"
                elif clf_bucket['roi'] > mdp_bucket['roi']:
                    winner = "⚠️ CLF Better"
                else:
                    winner = "🤝 Tie"
            
            clf_str = f"{clf_bucket['roi']:+.1%} ({clf_bucket['bets']})" if clf_bucket['bets'] > 0 else "N/A"
            mdp_str = f"{mdp_bucket['roi']:+.1%} ({mdp_bucket['bets']})" if mdp_bucket['bets'] > 0 else "N/A"
            
            print(f"{bucket:<20} | {clf_str:<15} | {mdp_str:<15} | {winner}")
    
    # =========================================================================
    # COMPARISON 3: CONFIDENCE LEVELS
    # =========================================================================
    print("\n" + "="*85)
    print("🎚️  COMPARISON 3: CONFIDENCE LEVELS (Does MDP predict dominance?)")
    print("="*85)
    
    print(f"\n{'METRIC':<30} | {'CLASSIFIER':<15} | {'MARGIN-DERIVED':<15}")
    print("-"*85)
    print(f"{'High Confidence (>80%)':<30} | {(prob_clf > 0.8).sum():<15} | {(prob_mdp > 0.8).sum():<15}")
    print(f"{'Very High (>90%)':<30} | {(prob_clf > 0.9).sum():<15} | {(prob_mdp > 0.9).sum():<15}")
    print(f"{'Extreme (>95%)':<30} | {(prob_clf > 0.95).sum():<15} | {(prob_mdp > 0.95).sum():<15}")
    print(f"{'Mean Probability':<30} | {prob_clf.mean():.3f}{'':11} | {prob_mdp.mean():.3f}{'':11}")
    print(f"{'Std Dev of Probability':<30} | {prob_clf.std():.3f}{'':11} | {prob_mdp.std():.3f}{'':11}")
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "="*85)
    print("🏆 FINAL VERDICT: WHICH ARCHITECTURE WINS?")
    print("="*85)
    
    improvements = []
    
    # Check log loss
    if logloss_mdp < logloss_clf:
        diff = (logloss_clf - logloss_mdp) / logloss_clf * 100
        print(f"✅ Log Loss: MDP is {diff:.1f}% better ({logloss_mdp:.5f} vs {logloss_clf:.5f})")
        improvements.append('log_loss')
    else:
        diff = (logloss_mdp - logloss_clf) / logloss_clf * 100
        print(f"❌ Log Loss: Classifier is {diff:.1f}% better ({logloss_clf:.5f} vs {logloss_mdp:.5f})")
    
    # Check favorite ROI
    if results_mdp['roi'] > results_clf['roi']:
        diff = results_mdp['roi'] - results_clf['roi']
        print(f"✅ Favorite ROI: MDP is {diff:.1%} better ({results_mdp['roi']:+.1%} vs {results_clf['roi']:+.1%})")
        improvements.append('favorite_roi')
    else:
        diff = results_clf['roi'] - results_mdp['roi']
        print(f"❌ Favorite ROI: Classifier is {diff:.1%} better ({results_clf['roi']:+.1%} vs {results_mdp['roi']:+.1%})")
    
    # Check favorite profit
    if results_mdp['total_profit'] > results_clf['total_profit']:
        diff = results_mdp['total_profit'] - results_clf['total_profit']
        print(f"✅ Favorite Profit: MDP is {diff:+.2f}u better ({results_mdp['total_profit']:+.2f}u vs {results_clf['total_profit']:+.2f}u)")
        improvements.append('favorite_profit')
    else:
        diff = results_clf['total_profit'] - results_mdp['total_profit']
        print(f"❌ Favorite Profit: Classifier is {diff:+.2f}u better ({results_clf['total_profit']:+.2f}u vs {results_mdp['total_profit']:+.2f}u)")
    
    # Check probability range (MDP should have wider range)
    if prob_mdp.std() > prob_clf.std():
        diff = (prob_mdp.std() - prob_clf.std()) / prob_clf.std() * 100
        print(f"✅ Confidence Range: MDP has {diff:.1f}% wider spread (better separation)")
        improvements.append('confidence_range')
    else:
        print(f"❌ Confidence Range: Classifier has wider spread")
    
    print("\n" + "="*85)
    if len(improvements) >= 3:
        print("🏆 WINNER: MARGIN-DERIVED PROBABILITY (MDP)")
        print("   Recommendation: Retrain production model as XGBoost Regressor")
        print("   Architecture: Predict margin → Convert to probability using Normal CDF")
    elif len(improvements) >= 1:
        print("🤔 MIXED RESULTS: MDP shows promise but not decisive")
        print("   Recommendation: Test on longer time period or ensemble both approaches")
    else:
        print("⚠️ WINNER: BINARY CLASSIFIER (Current)")
        print("   Recommendation: Keep current architecture, focus on calibration")
    print("="*85)

if __name__ == "__main__":
    test_architectures()
