import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from scipy.stats import pearsonr
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = 'data/training_data_GOLD_ELO_22_features.csv'
SPLIT_DATE = '2024-10-01'  # Start of 2024-25 Season

# IMPORT FEATURES FROM CONFIG
try:
    from production_config import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS
except ImportError:
    # Fallback if config is messy
    print("⚠️ Config missing. Using 'Variant D' defaults.")
    ACTIVE_FEATURES = [
        'off_elo_diff', 'def_elo_diff', 'home_composite_elo',           
        'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
        'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',          
        'injury_matchup_advantage', 'injury_shock_diff', 'star_power_leverage',
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

def validate_integrity():
    print("\n" + "="*80)
    print("🕵️ MODEL INTEGRITY & LEAKAGE CHECK")
    print("="*80)
    
    # 1. LOAD DATA
    print(f"📂 Loading {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Handle date column naming
        if 'date' in df.columns:
            df['game_date'] = pd.to_datetime(df['date'])
        elif 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        else:
            print("❌ No date column found")
            return
            
        df = df.sort_values('game_date').reset_index(drop=True)
        
        # Handle target column naming
        if 'target' not in df.columns and 'target_moneyline_win' in df.columns:
            df['target'] = df['target_moneyline_win']
            print("   ℹ️  Using 'target_moneyline_win' as target")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"   ✓ Loaded {len(df):,} games from {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    # Check for features
    missing = [f for f in ACTIVE_FEATURES if f not in df.columns]
    if missing:
        print(f"❌ MISSING FEATURES: {missing}")
        return

    # -------------------------------------------------------------------------
    # TEST 1: THE LEAKAGE HUNTER (Correlation Check)
    # -------------------------------------------------------------------------
    print("\n🔍 TEST 1: LEAKAGE HUNTER (Correlation with Target)")
    print("   Goal: Find features that 'know' the result too well (>0.3 is suspicious)")
    print("-" * 60)
    
    suspicious = []
    print(f"{'FEATURE':<35} | {'CORRELATION':<10} | {'STATUS'}")
    print("-" * 60)
    
    for feat in ACTIVE_FEATURES:
        # Drop NaNs for correlation
        tmp = df[[feat, 'target']].dropna()
        if len(tmp) == 0:
            continue
        corr, _ = pearsonr(tmp[feat], tmp['target'])
        
        status = "✅ Safe"
        if abs(corr) > 0.5: 
            status = "🚨 LEAK?" 
            suspicious.append((feat, corr))
        elif abs(corr) > 0.3: 
            status = "⚠️ High"
        
        print(f"{feat:<35} | {corr:+.4f}     | {status}")
        
    if suspicious:
        print(f"\n❌ WARNING: {len(suspicious)} features look dangerously correlated!")
        for feat, corr in suspicious:
            print(f"   • {feat}: {corr:+.4f}")
    else:
        print("\n✅ PASS: No obvious single-feature leaks found.")

    # -------------------------------------------------------------------------
    # TEST 2: WALK-FORWARD SIMULATION (2024-25 Season)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("📅 TEST 2: 2024-25 WALK-FORWARD SIMULATION")
    print("   Goal: Replicate real-world conditions (Train on Past, Test on Future)")
    print("="*80)
    
    train_df = df[df['game_date'] < SPLIT_DATE].copy()
    test_df = df[df['game_date'] >= SPLIT_DATE].copy()
    
    print(f"📚 Training: {len(train_df):,} games (History before {SPLIT_DATE})")
    print(f"🔮 Testing:  {len(test_df):,} games (2024-25 Season)")
    
    if len(test_df) == 0:
        print("❌ No 2024-25 games found. Check SPLIT_DATE.")
        return

    # Train Model
    print("⚙️  Training 'Variant D' Model...")
    X_train = train_df[ACTIVE_FEATURES]
    y_train = train_df['target']
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    
    # Predict
    X_test = test_df[ACTIVE_FEATURES]
    dtest = xgb.DMatrix(X_test)
    preds = model.predict(dtest)
    test_df['model_prob'] = preds
    
    # Metrics
    loss = log_loss(test_df['target'], preds)
    acc = accuracy_score(test_df['target'], (preds > 0.5).astype(int))
    brier = brier_score_loss(test_df['target'], preds)
    
    print("-" * 60)
    print(f"🏆 SIMULATION RESULTS")
    print(f"   Log Loss: {loss:.5f}")
    print(f"   Accuracy: {acc:.2%}")
    print(f"   Brier Score: {brier:.5f}")
    
    if loss < 0.60:
        print("   ⚠️  RESULT IS EXTREMELY GOOD. Verify data carefully.")
    elif loss < 0.65:
        print("   ✅ STRONG performance - but still realistic for NBA.")
    else:
        print("   ℹ️  Standard performance for NBA prediction.")
    
    # -------------------------------------------------------------------------
    # TEST 3: REALITY CHECK (Win Rates by Bucket)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("⚖️ TEST 3: REALITY CHECK (Probability Buckets)")
    print("   Goal: Does '70% Confidence' actually win 70% of the time?")
    print("="*80)
    
    # Create Buckets
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ['0-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%']
    test_df['conf_bin'] = pd.cut(test_df['model_prob'], bins=bins, labels=labels)
    
    print(f"{'CONFIDENCE':<12} | {'GAMES':<6} | {'PREDICTED %':<12} | {'ACTUAL WIN %':<12} | {'DIFF'}")
    print("-" * 75)
    
    calibration_errors = []
    for label in labels:
        subset = test_df[test_df['conf_bin'] == label]
        if len(subset) > 0:
            avg_pred = subset['model_prob'].mean()
            actual_win = subset['target'].mean()
            diff = actual_win - avg_pred
            
            flag = ""
            if abs(diff) > 0.15: 
                flag = "🚨 MAJOR ERROR"
                calibration_errors.append((label, diff))
            elif abs(diff) > 0.10: 
                flag = "⚠️ CALIBRATION OFF"
                calibration_errors.append((label, diff))
            
            print(f"{label:<12} | {len(subset):<6} | {avg_pred:.1%}       | {actual_win:.1%}       | {diff:+.1%} {flag}")
    
    if calibration_errors:
        print(f"\n⚠️  {len(calibration_errors)} buckets show calibration issues")
    else:
        print(f"\n✅ PASS: Model probabilities match reality well")

    # -------------------------------------------------------------------------
    # TEST 4: FINANCIAL REALITY (Moneyline Audit)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("💰 TEST 4: MONEYLINE PRICING AUDIT")
    print("   Goal: Are we winning games we shouldn't?")
    print("="*80)
    
    # Try to load odds data
    odds_loaded = False
    try:
        odds_2024 = pd.read_csv('data/closing_odds_2024_25_CLEANED.csv')
        odds_2024['game_date'] = pd.to_datetime(odds_2024['game_date'])
        
        # Merge with test_df
        test_df = test_df.merge(
            odds_2024[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
            on=['game_date', 'home_team', 'away_team'],
            how='left'
        )
        
        # Check if merge worked
        if test_df['home_ml_odds'].notna().sum() > 0:
            odds_loaded = True
            print(f"   ✓ Loaded odds for {test_df['home_ml_odds'].notna().sum():,} games")
    except Exception as e:
        print(f"   ⚠️ Could not load odds data: {e}")
    
    if odds_loaded and 'home_ml_odds' in test_df.columns:
        # Helper for Payouts
        def get_payout(ml): 
            if pd.isna(ml):
                return np.nan
            return (ml/100)+1 if ml>0 else (100/abs(ml))+1
        
        test_df['home_payout'] = test_df['home_ml_odds'].apply(get_payout)
        test_df['away_payout'] = test_df['away_ml_odds'].apply(get_payout)
        
        # Calculate ROI by "Odds Bucket" (Implied Probability)
        # We want to know: Are we profitable on Longshots (+200)? Or Favorites (-200)?
        
        # Determine Bet (Simple >50% Strategy for Audit)
        test_df['bet_home'] = test_df['model_prob'] > 0.5
        
        # Assign Odds of the Team we Bet on
        test_df['bet_odds'] = np.where(test_df['bet_home'], test_df['home_ml_odds'], test_df['away_ml_odds'])
        test_df['bet_payout'] = np.where(test_df['bet_home'], test_df['home_payout'], test_df['away_payout'])
        
        # Did we win?
        test_df['bet_won'] = np.where(test_df['bet_home'], test_df['target'] == 1, test_df['target'] == 0)
        
        # Filter to bets where we have odds
        bet_df = test_df[test_df['bet_odds'].notna()].copy()
        bet_df['profit'] = np.where(bet_df['bet_won'], bet_df['bet_payout'] - 1, -1)
        
        # Bucket by ODDS
        def bucket_odds(ml):
            if ml >= 200: return "Longshot (+200+)"
            if ml >= 100: return "Underdog (+100 to +200)"
            if ml >= -150: return "Tossup (-150 to +100)"
            return "Favorite (-150+)"
            
        bet_df['odds_bucket'] = bet_df['bet_odds'].apply(bucket_odds)
        
        print(f"{'ODDS BUCKET':<25} | {'BETS':<6} | {'WIN %':<8} | {'ROI':<8} | {'PROFIT'}")
        print("-" * 75)
        
        suspicious_results = []
        for bucket in ["Longshot (+200+)", "Underdog (+100 to +200)", "Tossup (-150 to +100)", "Favorite (-150+)"]:
            subset = bet_df[bet_df['odds_bucket'] == bucket]
            if len(subset) > 0:
                roi = subset['profit'].sum() / len(subset)
                win_rate = subset['bet_won'].mean()
                profit = subset['profit'].sum()
                
                flag = ""
                # Check for suspicious win rates
                if bucket == "Longshot (+200+)" and win_rate > 0.55:
                    flag = "🚨 TOO GOOD"
                    suspicious_results.append((bucket, win_rate))
                elif bucket == "Favorite (-150+)" and win_rate > 0.75:
                    flag = "🚨 TOO GOOD"
                    suspicious_results.append((bucket, win_rate))
                
                print(f"{bucket:<25} | {len(subset):<6} | {win_rate:.1%}  | {roi:+.1%}  | {profit:+.1f}u {flag}")
        
        overall_roi = bet_df['profit'].sum() / len(bet_df)
        overall_winrate = bet_df['bet_won'].mean()
        print("-" * 75)
        print(f"{'OVERALL':<25} | {len(bet_df):<6} | {overall_winrate:.1%}  | {overall_roi:+.1%}  | {bet_df['profit'].sum():+.1f}u")
        
        if suspicious_results:
            print(f"\n🚨 SUSPICIOUS: {len(suspicious_results)} buckets have unrealistic win rates")
            for bucket, wr in suspicious_results:
                print(f"   • {bucket}: {wr:.1%}")
        else:
            print(f"\n✅ PASS: Win rates are realistic for each odds bucket")
            
    else:
        print("⚠️ No Moneyline Data found. Skipping Financial Audit.")
        print("   To enable: Place cleaned odds CSVs in data/ folder")

    # -------------------------------------------------------------------------
    # FINAL VERDICT
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🎯 FINAL INTEGRITY VERDICT")
    print("="*80)
    
    issues = []
    if suspicious:
        issues.append(f"🚨 Data Leakage Risk: {len(suspicious)} features highly correlated")
    if loss < 0.60:
        issues.append(f"⚠️ Suspiciously Low Log Loss: {loss:.4f} (verify temporal integrity)")
    if calibration_errors and len(calibration_errors) > 3:
        issues.append(f"⚠️ Calibration Issues: {len(calibration_errors)} buckets off by >10%")
    if odds_loaded and suspicious_results:
        issues.append(f"🚨 Financial Red Flags: Unrealistic win rates detected")
    
    if issues:
        print("❌ INTEGRITY CONCERNS DETECTED:\n")
        for issue in issues:
            print(f"   {issue}")
        print("\n🔬 Recommendation: Review features and data pipeline for leakage")
    else:
        print("✅ INTEGRITY CHECK PASSED")
        print("   • No obvious data leakage detected")
        print("   • Performance metrics are realistic")
        print("   • Calibration within acceptable bounds")
        print("   • Financial results align with expectations")

if __name__ == "__main__":
    validate_integrity()
