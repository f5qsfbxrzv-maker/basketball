import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from scipy.stats import pearsonr
import sys
import warnings

warnings.filterwarnings("ignore")

# CONFIGURATION
DATA_PATH = 'data/training_data_GOLD_ELO_22_features.csv'
ODDS_PATH = 'data/closing_odds_2024_25_CLEANED.csv'
SPLIT_DATE = '2024-10-22'  # Start of 2024-25 Season

# Import your Active Features
try:
    from production_config import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS
except ImportError:
    print("❌ Critical: production_config.py missing.")
    sys.exit()

def forensic_backtest():
    print("🕵️ STARTING COMPREHENSIVE 2024-25 FORENSIC BACKTEST...")
    print(f"   Target: Diagnosing the 'Broken Favorites' Issue")
    print("-" * 75)

    # 1. LOAD & PREP DATA
    print("📂 Loading training data...")
    df = pd.read_csv(DATA_PATH)
    
    # Handle date column
    if 'date' in df.columns:
        df['game_date'] = pd.to_datetime(df['date'])
    else:
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Handle target column
    if 'target' not in df.columns and 'target_moneyline_win' in df.columns:
        df['target'] = df['target_moneyline_win']
    
    # Split: Train on history, test on 2024-25
    train_df = df[df['game_date'] < SPLIT_DATE].copy()
    season_df = df[df['game_date'] >= SPLIT_DATE].copy()
    
    if len(season_df) == 0:
        print("❌ Error: No games found after 2024-10-22.")
        return

    print(f"   ✓ Training set: {len(train_df):,} games")
    print(f"   ✓ Testing set (2024-25): {len(season_df):,} games")

    # 2. LOAD ODDS DATA
    print("📂 Loading odds data...")
    try:
        odds_df = pd.read_csv(ODDS_PATH)
        odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])
        
        # Merge odds with season data
        season_df = season_df.merge(
            odds_df[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
            on=['game_date', 'home_team', 'away_team'],
            how='left'
        )
        
        odds_coverage = season_df['home_ml_odds'].notna().sum()
        print(f"   ✓ Odds coverage: {odds_coverage}/{len(season_df)} games ({odds_coverage/len(season_df)*100:.1f}%)")
        
        if odds_coverage == 0:
            print("❌ Error: No odds data matched. Cannot perform forensic audit.")
            return
            
        # Filter to games with odds
        season_df = season_df[season_df['home_ml_odds'].notna()].copy()
        
    except Exception as e:
        print(f"❌ Error loading odds: {e}")
        return

    # 3. TRAIN MODEL ON PRE-2024 DATA
    print("⚙️  Training model on historical data...")
    X_train = train_df[ACTIVE_FEATURES]
    y_train = train_df['target']
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    
    # 4. GENERATE PREDICTIONS FOR 2024-25
    print("🔮 Generating predictions for 2024-25 season...")
    dtest = xgb.DMatrix(season_df[ACTIVE_FEATURES])
    season_df['model_prob'] = model.predict(dtest)

    # 5. CALCULATE IMPLIED PROBABILITIES & PAYOUTS
    def get_implied(ml): 
        if pd.isna(ml):
            return np.nan
        return (-ml)/(-ml+100) if ml<0 else 100/(ml+100)
    
    def get_payout(ml): 
        if pd.isna(ml):
            return np.nan
        return (ml/100)+1 if ml>0 else (100/abs(ml))+1

    season_df['home_implied'] = season_df['home_ml_odds'].apply(get_implied)
    season_df['away_implied'] = season_df['away_ml_odds'].apply(get_implied)
    season_df['home_payout'] = season_df['home_ml_odds'].apply(get_payout)
    season_df['away_payout'] = season_df['away_ml_odds'].apply(get_payout)

    # Determine Vegas Favorite vs Underdog
    season_df['home_is_vegas_fav'] = season_df['home_implied'] > season_df['away_implied']

    # ------------------------------------------------------------------
    # REPORT 1: THE FAVORITES AUDIT
    # ------------------------------------------------------------------
    print("\n" + "="*75)
    print("📉 REPORT 1: FAVORITES PERFORMANCE AUDIT (Why are we losing?)")
    print("   (Betting on the Vegas Favorite when Model agrees)")
    print("="*75)
    
    # Logic: Model agrees with Vegas favorite
    # If Home is Vegas Fav, Model must say Home Win (>50%). If Away is Fav, Model says Away Win (<50%).
    fav_bets = season_df.copy()
    fav_bets['bet_on_fav'] = np.where(
        fav_bets['home_is_vegas_fav'], 
        fav_bets['model_prob'] > 0.5, 
        fav_bets['model_prob'] < 0.5
    )
    
    # Filter to only rows where Model LIKED the Favorite
    model_likes_fav = fav_bets[fav_bets['bet_on_fav']].copy()
    
    # Did the Favorite win?
    model_likes_fav['fav_won'] = np.where(
        model_likes_fav['home_is_vegas_fav'], 
        model_likes_fav['target'] == 1, 
        model_likes_fav['target'] == 0
    )
    
    # Calculate Payout
    model_likes_fav['payout'] = np.where(
        model_likes_fav['home_is_vegas_fav'], 
        model_likes_fav['home_payout'], 
        model_likes_fav['away_payout']
    )
    model_likes_fav['profit'] = np.where(model_likes_fav['fav_won'], model_likes_fav['payout'] - 1, -1)

    print(f"Total Favorites Backed by Model: {len(model_likes_fav)}")
    print(f"Win Rate: {model_likes_fav['fav_won'].mean():.1%} (Actual Wins / Bets)")
    print(f"Total Profit: {model_likes_fav['profit'].sum():.2f} units")
    print(f"ROI: {model_likes_fav['profit'].mean():.2%}")
    
    # ------------------------------------------------------------------
    # REPORT 2: PRICING BUCKETS (Where is the leak?)
    # ------------------------------------------------------------------
    print("\n" + "="*75)
    print("📊 REPORT 2: FAVORITE PRICING BUCKETS")
    print("   (Are we failing on 'Heavy' favorites or 'Cheap' ones?)")
    print("-" * 75)
    print(f"{'ODDS RANGE':<20} | {'BETS':<6} | {'WIN %':<8} | {'ROI':<8} | {'PROFIT'}")
    print("-" * 75)
    
    # Get Odds of the Favorite
    model_likes_fav['fav_ml'] = np.where(
        model_likes_fav['home_is_vegas_fav'], 
        model_likes_fav['home_ml_odds'], 
        model_likes_fav['away_ml_odds']
    )
    
    bins = [-10000, -500, -250, -150, -110]
    labels = ['Locks (-500+)', 'Heavy (-250 to -500)', 'Moderate (-150 to -250)', 'Cheap (-110 to -150)']
    
    model_likes_fav['odds_bin'] = pd.cut(model_likes_fav['fav_ml'], bins=bins, labels=labels)
    
    for label in labels:
        subset = model_likes_fav[model_likes_fav['odds_bin'] == label]
        if len(subset) > 0:
            roi = subset['profit'].mean()
            win_rate = subset['fav_won'].mean()
            profit = subset['profit'].sum()
            print(f"{label:<20} | {len(subset):<6} | {win_rate:.1%}  | {roi:+.1%}  | {profit:+.2f}u")

    # ------------------------------------------------------------------
    # REPORT 3: THE "SMOKING GUN" (Disagreement Analysis)
    # ------------------------------------------------------------------
    print("\n" + "="*75)
    print("🔥 REPORT 3: THE SMOKING GUN (Market Disagreement)")
    print("   (Does the model know something Vegas doesn't?)")
    print("="*75)
    
    # Correlation Check
    # How closely does your Model Prob match Vegas Implied Prob?
    corr, _ = pearsonr(season_df['model_prob'], season_df['home_implied'])
    
    print(f"Correlation with Vegas: {corr:.4f}")
    if corr > 0.85:
        print("👉 DIAGNOSIS: Model is a 'Vegas Clone'. It mimics the line but adds no alpha.")
    elif corr < 0.60:
        print("👉 DIAGNOSIS: Model is a 'Contrarian'. It fundamentally disagrees with lines.")
    else:
        print("👉 DIAGNOSIS: Healthy disagreement. Model respects market but finds edges.")

    # Edge Analysis
    season_df['home_edge'] = season_df['model_prob'] - season_df['home_implied']
    season_df['away_edge'] = (1 - season_df['model_prob']) - season_df['away_implied']
    
    print(f"\nEdge Statistics:")
    print(f"   Home Edge - Mean: {season_df['home_edge'].mean():.3f}, Std: {season_df['home_edge'].std():.3f}")
    print(f"   Away Edge - Mean: {season_df['away_edge'].mean():.3f}, Std: {season_df['away_edge'].std():.3f}")
    print(f"   Games with >5% edge (either side): {((season_df['home_edge'].abs() > 0.05) | (season_df['away_edge'].abs() > 0.05)).sum()}")

    # ------------------------------------------------------------------
    # REPORT 4: THE "BUST" FEATURES
    # ------------------------------------------------------------------
    print("\n" + "="*75)
    print("🧬 REPORT 4: ANATOMY OF A FAILED FAVORITE")
    print("   (Comparing Winning Favorites vs. Losing Favorites)")
    print("-" * 75)
    
    # We analyze the "Heavy" favorites (-150 or worse) that LOST.
    busted_favs = model_likes_fav[
        (model_likes_fav['fav_won'] == False) & 
        (model_likes_fav['fav_ml'] < -150)
    ].copy()
    
    winning_favs = model_likes_fav[
        (model_likes_fav['fav_won'] == True) & 
        (model_likes_fav['fav_ml'] < -150)
    ].copy()

    # Compare Stats - focus on HOME favorites for cleaner analysis
    home_winners = winning_favs[winning_favs['home_is_vegas_fav']]
    home_busts = busted_favs[busted_favs['home_is_vegas_fav']]
    
    if len(home_winners) > 0 and len(home_busts) > 0:
        features_to_check = [
            'off_elo_diff', 'def_elo_diff', 'net_fatigue_score', 
            'injury_impact_diff', 'season_progress', 'ewma_pace_diff',
            'star_power_leverage', 'projected_possession_margin'
        ]
        
        # Filter to features that exist
        features_to_check = [f for f in features_to_check if f in home_winners.columns]
        
        print(f"{'FEATURE':<30} | {'WINNERS AVG':<12} | {'BUST AVG':<12} | {'DIFF':<10} | {'STATUS'}")
        print("-" * 80)
        
        red_flags = []
        for feat in features_to_check:
            w_avg = home_winners[feat].mean()
            l_avg = home_busts[feat].mean()
            diff = l_avg - w_avg
            diff_pct = (diff / abs(w_avg) * 100) if w_avg != 0 else 0
            
            status = ""
            if abs(diff_pct) > 15:
                status = "🚨 RED FLAG"
                red_flags.append((feat, diff_pct))
            elif abs(diff_pct) > 10:
                status = "⚠️ WARNING"
            
            print(f"{feat:<30} | {w_avg:12.4f} | {l_avg:12.4f} | {diff:+10.4f} | {status}")
        
        print("-" * 80)
        print(f"Analyzed {len(home_busts)} Home Favorites that Busted vs {len(home_winners)} Winners.")
        
        if red_flags:
            print(f"\n🚨 {len(red_flags)} RED FLAGS DETECTED:")
            for feat, pct in red_flags:
                print(f"   • {feat}: {pct:+.1f}% difference")
    else:
        print("⚠️ Insufficient data for Home Favorite comparison")
        print(f"   Winners: {len(home_winners)}, Busts: {len(home_busts)}")

    # ------------------------------------------------------------------
    # REPORT 5: UNDERDOG PERFORMANCE (The Control Group)
    # ------------------------------------------------------------------
    print("\n" + "="*75)
    print("🐕 REPORT 5: UNDERDOG PERFORMANCE (For Comparison)")
    print("   (When model likes the underdog)")
    print("="*75)
    
    # Model likes underdog (disagrees with Vegas)
    fav_bets['bet_on_dog'] = ~fav_bets['bet_on_fav']
    model_likes_dog = fav_bets[fav_bets['bet_on_dog']].copy()
    
    if len(model_likes_dog) > 0:
        # Did the underdog win?
        model_likes_dog['dog_won'] = np.where(
            model_likes_dog['home_is_vegas_fav'], 
            model_likes_dog['target'] == 0,  # Away underdog won
            model_likes_dog['target'] == 1   # Home underdog won
        )
        
        # Calculate Payout
        model_likes_dog['payout'] = np.where(
            model_likes_dog['home_is_vegas_fav'], 
            model_likes_dog['away_payout'], 
            model_likes_dog['home_payout']
        )
        model_likes_dog['profit'] = np.where(model_likes_dog['dog_won'], model_likes_dog['payout'] - 1, -1)
        
        print(f"Total Underdogs Backed by Model: {len(model_likes_dog)}")
        print(f"Win Rate: {model_likes_dog['dog_won'].mean():.1%}")
        print(f"Total Profit: {model_likes_dog['profit'].sum():.2f} units")
        print(f"ROI: {model_likes_dog['profit'].mean():.2%}")
        
        print(f"\n📊 Comparison:")
        print(f"   Favorites: {len(model_likes_fav)} bets, {model_likes_fav['profit'].sum():+.2f}u")
        print(f"   Underdogs: {len(model_likes_dog)} bets, {model_likes_dog['profit'].sum():+.2f}u")
        print(f"   Total:     {len(model_likes_fav) + len(model_likes_dog)} bets, {model_likes_fav['profit'].sum() + model_likes_dog['profit'].sum():+.2f}u")
    
    # ------------------------------------------------------------------
    # FINAL DIAGNOSIS
    # ------------------------------------------------------------------
    print("\n" + "="*75)
    print("🎯 FINAL DIAGNOSIS")
    print("="*75)
    
    fav_roi = model_likes_fav['profit'].mean()
    dog_roi = model_likes_dog['profit'].mean() if len(model_likes_dog) > 0 else 0
    
    if fav_roi < -0.05:
        print("🚨 CRITICAL: Favorites are bleeding money (ROI < -5%)")
        print("   → Model is overvaluing favorites or Vegas is sharper on favorites")
        print("   → Recommendation: Increase minimum edge threshold for favorites")
    elif fav_roi < 0:
        print("⚠️ WARNING: Favorites are slightly unprofitable")
        print("   → Consider avoiding favorites or requiring higher edge")
    else:
        print("✅ Favorites are profitable")
    
    if dog_roi > 0.05:
        print("✅ STRENGTH: Underdogs are highly profitable (ROI > 5%)")
        print("   → Model excels at finding value in underdogs")
        print("   → Recommendation: Focus betting strategy on underdogs")
    
    if corr > 0.85:
        print("\n⚠️ Model is too similar to Vegas - limited edge potential")
    elif corr < 0.60:
        print("\n⚠️ Model strongly disagrees with Vegas - verify data quality")

if __name__ == "__main__":
    forensic_backtest()
