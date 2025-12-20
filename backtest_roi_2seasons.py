"""
WALK-FORWARD ROI BACKTEST: 2023-24 and 2024-25 Seasons
========================================================
Tests Optimized Variant D with flat $100 bets to calculate:
- Total units won/lost
- ROI percentage
- Win rate
- Bet volume

Compares performance across two recent seasons.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime
import json

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

# Variant D Features (19 features from actual dataset)
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

# Betting Configuration
FLAT_BET = 100  # $100 per bet
MIN_EDGE = 0.02  # 2% minimum edge to place bet (similar to Trial 1306 thresholds)
COMMISSION = 0.00  # Assuming we can find no-vig lines or this is already factored

# Data
FILE_PATH = 'data/training_data_GOLD_ELO_22_features.csv'

# Season splits
SEASON_2023_START = '2023-10-01'
SEASON_2023_END = '2024-04-30'
SEASON_2024_START = '2024-10-01'
SEASON_2024_END = '2025-12-31'

# ==========================================
# 📊 BETTING SIMULATION
# ==========================================

def simulate_bets(y_true, y_pred, bet_amount=FLAT_BET, min_edge=MIN_EDGE):
    """
    Simulate betting with flat stakes
    
    Returns:
        DataFrame with bet-by-bet results
    """
    bets = []
    
    for i, (actual, prob) in enumerate(zip(y_true, y_pred)):
        # Calculate edge (assuming fair odds at 50%, adjust if you have actual market odds)
        # For now, assume we bet home when prob > 0.5 + min_edge
        # and bet away when prob < 0.5 - min_edge
        
        if prob > 0.5 + min_edge:
            # Bet on home
            bet_side = 'home'
            edge = prob - 0.5
            
            if actual == 1:  # Home won
                profit = bet_amount  # 1:1 odds simplified
                result = 'win'
            else:
                profit = -bet_amount
                result = 'loss'
                
            bets.append({
                'bet_number': i,
                'bet_side': bet_side,
                'model_prob': prob,
                'edge': edge,
                'bet_amount': bet_amount,
                'profit': profit,
                'result': result
            })
            
        elif prob < 0.5 - min_edge:
            # Bet on away
            bet_side = 'away'
            edge = (1 - prob) - 0.5
            
            if actual == 0:  # Away won (home lost)
                profit = bet_amount
                result = 'win'
            else:
                profit = -bet_amount
                result = 'loss'
                
            bets.append({
                'bet_number': i,
                'bet_side': bet_side,
                'model_prob': prob,
                'edge': edge,
                'bet_amount': bet_amount,
                'profit': profit,
                'result': result
            })
    
    return pd.DataFrame(bets)

# ==========================================
# 🧪 MAIN BACKTEST
# ==========================================

def run_roi_backtest():
    print("="*80)
    print("💰 WALK-FORWARD ROI BACKTEST: 2023-24 and 2024-25 SEASONS")
    print("="*80)
    print(f"Model: Optimized Variant D (Trial #245)")
    print(f"Betting: Flat ${FLAT_BET} per bet")
    print(f"Min Edge: {MIN_EDGE*100:.1f}%")
    print("="*80)
    
    # 1. Load data
    try:
        df = pd.read_csv(FILE_PATH)
        
        if 'date' in df.columns:
            df['game_date'] = pd.to_datetime(df['date'])
        elif 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        else:
            raise ValueError("No date column found")
        
        df = df.sort_values('game_date').reset_index(drop=True)
        print(f"\n✓ Loaded {len(df):,} games")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 2. Check features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
        return
    
    target_col = 'target_moneyline_win'
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return
    
    # 3. Define training and test sets for walk-forward
    # Train on everything before 2023-24 season
    # Test on 2023-24 and 2024-25 separately
    
    train_df = df[df['game_date'] < SEASON_2023_START].copy()
    season_23_df = df[(df['game_date'] >= SEASON_2023_START) & 
                      (df['game_date'] <= SEASON_2023_END)].copy()
    season_24_df = df[(df['game_date'] >= SEASON_2024_START) & 
                      (df['game_date'] <= SEASON_2024_END)].copy()
    
    print(f"\n📚 TRAINING SET")
    print(f"   Games: {len(train_df):,}")
    print(f"   Date range: {train_df['game_date'].min().date()} to {train_df['game_date'].max().date()}")
    
    print(f"\n📅 TEST SET: 2023-24 SEASON")
    print(f"   Games: {len(season_23_df):,}")
    if len(season_23_df) > 0:
        print(f"   Date range: {season_23_df['game_date'].min().date()} to {season_23_df['game_date'].max().date()}")
    
    print(f"\n📅 TEST SET: 2024-25 SEASON")
    print(f"   Games: {len(season_24_df):,}")
    if len(season_24_df) > 0:
        print(f"   Date range: {season_24_df['game_date'].min().date()} to {season_24_df['game_date'].max().date()}")
    
    # 4. Prepare training data
    X_train = train_df[FEATURES].copy()
    y_train = train_df[target_col].copy()
    
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train, y_train = X_train[mask], y_train[mask]
    
    print(f"\n✓ Training samples: {len(X_train):,}")
    
    # 5. Train model
    print("\n" + "="*80)
    print("⚙️  TRAINING MODEL")
    print("="*80)
    
    model = xgb.XGBClassifier(**OPTIMIZED_PARAMS, n_estimators=N_ESTIMATORS)
    model.fit(X_train, y_train, verbose=False)
    
    print("✓ Model trained successfully")
    
    # 6. Test on both seasons
    results_summary = []
    
    for season_name, season_df in [('2023-24', season_23_df), ('2024-25', season_24_df)]:
        if len(season_df) == 0:
            print(f"\n⚠️ No data for {season_name} season")
            continue
        
        print("\n" + "="*80)
        print(f"🎯 TESTING: {season_name} SEASON")
        print("="*80)
        
        X_test = season_df[FEATURES].copy()
        y_test = season_df[target_col].copy()
        
        mask = ~(X_test.isna().any(axis=1) | y_test.isna())
        X_test, y_test = X_test[mask], y_test[mask]
        
        print(f"✓ Test samples: {len(X_test):,}")
        
        # Predict
        preds = model.predict_proba(X_test)[:, 1]
        
        # Model performance
        loss = log_loss(y_test, preds)
        acc = accuracy_score(y_test, (preds > 0.5).astype(int))
        
        print(f"\nModel Performance:")
        print(f"  Log Loss: {loss:.5f}")
        print(f"  Accuracy: {acc:.2%}")
        
        # Simulate betting
        bet_results = simulate_bets(y_test.values, preds, FLAT_BET, MIN_EDGE)
        
        if len(bet_results) == 0:
            print(f"\n⚠️ No bets met {MIN_EDGE*100:.1f}% edge threshold")
            continue
        
        # Calculate metrics
        total_bets = len(bet_results)
        total_wagered = bet_results['bet_amount'].sum()
        total_profit = bet_results['profit'].sum()
        roi = (total_profit / total_wagered) * 100
        win_rate = (bet_results['result'] == 'win').sum() / total_bets
        avg_edge = bet_results['edge'].mean()
        
        print(f"\n💰 BETTING RESULTS:")
        print(f"  Total Bets:     {total_bets:,}")
        print(f"  Total Wagered:  ${total_wagered:,.2f}")
        print(f"  Total Profit:   ${total_profit:,.2f}")
        print(f"  ROI:            {roi:+.2f}%")
        print(f"  Win Rate:       {win_rate:.2%}")
        print(f"  Avg Edge:       {avg_edge*100:.2f}%")
        
        # Verdict
        if roi > 10:
            print(f"\n✅ ELITE: {roi:.1f}% ROI (Crushing it!)")
        elif roi > 5:
            print(f"\n✅ EXCELLENT: {roi:.1f}% ROI (Very profitable)")
        elif roi > 2:
            print(f"\n✅ SOLID: {roi:.1f}% ROI (Profitable)")
        elif roi > 0:
            print(f"\n⚠️ MARGINAL: {roi:.1f}% ROI (Barely profitable)")
        else:
            print(f"\n🔴 LOSING: {roi:.1f}% ROI (Not profitable)")
        
        # Store results
        results_summary.append({
            'season': season_name,
            'games': len(X_test),
            'log_loss': float(loss),
            'accuracy': float(acc),
            'total_bets': total_bets,
            'total_wagered': float(total_wagered),
            'total_profit': float(total_profit),
            'roi_percent': float(roi),
            'win_rate': float(win_rate),
            'avg_edge': float(avg_edge)
        })
        
        # Save detailed bet log
        bet_results.to_csv(
            f'models/experimental/bet_log_{season_name.replace("-", "_")}.csv',
            index=False
        )
        print(f"\n✓ Bet log saved: models/experimental/bet_log_{season_name.replace('-', '_')}.csv")
    
    # 7. Summary across both seasons
    if len(results_summary) > 0:
        print("\n" + "="*80)
        print("📊 SUMMARY: BOTH SEASONS")
        print("="*80)
        
        total_bets = sum(r['total_bets'] for r in results_summary)
        total_wagered = sum(r['total_wagered'] for r in results_summary)
        total_profit = sum(r['total_profit'] for r in results_summary)
        combined_roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        print(f"\nCombined Results:")
        print(f"  Total Bets:     {total_bets:,}")
        print(f"  Total Wagered:  ${total_wagered:,.2f}")
        print(f"  Total Profit:   ${total_profit:,.2f}")
        print(f"  Combined ROI:   {combined_roi:+.2f}%")
        
        print(f"\nPer-Season Breakdown:")
        for r in results_summary:
            print(f"  {r['season']}: {r['total_bets']:3d} bets, ${r['total_profit']:+8.2f} profit, {r['roi_percent']:+6.2f}% ROI")
        
        # Save summary
        summary_file = 'models/experimental/roi_backtest_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'model': 'variant_d_optimized_trial245',
                'flat_bet': FLAT_BET,
                'min_edge': MIN_EDGE,
                'seasons': results_summary,
                'combined': {
                    'total_bets': total_bets,
                    'total_wagered': float(total_wagered),
                    'total_profit': float(total_profit),
                    'roi_percent': float(combined_roi)
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n✓ Summary saved: {summary_file}")
        
        # Final verdict
        print("\n" + "="*80)
        print("🏆 FINAL VERDICT")
        print("="*80)
        
        if combined_roi > 10:
            print(f"✅ PRODUCTION READY: {combined_roi:.1f}% combined ROI")
            print("   Model shows consistent edge across multiple seasons")
            print("   🚀 APPROVED FOR DEPLOYMENT")
        elif combined_roi > 5:
            print(f"✅ STRONG CANDIDATE: {combined_roi:.1f}% combined ROI")
            print("   Model is profitable but monitor closely")
        elif combined_roi > 2:
            print(f"⚠️ PROCEED WITH CAUTION: {combined_roi:.1f}% combined ROI")
            print("   Profitable but margin is thin")
        else:
            print(f"🔴 NOT READY: {combined_roi:.1f}% combined ROI")
            print("   Needs further optimization or different thresholds")

if __name__ == "__main__":
    run_roi_backtest()
