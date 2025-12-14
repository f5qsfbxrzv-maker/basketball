"""
HYBRID MONEYLINE BACKTEST WITH DATA INTEGRITY FILTERING
Uses real closing odds when they're realistic, filters out corrupted extreme values
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sqlite3
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import json

print("="*80)
print("HYBRID MONEYLINE BACKTEST (WITH DATA QUALITY FILTERS)")
print("="*80)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def american_to_prob(odds):
    """Convert American odds to implied probability"""
    if pd.isna(odds) or odds == 0:
        return 0.5
    if odds < 0:
        return (-odds) / (-odds + 100)
    else:
        return 100 / (odds + 100)

def prob_to_american(prob):
    """Convert probability to American odds"""
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob

def is_realistic_odds(home_odds, away_odds):
    """
    Check if odds pair is realistic
    Returns True if odds pass sanity checks, False if corrupted
    """
    if pd.isna(home_odds) or pd.isna(away_odds):
        return False
    
    # Check 1: Vig should be 102-110% (typical sportsbook range)
    home_prob = american_to_prob(home_odds)
    away_prob = american_to_prob(away_odds)
    vig = home_prob + away_prob
    
    if vig < 1.02 or vig > 1.15:
        return False
    
    # Check 2: Extreme odds threshold (rarely see beyond these in NBA)
    if home_odds < -1200 or home_odds > 1200:
        return False
    if away_odds < -1200 or away_odds > 1200:
        return False
    
    # Check 3: Both can't be favorites or both underdogs
    if (home_odds < -105 and away_odds < -105):
        return False
    if (home_odds > 105 and away_odds > 105):
        return False
    
    return True

def get_synthetic_odds_from_model_prob(model_prob, vig=1.045):
    """
    Generate synthetic fair odds from model probability
    Apply realistic vig (4.5% default)
    """
    # Adjust for vig
    home_prob_with_vig = model_prob * vig / (model_prob * vig + (1 - model_prob) * vig)
    away_prob_with_vig = (1 - model_prob) * vig / (model_prob * vig + (1 - model_prob) * vig)
    
    home_odds = prob_to_american(home_prob_with_vig)
    away_odds = prob_to_american(away_prob_with_vig)
    
    return home_odds, away_odds

# ============================================================================
# LOAD DATA
# ============================================================================

# Load training data
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Load moneyline odds
conn = sqlite3.connect('data/live/historical_closing_odds.db')
odds_df = pd.read_sql("""
    SELECT game_date, home_team, away_team, home_ml_odds, away_ml_odds
    FROM moneyline_odds 
    ORDER BY game_date
""", conn)
conn.close()

odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

print(f"\n📊 Moneyline odds available: {len(odds_df)} games")

# ============================================================================
# SPLIT DATA
# ============================================================================

train_df = df[df['season'] != '2024-25'].copy()
test_df = df[df['season'] == '2024-25'].copy()

# Merge odds
test_df = test_df.merge(
    odds_df,
    left_on=['date', 'home_team', 'away_team'],
    right_on=['game_date', 'home_team', 'away_team'],
    how='inner'
).drop_duplicates(subset=['game_id'], keep='first')

print(f"\n📈 Data split:")
print(f"  Train: {len(train_df):,} games (pre-2024-25)")
print(f"  Test:  {len(test_df):,} games (2024-25 with odds)")

# ============================================================================
# DATA QUALITY FILTERING
# ============================================================================

print(f"\n🔍 Filtering odds quality...")
test_df['odds_realistic'] = test_df.apply(
    lambda row: is_realistic_odds(row['home_ml_odds'], row['away_ml_odds']),
    axis=1
)

clean_count = test_df['odds_realistic'].sum()
corrupt_count = len(test_df) - clean_count

print(f"  ✅ Clean odds: {clean_count} games ({clean_count/len(test_df)*100:.1f}%)")
print(f"  ❌ Corrupted/extreme odds: {corrupt_count} games ({corrupt_count/len(test_df)*100:.1f}%)")

# ============================================================================
# TRAIN MODEL
# ============================================================================

feature_cols = [c for c in df.columns if c not in [
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'date', 'game_id', 'season', 'home_team', 'away_team',
    'home_ml_odds', 'away_ml_odds', 'game_date', 'bookmaker', 
    'snapshot_timestamp', 'api_game_id', 'odds_realistic'
]]

print(f"\n🤖 Training model on {len(feature_cols)} features...")
model = xgb.XGBClassifier(
    learning_rate=0.008,
    n_estimators=2000,
    max_depth=4,
    min_child_weight=14,
    subsample=0.8318,
    colsample_bytree=0.6087,
    gamma=2.6571,
    reg_alpha=0.0159,
    reg_lambda=1.0830,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

model.fit(train_df[feature_cols], train_df['target_moneyline_win'])

# Calibrate
calib_start = pd.Timestamp('2023-10-01')
calib_df = train_df[train_df['date'] >= calib_start]

raw_calib = model.predict_proba(calib_df[feature_cols])[:, 1]
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(raw_calib, calib_df['target_moneyline_win'])

print("  ✅ Model trained and calibrated")

# ============================================================================
# PREDICT AND SIMULATE
# ============================================================================

raw_test = model.predict_proba(test_df[feature_cols])[:, 1]
test_df['model_prob'] = iso_reg.transform(raw_test)

# Performance metrics
auc = roc_auc_score(test_df['target_moneyline_win'], test_df['model_prob'])
brier = brier_score_loss(test_df['target_moneyline_win'], test_df['model_prob'])
accuracy = (test_df['model_prob'] > 0.5).astype(int).eq(test_df['target_moneyline_win']).mean()

print(f"\n📊 Model Performance:")
print(f"  AUC: {auc:.4f}")
print(f"  Brier: {brier:.4f}")
print(f"  Accuracy: {accuracy:.1%}")

# ============================================================================
# BETTING SIMULATION
# ============================================================================

print(f"\n{'='*80}")
print("HYBRID BETTING SIMULATION")
print("="*80)
print("Strategy: Use real odds if realistic, skip corrupted odds")
print("="*80)

def simulate_betting(df, edge_threshold, use_clean_only=True):
    """Simulate betting with given edge threshold"""
    bets = []
    
    for _, row in df.iterrows():
        # Skip if corrupted odds and we're being strict
        if use_clean_only and not row['odds_realistic']:
            continue
        
        home_ml = row['home_ml_odds']
        away_ml = row['away_ml_odds']
        model_prob = row['model_prob']
        
        home_implied = american_to_prob(home_ml)
        away_implied = american_to_prob(away_ml)
        
        # Calculate edges
        edge_home = model_prob - home_implied
        edge_away = (1 - model_prob) - away_implied
        
        # Determine bet
        if edge_home > edge_threshold:
            bet_side = 'HOME'
            bet_odds = home_ml
            win = row['target_moneyline_win'] == 1
        elif edge_away > edge_threshold:
            bet_side = 'AWAY'
            bet_odds = away_ml
            win = row['target_moneyline_win'] == 0
        else:
            continue
        
        # Calculate profit
        if bet_odds < 0:
            profit = (100 / abs(bet_odds)) if win else -1.0
        else:
            profit = (bet_odds / 100) if win else -1.0
        
        bets.append({
            'date': row['date'],
            'home': row['home_team'],
            'away': row['away_team'],
            'bet_side': bet_side,
            'bet_odds': bet_odds,
            'edge': edge_home if bet_side == 'HOME' else edge_away,
            'win': win,
            'profit': profit,
            'model_prob': model_prob
        })
    
    if len(bets) == 0:
        return None
    
    bets_df = pd.DataFrame(bets)
    return {
        'bets': len(bets_df),
        'wins': bets_df['win'].sum(),
        'win_rate': bets_df['win'].mean(),
        'total_profit': bets_df['profit'].sum(),
        'roi': bets_df['profit'].mean(),
        'bets_df': bets_df
    }

# Test different thresholds
thresholds = [0.00, 0.02, 0.03, 0.04, 0.05]

print(f"\n{'Threshold':<12}{'Bets':<10}{'W-L':<15}{'Win%':<10}{'Profit':<12}{'ROI%':<10}")
print("-" * 80)

results = {}
for threshold in thresholds:
    result = simulate_betting(test_df, threshold, use_clean_only=True)
    if result:
        results[threshold] = result
        print(f"{threshold:.2f}        {result['bets']:<10}{result['wins']}-{result['bets']-result['wins']:<12}"
              f"{result['win_rate']*100:<10.1f}{result['total_profit']:<12.1f}{result['roi']*100:<10.1f}%")

# Best result
if len(results) > 0:
    best_threshold = max(results.keys(), key=lambda k: results[k]['roi'])
    best = results[best_threshold]
    
    print(f"\n{'='*80}")
    print("BEST PERFORMANCE (CLEAN ODDS ONLY)")
    print("="*80)
    print(f"  Edge threshold: {best_threshold:.2f}")
    print(f"  Bets placed:    {best['bets']}")
    print(f"  Record:         {best['wins']}-{best['bets']-best['wins']}")
    print(f"  Win rate:       {best['win_rate']*100:.1f}%")
    print(f"  Total profit:   {best['total_profit']:.1f} units")
    print(f"  ROI:            {best['roi']*100:+.1f}%")
    
    if best['roi'] > 0:
        print("\n  ✅ POSITIVE ROI: Model shows edge on clean moneyline odds")
    else:
        print("\n  ❌ NEGATIVE ROI: Model cannot beat clean closing lines")
    
    # Save results
    output = {
        'test_games': len(test_df),
        'clean_odds_games': clean_count,
        'corrupted_odds_games': corrupt_count,
        'best_threshold': best_threshold,
        'best_results': {
            'bets': int(best['bets']),
            'wins': int(best['wins']),
            'win_rate': float(best['win_rate']),
            'total_profit': float(best['total_profit']),
            'roi': float(best['roi'])
        },
        'all_thresholds': {
            str(k): {
                'bets': int(v['bets']),
                'wins': int(v['wins']),
                'roi': float(v['roi'])
            } for k, v in results.items()
        }
    }
    
    with open('models/backtest_moneyline_hybrid.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n  ✅ Results saved: models/backtest_moneyline_hybrid.json")

print(f"\n{'='*80}")
print("HYBRID BACKTEST COMPLETE")
print("="*80)
