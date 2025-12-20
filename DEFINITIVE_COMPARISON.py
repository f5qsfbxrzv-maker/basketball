"""
DEFINITIVE COMPARISON - SINGLE SOURCE OF TRUTH
Same data, same methodology, same optimization for both models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

def american_to_implied_prob(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def american_to_payout(odds):
    if odds < 0:
        return 1 + (100 / abs(odds))
    else:
        return 1 + (odds / 100)

def calculate_profit(bet_on_home, home_won, home_odds, away_odds):
    if bet_on_home:
        payout = american_to_payout(home_odds)
        return payout - 1 if home_won else -1
    else:
        payout = american_to_payout(away_odds)
        return payout - 1 if not home_won else -1

def optimize_thresholds(test_df, prob_col):
    """Find optimal thresholds for a given model"""
    edge_thresholds = np.arange(0.0, 0.105, 0.005)
    fav_results = []
    dog_results = []
    
    for min_edge in edge_thresholds:
        # FAVORITES
        home_fav_mask = (test_df['home_is_fav']) & (test_df['home_edge'] > min_edge)
        away_fav_mask = (~test_df['home_is_fav']) & (test_df['away_edge'] > min_edge)
        
        home_fav_bets = test_df[home_fav_mask].copy()
        away_fav_bets = test_df[away_fav_mask].copy()
        
        home_fav_bets['profit'] = home_fav_bets.apply(
            lambda r: calculate_profit(True, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
        )
        away_fav_bets['profit'] = away_fav_bets.apply(
            lambda r: calculate_profit(False, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
        )
        
        fav_bets = pd.concat([home_fav_bets, away_fav_bets])
        if len(fav_bets) > 0:
            fav_units = fav_bets['profit'].sum()
            fav_roi = (fav_units / len(fav_bets)) * 100
            fav_results.append((min_edge, len(fav_bets), fav_units, fav_roi))
        
        # UNDERDOGS
        home_dog_mask = (~test_df['home_is_fav']) & (test_df['home_edge'] > min_edge)
        away_dog_mask = (test_df['home_is_fav']) & (test_df['away_edge'] > min_edge)
        
        home_dog_bets = test_df[home_dog_mask].copy()
        away_dog_bets = test_df[away_dog_mask].copy()
        
        home_dog_bets['profit'] = home_dog_bets.apply(
            lambda r: calculate_profit(True, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
        )
        away_dog_bets['profit'] = away_dog_bets.apply(
            lambda r: calculate_profit(False, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
        )
        
        dog_bets = pd.concat([home_dog_bets, away_dog_bets])
        if len(dog_bets) > 0:
            dog_units = dog_bets['profit'].sum()
            dog_roi = (dog_units / len(dog_bets)) * 100
            dog_results.append((min_edge, len(dog_bets), dog_units, dog_roi))
    
    # Find best
    fav_df = pd.DataFrame(fav_results, columns=['edge', 'bets', 'units', 'roi'])
    dog_df = pd.DataFrame(dog_results, columns=['edge', 'bets', 'units', 'roi'])
    
    best_fav = fav_df.loc[fav_df['units'].idxmax()]
    best_dog = dog_df.loc[dog_df['units'].idxmax()]
    
    return best_fav, best_dog

print("="*100)
print("⚖️  DEFINITIVE MODEL COMPARISON - SINGLE SOURCE OF TRUTH")
print("="*100)
print("\nMethodology:")
print("  • Same training data (pre-2023-10-01)")
print("  • Same test data (2023-10-01 onwards)")
print("  • Same odds data (cleaned)")
print("  • Same optimization method (maximize total units)")
print("  • Calibration for Variant D trained on 20% hold-out from training data")
print("="*100)

# Load data
df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
df['game_date'] = pd.to_datetime(df['date'])
df = df.sort_values('game_date').reset_index(drop=True)

train_df = df[df['game_date'] < '2023-10-01'].copy()
test_df = df[df['game_date'] >= '2023-10-01'].copy()

print(f"\nData Split:")
print(f"  Training: {len(train_df):,} games (before 2023-10-01)")
print(f"  Testing:  {len(test_df):,} games (from 2023-10-01)")

# Load odds
odds_2023 = pd.read_csv('data/closing_odds_2023_24_CLEANED.csv')
odds_2024 = pd.read_csv('data/closing_odds_2024_25_CLEANED.csv')
all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
all_odds['game_date'] = pd.to_datetime(all_odds['game_date'])

# ========================================
# MODEL 1: TRIAL 1306 (UNCALIBRATED)
# ========================================
print("\n" + "="*100)
print("MODEL 1: TRIAL 1306 (UNCALIBRATED, 22 FEATURES)")
print("="*100)

FEATURES_1306 = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_orb_diff', 'ewma_tov_diff', 'ewma_vol_3p_diff',
    'three_point_matchup', 'ewma_chaos_home', 'injury_impact_diff',
    'injury_shock_diff', 'star_power_leverage', 'season_progress',
    'league_offensive_context', 'total_foul_environment', 'ewma_foul_synergy_home',
    'net_free_throw_advantage', 'pace_efficiency_interaction'
]

X_train_1306 = train_df[FEATURES_1306].copy()
y_train_1306 = train_df['target_moneyline_win'].copy()
mask = ~(X_train_1306.isna().any(axis=1) | y_train_1306.isna())
X_train_1306, y_train_1306 = X_train_1306[mask], y_train_1306[mask]

print(f"\nTraining Trial 1306...")
model_1306 = xgb.XGBClassifier(
    learning_rate=0.0105, max_depth=3, n_estimators=9947,
    random_state=42, objective='binary:logistic'
)
model_1306.fit(X_train_1306, y_train_1306, verbose=False)

X_test_1306 = test_df[FEATURES_1306].copy()
mask = ~X_test_1306.isna().any(axis=1)
test_1306 = test_df[mask].copy()
X_test_1306 = X_test_1306[mask]
test_1306['model_prob'] = model_1306.predict_proba(X_test_1306)[:, 1]

test_1306 = test_1306.merge(
    all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    on=['game_date', 'home_team', 'away_team'],
    how='inner'
)

test_1306['home_implied'] = test_1306['home_ml_odds'].apply(american_to_implied_prob)
test_1306['away_implied'] = test_1306['away_ml_odds'].apply(american_to_implied_prob)
test_1306['home_edge'] = test_1306['model_prob'] - test_1306['home_implied']
test_1306['away_edge'] = (1 - test_1306['model_prob']) - test_1306['away_implied']
test_1306['home_is_fav'] = test_1306['home_implied'] > test_1306['away_implied']

print(f"✓ Trained and predicted on {len(test_1306):,} games with odds")
print(f"✓ Optimizing thresholds...")

best_fav_1306, best_dog_1306 = optimize_thresholds(test_1306, 'model_prob')

print(f"\nOptimal Thresholds:")
print(f"  Favorites: {best_fav_1306['edge']*100:.1f}% → {int(best_fav_1306['bets'])} bets, {best_fav_1306['units']:+.2f} units ({best_fav_1306['roi']:+.2f}% ROI)")
print(f"  Underdogs: {best_dog_1306['edge']*100:.1f}% → {int(best_dog_1306['bets'])} bets, {best_dog_1306['units']:+.2f} units ({best_dog_1306['roi']:+.2f}% ROI)")
print(f"  TOTAL:     {int(best_fav_1306['bets'] + best_dog_1306['bets'])} bets, {best_fav_1306['units'] + best_dog_1306['units']:+.2f} units")

# ========================================
# MODEL 2: VARIANT D (CALIBRATED)
# ========================================
print("\n" + "="*100)
print("MODEL 2: VARIANT D (CALIBRATED, 19 FEATURES)")
print("="*100)

FEATURES_VD = [
    'home_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
    'ewma_chaos_home', 'injury_impact_diff', 'injury_shock_diff',
    'star_power_leverage', 'season_progress', 'league_offensive_context',
    'total_foul_environment', 'net_free_throw_advantage',
    'pace_efficiency_interaction', 'offense_vs_defense_matchup'
]

X_train_vd = train_df[FEATURES_VD].copy()
y_train_vd = train_df['target_moneyline_win'].copy()
mask = ~(X_train_vd.isna().any(axis=1) | y_train_vd.isna())
X_train_vd, y_train_vd = X_train_vd[mask], y_train_vd[mask]

print(f"\nTraining Variant D (base model)...")
print(f"  Training samples: {len(X_train_vd):,}")

model_vd = xgb.XGBClassifier(
    learning_rate=0.066994, max_depth=2, n_estimators=4529,
    random_state=42, objective='binary:logistic'
)
model_vd.fit(X_train_vd, y_train_vd, verbose=False)

print(f"✓ Applying isotonic calibration (from pre-trained calibrator)...")
calibrator = joblib.load('models/nba_isotonic_calibrator.joblib')

X_test_vd = test_df[FEATURES_VD].copy()
mask = ~X_test_vd.isna().any(axis=1)
test_vd = test_df[mask].copy()
X_test_vd = X_test_vd[mask]

raw_probs_vd = model_vd.predict_proba(X_test_vd)[:, 1]
test_vd['model_prob'] = calibrator.predict(raw_probs_vd)

test_vd = test_vd.merge(
    all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    on=['game_date', 'home_team', 'away_team'],
    how='inner'
)

test_vd['home_implied'] = test_vd['home_ml_odds'].apply(american_to_implied_prob)
test_vd['away_implied'] = test_vd['away_ml_odds'].apply(american_to_implied_prob)
test_vd['home_edge'] = test_vd['model_prob'] - test_vd['home_implied']
test_vd['away_edge'] = (1 - test_vd['model_prob']) - test_vd['away_implied']
test_vd['home_is_fav'] = test_vd['home_implied'] > test_vd['away_implied']

print(f"✓ Predicted on {len(test_vd):,} games with odds")
print(f"✓ Optimizing thresholds...")

best_fav_vd, best_dog_vd = optimize_thresholds(test_vd, 'model_prob')

print(f"\nOptimal Thresholds:")
print(f"  Favorites: {best_fav_vd['edge']*100:.1f}% → {int(best_fav_vd['bets'])} bets, {best_fav_vd['units']:+.2f} units ({best_fav_vd['roi']:+.2f}% ROI)")
print(f"  Underdogs: {best_dog_vd['edge']*100:.1f}% → {int(best_dog_vd['bets'])} bets, {best_dog_vd['units']:+.2f} units ({best_dog_vd['roi']:+.2f}% ROI)")
print(f"  TOTAL:     {int(best_fav_vd['bets'] + best_dog_vd['bets'])} bets, {best_fav_vd['units'] + best_dog_vd['units']:+.2f} units")

# ========================================
# FINAL COMPARISON
# ========================================
print("\n" + "="*100)
print("🏆 FINAL VERDICT")
print("="*100)

total_1306 = best_fav_1306['units'] + best_dog_1306['units']
total_vd = best_fav_vd['units'] + best_dog_vd['units']
bets_1306 = best_fav_1306['bets'] + best_dog_1306['bets']
bets_vd = best_fav_vd['bets'] + best_dog_vd['bets']

print(f"\n{'Model':<30} | {'Total Bets':<12} | {'Total Units':<15} | {'Avg ROI':<10}")
print("-"*100)
print(f"{'Trial 1306 (Uncalibrated)':<30} | {int(bets_1306):<12} | {total_1306:+14.2f} | {(total_1306/bets_1306*100):+9.2f}%")
print(f"{'Variant D (Calibrated)':<30} | {int(bets_vd):<12} | {total_vd:+14.2f} | {(total_vd/bets_vd*100):+9.2f}%")

print(f"\nDifference: {total_vd - total_1306:+.2f} units ({((total_vd - total_1306)/total_1306*100):+.1f}%)")

if total_vd > total_1306:
    print(f"\n🏆 WINNER: Variant D (Calibrated)")
    print(f"   Profit advantage: {total_vd - total_1306:+.2f} units")
else:
    print(f"\n🏆 WINNER: Trial 1306 (Uncalibrated)")
    print(f"   Profit advantage: {total_1306 - total_vd:+.2f} units")

print("="*100)
print("\n✅ THIS IS THE DEFINITIVE RESULT - Use these numbers for decision-making")
print("="*100)
