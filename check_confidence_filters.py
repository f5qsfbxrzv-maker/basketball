"""
CHECK IMPACT OF MINIMUM CONFIDENCE FILTERS
Test both models with various minimum probability thresholds
"""

import pandas as pd
import numpy as np
import xgboost as xgb

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

def test_with_confidence_filter(test_df, model_prob_col, min_confidence, fav_edge_thresh, dog_edge_thresh):
    """Test performance with a minimum confidence filter"""
    
    # Filter by confidence first
    confident_bets = test_df[
        ((test_df[model_prob_col] >= min_confidence) | 
         ((1 - test_df[model_prob_col]) >= min_confidence))
    ].copy()
    
    if len(confident_bets) == 0:
        return None
    
    # Apply edge thresholds
    fav_mask = (
        ((confident_bets['home_is_fav']) & (confident_bets['home_edge'] > fav_edge_thresh)) |
        ((~confident_bets['home_is_fav']) & (confident_bets['away_edge'] > fav_edge_thresh))
    )
    
    dog_mask = (
        ((~confident_bets['home_is_fav']) & (confident_bets['home_edge'] > dog_edge_thresh)) |
        ((confident_bets['home_is_fav']) & (confident_bets['away_edge'] > dog_edge_thresh))
    )
    
    # Calculate profits
    total_bets = 0
    total_profit = 0
    
    for mask, is_fav in [(fav_mask, True), (dog_mask, False)]:
        bets = confident_bets[mask].copy()
        if len(bets) == 0:
            continue
            
        for _, row in bets.iterrows():
            # Determine which side we're betting
            if row['home_is_fav']:
                # Home is favorite
                if is_fav and row['home_edge'] > fav_edge_thresh:
                    # Bet home (favorite)
                    profit = calculate_profit(True, row['target_moneyline_win']==1, row['home_ml_odds'], row['away_ml_odds'])
                    total_profit += profit
                    total_bets += 1
                elif not is_fav and row['away_edge'] > dog_edge_thresh:
                    # Bet away (underdog)
                    profit = calculate_profit(False, row['target_moneyline_win']==1, row['home_ml_odds'], row['away_ml_odds'])
                    total_profit += profit
                    total_bets += 1
            else:
                # Away is favorite
                if is_fav and row['away_edge'] > fav_edge_thresh:
                    # Bet away (favorite)
                    profit = calculate_profit(False, row['target_moneyline_win']==1, row['home_ml_odds'], row['away_ml_odds'])
                    total_profit += profit
                    total_bets += 1
                elif not is_fav and row['home_edge'] > dog_edge_thresh:
                    # Bet home (underdog)
                    profit = calculate_profit(True, row['target_moneyline_win']==1, row['home_ml_odds'], row['away_ml_odds'])
                    total_profit += profit
                    total_bets += 1
    
    roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    return {
        'bets': total_bets,
        'units': total_profit,
        'roi': roi
    }

print("="*100)
print("🎯 CONFIDENCE FILTER IMPACT ANALYSIS")
print("="*100)

# Load data
df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
df['game_date'] = pd.to_datetime(df['date'])
train_df = df[df['game_date'] < '2023-10-01'].copy()
test_df = df[df['game_date'] >= '2023-10-01'].copy()

# Load odds
odds_2023 = pd.read_csv('data/closing_odds_2023_24_CLEANED.csv')
odds_2024 = pd.read_csv('data/closing_odds_2024_25_CLEANED.csv')
all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
all_odds['game_date'] = pd.to_datetime(all_odds['game_date'])

# ========================================
# TRIAL 1306 (UNCALIBRATED)
# ========================================
print("\n" + "="*100)
print("MODEL 1: TRIAL 1306 (UNCALIBRATED)")
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

X_train = train_df[FEATURES_1306].copy()
y_train = train_df['target_moneyline_win'].copy()
mask = ~(X_train.isna().any(axis=1) | y_train.isna())
X_train, y_train = X_train[mask], y_train[mask]

model_1306 = xgb.XGBClassifier(
    learning_rate=0.0105, max_depth=3, n_estimators=9947,
    random_state=42, objective='binary:logistic'
)
model_1306.fit(X_train, y_train, verbose=False)

X_test = test_df[FEATURES_1306].copy()
mask = ~X_test.isna().any(axis=1)
test_1306 = test_df[mask].copy()
X_test = X_test[mask]
test_1306['model_prob_uncal'] = model_1306.predict_proba(X_test)[:, 1]

test_1306 = test_1306.merge(
    all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    on=['game_date', 'home_team', 'away_team'],
    how='inner'
)

test_1306['home_implied'] = test_1306['home_ml_odds'].apply(american_to_implied_prob)
test_1306['away_implied'] = test_1306['away_ml_odds'].apply(american_to_implied_prob)
test_1306['home_edge'] = test_1306['model_prob_uncal'] - test_1306['home_implied']
test_1306['away_edge'] = (1 - test_1306['model_prob_uncal']) - test_1306['away_implied']
test_1306['home_is_fav'] = test_1306['home_implied'] > test_1306['away_implied']

# Optimal thresholds for Trial 1306
FAV_THRESH_1306 = 0.085
DOG_THRESH_1306 = 0.050

print("\nOptimal Thresholds: Fav=8.5%, Dog=5.0%")
print(f"{'Min Confidence':<20} | {'Bets':<8} | {'Units':<12} | {'ROI':<10} | {'Change':<15}")
print("-"*100)

baseline_1306 = test_with_confidence_filter(test_1306, 'model_prob_uncal', 0.50, FAV_THRESH_1306, DOG_THRESH_1306)

for min_conf in [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65]:
    result = test_with_confidence_filter(test_1306, 'model_prob_uncal', min_conf, FAV_THRESH_1306, DOG_THRESH_1306)
    if result:
        change = f"{result['units'] - baseline_1306['units']:+.2f} units"
        print(f"{min_conf:.0%}                 | {result['bets']:<8} | {result['units']:+11.2f} | {result['roi']:+9.2f}% | {change:<15}")

# ========================================
# VARIANT D (CALIBRATED)
# ========================================
print("\n" + "="*100)
print("MODEL 2: VARIANT D (CALIBRATED)")
print("="*100)

FEATURES_VARIANT_D = [
    'home_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
    'ewma_chaos_home', 'injury_impact_diff', 'injury_shock_diff',
    'star_power_leverage', 'season_progress', 'league_offensive_context',
    'total_foul_environment', 'net_free_throw_advantage',
    'pace_efficiency_interaction', 'offense_vs_defense_matchup'
]

X_train_vd = train_df[FEATURES_VARIANT_D].copy()
y_train_vd = train_df['target_moneyline_win'].copy()
mask = ~(X_train_vd.isna().any(axis=1) | y_train_vd.isna())
X_train_vd, y_train_vd = X_train_vd[mask], y_train_vd[mask]

# Load calibrator
import joblib
calibrator = joblib.load('models/nba_isotonic_calibrator.joblib')

model_vd = xgb.XGBClassifier(
    learning_rate=0.066994, max_depth=2, n_estimators=4529,
    random_state=42, objective='binary:logistic'
)
model_vd.fit(X_train_vd, y_train_vd, verbose=False)

X_test_vd = test_df[FEATURES_VARIANT_D].copy()
mask = ~X_test_vd.isna().any(axis=1)
test_vd = test_df[mask].copy()
X_test_vd = X_test_vd[mask]

raw_probs = model_vd.predict_proba(X_test_vd)[:, 1]
test_vd['model_prob_cal'] = calibrator.predict(raw_probs)

test_vd = test_vd.merge(
    all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    on=['game_date', 'home_team', 'away_team'],
    how='inner'
)

test_vd['home_implied'] = test_vd['home_ml_odds'].apply(american_to_implied_prob)
test_vd['away_implied'] = test_vd['away_ml_odds'].apply(american_to_implied_prob)
test_vd['home_edge'] = test_vd['model_prob_cal'] - test_vd['home_implied']
test_vd['away_edge'] = (1 - test_vd['model_prob_cal']) - test_vd['away_implied']
test_vd['home_is_fav'] = test_vd['home_implied'] > test_vd['away_implied']

# Optimal thresholds for Variant D
FAV_THRESH_VD = 0.080
DOG_THRESH_VD = 0.035

print("\nOptimal Thresholds: Fav=8.0%, Dog=3.5%")
print(f"{'Min Confidence':<20} | {'Bets':<8} | {'Units':<12} | {'ROI':<10} | {'Change':<15}")
print("-"*100)

baseline_vd = test_with_confidence_filter(test_vd, 'model_prob_cal', 0.50, FAV_THRESH_VD, DOG_THRESH_VD)

for min_conf in [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65]:
    result = test_with_confidence_filter(test_vd, 'model_prob_cal', min_conf, FAV_THRESH_VD, DOG_THRESH_VD)
    if result:
        change = f"{result['units'] - baseline_vd['units']:+.2f} units"
        print(f"{min_conf:.0%}                 | {result['bets']:<8} | {result['units']:+11.2f} | {result['roi']:+9.2f}% | {change:<15}")

print("\n" + "="*100)
print("💡 ANALYSIS")
print("="*100)
print("\nCurrently we have NO minimum confidence filter on either model.")
print("Both models bet whenever edge > threshold, regardless of confidence.")
print("\nAdding a confidence filter could:")
print("  ✓ Reduce total bets (only bet high-confidence games)")
print("  ✓ Potentially improve ROI (if low-confidence bets are unprofitable)")
print("  ✗ Reduce total units (fewer betting opportunities)")
print("="*100)
