"""
VERIFY ACTUAL PERFORMANCE - NO FILTERS
Use exact same methodology as check_trial1306_roi.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

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

print("="*100)
print("🔍 VERIFICATION: ACTUAL PERFORMANCE WITH NO FILTERS")
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
# TRIAL 1306
# ========================================
print("\n" + "="*100)
print("TRIAL 1306 (UNCALIBRATED)")
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

model = xgb.XGBClassifier(learning_rate=0.0105, max_depth=3, n_estimators=9947, random_state=42, objective='binary:logistic')
model.fit(X_train, y_train, verbose=False)

X_test = test_df[FEATURES_1306].copy()
mask = ~X_test.isna().any(axis=1)
test_clean = test_df[mask].copy()
X_test = X_test[mask]
test_clean['model_prob'] = model.predict_proba(X_test)[:, 1]

test_clean = test_clean.merge(
    all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    on=['game_date', 'home_team', 'away_team'],
    how='inner'
)

test_clean['home_implied'] = test_clean['home_ml_odds'].apply(american_to_implied_prob)
test_clean['away_implied'] = test_clean['away_ml_odds'].apply(american_to_implied_prob)
test_clean['home_edge'] = test_clean['model_prob'] - test_clean['home_implied']
test_clean['away_edge'] = (1 - test_clean['model_prob']) - test_clean['away_implied']
test_clean['home_is_fav'] = test_clean['home_implied'] > test_clean['away_implied']

# Use optimal thresholds: 8.5% fav, 5.0% dog
FAV_EDGE = 0.085
DOG_EDGE = 0.050

# FAVORITES
home_fav_mask = (test_clean['home_is_fav']) & (test_clean['home_edge'] > FAV_EDGE)
away_fav_mask = (~test_clean['home_is_fav']) & (test_clean['away_edge'] > FAV_EDGE)

home_fav_bets = test_clean[home_fav_mask].copy()
away_fav_bets = test_clean[away_fav_mask].copy()

home_fav_bets['profit'] = home_fav_bets.apply(
    lambda r: calculate_profit(True, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
)
away_fav_bets['profit'] = away_fav_bets.apply(
    lambda r: calculate_profit(False, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
)

fav_bets = pd.concat([home_fav_bets, away_fav_bets])
fav_units = fav_bets['profit'].sum()
fav_count = len(fav_bets)

# UNDERDOGS
home_dog_mask = (~test_clean['home_is_fav']) & (test_clean['home_edge'] > DOG_EDGE)
away_dog_mask = (test_clean['home_is_fav']) & (test_clean['away_edge'] > DOG_EDGE)

home_dog_bets = test_clean[home_dog_mask].copy()
away_dog_bets = test_clean[away_dog_mask].copy()

home_dog_bets['profit'] = home_dog_bets.apply(
    lambda r: calculate_profit(True, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
)
away_dog_bets['profit'] = away_dog_bets.apply(
    lambda r: calculate_profit(False, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
)

dog_bets = pd.concat([home_dog_bets, away_dog_bets])
dog_units = dog_bets['profit'].sum()
dog_count = len(dog_bets)

print(f"\nThresholds: Favorites 8.5%, Underdogs 5.0%")
print(f"  Favorites: {fav_count} bets, {fav_units:+.2f} units, {fav_units/fav_count*100:+.2f}% ROI")
print(f"  Underdogs: {dog_count} bets, {dog_units:+.2f} units, {dog_units/dog_count*100:+.2f}% ROI")
print(f"  TOTAL:     {fav_count+dog_count} bets, {fav_units+dog_units:+.2f} units, {(fav_units+dog_units)/(fav_count+dog_count)*100:+.2f}% ROI")

# ========================================
# VARIANT D CALIBRATED  
# ========================================
print("\n" + "="*100)
print("VARIANT D (CALIBRATED)")
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

calibrator = joblib.load('models/nba_isotonic_calibrator.joblib')

model_vd = xgb.XGBClassifier(learning_rate=0.066994, max_depth=2, n_estimators=4529, random_state=42, objective='binary:logistic')
model_vd.fit(X_train_vd, y_train_vd, verbose=False)

X_test_vd = test_df[FEATURES_VD].copy()
mask = ~X_test_vd.isna().any(axis=1)
test_vd = test_df[mask].copy()
X_test_vd = X_test_vd[mask]

raw_probs = model_vd.predict_proba(X_test_vd)[:, 1]
test_vd['model_prob'] = calibrator.predict(raw_probs)

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

# Use optimal thresholds: 8.0% fav, 3.5% dog
FAV_EDGE_VD = 0.080
DOG_EDGE_VD = 0.035

# FAVORITES
home_fav_mask = (test_vd['home_is_fav']) & (test_vd['home_edge'] > FAV_EDGE_VD)
away_fav_mask = (~test_vd['home_is_fav']) & (test_vd['away_edge'] > FAV_EDGE_VD)

home_fav_bets = test_vd[home_fav_mask].copy()
away_fav_bets = test_vd[away_fav_mask].copy()

home_fav_bets['profit'] = home_fav_bets.apply(
    lambda r: calculate_profit(True, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
)
away_fav_bets['profit'] = away_fav_bets.apply(
    lambda r: calculate_profit(False, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
)

fav_bets_vd = pd.concat([home_fav_bets, away_fav_bets])
fav_units_vd = fav_bets_vd['profit'].sum()
fav_count_vd = len(fav_bets_vd)

# UNDERDOGS
home_dog_mask = (~test_vd['home_is_fav']) & (test_vd['home_edge'] > DOG_EDGE_VD)
away_dog_mask = (test_vd['home_is_fav']) & (test_vd['away_edge'] > DOG_EDGE_VD)

home_dog_bets = test_vd[home_dog_mask].copy()
away_dog_bets = test_vd[away_dog_mask].copy()

home_dog_bets['profit'] = home_dog_bets.apply(
    lambda r: calculate_profit(True, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
)
away_dog_bets['profit'] = away_dog_bets.apply(
    lambda r: calculate_profit(False, r['target_moneyline_win']==1, r['home_ml_odds'], r['away_ml_odds']), axis=1
)

dog_bets_vd = pd.concat([home_dog_bets, away_dog_bets])
dog_units_vd = dog_bets_vd['profit'].sum()
dog_count_vd = len(dog_bets_vd)

print(f"\nThresholds: Favorites 8.0%, Underdogs 3.5%")
print(f"  Favorites: {fav_count_vd} bets, {fav_units_vd:+.2f} units, {fav_units_vd/fav_count_vd*100:+.2f}% ROI")
print(f"  Underdogs: {dog_count_vd} bets, {dog_units_vd:+.2f} units, {dog_units_vd/dog_count_vd*100:+.2f}% ROI")
print(f"  TOTAL:     {fav_count_vd+dog_count_vd} bets, {fav_units_vd+dog_units_vd:+.2f} units, {(fav_units_vd+dog_units_vd)/(fav_count_vd+dog_count_vd)*100:+.2f}% ROI")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"{'Model':<30} | {'Total Bets':<12} | {'Total Units':<15} | {'ROI':<10}")
print("-"*100)
print(f"{'Trial 1306':<30} | {fav_count+dog_count:<12} | {fav_units+dog_units:+14.2f} | {(fav_units+dog_units)/(fav_count+dog_count)*100:+9.2f}%")
print(f"{'Variant D Calibrated':<30} | {fav_count_vd+dog_count_vd:<12} | {fav_units_vd+dog_units_vd:+14.2f} | {(fav_units_vd+dog_units_vd)/(fav_count_vd+dog_count_vd)*100:+9.2f}%")
print("="*100)
