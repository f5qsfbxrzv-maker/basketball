"""
CALIBRATED EDGE OPTIMIZATION
Apply isotonic calibration before finding optimal thresholds
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Features
FEATURES = [
    'home_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
    'ewma_chaos_home', 'injury_impact_diff', 'injury_shock_diff',
    'star_power_leverage', 'season_progress', 'league_offensive_context',
    'total_foul_environment', 'net_free_throw_advantage',
    'pace_efficiency_interaction', 'offense_vs_defense_matchup'
]

# Trial #245 params
OPTIMIZED_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'learning_rate': 0.066994,
    'max_depth': 2,
    'n_estimators': 4529,
    'min_child_weight': 12,
    'gamma': 2.025432,
    'subsample': 0.630135,
    'colsample_bytree': 0.903401,
    'colsample_bylevel': 0.959686,
    'reg_alpha': 1.081072,
    'reg_lambda': 5.821363,
}

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

print("="*80)
print("🎯 CALIBRATED EDGE OPTIMIZATION")
print("="*80)

# Load data
df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
df['game_date'] = pd.to_datetime(df['date'])
df = df.sort_values('game_date').reset_index(drop=True)

# Split by date (pre-2023 for training, 2023+ for testing)
train_df = df[df['game_date'] < '2023-10-01'].copy()
test_df = df[df['game_date'] >= '2023-10-01'].copy()

print(f"\n📚 Training: {len(train_df):,} games (before 2023-10-01)")
print(f"🔮 Testing:  {len(test_df):,} games (from 2023-10-01 onward)")

# Prepare training data with CALIBRATION SET
X_train_full = train_df[FEATURES].copy()
y_train_full = train_df['target_moneyline_win'].copy()
mask = ~(X_train_full.isna().any(axis=1) | y_train_full.isna())
X_train_full, y_train_full = X_train_full[mask], y_train_full[mask]

# Split training into model training (80%) and calibration (20%)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"\n⚙️  Model training: {len(X_train):,} games")
print(f"📐 Calibration set: {len(X_cal):,} games")

# Train base XGBoost model
print("\n🏋️ Training base model...")
base_model = xgb.XGBClassifier(**OPTIMIZED_PARAMS)
base_model.fit(X_train, y_train, verbose=False)
print("✓ Base model trained")

# Apply isotonic calibration
print("\n🔧 Applying isotonic calibration...")
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='isotonic', 
    cv='prefit'
)
calibrated_model.fit(X_cal, y_cal)
print("✓ Calibration complete")

# Test calibration quality
print("\n📊 Checking calibration quality...")
X_test = test_df[FEATURES].copy()
y_test = test_df['target_moneyline_win'].copy()
mask = ~(X_test.isna().any(axis=1) | y_test.isna())
test_clean = test_df[mask].copy()
X_test = X_test[mask]
y_test = y_test[mask]

# Get both uncalibrated and calibrated predictions
test_clean['model_prob_uncal'] = base_model.predict_proba(X_test)[:, 1]
test_clean['model_prob_cal'] = calibrated_model.predict_proba(X_test)[:, 1]

# Check calibration by bin
for name, col in [('Uncalibrated', 'model_prob_uncal'), ('Calibrated', 'model_prob_cal')]:
    test_clean['prob_bin'] = pd.cut(test_clean[col], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    cal_check = test_clean.groupby('prob_bin', observed=False).agg({
        col: 'mean',
        'target_moneyline_win': 'mean',
        'game_date': 'count'
    })
    cal_check.columns = ['Avg_Model_Prob', 'Actual_Win_Rate', 'Count']
    print(f"\n{name}:")
    print(cal_check)

# Load odds
print("\n📥 Loading odds...")
odds_2023 = pd.read_csv('data/closing_odds_2023_24_CLEANED.csv')
odds_2024 = pd.read_csv('data/closing_odds_2024_25_CLEANED.csv')
all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
all_odds['game_date'] = pd.to_datetime(all_odds['game_date'])

# Merge with test data
test_clean = test_clean.merge(
    all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    on=['game_date', 'home_team', 'away_team'],
    how='inner'
)

print(f"✓ {len(test_clean):,} games matched with odds")

# Calculate implied probabilities and edges (CALIBRATED)
test_clean['home_implied'] = test_clean['home_ml_odds'].apply(american_to_implied_prob)
test_clean['away_implied'] = test_clean['away_ml_odds'].apply(american_to_implied_prob)
test_clean['home_edge'] = test_clean['model_prob_cal'] - test_clean['home_implied']
test_clean['away_edge'] = (1 - test_clean['model_prob_cal']) - test_clean['away_implied']
test_clean['home_is_fav'] = test_clean['home_implied'] > test_clean['away_implied']

# Test edge thresholds
print("\n" + "="*80)
print("🔍 TESTING EDGE THRESHOLDS (CALIBRATED MODEL)")
print("="*80)

edge_thresholds = np.arange(0.0, 0.105, 0.005)
fav_results = []
dog_results = []

for min_edge in edge_thresholds:
    # FAVORITES
    home_fav_mask = (test_clean['home_is_fav']) & (test_clean['home_edge'] > min_edge)
    away_fav_mask = (~test_clean['home_is_fav']) & (test_clean['away_edge'] > min_edge)
    
    home_fav_bets = test_clean[home_fav_mask].copy()
    away_fav_bets = test_clean[away_fav_mask].copy()
    
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
    home_dog_mask = (~test_clean['home_is_fav']) & (test_clean['home_edge'] > min_edge)
    away_dog_mask = (test_clean['home_is_fav']) & (test_clean['away_edge'] > min_edge)
    
    home_dog_bets = test_clean[home_dog_mask].copy()
    away_dog_bets = test_clean[away_dog_mask].copy()
    
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

# Display results
print("\n" + "="*80)
print("🦁 FAVORITES: OPTIMAL EDGE THRESHOLDS")
print("="*80)
print("MIN EDGE     | BETS     | TOTAL UNITS     | ROI        | UNITS/BET")
print("-"*80)

fav_df = pd.DataFrame(fav_results, columns=['edge', 'bets', 'units', 'roi'])
fav_df = fav_df.sort_values('units', ascending=False).head(10)
for _, row in fav_df.iterrows():
    print(f"+ {row['edge']*100:5.1f}%      | {int(row['bets']):7d} | {row['units']:+14.2f} | {row['roi']:+9.2f}% | {row['units']/row['bets']:+7.3f}")

print("\n" + "="*80)
print("🐶 UNDERDOGS: OPTIMAL EDGE THRESHOLDS")
print("="*80)
print("MIN EDGE     | BETS     | TOTAL UNITS     | ROI        | UNITS/BET")
print("-"*80)

dog_df = pd.DataFrame(dog_results, columns=['edge', 'bets', 'units', 'roi'])
dog_df = dog_df.sort_values('units', ascending=False).head(10)
for _, row in dog_df.iterrows():
    print(f"+ {row['edge']*100:5.1f}%      | {int(row['bets']):7d} | {row['units']:+14.2f} | {row['roi']:+9.2f}% | {row['units']/row['bets']:+7.3f}")

# Best thresholds
best_fav = fav_df.iloc[0]
best_dog = dog_df.iloc[0]

print("\n" + "="*80)
print("🎯 RECOMMENDED THRESHOLDS (CALIBRATED)")
print("="*80)
print(f"Favorites: Bet when edge > +{best_fav['edge']*100:.1f}%")
print(f"Underdogs: Bet when edge > +{best_dog['edge']*100:.1f}%")
print(f"\nExpected Performance:")
print(f"  Favorites: {best_fav['units']:+.2f} units on {best_fav['bets']:.0f} bets ({best_fav['roi']:+.2f}% ROI)")
print(f"  Underdogs: {best_dog['units']:+.2f} units on {best_dog['bets']:.0f} bets ({best_dog['roi']:+.2f}% ROI)")
print(f"  COMBINED:  {best_fav['units']+best_dog['units']:+.2f} units on {best_fav['bets']+best_dog['bets']:.0f} bets")
