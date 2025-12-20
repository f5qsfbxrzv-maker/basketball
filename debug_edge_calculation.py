"""
DEBUG: Verify edge calculations are correct
"""

import pandas as pd
import numpy as np
import xgboost as xgb

# Load data
df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
df['game_date'] = pd.to_datetime(df['date'])
df = df.sort_values('game_date').reset_index(drop=True)

# Load odds
odds_2023 = pd.read_csv('data/closing_odds_2023_24_CLEANED.csv')
odds_2024 = pd.read_csv('data/closing_odds_2024_25_CLEANED.csv')
all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
all_odds['game_date'] = pd.to_datetime(all_odds['game_date'])

# Split data
train_df = df[df['game_date'] < '2023-10-01'].copy()
test_df = df[df['game_date'] >= '2023-10-01'].copy()

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

# Train model
X_train = train_df[FEATURES].copy()
y_train = train_df['target_moneyline_win'].copy()
mask = ~(X_train.isna().any(axis=1) | y_train.isna())
X_train, y_train = X_train[mask], y_train[mask]

model = xgb.XGBClassifier(
    learning_rate=0.066994, max_depth=2, n_estimators=4529,
    random_state=42, objective='binary:logistic'
)
model.fit(X_train, y_train, verbose=False)

# Predict on test
X_test = test_df[FEATURES].copy()
mask = ~X_test.isna().any(axis=1)
test_clean = test_df[mask].copy()
X_test = X_test[mask]

test_clean['model_prob'] = model.predict_proba(X_test)[:, 1]

# Merge odds
test_clean = test_clean.merge(
    all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
    on=['game_date', 'home_team', 'away_team'],
    how='inner'
)

# Calculate implied probabilities
def american_to_implied(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

test_clean['home_implied'] = test_clean['home_ml_odds'].apply(american_to_implied)
test_clean['away_implied'] = test_clean['away_ml_odds'].apply(american_to_implied)

# Calculate edges
test_clean['home_edge'] = test_clean['model_prob'] - test_clean['home_implied']
test_clean['away_edge'] = (1 - test_clean['model_prob']) - test_clean['away_implied']

# Check for vig issues
test_clean['total_implied'] = test_clean['home_implied'] + test_clean['away_implied']
test_clean['vig_pct'] = (test_clean['total_implied'] - 1) * 100

print("="*80)
print("🔍 EDGE CALCULATION DEBUG")
print("="*80)

print(f"\nTotal games with odds: {len(test_clean):,}")
print(f"\nVig Statistics:")
print(f"  Average vig: {test_clean['vig_pct'].mean():.2f}%")
print(f"  Min vig: {test_clean['vig_pct'].min():.2f}%")
print(f"  Max vig: {test_clean['vig_pct'].max():.2f}%")

print("\n" + "="*80)
print("SAMPLE GAMES: First 10 with bets")
print("="*80)

# Find games where we'd bet (edge > 0.5%)
test_clean['max_edge'] = test_clean[['home_edge', 'away_edge']].max(axis=1)
test_clean['bet_side'] = test_clean[['home_edge', 'away_edge']].idxmax(axis=1)
test_clean['bet_side'] = test_clean['bet_side'].map({'home_edge': 'HOME', 'away_edge': 'AWAY'})

samples = test_clean[test_clean['max_edge'] > 0.005].head(10)

for idx, row in samples.iterrows():
    print(f"\n{row['game_date'].date()} | {row['away_team']} @ {row['home_team']}")
    print(f"  Model prob (home win): {row['model_prob']:.3f}")
    print(f"  Home odds: {row['home_ml_odds']:+.0f} → Implied: {row['home_implied']:.3f} → Edge: {row['home_edge']:+.3f} ({row['home_edge']*100:+.1f}%)")
    print(f"  Away odds: {row['away_ml_odds']:+.0f} → Implied: {row['away_implied']:.3f} → Edge: {row['away_edge']:+.3f} ({row['away_edge']*100:+.1f}%)")
    print(f"  Vig: {row['vig_pct']:.2f}%")
    print(f"  → BET: {row['bet_side']} (edge = {row['max_edge']*100:.1f}%)")
    print(f"  Result: {'HOME' if row['target_moneyline_win'] == 1 else 'AWAY'} won")

print("\n" + "="*80)
print("EDGE DISTRIBUTION")
print("="*80)

print("\nHome Edge Distribution:")
print(test_clean['home_edge'].describe())

print("\nAway Edge Distribution:")
print(test_clean['away_edge'].describe())

print("\n" + "="*80)
print("BETS BY EDGE THRESHOLD")
print("="*80)

for threshold in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10]:
    home_bets = (test_clean['home_edge'] > threshold).sum()
    away_bets = (test_clean['away_edge'] > threshold).sum()
    total_bets = home_bets + away_bets
    print(f"  {threshold*100:4.1f}% edge: {total_bets:4d} bets (Home: {home_bets:3d}, Away: {away_bets:3d})")

print("\n" + "="*80)
print("CHECKING IF MODEL IS CALIBRATED")
print("="*80)

# Bin by model probability and check accuracy
test_clean['prob_bin'] = pd.cut(test_clean['model_prob'], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
calibration = test_clean.groupby('prob_bin').agg({
    'model_prob': 'mean',
    'target_moneyline_win': 'mean',
    'game_date': 'count'
})
calibration.columns = ['Avg_Model_Prob', 'Actual_Win_Rate', 'Count']
print(calibration)

print("\n💡 If Avg_Model_Prob ≈ Actual_Win_Rate, model is calibrated")
print("💡 If model is well-calibrated, we shouldn't need huge edge buffers")
