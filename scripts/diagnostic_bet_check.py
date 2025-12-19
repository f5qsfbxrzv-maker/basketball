"""
Diagnostic - Check Actual Bets from Backtest
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna

print("="*90)
print("BACKTEST BET DIAGNOSTIC")
print("="*90)

# Load Trial #215 parameters
study_name = 'nba_matchup_optimized_2000trials'
db_path = f'models/{study_name}.db'
storage = f'sqlite:///{db_path}'
study = optuna.load_study(study_name=study_name, storage=storage)
trial_215 = study.trials[215]

# Load data
df = pd.read_csv('data/training_data_matchup_optimized.csv')
df['date'] = pd.to_datetime(df['date'])

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Train/test split
train_df = df[df['date'] < '2024-10-01'].copy()
test_df = df[df['date'] >= '2024-10-01'].copy()

# Train model
params = {
    **trial_215.params,
    'n_estimators': 1000,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'random_state': 42,
    'early_stopping_rounds': 50
}

X_train = train_df[feature_cols]
y_train = train_df['target_moneyline_win']
X_test = test_df[feature_cols]

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, test_df['target_moneyline_win'])], verbose=False)

# Generate predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
test_df['model_prob_home'] = y_pred_proba
test_df['model_prob_away'] = 1 - y_pred_proba

# Load odds
odds_df = pd.read_csv('data/live/closing_odds_2024_25.csv')
odds_df = odds_df.rename(columns={
    'game_date': 'date',
    'home_ml_odds': 'home_ml',
    'away_ml_odds': 'away_ml'
})
odds_df['date'] = pd.to_datetime(odds_df['date'])
odds_df = odds_df.sort_values('snapshot_timestamp').groupby(['date', 'home_team', 'away_team']).last().reset_index()

# Merge
merged = test_df.merge(odds_df, on=['date', 'home_team', 'away_team'], how='inner')

# Calculate fair probabilities
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

merged['home_market_prob'] = merged['home_ml'].apply(american_to_prob)
merged['away_market_prob'] = merged['away_ml'].apply(american_to_prob)

# Remove vig
total_prob = merged['home_market_prob'] + merged['away_market_prob']
merged['home_fair_prob'] = merged['home_market_prob'] / total_prob
merged['away_fair_prob'] = merged['away_market_prob'] / total_prob

# Calculate edge
merged['edge_home'] = merged['model_prob_home'] - merged['home_fair_prob']
merged['edge_away'] = merged['model_prob_away'] - merged['away_fair_prob']

# Find bets at 15% threshold
threshold = 0.15

print(f"\nThreshold: {threshold:.1%}")
print("="*90)

# Home bets
home_bets = merged[merged['edge_home'] > threshold].copy()
home_bets['bet_team'] = 'home'
home_bets['bet_odds'] = home_bets['home_ml']
home_bets['edge'] = home_bets['edge_home']
home_bets['won'] = home_bets['target_moneyline_win'] == 1

# Away bets  
away_bets = merged[merged['edge_away'] > threshold].copy()
away_bets['bet_team'] = 'away'
away_bets['bet_odds'] = away_bets['away_ml']
away_bets['edge'] = away_bets['edge_away']
away_bets['won'] = away_bets['target_moneyline_win'] == 0

# Combine
all_bets = pd.concat([home_bets, away_bets])
all_bets['is_favorite'] = all_bets['bet_odds'] < 0

print(f"\nTotal bets: {len(all_bets)}")
print(f"Home bets: {len(home_bets)}")
print(f"Away bets: {len(away_bets)}")

print(f"\nBy classification:")
print(f"Favorites (odds < 0): {(all_bets['bet_odds'] < 0).sum()}")
print(f"Underdogs (odds > 0): {(all_bets['bet_odds'] > 0).sum()}")

print(f"\n" + "="*90)
print("SAMPLE FAVORITE BETS")
print("="*90)
fav_bets = all_bets[all_bets['bet_odds'] < 0].sort_values('date').head(10)
print(fav_bets[['date', 'home_team', 'away_team', 'bet_team', 'bet_odds', 'edge', 'won']])

print(f"\n" + "="*90)
print("SAMPLE UNDERDOG BETS")
print("="*90)
dog_bets = all_bets[all_bets['bet_odds'] > 0].sort_values('date').head(10)
print(dog_bets[['date', 'home_team', 'away_team', 'bet_team', 'bet_odds', 'edge', 'won']])

print(f"\n" + "="*90)
print("EDGE DISTRIBUTION BY BET TYPE")
print("="*90)
print(f"\nFavorites:")
print(f"  Count: {(all_bets['bet_odds'] < 0).sum()}")
print(f"  Avg edge: {all_bets[all_bets['bet_odds'] < 0]['edge'].mean()*100:.2f}%")
print(f"  Edge range: {all_bets[all_bets['bet_odds'] < 0]['edge'].min()*100:.2f}% to {all_bets[all_bets['bet_odds'] < 0]['edge'].max()*100:.2f}%")
print(f"  Win rate: {all_bets[all_bets['bet_odds'] < 0]['won'].mean()*100:.1f}%")

print(f"\nUnderdogs:")
print(f"  Count: {(all_bets['bet_odds'] > 0).sum()}")
print(f"  Avg edge: {all_bets[all_bets['bet_odds'] > 0]['edge'].mean()*100:.2f}%")
print(f"  Edge range: {all_bets[all_bets['bet_odds'] > 0]['edge'].min()*100:.2f}% to {all_bets[all_bets['bet_odds'] > 0]['edge'].max()*100:.2f}%")
print(f"  Win rate: {all_bets[all_bets['bet_odds'] > 0]['won'].mean()*100:.1f}%")
