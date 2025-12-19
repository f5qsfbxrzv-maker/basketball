"""
Split Backtest - Maximize Total Units
Treats favorites and underdogs as separate asset classes
Optimizes for total profit (units) not just ROI percentage
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import log_loss, roc_auc_score

print("="*90)
print("SPLIT BACKTEST: MAXIMIZING TOTAL UNITS (TRIAL #215)")
print("="*90)

# ==============================================================================
# 1. LOAD & PREPARE
# ==============================================================================
print("\n[1/4] Loading data and model...")

# Load Trial #215
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

print(f"  Train samples: {len(train_df):,}")
print(f"  Test samples: {len(test_df):,}")

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
y_test = test_df['target_moneyline_win']

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Generate predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
test_df['model_prob_home'] = y_pred_proba
test_df['model_prob_away'] = 1 - y_pred_proba

print(f"  Test LogLoss: {log_loss(y_test, y_pred_proba):.6f}")
print(f"  Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# ==============================================================================
# 2. MERGE WITH ODDS
# ==============================================================================
print("\n[2/4] Merging with odds data...")

odds_df = pd.read_csv('data/live/closing_odds_2024_25.csv')
odds_df = odds_df.rename(columns={
    'game_date': 'date',
    'home_ml_odds': 'home_ml',
    'away_ml_odds': 'away_ml'
})
odds_df['date'] = pd.to_datetime(odds_df['date'])
odds_df = odds_df.sort_values('snapshot_timestamp').groupby(['date', 'home_team', 'away_team']).last().reset_index()

merged = test_df.merge(odds_df, on=['date', 'home_team', 'away_team'], how='inner')

print(f"  Merged records: {len(merged):,}")

# Convert to decimal odds
def american_to_decimal(odds):
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))

merged['home_decimal'] = merged['home_ml'].apply(american_to_decimal)
merged['away_decimal'] = merged['away_ml'].apply(american_to_decimal)

# Calculate fair probabilities (remove vig)
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

merged['home_market_prob'] = merged['home_ml'].apply(american_to_prob)
merged['away_market_prob'] = merged['away_ml'].apply(american_to_prob)

total_prob = merged['home_market_prob'] + merged['away_market_prob']
merged['home_fair_prob'] = merged['home_market_prob'] / total_prob
merged['away_fair_prob'] = merged['away_market_prob'] / total_prob

# Calculate edges
merged['edge_home'] = merged['model_prob_home'] - merged['home_fair_prob']
merged['edge_away'] = merged['model_prob_away'] - merged['away_fair_prob']

# ==============================================================================
# 3. SPLIT INTO FAVORITES AND UNDERDOGS
# ==============================================================================
print("\n[3/4] Splitting universe into favorites and underdogs...")

# Create bet records (one row per potential bet)
home_bets = merged[['date', 'home_team', 'away_team', 'home_ml', 'home_decimal', 
                     'edge_home', 'target_moneyline_win']].copy()
home_bets.columns = ['date', 'bet_team', 'opponent', 'odds', 'decimal_odds', 'edge', 'won']
home_bets['won'] = home_bets['won'] == 1

away_bets = merged[['date', 'home_team', 'away_team', 'away_ml', 'away_decimal', 
                     'edge_away', 'target_moneyline_win']].copy()
away_bets.columns = ['date', 'opponent', 'bet_team', 'odds', 'decimal_odds', 'edge', 'won']
away_bets['won'] = away_bets['won'] == 0

all_bets = pd.concat([home_bets, away_bets], ignore_index=True)

# Split by odds threshold (2.00 = +100 = pick'em)
favorites = all_bets[all_bets['decimal_odds'] < 2.00].copy()
underdogs = all_bets[all_bets['decimal_odds'] >= 2.00].copy()

print(f"  Total potential bets: {len(all_bets):,}")
print(f"  Favorites pool (odds < 2.00): {len(favorites):,}")
print(f"  Underdogs pool (odds >= 2.00): {len(underdogs):,}")

# ==============================================================================
# 4. OPTIMIZATION FUNCTION
# ==============================================================================
def run_unit_optimizer(data, thresholds, label):
    print(f"\n{'-'*90}")
    print(f"OPTIMIZATION: {label}")
    print(f"{'-'*90}")
    print(f"{'Threshold':<10} | {'Bets':<6} | {'Win%':<7} | {'Units':<11} | {'ROI':<8} | {'Avg Odds':<8}")
    print("-" * 90)
    
    best_units = -1000
    best_thresh = 0
    best_stats = {}
    
    for thresh in thresholds:
        bets = data[data['edge'] >= thresh].copy()
        
        if len(bets) < 10:
            continue
        
        n_bets = len(bets)
        wins = bets['won'].sum()
        win_rate = (wins / n_bets) * 100
        
        # Calculate units (profit = (decimal_odds - 1) for wins, -1 for losses)
        bets['units'] = np.where(bets['won'], bets['decimal_odds'] - 1, -1)
        total_units = bets['units'].sum()
        roi = (total_units / n_bets) * 100
        avg_odds = bets['decimal_odds'].mean()
        
        print(f"{thresh*100:>4.1f}%      | {n_bets:<6} | {win_rate:>6.1f}% | {total_units:>+10.2f}u | {roi:>7.2f}% | {avg_odds:>7.2f}")
        
        # Maximize units with positive ROI requirement
        if total_units > best_units and roi > 0:
            best_units = total_units
            best_thresh = thresh
            best_stats = {
                'roi': roi,
                'bets': n_bets,
                'win_rate': win_rate,
                'wins': wins,
                'avg_odds': avg_odds
            }
    
    return best_thresh, best_units, best_stats

# ==============================================================================
# 5. RUN OPTIMIZATIONS
# ==============================================================================
print("\n[4/4] Running split optimizations...")

# Favorites: Test sensitive thresholds (0.5% to 7%)
fav_thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07]
best_fav_thresh, fav_units, fav_stats = run_unit_optimizer(
    favorites, fav_thresholds, "FAVORITES (Odds < 2.00 / -100)"
)

# Underdogs: Test aggressive thresholds (5% to 20%)
dog_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
best_dog_thresh, dog_units, dog_stats = run_unit_optimizer(
    underdogs, dog_thresholds, "UNDERDOGS (Odds >= 2.00 / +100)"
)

# ==============================================================================
# 6. FINAL PORTFOLIO SUMMARY
# ==============================================================================
print(f"\n{'='*90}")
print("OPTIMAL PORTFOLIO (MAXIMUM UNITS STRATEGY)")
print(f"{'='*90}")

total_bets = fav_stats.get('bets', 0) + dog_stats.get('bets', 0)
total_units = fav_units + dog_units
total_roi = (total_units / total_bets * 100) if total_bets > 0 else 0

print(f"\n1. FAVORITES STRATEGY:")
print(f"   Threshold:    {best_fav_thresh*100:.1f}% Edge")
print(f"   Bets:         {fav_stats.get('bets', 0)}")
print(f"   Win Rate:     {fav_stats.get('win_rate', 0):.1f}% ({fav_stats.get('wins', 0)}/{fav_stats.get('bets', 0)})")
print(f"   Avg Odds:     {fav_stats.get('avg_odds', 0):.2f}")
print(f"   Performance:  {fav_units:+.2f} units  |  {fav_stats.get('roi', 0):.2f}% ROI")

print(f"\n2. UNDERDOGS STRATEGY:")
print(f"   Threshold:    {best_dog_thresh*100:.1f}% Edge")
print(f"   Bets:         {dog_stats.get('bets', 0)}")
print(f"   Win Rate:     {dog_stats.get('win_rate', 0):.1f}% ({dog_stats.get('wins', 0)}/{dog_stats.get('bets', 0)})")
print(f"   Avg Odds:     {dog_stats.get('avg_odds', 0):.2f}")
print(f"   Performance:  {dog_units:+.2f} units  |  {dog_stats.get('roi', 0):.2f}% ROI")

print(f"\n{'-'*90}")
print(f"COMBINED PERFORMANCE:")
print(f"   Total Bets:   {total_bets}")
print(f"   Favorites:    {fav_stats.get('bets', 0)} ({fav_stats.get('bets', 0)/total_bets*100:.1f}%)")
print(f"   Underdogs:    {dog_stats.get('bets', 0)} ({dog_stats.get('bets', 0)/total_bets*100:.1f}%)")
print(f"   Total Units:  {total_units:+.2f}u")
print(f"   Total ROI:    {total_roi:.2f}%")
print(f"{'='*90}")

# Save results
import json
results = {
    'strategy': 'split_maximize_units',
    'favorites': {
        'threshold': best_fav_thresh,
        'bets': fav_stats.get('bets', 0),
        'win_rate': fav_stats.get('win_rate', 0),
        'units': float(fav_units),
        'roi': fav_stats.get('roi', 0)
    },
    'underdogs': {
        'threshold': best_dog_thresh,
        'bets': dog_stats.get('bets', 0),
        'win_rate': dog_stats.get('win_rate', 0),
        'units': float(dog_units),
        'roi': dog_stats.get('roi', 0)
    },
    'combined': {
        'total_bets': total_bets,
        'total_units': float(total_units),
        'total_roi': total_roi
    }
}

with open('models/backtest_trial215_split_optimized.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: models/backtest_trial215_split_optimized.json")
print("="*90)
