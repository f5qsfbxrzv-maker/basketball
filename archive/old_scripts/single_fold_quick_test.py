import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season', 'target_spread', 'target_spread_cover', 'target_moneyline_win', 'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']
features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df['target_spread_cover']

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    pass

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

print('SINGLE FOLD TEST (Fold 5 - Most Recent)')
print(f'Train: {len(X_train)} games, Val: {len(X_val)} games')

params = {'max_depth': 6, 'learning_rate': 0.02, 'n_estimators': 1500, 'colsample_bytree': 0.8, 'subsample': 0.8, 'min_child_weight': 10, 'gamma': 1.0, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'random_state': 42, 'verbosity': 0}

print('\nUNCONSTRAINED:')
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
preds = model.predict_proba(X_val)[:, 1]
auc_unc = roc_auc_score(y_val, preds)
print(f'  AUC: {auc_unc:.5f}')

constraint_map = {'off_elo_diff': 1, 'def_elo_diff': 1, 'home_composite_elo': 1, 'ewma_efg_diff': 1, 'home_ewma_3p_pct': 1, 'away_back_to_back': -1, 'home_back_to_back': -1, 'away_3in4': -1, 'rest_advantage': 1, 'injury_shock_diff': 1, 'injury_shock_home': -1, 'injury_impact_abs': -1, 'injury_impact_diff': 1, 'away_star_missing': 1, 'home_star_missing': -1, 'star_mismatch': 1, 'ewma_tov_diff': -1, 'ewma_orb_diff': 1, 'home_orb': 1, 'home_drb': 1, 'home_ewma_fta_rate': 1, 'ewma_foul_synergy_home': 1, 'fatigue_mismatch': 1}
constraints = tuple(constraint_map.get(col, 0) for col in features)
params['monotone_constraints'] = constraints

print('\nCONSTRAINED (21 constraints):')
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
preds = model.predict_proba(X_val)[:, 1]
auc_con = roc_auc_score(y_val, preds)
print(f'  AUC: {auc_con:.5f}')

print(f'\nDifference: {(auc_con - auc_unc)*100:+.2f}%')
print(f'Decision: {"Use CONSTRAINED" if auc_con >= auc_unc - 0.01 else "Use UNCONSTRAINED"}')
