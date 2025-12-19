"""
Production Betting Strategy - Trial #215 Split-Optimized
Uses locked-in thresholds: 1.0% favorites, 15.0% underdogs
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from datetime import datetime
import json
import sys
sys.path.append('.')
from betting_strategy_config import (
    FAVORITE_EDGE_THRESHOLD,
    UNDERDOG_EDGE_THRESHOLD,
    ODDS_SPLIT_THRESHOLD,
    MODEL_TRIAL,
    MODEL_STUDY
)

print("="*90)
print("PRODUCTION BETTING STRATEGY - SPLIT-OPTIMIZED")
print("="*90)
print(f"\nStrategy Version: 1.0")
print(f"Model: Trial #{MODEL_TRIAL}")
print(f"Locked Thresholds:")
print(f"  Favorites (< {ODDS_SPLIT_THRESHOLD:.2f}): {FAVORITE_EDGE_THRESHOLD*100:.1f}% edge")
print(f"  Underdogs (>= {ODDS_SPLIT_THRESHOLD:.2f}): {UNDERDOG_EDGE_THRESHOLD*100:.1f}% edge")

# ==============================================================================
# 1. LOAD MODEL
# ==============================================================================
print(f"\n[1/3] Loading model...")

study = optuna.load_study(
    study_name=MODEL_STUDY,
    storage=f'sqlite:///models/{MODEL_STUDY}.db'
)
trial = study.trials[MODEL_TRIAL]

df = pd.read_csv('data/training_data_matchup_optimized.csv')
df['date'] = pd.to_datetime(df['date'])

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

train_df = df[df['date'] < '2024-10-01'].copy()

params = {
    **trial.params,
    'n_estimators': 1000,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'random_state': 42
}

X_train = train_df[feature_cols]
y_train = train_df['target_moneyline_win']

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, verbose=False)

print(f"  Model trained on {len(train_df):,} games")

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def american_to_decimal(odds):
    """Convert American odds to decimal"""
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))

def american_to_prob(odds):
    """Convert American odds to probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def remove_vig(home_prob, away_prob):
    """Remove vig to get fair probabilities"""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total

def calculate_edge(model_prob, fair_prob):
    """Calculate betting edge"""
    return model_prob - fair_prob

def classify_bet(decimal_odds, edge):
    """Determine if bet qualifies under split thresholds"""
    if decimal_odds < ODDS_SPLIT_THRESHOLD:
        # Favorite
        return 'favorite' if edge >= FAVORITE_EDGE_THRESHOLD else None
    else:
        # Underdog
        return 'underdog' if edge >= UNDERDOG_EDGE_THRESHOLD else None

# ==============================================================================
# 3. GENERATE BETS FUNCTION
# ==============================================================================

def generate_bets(game_date, home_team, away_team, home_ml, away_ml):
    """
    Generate bet recommendations for a single game
    
    Parameters:
    -----------
    game_date : str or datetime
        Game date
    home_team : str
        Home team abbreviation
    away_team : str
        Away team abbreviation
    home_ml : int
        Home moneyline (American odds)
    away_ml : int
        Away moneyline (American odds)
    
    Returns:
    --------
    list of dict : Bet recommendations (empty if no qualifying bets)
    """
    
    # Find game in dataset
    game_df = df[(df['date'] == pd.to_datetime(game_date)) & 
                 (df['home_team'] == home_team) & 
                 (df['away_team'] == away_team)]
    
    if len(game_df) == 0:
        return []
    
    # Get features
    game_features = game_df[feature_cols].iloc[0]
    
    # Generate predictions
    model_prob_home = model.predict_proba([game_features])[0][1]
    model_prob_away = 1 - model_prob_home
    
    # Calculate fair probabilities
    home_market_prob = american_to_prob(home_ml)
    away_market_prob = american_to_prob(away_ml)
    home_fair_prob, away_fair_prob = remove_vig(home_market_prob, away_market_prob)
    
    # Calculate edges
    edge_home = calculate_edge(model_prob_home, home_fair_prob)
    edge_away = calculate_edge(model_prob_away, away_fair_prob)
    
    # Convert to decimal odds
    home_decimal = american_to_decimal(home_ml)
    away_decimal = american_to_decimal(away_ml)
    
    # Check for qualifying bets
    bets = []
    
    # Home bet
    home_class = classify_bet(home_decimal, edge_home)
    if home_class:
        bets.append({
            'game_date': str(game_date),
            'bet_team': home_team,
            'opponent': away_team,
            'location': 'home',
            'bet_type': home_class,
            'odds_american': home_ml,
            'odds_decimal': home_decimal,
            'model_prob': model_prob_home,
            'fair_prob': home_fair_prob,
            'edge': edge_home,
            'edge_pct': edge_home * 100
        })
    
    # Away bet
    away_class = classify_bet(away_decimal, edge_away)
    if away_class:
        bets.append({
            'game_date': str(game_date),
            'bet_team': away_team,
            'opponent': home_team,
            'location': 'away',
            'bet_type': away_class,
            'odds_american': away_ml,
            'odds_decimal': away_decimal,
            'model_prob': model_prob_away,
            'fair_prob': away_fair_prob,
            'edge': edge_away,
            'edge_pct': edge_away * 100
        })
    
    return bets

# ==============================================================================
# 4. EXAMPLE USAGE
# ==============================================================================

print(f"\n[2/3] Testing bet generation...")

# Example: Load recent game from odds file
odds_df = pd.read_csv('data/live/closing_odds_2024_25.csv')
odds_df = odds_df.rename(columns={
    'game_date': 'date',
    'home_ml_odds': 'home_ml',
    'away_ml_odds': 'away_ml'
})

# Test on first few games
print(f"\nExample bets from recent games:")
print("-" * 90)

for idx, row in odds_df.head(10).iterrows():
    bets = generate_bets(
        row['date'],
        row['home_team'],
        row['away_team'],
        row['home_ml'],
        row['away_ml']
    )
    
    if bets:
        for bet in bets:
            print(f"✓ {bet['game_date']}: {bet['bet_team']} ({bet['location']}) "
                  f"{bet['odds_american']:+} | {bet['bet_type'].upper()} | "
                  f"Edge: {bet['edge_pct']:.1f}% | Model: {bet['model_prob']*100:.1f}%")

print(f"\n[3/3] Production strategy ready!")
print("="*90)

# Save strategy summary
strategy_summary = {
    'version': '1.0',
    'locked_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_trial': MODEL_TRIAL,
    'thresholds': {
        'favorites': FAVORITE_EDGE_THRESHOLD,
        'underdogs': UNDERDOG_EDGE_THRESHOLD,
        'odds_split': ODDS_SPLIT_THRESHOLD
    },
    'expected_performance': {
        'total_bets_per_season': 718,
        'favorites_pct': 35.8,
        'underdogs_pct': 64.2,
        'expected_roi': 7.80,
        'expected_units_per_season': 55.99
    }
}

with open('models/production_strategy_v1.json', 'w') as f:
    json.dump(strategy_summary, f, indent=2)

print(f"Saved: models/production_strategy_v1.json")
print(f"\nTo use: import generate_bets from this module")
print("="*90)
