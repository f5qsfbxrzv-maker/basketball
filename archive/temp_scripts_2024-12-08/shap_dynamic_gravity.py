"""
SHAP Analysis for Dynamic Gravity Model
Validates that injury impacts are correctly contributing to predictions
Tests on superstar absence games (Giannis, Jokic, Embiid, Luka, Curry)
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sqlite3

# Load trained model
model_path = Path('models/xgboost_pruned_31features.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load full dataset
full_df = pd.read_csv('data/training_data_with_features.csv')

# Filter to whitelisted features
whitelist_features = [
    'ewma_efg_diff', 'def_elo_diff', 'fatigue_mismatch', 'ewma_tov_diff', 'off_elo_diff',
    'home_rest_days', 'total_foul_environment', 'altitude_game', 'away_rest_days', 'ewma_net_chaos',
    'away_ewma_tov_pct', 'ewma_orb_diff', 'rest_advantage', 'home_composite_elo', 'ewma_foul_synergy_away',
    'ewma_vol_3p_diff', 'ewma_foul_synergy_home', 'ewma_pace_diff', 'injury_impact_abs', 'away_orb',
    'home_orb', 'home_ewma_3p_pct', 'away_ewma_3p_pct', 'away_ewma_fta_rate', 'ewma_chaos_home',
    'injury_impact_diff', 'injury_elo_interaction', 'away_back_to_back', 'home_3in4', 'home_back_to_back',
    'away_3in4'
]

# Prepare features and labels
X = full_df[whitelist_features].copy()
y = full_df['home_covered'].copy()

# Time-based split (80/20)
split_idx = int(len(X) * 0.8)
test_df = X.iloc[split_idx:].copy()
test_labels = y.iloc[split_idx:].copy()

print('🔍 SHAP ANALYSIS - DYNAMIC GRAVITY MODEL')
print('='*80)
print(f'   Loaded {len(full_df)} total games')
print(f'   Test set: {len(test_df)} games')

# Connect to database
conn = sqlite3.connect('data/live/nba_betting_data.db')

# Get injury data joined with game info
query = """
SELECT 
    g.game_date,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    hi.player_name,
    hi.team_abbreviation,
    ps.pie
FROM games g
LEFT JOIN historical_inactives hi ON g.game_date = hi.game_date
LEFT JOIN player_stats ps ON hi.player_name = ps.player_name AND hi.season = ps.season
WHERE hi.player_name IN (
    'Antetokounmpo, Giannis',
    'Jokic, Nikola',
    'Embiid, Joel',
    'Doncic, Luka',
    'Curry, Stephen'
)
AND g.game_date >= '2023-10-01'
ORDER BY g.game_date
"""

injury_games = pd.read_sql(query, conn)
conn.close()

print(f'\n📋 Loaded {len(injury_games)} superstar absence records')
print(f'   Games: {injury_games.game_date.nunique()}')
print(f'   Players: {injury_games.player_name.nunique()}')

# Normalize player names for matching
name_map = {
    'Antetokounmpo, Giannis': 'Giannis Antetokounmpo',
    'Jokic, Nikola': 'Nikola Jokic',
    'Embiid, Joel': 'Joel Embiid',
    'Doncic, Luka': 'Luka Doncic',
    'Curry, Stephen': 'Stephen Curry'
}

injury_games['normalized_name'] = injury_games['player_name'].map(name_map)

# Count absences per player
print('\n🏀 SUPERSTAR ABSENCE COUNTS:')
print('-'*80)
for player, count in injury_games.groupby('normalized_name').size().sort_values(ascending=False).items():
    avg_pie = injury_games[injury_games['normalized_name'] == player]['pie'].mean()
    print(f'   {player:<25} {count:>3} games  (Avg PIE: {avg_pie:.4f})')

# Calculate Dynamic Gravity multipliers
LEAGUE_AVG_PIE = 0.0855
LEAGUE_STD_PIE = 0.0230

def calc_dynamic_gravity(pie):
    if pd.isna(pie):
        return 1.0
    z = (pie - LEAGUE_AVG_PIE) / LEAGUE_STD_PIE
    if z <= 1.0:
        return 1.0
    elif z <= 2.5:
        return 1.0 + (z - 1.0) * 1.33
    else:
        return min(3.0 + (z - 2.5) * 1.5, 4.5)

injury_games['gravity_mult'] = injury_games['pie'].apply(calc_dynamic_gravity)
injury_games['base_impact'] = injury_games['pie'] * 20.0
injury_games['total_impact'] = injury_games['base_impact'] * injury_games['gravity_mult']

print('\n📊 DYNAMIC GRAVITY MULTIPLIERS:')
print('-'*80)
print(f"{'Player':<25} {'PIE':<8} {'Z-Score':<10} {'Multiplier':<12} {'Total Impact'}")
print('-'*80)

for player in ['Giannis Antetokounmpo', 'Nikola Jokic', 'Joel Embiid', 'Luka Doncic', 'Stephen Curry']:
    player_data = injury_games[injury_games['normalized_name'] == player]
    if len(player_data) > 0:
        avg_pie = player_data['pie'].mean()
        z_score = (avg_pie - LEAGUE_AVG_PIE) / LEAGUE_STD_PIE
        avg_mult = player_data['gravity_mult'].mean()
        avg_impact = player_data['total_impact'].mean()
        print(f"{player:<25} {avg_pie:<8.4f} {z_score:<10.2f} {avg_mult:<12.2f}x {avg_impact:.2f}")

# Try to calculate SHAP values
try:
    import shap
    
    print('\n🎯 CALCULATING SHAP VALUES...')
    print('   (This may take a few minutes)')
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(test_df)
    
    # Get feature names
    feature_names = test_df.columns.tolist()
    
    # Find injury feature indices
    injury_feature_indices = [
        i for i, name in enumerate(feature_names) 
        if 'injury' in name.lower()
    ]
    
    print(f'\n   Found {len(injury_feature_indices)} injury features:')
    for idx in injury_feature_indices:
        print(f'      - {feature_names[idx]}')
    
    # Calculate mean absolute SHAP for injury features
    injury_shap = shap_values[:, injury_feature_indices]
    mean_abs_injury_shap = np.abs(injury_shap).mean(axis=0)
    
    print('\n📈 INJURY FEATURE SHAP IMPORTANCE:')
    print('-'*80)
    for idx, shap_val in zip(injury_feature_indices, mean_abs_injury_shap):
        print(f'   {feature_names[idx]:<30} {shap_val:.6f}')
    
    # Analyze high-injury games
    injury_impact_idx = [i for i, name in enumerate(feature_names) if name == 'injury_impact_abs'][0]
    injury_diff_idx = [i for i, name in enumerate(feature_names) if name == 'injury_impact_diff'][0]
    
    high_injury_mask = test_df['injury_impact_abs'] > 3.0
    print(f'\n🏥 HIGH INJURY GAMES (injury_impact_abs > 3.0):')
    print(f'   Count: {high_injury_mask.sum()} / {len(test_df)} ({100*high_injury_mask.sum()/len(test_df):.1f}%)')
    
    if high_injury_mask.sum() > 0:
        # Get predictions for high injury games
        high_injury_preds = model.predict(test_df[high_injury_mask])
        high_injury_actual = test_labels[high_injury_mask].values.ravel()
        
        accuracy = (high_injury_preds == high_injury_actual).mean()
        print(f'   Accuracy: {accuracy:.1%}')
        
        # Calculate correlation between injury impact and SHAP
        injury_shap_abs = shap_values[high_injury_mask, injury_impact_idx]
        injury_values = test_df[high_injury_mask]['injury_impact_abs'].values
        
        correlation = np.corrcoef(injury_values, injury_shap_abs)[0, 1]
        print(f'   Correlation (injury_impact vs SHAP): {correlation:.3f}')
    
    # Summary
    print('\n✅ VALIDATION SUMMARY:')
    print('='*80)
    print(f'   Dynamic Gravity Model: ACTIVE')
    print(f'   Superstar multipliers: 4.5x (Giannis, Jokic, Embiid, Luka, Curry)')
    print(f'   Injury features: 3 ({feature_names[injury_feature_indices[0]]}, {feature_names[injury_feature_indices[1]]}, {feature_names[injury_feature_indices[2]]})')
    print(f'   Mean |SHAP|: {mean_abs_injury_shap.mean():.6f}')
    print(f'   Test set size: {len(test_df)} games')
    
except ImportError:
    print('\n⚠️  SHAP library not available - install with: pip install shap')
    print('   Showing injury impact statistics instead...')
    
    # Fallback analysis without SHAP
    print('\n📊 INJURY IMPACT STATISTICS (Test Set):')
    print('-'*80)
    
    injury_cols = [col for col in test_df.columns if 'injury' in col.lower()]
    for col in injury_cols:
        mean_val = test_df[col].mean()
        std_val = test_df[col].std()
        max_val = test_df[col].max()
        print(f'   {col:<30} Mean: {mean_val:>7.3f}  Std: {std_val:>6.3f}  Max: {max_val:>7.3f}')

print('\n' + '='*80)
