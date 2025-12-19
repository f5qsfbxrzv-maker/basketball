import xgboost as xgb
import pandas as pd

# Load the trained model
model = xgb.Booster()
model.load_model('models/xgboost_25features_optuna_20251215_144548.json')

# Feature names in order
features = [
    'home_composite_elo',
    'away_composite_elo',
    'off_elo_diff',
    'def_elo_diff',
    'net_fatigue_score',
    'ewma_efg_diff',
    'ewma_pace_diff',
    'ewma_tov_diff',
    'ewma_orb_diff',
    'ewma_vol_3p_diff',
    'injury_impact_diff',
    'injury_shock_diff',
    'star_mismatch',
    'ewma_chaos_home',
    'ewma_foul_synergy_home',
    'total_foul_environment',
    'league_offensive_context',
    'season_progress',
    'pace_efficiency_interaction',
    'projected_possession_margin',
    'three_point_matchup',
    'net_free_throw_advantage',
    'star_power_leverage',
    'offense_vs_defense_matchup',
    'injury_matchup_advantage'
]

# Get feature importance by gain
importance = model.get_score(importance_type='gain')

# Create DataFrame
df = pd.DataFrame({
    'feature': features,
    'importance': [importance.get(f'f{i}', 0) for i in range(len(features))]
})

# Sort by importance
df = df.sort_values('importance', ascending=False)
df['rank'] = range(1, len(df) + 1)

# Display results
print('\nFEATURE IMPORTANCE (sorted by gain):')
print('=' * 70)
print(f'{"Rank":<6} {"Feature":<40} {"Importance":>12}')
print('-' * 70)

for idx, row in df.iterrows():
    print(f"{row['rank']:<6} {row['feature']:<40} {row['importance']:>12.1f}")

print()
print('=' * 70)

# Find injury feature rank
injury_rank = df[df['feature'] == 'injury_matchup_advantage']['rank'].values[0]
injury_importance = df[df['feature'] == 'injury_matchup_advantage']['importance'].values[0]
print(f"\ninjury_matchup_advantage: Rank #{injury_rank}/25, Importance: {injury_importance:.1f}")
