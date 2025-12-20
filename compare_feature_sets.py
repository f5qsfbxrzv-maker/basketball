import pandas as pd

print("=" * 80)
print("MDP MODEL FEATURE COMPARISON")
print("=" * 80)

# 1. Config file says we have these 19 features
config_features = [
    'off_elo_diff', 
    'def_elo_diff', 
    'home_composite_elo',
    'projected_possession_margin', 
    'ewma_pace_diff', 
    'net_fatigue_score',
    'ewma_efg_diff', 
    'ewma_vol_3p_diff', 
    'three_point_matchup',
    'injury_matchup_advantage', 
    'injury_shock_diff', 
    'star_power_leverage',
    'season_progress', 
    'league_offensive_context',
    'total_foul_environment', 
    'net_free_throw_advantage',
    'offense_vs_defense_matchup', 
    'pace_efficiency_interaction', 
    'star_mismatch'
]

print(f"\n1. PRODUCTION CONFIG (production_config_mdp.py): {len(config_features)} features")
for i, f in enumerate(config_features, 1):
    print(f"  {i:2d}. {f}")

# 2. Check what's in the training data
try:
    df = pd.read_csv('data/training_data_MDP_with_margins.csv', nrows=5)
    training_features = [c for c in df.columns if c not in [
        'date', 'game_date', 'target_ats_margin', 'home_team', 'away_team', 
        'target_moneyline_win', 'home_score', 'away_score', 'target_total', 
        'season', 'Unnamed: 0'
    ]]
    
    print(f"\n2. TRAINING DATA (training_data_MDP_with_margins.csv): {len(training_features)} features")
    for i, f in enumerate(training_features, 1):
        print(f"  {i:2d}. {f}")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    config_set = set(config_features)
    training_set = set(training_features)
    
    only_in_config = config_set - training_set
    only_in_training = training_set - config_set
    in_both = config_set & training_set
    
    print(f"\n✅ Features in BOTH ({len(in_both)}):")
    for f in sorted(in_both):
        print(f"  - {f}")
    
    if only_in_config:
        print(f"\n⚠️  Only in CONFIG, NOT in training data ({len(only_in_config)}):")
        for f in sorted(only_in_config):
            print(f"  - {f}")
    
    if only_in_training:
        print(f"\n⚠️  Only in TRAINING DATA, not in config ({len(only_in_training)}):")
        for f in sorted(only_in_training):
            print(f"  - {f}")
    
except FileNotFoundError:
    print("\n❌ training_data_MDP_with_margins.csv not found")

# 3. Check other training data files
print("\n" + "=" * 80)
print("CHECKING OTHER TRAINING FILES")
print("=" * 80)

other_files = [
    'data/training_data_GOLD_ELO_22_features.csv',
    'data/training_data_SYNDICATE_CLEANED_14_features.csv',
]

for filepath in other_files:
    try:
        df = pd.read_csv(filepath, nrows=5)
        features = [c for c in df.columns if c not in [
            'date', 'game_date', 'target_ats_margin', 'home_team', 'away_team',
            'target_moneyline_win', 'home_score', 'away_score', 'target_total',
            'season', 'Unnamed: 0'
        ]]
        print(f"\n{filepath}: {len(features)} features")
        for i, f in enumerate(features, 1):
            print(f"  {i:2d}. {f}")
    except Exception as e:
        print(f"\n{filepath}: ❌ {e}")
