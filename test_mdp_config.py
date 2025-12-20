from production_config_mdp import *

print('✓ MDP Config loaded successfully')
print(f'Architecture: {MODEL_TYPE}')
print(f'Objective: {XGB_PARAMS["objective"]}')
print(f'StdDev: {NBA_STD_DEV}')
print(f'Favorite Edge: {MIN_EDGE_FAVORITE:.1%}')
print(f'Underdog Edge: {MIN_EDGE_UNDERDOG:.1%}')
print(f'Max Favorite Odds: {MAX_FAVORITE_ODDS}')
print(f'N Estimators: {N_ESTIMATORS}')
print(f'Features: {len(ACTIVE_FEATURES)}')
