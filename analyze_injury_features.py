"""Analyze all available injury features and data sources"""
import sqlite3
import pandas as pd
from pathlib import Path

print('=' * 80)
print('INJURY FEATURES ANALYSIS')
print('=' * 80)

# Check database schema
db_path = 'data/nba_betting_data.db'
if Path(db_path).exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print('\n1. DATABASE TABLES:')
    injury_tables = [t for t in tables if 'injury' in t.lower() or 'injured' in t.lower()]
    for table in injury_tables:
        print(f'   ✅ {table}')
        # Get schema
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        for col in columns:
            print(f'      - {col[1]} ({col[2]})')
        
        # Get sample row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f'      Total rows: {count}')
        print()
    
    conn.close()
else:
    print(f'   ❌ Database not found: {db_path}')

# Check feature calculator implementation
print('\n2. FEATURE CALCULATOR INJURY METHODS:')
print()

try:
    from src.features.feature_calculator_v5 import FeatureCalculatorV5
    import inspect
    
    methods = inspect.getmembers(FeatureCalculatorV5, predicate=inspect.isfunction)
    injury_methods = [m for m in methods if 'injury' in m[0].lower()]
    
    for method_name, method_obj in injury_methods:
        print(f'   ✅ {method_name}()')
        # Get docstring
        doc = inspect.getdoc(method_obj)
        if doc:
            lines = doc.split('\n')[:3]  # First 3 lines
            for line in lines:
                print(f'      {line.strip()}')
        print()
        
except Exception as e:
    print(f'   ❌ Error: {e}')

# Check injury replacement model
print('\n3. INJURY REPLACEMENT MODEL:')
print()

try:
    from src.features.injury_replacement_model import (
        calculate_team_injury_impact_simple,
        get_injured_players
    )
    
    print('   ✅ calculate_team_injury_impact_simple()')
    print('      Calculates PIE-based injury impact for a team')
    print('      Returns: float (sum of injured players PIE scores)')
    print()
    
    print('   ✅ get_injured_players()')
    print('      Retrieves list of injured players from database')
    print('      Returns: list of dicts with player, status, injury')
    print()
    
except Exception as e:
    print(f'   ❌ Error: {e}')

# Check what the 43-feature model used
print('\n4. HISTORICAL INJURY FEATURES (43-feature model):')
print()

import xgboost as xgb
model = xgb.Booster()
model.load_model('models/xgboost_final_trial98_REFERENCE_43features.json')
features = model.feature_names
scores = model.get_score(importance_type='gain')

injury_features = [f for f in features if 'injury' in f.lower() or 'star' in f.lower()]
for feat in injury_features:
    importance = scores.get(feat, 0.0)
    print(f'   {feat:35s} importance: {importance:6.1f}')

print('\n' + '=' * 80)
print('INJURY FEATURE ENGINEERING OPTIONS')
print('=' * 80)
print()

print('OPTION 1: Simple PIE-based Impact')
print('  - injury_impact_home: Sum of injured players PIE for home team')
print('  - injury_impact_away: Sum of injured players PIE for away team')
print('  - injury_impact_diff: home - away (advantage metric)')
print()

print('OPTION 2: Shock-Based (New vs Expected)')
print('  - injury_shock_home: Current injury - EWMA baseline')
print('  - injury_shock_away: Current injury - EWMA baseline')
print('  - injury_shock_diff: Unexpected injury change')
print('  - Captures "news" rather than baseline roster issues')
print()

print('OPTION 3: Star Binary Flags')
print('  - home_star_missing: 1 if PIE >= 4.0 player out')
print('  - away_star_missing: 1 if PIE >= 4.0 player out')
print('  - star_mismatch: Asymmetric star advantage')
print('  - "Loud signal" for tree models')
print()

print('OPTION 4: COMPREHENSIVE MATCHUP METRIC (RECOMMENDED)')
print('  Formula: injury_matchup_advantage =')
print('    0.4 * injury_impact_diff          # Baseline talent gap')
print('    + 0.3 * injury_shock_diff         # Surprise factor')
print('    + 0.2 * star_mismatch * 5.0       # Star power (scaled)')
print('    + 0.1 * injury_impact_abs * sign  # Total injury load')
print()
print('  Benefits:')
print('    - Single feature reduces dimensionality')
print('    - Weighted by empirical importance')
print('    - Captures multiple injury aspects')
print('    - Easier to interpret and monitor')
print()

print('=' * 80)
print('RECOMMENDATION')
print('=' * 80)
print()
print('Add ONE comprehensive injury feature to 19-feature model:')
print()
print('  injury_matchup_advantage = (')
print('    0.4 * (home_injury_pie - away_injury_pie)')
print('    + 0.3 * (home_shock - away_shock)')
print('    + 0.2 * (home_star - away_star) * 5.0')
print('    + 0.1 * abs(home_injury_pie + away_injury_pie) * sign(home-away)')
print('  )')
print()
print('This creates a 20-feature model with injury integration.')
