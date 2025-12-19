"""
Clean feature validation without emojis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import sqlite3
from datetime import datetime
from config.settings import DB_PATH
from src.features.feature_calculator_live import FeatureCalculatorV5

print("="*80)
print("FEATURE VALIDATION - FULL OUTPUT")
print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Data freshness
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

cursor.execute("SELECT MAX(GAME_DATE) FROM game_logs")
latest_game = cursor.fetchone()[0]

cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
latest_elo = cursor.fetchone()[0]

print(f"\nLatest game: {latest_game}")
print(f"Latest ELO: {latest_elo}")

# Test games
calc = FeatureCalculatorV5()
sample_games = [
    ('OKC', 'SAS', 'Oklahoma City vs San Antonio'),
    ('NYK', 'ORL', 'New York @ Orlando')
]

for i, (home, away, desc) in enumerate(sample_games, 1):
    print(f"\n{'='*80}")
    print(f"GAME {i}: {desc}")
    print(f"{'='*80}")
    
    try:
        features = calc.calculate_game_features(home, away, game_date='2025-12-14')
        
        print(f"\n[OK] Generated {len(features)} total features")
        
        # Get ELO from database
        cursor.execute("SELECT off_elo, def_elo, composite_elo, game_date FROM elo_ratings WHERE team = ? ORDER BY game_date DESC LIMIT 1", (home,))
        home_elo = cursor.fetchone()
        
        cursor.execute("SELECT off_elo, def_elo, composite_elo, game_date FROM elo_ratings WHERE team = ? ORDER BY game_date DESC LIMIT 1", (away,))
        away_elo = cursor.fetchone()
        
        print(f"\n--- DATABASE ELO (as of {home_elo[3] if home_elo else 'N/A'}) ---")
        if home_elo:
            print(f"{home}: Off={home_elo[0]:.1f}, Def={home_elo[1]:.1f}, Comp={home_elo[2]:.1f}")
        if away_elo:
            print(f"{away}: Off={away_elo[0]:.1f}, Def={away_elo[1]:.1f}, Comp={away_elo[2]:.1f}")
        
        print(f"\n--- CALCULATED FEATURES ---")
        
        # ELO features
        elo_feats = {k: v for k, v in features.items() if 'elo' in k}
        print(f"\nELO Features ({len(elo_feats)}):")
        for k, v in sorted(elo_feats.items()):
            print(f"  {k:35s}: {v:10.2f}")
        
        # Injury features
        inj_feats = {k: v for k, v in features.items() if 'injury' in k or 'star' in k}
        print(f"\nInjury Features ({len(inj_feats)}):")
        for k, v in sorted(inj_feats.items()):
            print(f"  {k:35s}: {v:10.2f}")
        
        # Rest features
        rest_feats = {k: v for k, v in features.items() if 'rest' in k or 'fatigue' in k or 'back_to_back' in k or '3in4' in k}
        print(f"\nRest/Fatigue Features ({len(rest_feats)}):")
        for k, v in sorted(rest_feats.items()):
            print(f"  {k:35s}: {v:10.2f}")
        
        # Validation checks
        print(f"\n--- VALIDATION CHECKS ---")
        issues = []
        
        # Check away_composite_elo
        if 'away_composite_elo' in features:
            if features['away_composite_elo'] == 0:
                issues.append("ERROR: away_composite_elo is 0")
        else:
            issues.append("WARNING: away_composite_elo not in features")
        
        # Check for 1500 defaults
        if 'home_composite_elo' in features and 1499 <= features['home_composite_elo'] <= 1501:
            issues.append("WARNING: home_composite_elo at default 1500")
        if 'away_composite_elo' in features and 1499 <= features['away_composite_elo'] <= 1501:
            issues.append("WARNING: away_composite_elo at default 1500")
        
        # Check for NaN
        nan_count = sum(1 for v in features.values() if str(v) == 'nan')
        if nan_count > 0:
            issues.append(f"ERROR: {nan_count} NaN values")
        
        # Check for None
        none_count = sum(1 for v in features.values() if v is None)
        if none_count > 0:
            issues.append(f"ERROR: {none_count} None values")
        
        if issues:
            print("ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("[OK] All features valid - no defaults, zeros, or NaN values")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate features: {e}")
        import traceback
        traceback.print_exc()

conn.close()

print(f"\n{'='*80}")
print("VALIDATION COMPLETE")
print(f"{'='*80}")
