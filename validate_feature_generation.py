"""
Full feature validation for sample games
Shows all features, data dates, and checks for defaults
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import sqlite3
from datetime import datetime
from config.settings import DB_PATH

print("="*80)
print("FEATURE GENERATION VALIDATION")
print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Step 1: Check data freshness
print("\n[DATA FRESHNESS CHECK]")
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

cursor.execute("SELECT MAX(GAME_DATE) FROM game_logs")
latest_game = cursor.fetchone()[0]
print(f"Latest game in database: {latest_game}")

cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
latest_elo = cursor.fetchone()[0]
print(f"Latest ELO update: {latest_elo}")

cursor.execute("SELECT COUNT(*) FROM active_injuries")
injury_count = cursor.fetchone()[0]
print(f"Active injuries tracked: {injury_count}")

# Step 2: Get sample matchups for today/tomorrow
print("\n[SAMPLE GAMES TO TEST]")
sample_games = [
    ('OKC', 'SAS', 'OKC vs SAS'),
    ('NYK', 'ORL', 'NYK @ ORL'),
    ('LAL', 'GSW', 'LAL vs GSW'),
    ('BOS', 'MIA', 'BOS vs MIA')
]

# Step 3: Calculate features for each game
from src.features.feature_calculator_live import FeatureCalculatorV5

calc = FeatureCalculatorV5()

for i, (home, away, desc) in enumerate(sample_games, 1):
    print("\n" + "="*80)
    print(f"GAME {i}: {desc} ({home} vs {away})")
    print(f"Test date: 2025-12-14")
    print("="*80)
    
    try:
        features = calc.calculate_game_features(home, away, game_date='2025-12-14')
        
        print(f"\nTotal features generated: {len(features)}")
        
        # Check for ELO data
        print("\n--- ELO FEATURES ---")
        elo_features = {k: v for k, v in features.items() if 'elo' in k}
        
        # Get actual ELO from database for comparison
        cursor.execute("""
            SELECT off_elo, def_elo, composite_elo, game_date
            FROM elo_ratings
            WHERE team = ?
            ORDER BY game_date DESC
            LIMIT 1
        """, (home,))
        home_elo_db = cursor.fetchone()
        
        cursor.execute("""
            SELECT off_elo, def_elo, composite_elo, game_date
            FROM elo_ratings
            WHERE team = ?
            ORDER BY game_date DESC
            LIMIT 1
        """, (away,))
        away_elo_db = cursor.fetchone()
        
        if home_elo_db:
            print(f"\n{home} ELO (from DB as of {home_elo_db[3]}):")
            print(f"  Off: {home_elo_db[0]:.1f}, Def: {home_elo_db[1]:.1f}, Comp: {home_elo_db[2]:.1f}")
        
        if away_elo_db:
            print(f"\n{away} ELO (from DB as of {away_elo_db[3]}):")
            print(f"  Off: {away_elo_db[0]:.1f}, Def: {away_elo_db[1]:.1f}, Comp: {away_elo_db[2]:.1f}")
        
        print(f"\nELO features calculated:")
        for k, v in sorted(elo_features.items()):
            print(f"  {k:35s}: {v:10.2f}")
        
        # Check for defaults/issues
        print("\n--- VALIDATION CHECKS ---")
        issues = []
        
        # Check for 1500 defaults
        if 'home_composite_elo' in features:
            if 1499 <= features['home_composite_elo'] <= 1501:
                issues.append(f"⚠️ Home composite ELO is default 1500")
        if 'away_composite_elo' in features:
            if 1499 <= features['away_composite_elo'] <= 1501:
                issues.append(f"⚠️ Away composite ELO is default 1500")
        
        # Check for zero values
        if 'off_elo_diff' in features and features['off_elo_diff'] == 0:
            issues.append(f"⚠️ Offensive ELO diff is 0")
        if 'def_elo_diff' in features and features['def_elo_diff'] == 0:
            issues.append(f"⚠️ Defensive ELO diff is 0")
        
        # Check for NaN
        nan_features = [k for k, v in features.items() if str(v) == 'nan']
        if nan_features:
            issues.append(f"⚠️ {len(nan_features)} NaN features: {nan_features[:3]}")
        
        # Check for None
        none_features = [k for k, v in features.items() if v is None]
        if none_features:
            issues.append(f"⚠️ {len(none_features)} None features: {none_features[:3]}")
        
        if issues:
            print("ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✅ No defaults or invalid values detected")
        
        # Show key feature categories
        print("\n--- FEATURE BREAKDOWN ---")
        injury_feats = [k for k in features.keys() if 'injury' in k or 'star' in k]
        rest_feats = [k for k in features.keys() if 'rest' in k or 'fatigue' in k]
        ewma_feats = [k for k in features.keys() if 'ewma' in k]
        
        print(f"  Injury features: {len(injury_feats)}")
        print(f"  Rest/Fatigue features: {len(rest_feats)}")
        print(f"  EWMA features: {len(ewma_feats)}")
        print(f"  ELO features: {len(elo_features)}")
        
        # Show sample of each category
        if injury_feats:
            print(f"\n  Sample injury features:")
            for k in injury_feats[:3]:
                print(f"    {k}: {features[k]:.4f}")
        
        if rest_feats:
            print(f"\n  Sample rest features:")
            for k in rest_feats[:3]:
                print(f"    {k}: {features[k]:.4f}")
        
    except Exception as e:
        print(f"\n❌ ERROR generating features: {e}")
        import traceback
        traceback.print_exc()

conn.close()

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
