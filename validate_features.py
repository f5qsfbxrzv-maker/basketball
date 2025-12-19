"""
Feature Validation - Verify model inputs are accurate and current
Shows sample game features with data freshness timestamps
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import sqlite3
from datetime import datetime
from config.settings import DB_PATH

print("="*80)
print("FEATURE VALIDATION - Model Input Verification")
print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Step 1: Check data freshness
print("\n[DATA FRESHNESS CHECK]")
print("-"*80)

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Check game_logs
cursor.execute("SELECT MAX(game_date), COUNT(*) FROM game_logs")
latest_game, total_games = cursor.fetchone()
print(f"Game Logs:")
print(f"  Latest game: {latest_game}")
print(f"  Total games: {total_games}")
days_old = (datetime.now().date() - datetime.strptime(latest_game, '%Y-%m-%d').date()).days
print(f"  Age: {days_old} day(s) old {'⚠️ STALE' if days_old > 1 else '✓ Fresh'}")

# Check ELO ratings
cursor.execute("SELECT MAX(game_date), COUNT(*) FROM elo_ratings")
latest_elo, total_elo = cursor.fetchone()
print(f"\nELO Ratings:")
print(f"  Latest update: {latest_elo}")
print(f"  Total ratings: {total_elo}")
elo_days_old = (datetime.now().date() - datetime.strptime(latest_elo, '%Y-%m-%d').date()).days
print(f"  Age: {elo_days_old} day(s) old {'⚠️ STALE' if elo_days_old > 1 else '✓ Fresh'}")

# Check injuries
cursor.execute("SELECT COUNT(*) FROM active_injuries")
injury_count = cursor.fetchone()[0]
print(f"\nActive Injuries:")
print(f"  Count: {injury_count} players")

# Step 2: Load predictions cache and show sample features
print("\n" + "="*80)
print("[SAMPLE GAME FEATURES]")
print("="*80)

cache_file = Path("predictions_cache.json")
if cache_file.exists():
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    
    print(f"\nLoaded {len(cache)} cached predictions")
    
    # Show 3 sample games with full features
    sample_count = 0
    for key, pred in cache.items():
        if sample_count >= 3:
            break
            
        print(f"\n{'='*80}")
        print(f"GAME {sample_count + 1}: {key}")
        print(f"{'='*80}")
        print(f"Matchup: {pred['away_team']} @ {pred['home_team']}")
        print(f"Date: {pred['game_date']}")
        print(f"Model Prediction: Home Win {pred['home_win_prob']:.1%}, Away Win {pred['away_win_prob']:.1%}")
        
        if pred.get('best_bet'):
            print(f"Best Bet: {pred['best_bet']['pick']} (Edge: {pred['best_bet']['edge']:.1%})")
        
        features = pred.get('features', {})
        if features:
            print(f"\n--- ELO FEATURES ---")
            print(f"{'Feature':<35} {'Home':<15} {'Away':<15}")
            print("-"*65)
            
            # Composite ELO
            home_comp_elo = features.get('home_composite_elo', 0)
            away_comp_elo = features.get('away_composite_elo', 0)
            if isinstance(home_comp_elo, (int, float)) and isinstance(away_comp_elo, (int, float)):
                print(f"{'Composite ELO':<35} {home_comp_elo:<15.1f} {away_comp_elo:<15.1f}")
            else:
                print(f"{'Composite ELO':<35} {str(home_comp_elo):<15} {str(away_comp_elo):<15}")
            
            # Get actual ELO from database for verification
            home_team = pred['home_team']
            away_team = pred['away_team']
            
            cursor.execute("""
                SELECT off_elo, def_elo, composite_elo, game_date 
                FROM elo_ratings 
                WHERE team = ? 
                ORDER BY game_date DESC 
                LIMIT 1
            """, (home_team,))
            home_elo_db = cursor.fetchone()
            
            cursor.execute("""
                SELECT off_elo, def_elo, composite_elo, game_date 
                FROM elo_ratings 
                WHERE team = ? 
                ORDER BY game_date DESC 
                LIMIT 1
            """, (away_team,))
            away_elo_db = cursor.fetchone()
            
            if home_elo_db and away_elo_db:
                print(f"\n{'DATABASE VERIFICATION:':<35}")
                print(f"  Home ({home_team}) ELO: Off={home_elo_db[0]:.1f}, Def={home_elo_db[1]:.1f}, Comp={home_elo_db[2]:.1f}")
                print(f"    Last updated: {home_elo_db[3]}")
                print(f"  Away ({away_team}) ELO: Off={away_elo_db[0]:.1f}, Def={away_elo_db[1]:.1f}, Comp={away_elo_db[2]:.1f}")
                print(f"    Last updated: {away_elo_db[3]}")
                
                # Check if feature matches database
                if abs(home_comp_elo - home_elo_db[2]) > 1:
                    print(f"  ⚠️  HOME ELO MISMATCH: Feature={home_comp_elo:.1f}, DB={home_elo_db[2]:.1f}")
                if abs(away_comp_elo - away_elo_db[2]) > 1:
                    print(f"  ⚠️  AWAY ELO MISMATCH: Feature={away_comp_elo:.1f}, DB={away_elo_db[2]:.1f}")
            else:
                print(f"  ⚠️  ELO data missing in database for one or both teams")
            
            # Other key features
            print(f"\n{'--- KEY FEATURES ---':<35}")
            key_features = [
                ('off_elo_diff', 'Offensive ELO Diff'),
                ('def_elo_diff', 'Defensive ELO Diff'),
                ('home_rest_days', 'Home Rest Days'),
                ('away_rest_days', 'Away Rest Days'),
                ('rest_advantage', 'Rest Advantage'),
                ('injury_shock_home', 'Home Injury Impact'),
                ('injury_shock_away', 'Away Injury Impact'),
                ('injury_shock_diff', 'Injury Differential'),
            ]
            
            for feat_key, feat_name in key_features:
                val = features.get(feat_key, 'N/A')
                if isinstance(val, (int, float)):
                    print(f"{feat_name:<35} {val:>15.2f}")
                else:
                    print(f"{feat_name:<35} {str(val):>15}")
            
            # Show all features
            print(f"\n{'--- ALL FEATURES ---':<35}")
            print(f"Total features: {len(features)}")
            
            # Group features by type
            elo_features = {k: v for k, v in features.items() if 'elo' in k.lower()}
            injury_features = {k: v for k, v in features.items() if 'injury' in k.lower() or 'shock' in k.lower()}
            rest_features = {k: v for k, v in features.items() if 'rest' in k.lower()}
            
            print(f"  ELO features: {len(elo_features)}")
            print(f"  Injury features: {len(injury_features)}")
            print(f"  Rest features: {len(rest_features)}")
            print(f"  Other features: {len(features) - len(elo_features) - len(injury_features) - len(rest_features)}")
            
            # Check for suspicious values
            print(f"\n{'--- SANITY CHECKS ---':<35}")
            issues = []
            
            # Check for default 1500 ELO (indicates missing data)
            if abs(home_comp_elo - 1500) < 1:
                issues.append(f"Home ELO is default 1500 (missing data?)")
            if abs(away_comp_elo - 1500) < 1:
                issues.append(f"Away ELO is default 1500 (missing data?)")
            
            # Check for NaN or None values
            nan_features = [k for k, v in features.items() if v is None or (isinstance(v, float) and str(v) == 'nan')]
            if nan_features:
                issues.append(f"{len(nan_features)} features are NaN/None: {', '.join(nan_features[:5])}")
            
            # Check rest days are reasonable (0-7)
            home_rest = features.get('home_rest_days', 0)
            away_rest = features.get('away_rest_days', 0)
            if home_rest > 7 or away_rest > 7:
                issues.append(f"Rest days unusually high: home={home_rest}, away={away_rest}")
            
            if issues:
                for issue in issues:
                    print(f"  ⚠️  {issue}")
            else:
                print(f"  ✓ All features look reasonable")
        
        sample_count += 1
else:
    print("\n⚠️  No predictions cache found - run dashboard to generate predictions")

conn.close()

# Summary
print("\n" + "="*80)
print("[VALIDATION SUMMARY]")
print("="*80)

if elo_days_old > 1:
    print(f"⚠️  WARNING: ELO data is {elo_days_old} days old")
    print(f"   Predictions are using outdated ratings from {latest_elo}")
    print(f"   Run: python daily_data_update.py")
else:
    print(f"✓ Data is current (ELO last updated {elo_days_old} day(s) ago)")

if days_old > 1:
    print(f"⚠️  WARNING: Game logs are {days_old} days old")
    print(f"   Missing recent games since {latest_game}")
else:
    print(f"✓ Game logs are current")

print(f"\n✓ Total features per game: {len(features) if 'features' in locals() else 'Unknown'}")
print(f"✓ Active injuries tracked: {injury_count}")

print("\nRecommendations:")
if elo_days_old > 1 or days_old > 1:
    print("  1. Run 'python daily_data_update.py' to refresh all data")
    print("  2. Delete predictions_cache.json and restart dashboard")
    print("  3. Verify predictions use updated ELO ratings")
else:
    print("  ✓ Data is fresh and ready for predictions")
