"""
SHAP Analysis: Injury Feature Impact
=====================================
Generate "Smoking Gun" evidence for superstar injury multipliers.

This script:
1. Loads the trained model and calculates SHAP values
2. Finds games where major stars were out (Giannis, Jokic, Luka, etc.)
3. Analyzes SHAP contribution of injury features in those games
4. Validates if multipliers (3.0x, 2.8x) are appropriately tuned
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import sqlite3
import pickle
import shap
from src.features.feature_calculator_v5 import FeatureCalculatorV5

print("\n" + "="*80)
print("🔍 SHAP ANALYSIS: SUPERSTAR INJURY IMPACT")
print("="*80)

# Load trained model
print("\n1. Loading trained model...")
with open('models/xgboost_pruned_31features.pkl', 'rb') as f:
    model = pickle.load(f)
print("   ✅ Model loaded")

# Initialize feature calculator
print("\n2. Initializing feature calculator...")
calc = FeatureCalculatorV5()

# Load game results
db_path = 'data/live/nba_betting_data.db'
with sqlite3.connect(db_path) as conn:
    games_df = pd.read_sql_query(
        """
        SELECT game_date, home_team, away_team, home_score, away_score
        FROM game_results
        WHERE game_date >= '2024-01-01' AND game_date <= '2025-11-01'
        ORDER BY game_date
        """,
        conn
    )

print(f"   ✅ Loaded {len(games_df)} games")

# Find superstar absences
print("\n3. Finding superstar absence games...")
SUPERSTARS = {
    'Giannis Antetokounmpo': ('MIL', 3.0),
    'Nikola Jokic': ('DEN', 2.8),
    'Luka Doncic': ('DAL', 2.8),
    'Joel Embiid': ('PHI', 2.6),
    'Shai Gilgeous-Alexander': ('OKC', 2.5),
    'Stephen Curry': ('GSW', 2.5),
}

superstar_games = []

with sqlite3.connect(db_path) as conn:
    for player_name, (team, multiplier) in SUPERSTARS.items():
        # Find games where this superstar was out
        query = """
        SELECT DISTINCT game_date
        FROM historical_inactives
        WHERE player_name LIKE ?
        AND team_abbreviation = ?
        AND game_date >= '2024-01-01'
        """
        
        # Handle name format (try both "Last, First" and "First Last")
        name_patterns = [
            f"%{player_name.split()[-1]}%",  # Last name
            f"%{player_name}%",  # Full name
        ]
        
        for pattern in name_patterns:
            absences = pd.read_sql_query(query, conn, params=(pattern, team))
            if not absences.empty:
                for date in absences['game_date']:
                    # Find the actual game
                    game = games_df[
                        (games_df['game_date'] == date) &
                        ((games_df['home_team'] == team) | (games_df['away_team'] == team))
                    ]
                    
                    if not game.empty:
                        superstar_games.append({
                            'player': player_name,
                            'multiplier': multiplier,
                            'date': date,
                            'team': team,
                            'home_team': game.iloc[0]['home_team'],
                            'away_team': game.iloc[0]['away_team'],
                            'home_score': game.iloc[0]['home_score'],
                            'away_score': game.iloc[0]['away_score'],
                        })
                break  # Found matches, no need to try other patterns

superstar_df = pd.DataFrame(superstar_games).drop_duplicates(subset=['date', 'team'])
print(f"   ✅ Found {len(superstar_df)} superstar absence games")

if superstar_df.empty:
    print("\n❌ No superstar games found. Check database.")
    sys.exit(1)

# Calculate features for these games
print("\n4. Calculating features for superstar games...")
features_list = []
for idx, row in superstar_df.iterrows():
    if idx % 10 == 0:
        print(f"   Progress: {idx}/{len(superstar_df)}")
    
    features = calc.calculate_game_features(
        row['home_team'],
        row['away_team'],
        game_date=row['date']
    )
    features_list.append(features)

features_df = pd.DataFrame(features_list)
print(f"   ✅ Features calculated: {features_df.shape}")

# Ensure feature order matches training
feature_cols = [
    'ewma_efg_diff', 'ewma_tov_diff', 'ewma_orb_diff', 'ewma_pace_diff', 
    'ewma_vol_3p_diff', 'home_ewma_3p_pct', 'away_ewma_3p_pct', 'away_ewma_tov_pct',
    'home_orb', 'away_orb', 'away_ewma_fta_rate', 'ewma_foul_synergy_home',
    'ewma_foul_synergy_away', 'total_foul_environment', 'ewma_chaos_home',
    'ewma_net_chaos', 'home_rest_days', 'away_rest_days', 'rest_advantage',
    'fatigue_mismatch', 'home_back_to_back', 'away_back_to_back', 'home_3in4',
    'away_3in4', 'home_composite_elo', 'altitude_game', 'injury_impact_diff',
    'injury_impact_abs', 'injury_elo_interaction', 'off_elo_diff', 'def_elo_diff'
]

X = features_df[feature_cols].fillna(0)

# Calculate SHAP values
print("\n5. Calculating SHAP values...")
print("   (This may take a few minutes...)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print(f"   ✅ SHAP values calculated: {shap_values.shape}")

# Analyze injury feature contributions
print("\n" + "="*80)
print("📊 INJURY FEATURE SHAP ANALYSIS")
print("="*80)

injury_feature_indices = {
    'injury_impact_diff': feature_cols.index('injury_impact_diff'),
    'injury_impact_abs': feature_cols.index('injury_impact_abs'),
    'injury_elo_interaction': feature_cols.index('injury_elo_interaction'),
}

# Create results table
results = []
for idx, row in superstar_df.iterrows():
    shap_idx = idx
    
    # Get SHAP contributions for injury features
    inj_diff_shap = shap_values[shap_idx, injury_feature_indices['injury_impact_diff']]
    inj_abs_shap = shap_values[shap_idx, injury_feature_indices['injury_impact_abs']]
    inj_elo_shap = shap_values[shap_idx, injury_feature_indices['injury_elo_interaction']]
    total_inj_shap = inj_diff_shap + inj_abs_shap + inj_elo_shap
    
    # Get actual feature values
    inj_diff_val = X.iloc[shap_idx]['injury_impact_diff']
    inj_abs_val = X.iloc[shap_idx]['injury_impact_abs']
    
    # Get model prediction
    pred_proba = model.predict_proba(X.iloc[[shap_idx]])[0, 1]
    
    # Actual outcome
    home_win = 1 if row['home_score'] > row['away_score'] else 0
    
    results.append({
        'player': row['player'],
        'multiplier': row['multiplier'],
        'date': row['date'],
        'matchup': f"{row['away_team']} @ {row['home_team']}",
        'team': row['team'],
        'is_home': row['team'] == row['home_team'],
        'inj_diff': inj_diff_val,
        'inj_abs': inj_abs_val,
        'shap_diff': inj_diff_shap,
        'shap_abs': inj_abs_shap,
        'shap_elo_int': inj_elo_shap,
        'total_shap': total_inj_shap,
        'pred_home_win': pred_proba,
        'actual_home_win': home_win,
        'correct': (pred_proba > 0.5) == home_win,
    })

results_df = pd.DataFrame(results)

# Summary by superstar
print("\n📈 SHAP IMPACT BY SUPERSTAR:")
print("="*80)
print(f"{'Player':<25} {'Games':<7} {'Avg SHAP':<12} {'Multiplier':<12} {'Tuning'}")
print("-"*80)

for player_name, (team, multiplier) in SUPERSTARS.items():
    player_games = results_df[results_df['player'] == player_name]
    if len(player_games) > 0:
        avg_shap = player_games['total_shap'].mean()
        avg_abs = player_games['inj_abs'].mean()
        
        # Expected SHAP based on multiplier (rough heuristic)
        # Higher multiplier should mean higher absolute SHAP
        expected_impact = multiplier * 0.05  # Rough scale
        
        if abs(avg_shap) < expected_impact * 0.5:
            tuning = "⬆️ INCREASE"
        elif abs(avg_shap) > expected_impact * 1.5:
            tuning = "⬇️ DECREASE"
        else:
            tuning = "✅ GOOD"
        
        print(f"{player_name:<25} {len(player_games):<7} {avg_shap:>11.4f} {multiplier:>11.1f}x {tuning}")

# Top 20 games by absolute SHAP contribution
print("\n" + "="*80)
print("🔥 TOP 20 GAMES BY INJURY SHAP IMPACT")
print("="*80)
print(f"{'Date':<12} {'Player':<20} {'Matchup':<20} {'SHAP':<10} {'Pred':<8} {'Result'}")
print("-"*80)

results_df['abs_shap'] = results_df['total_shap'].abs()
top_games = results_df.nlargest(20, 'abs_shap')
for _, row in top_games.iterrows():
    shap_str = f"{row['total_shap']:>+8.4f}"
    pred_str = f"{row['pred_home_win']:.2f}"
    result = "✅" if row['correct'] else "❌"
    team_side = "HOME" if row['is_home'] else "AWAY"
    
    print(f"{row['date']:<12} {row['player']:<20} {row['matchup']:<20} {shap_str:<10} {pred_str:<8} {result} ({team_side})")

# Accuracy analysis
print("\n" + "="*80)
print("🎯 PREDICTION ACCURACY")
print("="*80)

overall_acc = results_df['correct'].mean()
print(f"\nOverall accuracy on superstar absence games: {overall_acc:.1%}")

# By impact level
high_impact = results_df[abs(results_df['total_shap']) > 0.1]
if len(high_impact) > 0:
    high_acc = high_impact['correct'].mean()
    print(f"High SHAP impact games (|SHAP| > 0.1): {high_acc:.1%} ({len(high_impact)} games)")

med_impact = results_df[(abs(results_df['total_shap']) > 0.05) & (abs(results_df['total_shap']) <= 0.1)]
if len(med_impact) > 0:
    med_acc = med_impact['correct'].mean()
    print(f"Medium SHAP impact games (0.05 < |SHAP| ≤ 0.1): {med_acc:.1%} ({len(med_impact)} games)")

# Directional accuracy (did injury hurt the team?)
print("\n" + "="*80)
print("📉 DIRECTIONAL IMPACT (Does injury lower win probability?)")
print("="*80)

# When home team has the injured superstar
home_injured = results_df[results_df['is_home'] == True]
if len(home_injured) > 0:
    # SHAP should be negative (lowering home win probability)
    home_negative = (home_injured['total_shap'] < 0).sum()
    print(f"\nHome team injured: {home_negative}/{len(home_injured)} ({home_negative/len(home_injured):.1%}) had negative SHAP")
    print(f"   Average SHAP: {home_injured['total_shap'].mean():.4f} (should be negative)")

# When away team has the injured superstar  
away_injured = results_df[results_df['is_home'] == False]
if len(away_injured) > 0:
    # SHAP should be positive (raising home win probability)
    away_positive = (away_injured['total_shap'] > 0).sum()
    print(f"\nAway team injured: {away_positive}/{len(away_injured)} ({away_positive/len(away_injured):.1%}) had positive SHAP")
    print(f"   Average SHAP: {away_injured['total_shap'].mean():.4f} (should be positive)")

# Multiplier recommendations
print("\n" + "="*80)
print("🎛️ MULTIPLIER TUNING RECOMMENDATIONS")
print("="*80)

print("\nBased on SHAP analysis:")
for player_name, (team, current_mult) in SUPERSTARS.items():
    player_games = results_df[results_df['player'] == player_name]
    if len(player_games) > 0:
        avg_shap = abs(player_games['total_shap'].mean())
        avg_abs = player_games['inj_abs'].mean()
        
        # Calculate recommended multiplier based on SHAP/abs ratio
        # Target: SHAP contribution around 0.10-0.15 for top superstars
        target_shap = 0.12 if current_mult >= 2.8 else 0.08
        
        if avg_abs > 0:
            # Rough calibration (this is heuristic)
            shap_per_unit = avg_shap / avg_abs if avg_abs > 0 else 0
            current_shap_efficiency = avg_shap / current_mult if current_mult > 0 else 0
            
            if avg_shap < target_shap * 0.6:
                recommended = current_mult * 1.3  # Increase by 30%
                print(f"\n{player_name}:")
                print(f"   Current: {current_mult:.1f}x → Recommended: {recommended:.1f}x")
                print(f"   Reason: SHAP too low ({avg_shap:.4f}, target ~{target_shap:.2f})")
            elif avg_shap > target_shap * 1.4:
                recommended = current_mult * 0.8  # Decrease by 20%
                print(f"\n{player_name}:")
                print(f"   Current: {current_mult:.1f}x → Recommended: {recommended:.1f}x")
                print(f"   Reason: SHAP too high ({avg_shap:.4f}, target ~{target_shap:.2f})")
            else:
                print(f"\n{player_name}:")
                print(f"   Current: {current_mult:.1f}x → ✅ WELL CALIBRATED")
                print(f"   SHAP: {avg_shap:.4f} (target ~{target_shap:.2f})")

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

# Save full results
results_df.to_csv('output/shap_superstar_analysis.csv', index=False)
print("   ✅ Saved: output/shap_superstar_analysis.csv")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
