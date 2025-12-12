"""
Feature Sparsity Audit - Validates injury and fatigue features are triggering properly
Checks:
1. Sparsity: How often features are non-zero (signal vs noise)
2. Magnitude: Are values large enough to matter?
3. Reality Check: Do high-impact games make sense?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import sqlite3
from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

print("=" * 80)
print("FEATURE SPARSITY AUDIT")
print("=" * 80)

# Initialize calculator
print("\n1. Initializing Feature Calculator V5...")
calc = FeatureCalculatorV5()

# Load games
print("\n2. Loading game results...")
conn = sqlite3.connect('data/live/nba_betting_data.db')
query = """
SELECT 
    game_id,
    game_date,
    home_team,
    away_team,
    home_score,
    away_score,
    home_won
FROM game_results
WHERE game_date >= '2023-01-01' AND game_date < '2025-11-01'
ORDER BY game_date
"""
games_df = pd.read_sql(query, conn)
conn.close()

print(f"   Loaded {len(games_df)} games")

# Generate features for sample
print("\n3. Generating features (sampling 1000 games for comprehensive audit)...")
sample_games = games_df.sample(min(1000, len(games_df)), random_state=42)
features_list = []
metadata_list = []

for idx, row in sample_games.iterrows():
    try:
        features = calc.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            game_date=row['game_date']
        )
        features_list.append(features)
        metadata_list.append({
            'date': row['game_date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': row['home_score'],
            'away_score': row['away_score']
        })
    except Exception as e:
        continue

df = pd.DataFrame(features_list)
metadata_df = pd.DataFrame(metadata_list)
df = pd.concat([metadata_df, df], axis=1)

print(f"   Generated {len(df)} feature sets")

# ============================================================================
# AUDIT: Injury Features
# ============================================================================
print("\n" + "=" * 80)
print("INJURY FEATURES AUDIT")
print("=" * 80)

injury_features = ['injury_impact_abs', 'injury_impact_diff']

for col in injury_features:
    if col not in df.columns:
        print(f"❌ Column '{col}' not found.")
        continue
    
    # Calculate sparsity
    active_games = df[df[col] != 0]
    sparsity = 1 - (len(active_games) / len(df))
    
    print(f"\n--- {col} ---")
    print(f"  Zeros (Healthy/No Data): {sparsity:.1%}")
    print(f"  Active Games: {len(active_games)} ({len(active_games)/len(df):.1%})")
    
    if len(active_games) > 0:
        print(f"  Mean Impact (when active): {active_games[col].mean():.2f}")
        print(f"  Median Impact: {active_games[col].median():.2f}")
        print(f"  Max Impact: {active_games[col].max():.2f}")
        print(f"  Std Dev: {active_games[col].std():.2f}")
    else:
        print("  ⚠️ CRITICAL: This feature is all zeros.")

# Reality check - Top injured games
print("\n" + "=" * 80)
print("🏥 TOP 10 MOST 'INJURED' GAMES")
print("=" * 80)

if 'injury_impact_abs' in df.columns:
    top_injuries = df[df['injury_impact_abs'] > 0].sort_values(
        by='injury_impact_abs', ascending=False
    ).head(10)
    
    print(f"\n{'Date':<12} {'Home':<5} {'Away':<5} {'Score':<10} {'Injury Impact':<15}")
    print("-" * 80)
    for _, row in top_injuries.iterrows():
        score = f"{row['home_score']}-{row['away_score']}"
        impact = row['injury_impact_abs']
        print(f"{row['date']:<12} {row['home_team']:<5} {row['away_team']:<5} {score:<10} {impact:<15.2f}")

# ============================================================================
# AUDIT: Fatigue Features
# ============================================================================
print("\n" + "=" * 80)
print("FATIGUE FEATURES AUDIT")
print("=" * 80)

fatigue_features = ['home_3in4', 'away_3in4', 'home_back_to_back', 'away_back_to_back', 
                    'fatigue_mismatch', 'rest_advantage']

for col in fatigue_features:
    if col not in df.columns:
        print(f"❌ Column '{col}' not found.")
        continue
    
    # For binary features
    if df[col].nunique() <= 2:
        positive_games = df[df[col] == 1]
        rate = len(positive_games) / len(df)
        
        print(f"\n--- {col} ---")
        print(f"  Positive Rate: {rate:.1%} ({len(positive_games)} games)")
        
        if col in ['home_3in4', 'away_3in4']:
            expected = "~12-18%"
        elif col in ['home_back_to_back', 'away_back_to_back']:
            expected = "~17-20%"
        elif col == 'fatigue_mismatch':
            expected = "~5-10%"
        else:
            expected = "varies"
        
        print(f"  Expected NBA Rate: {expected}")
        
    else:
        # Continuous features
        active_games = df[df[col] != 0]
        sparsity = 1 - (len(active_games) / len(df))
        
        print(f"\n--- {col} ---")
        print(f"  Zeros: {sparsity:.1%}")
        print(f"  Active Games: {len(active_games)} ({len(active_games)/len(df):.1%})")
        
        if len(active_games) > 0:
            print(f"  Mean (when active): {active_games[col].mean():.2f}")
            print(f"  Max: {active_games[col].max():.2f}")

# ============================================================================
# COMPARATIVE MAGNITUDE CHECK
# ============================================================================
print("\n" + "=" * 80)
print("MAGNITUDE COMPARISON (All Features)")
print("=" * 80)
print("\nChecking if injury/fatigue signals are comparable to other features...")

comparison_features = ['ewma_efg_diff', 'def_elo_diff', 'injury_impact_diff', 
                       'injury_impact_abs', 'away_3in4', 'fatigue_mismatch']

print(f"\n{'Feature':<30} {'Mean':<10} {'Std Dev':<10} {'Max':<10} {'Range'}")
print("-" * 80)

for col in comparison_features:
    if col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        max_val = df[col].max()
        min_val = df[col].min()
        range_val = max_val - min_val
        print(f"{col:<30} {mean_val:<10.3f} {std_val:<10.3f} {max_val:<10.3f} {range_val:.3f}")

print("\n" + "=" * 80)
print("✅ AUDIT COMPLETE")
print("=" * 80)
print("\nInterpretation Guide:")
print("  • Injury active rate 30-60% = GOOD (enough signal without noise)")
print("  • 3-in-4 rate 12-18% = GOOD (matches NBA schedule reality)")
print("  • Back-to-back 17-20% = GOOD (typical NBA frequency)")
print("  • Max injury impact 15-30 = GOOD (capped properly)")
print("  • Feature magnitudes comparable = GOOD (model can learn)")
