"""
Data Cleaning & Repair Script
Fixes critical data quality issues identified in audit:
1. Rest Days Bug (286 days = off-season contamination)
2. ELO Inflation (10.7% drift over time)
3. ELO Diff Outliers (>400 point spreads)
"""

import pandas as pd
import numpy as np

print("="*70)
print("DATA CLEANING & REPAIR")
print("="*70)

# Load datasets
datasets = {
    'training_data_with_features.csv': 'Original (36 features)',
    'training_data_lean.csv': 'Lean (21 features)',
    'training_data_enhanced.csv': 'Enhanced (29 features)'
}

for filename, description in datasets.items():
    filepath = f"data/{filename}"
    
    try:
        print(f"\n{'='*70}")
        print(f"CLEANING: {description}")
        print(f"File: {filepath}")
        print(f"{'='*70}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded: {len(df):,} games")
        
        # 1. FIX REST DAYS BUG
        print("\n1. FIXING REST DAYS BUG (Off-season Contamination)")
        print("-"*70)
        
        if 'away_rest_days' in df.columns:
            before_away = df['away_rest_days'].describe()
            print(f"BEFORE - away_rest_days: max={df['away_rest_days'].max():.0f}")
            
            # Reset season openers (>15 days) to 3 days (neutral)
            # Rationale: Not "max rested" (15) because of rust, not 0 because they did rest
            season_openers_away = df['away_rest_days'] > 15
            df.loc[season_openers_away, 'away_rest_days'] = 3
            print(f"  → Fixed {season_openers_away.sum():,} season opener games")
            print(f"AFTER  - away_rest_days: max={df['away_rest_days'].max():.0f}")
        
        if 'home_rest_days' in df.columns:
            before_home = df['home_rest_days'].describe()
            print(f"\nBEFORE - home_rest_days: max={df['home_rest_days'].max():.0f}")
            
            season_openers_home = df['home_rest_days'] > 15
            df.loc[season_openers_home, 'home_rest_days'] = 3
            print(f"  → Fixed {season_openers_home.sum():,} season opener games")
            print(f"AFTER  - home_rest_days: max={df['home_rest_days'].max():.0f}")
        
        # 2. FIX ELO INFLATION
        print("\n2. FIXING ELO INFLATION (Z-Score Normalization per Season)")
        print("-"*70)
        
        df['season_year'] = pd.to_datetime(df['date']).dt.year
        
        # Find ELO columns (not diffs)
        elo_cols = [c for c in df.columns if 'elo' in c.lower() and 'diff' not in c.lower()]
        
        for col in elo_cols:
            print(f"\n{col}:")
            
            # Show before
            before_mean = df.groupby('season_year')[col].mean()
            print(f"  BEFORE - Mean by season:")
            print(f"    2015: {before_mean.get(2015, 0):.1f}")
            print(f"    2025: {before_mean.get(2025, 0):.1f}")
            print(f"    Drift: {before_mean.get(2025, 0) - before_mean.get(2015, 0):+.1f}")
            
            # Apply z-score normalization per season, then scale to 1500 ± std
            for year in df['season_year'].unique():
                mask = df['season_year'] == year
                season_data = df.loc[mask, col]
                
                # Z-score: (x - mean) / std
                season_mean = season_data.mean()
                season_std = season_data.std()
                
                if season_std > 0:
                    z_scores = (season_data - season_mean) / season_std
                    # Rescale to 1500 ± 100 (typical ELO spread)
                    df.loc[mask, col] = 1500 + (z_scores * 100)
            
            # Show after
            after_mean = df.groupby('season_year')[col].mean()
            print(f"  AFTER  - Mean by season:")
            print(f"    2015: {after_mean.get(2015, 0):.1f}")
            print(f"    2025: {after_mean.get(2025, 0):.1f}")
            print(f"    Drift: {after_mean.get(2025, 0) - after_mean.get(2015, 0):+.1f}")
            print(f"  ✓ Normalized to 1500 ± 100")
        
        # 3. RECALCULATE ELO DIFFS (after fixing base ELO)
        print("\n3. RECALCULATING ELO DIFFS")
        print("-"*70)
        
        # Note: We can't recalculate diffs without home/away ELO separately
        # Instead, we'll clip the existing diffs
        elo_diff_cols = [c for c in df.columns if 'elo' in c.lower() and 'diff' in c.lower()]
        
        for col in elo_diff_cols:
            before_range = (df[col].min(), df[col].max())
            print(f"\n{col}:")
            print(f"  BEFORE: [{before_range[0]:.1f}, {before_range[1]:.1f}]")
            
            # Clip extreme outliers to ±400
            df[col] = df[col].clip(-400, 400)
            
            after_range = (df[col].min(), df[col].max())
            print(f"  AFTER:  [{after_range[0]:.1f}, {after_range[1]:.1f}]")
            
            clipped_count = ((df[col] == -400) | (df[col] == 400)).sum()
            print(f"  → Clipped {clipped_count:,} extreme outliers")
        
        # 4. RECALCULATE INTERACTION FEATURES (if they exist)
        print("\n4. RECALCULATING INTERACTION FEATURES")
        print("-"*70)
        
        # Since we modified ELO, we need to recalculate interactions involving ELO
        if 'form_x_defense' in df.columns and 'def_elo_diff' in df.columns and 'ewma_tov_diff' in df.columns:
            df['form_x_defense'] = df['def_elo_diff'] * df['ewma_tov_diff']
            print("  ✓ Recalculated: form_x_defense = def_elo_diff × ewma_tov_diff")
        
        if 'efficiency_x_pace' in df.columns and 'ewma_efg_diff' in df.columns and 'ewma_pace_diff' in df.columns:
            df['efficiency_x_pace'] = df['ewma_efg_diff'] * df['ewma_pace_diff']
            print("  ✓ Recalculated: efficiency_x_pace = ewma_efg_diff × ewma_pace_diff")
        
        if 'tired_altitude' in df.columns and 'altitude_game' in df.columns and 'away_back_to_back' in df.columns:
            df['tired_altitude'] = df['altitude_game'] * df['away_back_to_back']
            print("  ✓ Recalculated: tired_altitude = altitude_game × away_back_to_back")
        
        # 5. VALIDATE REPAIRS
        print("\n5. VALIDATION")
        print("-"*70)
        
        issues = []
        
        # Check rest days
        if 'away_rest_days' in df.columns:
            if df['away_rest_days'].max() > 15:
                issues.append(f"away_rest_days still has outliers: max={df['away_rest_days'].max()}")
            else:
                print("  ✓ away_rest_days: All values ≤15")
        
        if 'home_rest_days' in df.columns:
            if df['home_rest_days'].max() > 15:
                issues.append(f"home_rest_days still has outliers: max={df['home_rest_days'].max()}")
            else:
                print("  ✓ home_rest_days: All values ≤15")
        
        # Check ELO normalization
        for col in elo_cols:
            yearly_mean = df.groupby('season_year')[col].mean()
            max_drift = yearly_mean.max() - yearly_mean.min()
            if max_drift > 50:
                issues.append(f"{col} drift still >50: {max_drift:.1f}")
            else:
                print(f"  ✓ {col}: Drift {max_drift:.1f} (within tolerance)")
        
        # Check ELO diff clipping
        for col in elo_diff_cols:
            if df[col].min() < -400 or df[col].max() > 400:
                issues.append(f"{col} still has values outside ±400")
            else:
                print(f"  ✓ {col}: All values within ±400")
        
        if issues:
            print("\n  ⚠️ VALIDATION ISSUES:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("\n  ✓ ALL VALIDATIONS PASSED")
        
        # 6. SAVE CLEANED DATA
        output_path = filepath.replace('.csv', '_cleaned.csv')
        df = df.drop(columns=['season_year'], errors='ignore')  # Remove temp column
        df.to_csv(output_path, index=False)
        print(f"\n6. SAVED CLEANED DATA")
        print(f"   → {output_path}")
        
    except FileNotFoundError:
        print(f"\n⚠️ File not found: {filepath}")
        continue

print("\n" + "="*70)
print("CLEANING COMPLETE")
print("="*70)
print("\nFIXES APPLIED:")
print("  ✓ Rest days capped at 15 (season openers reset to 3)")
print("  ✓ ELO normalized to 1500 ± 100 per season (drift eliminated)")
print("  ✓ ELO diffs clipped to ±400 (extreme outliers removed)")
print("  ✓ Interaction features recalculated")
print("\nCLEANED FILES:")
print("  - data/training_data_with_features_cleaned.csv")
print("  - data/training_data_lean_cleaned.csv")
print("  - data/training_data_enhanced_cleaned.csv")
print("\nNEXT STEPS:")
print("  1. Run audit again: python scripts/audit_data_quality.py")
print("  2. Train on cleaned data: python scripts/train_lean_model.py")
print("  3. Compare AUC: cleaned vs original")
print("="*70)
