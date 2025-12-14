"""
Data Quality Audit - Sanity Checks & Validation
1. Impossible Values (infinities, NaNs, outliers)
2. Duplicate Information (perfect correlations)
3. ELO Inflation (mean drift over time)
4. Feature Logic Verification
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DATA QUALITY AUDIT - SANITY CHECKS")
print("="*70)

# Load all datasets
datasets = {
    'Original (36 features)': 'data/training_data_with_features.csv',
    'Lean (21 features)': 'data/training_data_lean.csv',
    'Enhanced (29 features)': 'data/training_data_enhanced.csv'
}

for name, path in datasets.items():
    try:
        df = pd.read_csv(path)
        print(f"\n{'='*70}")
        print(f"AUDITING: {name}")
        print(f"File: {path}")
        print(f"{'='*70}")
        
        # 1. INFINITIES & NANS
        print("\n1. INFINITIES & NaNs CHECK")
        print("-"*70)
        nulls = df.isnull().sum().sum()
        numeric_cols = df.select_dtypes(include=np.number).columns
        infs = np.isinf(df[numeric_cols]).sum().sum()
        
        print(f"Total NaNs:  {nulls:,}")
        print(f"Total Infs:  {infs:,}")
        
        if nulls > 0:
            print("\n⚠️ Columns with NaNs:")
            null_cols = df.isnull().sum()
            null_cols = null_cols[null_cols > 0].sort_values(ascending=False)
            for col, count in null_cols.items():
                print(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")
        
        if infs > 0:
            print("\n⚠️ Columns with Infinities:")
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    print(f"  {col}: {inf_count:,}")
        
        # 2. IMPOSSIBLE VALUES
        print("\n2. IMPOSSIBLE VALUES CHECK")
        print("-"*70)
        
        # Define expected ranges for key features
        checks = {
            'ewma_efg_diff': (-0.3, 0.3, 'eFG% diff >30% is suspicious'),
            'ewma_pace_diff': (-20, 20, 'Pace diff >20 is very rare'),
            'away_rest_days': (0, 15, '>15 days rest suggests All-Star break/data error'),
            'home_rest_days': (0, 15, '>15 days rest suggests All-Star break/data error'),
            'off_elo_diff': (-400, 400, 'ELO diff >400 is extreme'),
            'def_elo_diff': (-400, 400, 'ELO diff >400 is extreme'),
            'home_composite_elo': (1200, 1800, 'ELO outside 1200-1800 suggests inflation/deflation'),
            'ewma_tov_diff': (-0.15, 0.15, 'TOV% diff >15% is suspicious'),
            'away_back_to_back': (0, 1, 'Back-to-back should be binary 0/1'),
            'home_back_to_back': (0, 1, 'Back-to-back should be binary 0/1'),
            'rolling_7_day_distance': (0, 15000, '>15k miles in 7 days is impossible'),
            'body_clock_lag': (0, 3, 'Max time zone shift is 3 hours'),
        }
        
        violations = []
        for col, (min_expected, max_expected, reason) in checks.items():
            if col in df.columns:
                actual_min = df[col].min()
                actual_max = df[col].max()
                
                if actual_min < min_expected or actual_max > max_expected:
                    violations.append({
                        'feature': col,
                        'expected_range': f"[{min_expected}, {max_expected}]",
                        'actual_range': f"[{actual_min:.4f}, {actual_max:.4f}]",
                        'reason': reason
                    })
                    
                    print(f"\n⚠️ {col}:")
                    print(f"  Expected: [{min_expected}, {max_expected}]")
                    print(f"  Actual:   [{actual_min:.4f}, {actual_max:.4f}]")
                    print(f"  Reason:   {reason}")
                    
                    # Show outlier examples
                    outliers = df[(df[col] < min_expected) | (df[col] > max_expected)]
                    if len(outliers) > 0:
                        print(f"  Outlier count: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
                        if col in ['away_rest_days', 'home_rest_days'] and len(outliers) > 0:
                            print("\n  Sample outliers:")
                            sample = outliers[['date', 'home_team', 'away_team', col]].head(5)
                            for _, row in sample.iterrows():
                                print(f"    {row['date']}: {row['away_team']}@{row['home_team']} - {col}={row[col]}")
        
        if not violations:
            print("✓ All features within expected ranges")
        
        # 3. PERFECT CORRELATIONS (Duplicate Information)
        print("\n3. DUPLICATE INFORMATION CHECK")
        print("-"*70)
        
        feature_cols = [c for c in df.columns if c not in ['date','game_id','home_team','away_team','season','target_spread','target_spread_cover','target_moneyline_win','target_game_total','target_over_under','target_home_cover','target_over']]
        
        if len(feature_cols) > 1:
            corr_matrix = df[feature_cols].corr().abs()
            
            # Find perfect correlations (>0.99, excluding diagonal)
            perfect_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.99:
                        perfect_corr.append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            if perfect_corr:
                print(f"⚠️ Found {len(perfect_corr)} near-perfect correlations (>0.99):")
                for pair in sorted(perfect_corr, key=lambda x: x['correlation'], reverse=True):
                    print(f"\n  {pair['feature_1']}")
                    print(f"  ↔ {pair['feature_2']}")
                    print(f"  r = {pair['correlation']:.6f}")
                    
                    # Show if one is a linear transformation of the other
                    ratio = (df[pair['feature_1']] / df[pair['feature_2']].replace(0, np.nan)).dropna()
                    if ratio.std() < 0.001:  # Nearly constant ratio
                        print(f"  ⚠️ Linear relationship: {pair['feature_1']} = {ratio.mean():.4f} × {pair['feature_2']}")
                        print(f"     → One feature is redundant!")
            else:
                print("✓ No perfect correlations found")
        
        # 4. ELO INFLATION CHECK
        print("\n4. ELO INFLATION CHECK")
        print("-"*70)
        
        df['season_year'] = pd.to_datetime(df['date']).dt.year
        
        elo_cols = [c for c in feature_cols if 'elo' in c.lower() and 'diff' not in c.lower()]
        
        if elo_cols:
            for col in elo_cols:
                print(f"\n{col} by Season:")
                yearly_mean = df.groupby('season_year')[col].mean()
                yearly_std = df.groupby('season_year')[col].std()
                
                for year in sorted(yearly_mean.index):
                    print(f"  {year}: μ={yearly_mean[year]:>7.1f}, σ={yearly_std[year]:>6.1f}")
                
                # Check for drift
                first_year = yearly_mean.iloc[0]
                last_year = yearly_mean.iloc[-1]
                drift = last_year - first_year
                drift_pct = (drift / first_year) * 100
                
                if abs(drift_pct) > 5:
                    print(f"\n  ⚠️ ELO DRIFT DETECTED:")
                    print(f"     {yearly_mean.index[0]} → {yearly_mean.index[-1]}: {drift:+.1f} ({drift_pct:+.1f}%)")
                    print(f"     This suggests ELO inflation/deflation over time")
                else:
                    print(f"\n  ✓ ELO stable ({drift_pct:+.1f}% drift)")
        
        # 5. FEATURE LOGIC VERIFICATION
        print("\n5. FEATURE LOGIC VERIFICATION")
        print("-"*70)
        
        # Check interaction features
        interaction_features = [
            ('efficiency_x_pace', 'ewma_efg_diff', 'ewma_pace_diff'),
            ('tired_altitude', 'altitude_game', 'away_back_to_back'),
            ('form_x_defense', 'def_elo_diff', 'ewma_tov_diff')
        ]
        
        for interaction, component1, component2 in interaction_features:
            if interaction in df.columns and component1 in df.columns and component2 in df.columns:
                # Verify interaction = component1 * component2
                expected = df[component1] * df[component2]
                actual = df[interaction]
                
                # Check if they match (within floating point tolerance)
                matches = np.allclose(expected, actual, rtol=1e-5, atol=1e-8)
                
                if matches:
                    print(f"✓ {interaction} = {component1} × {component2}")
                else:
                    diff = (expected - actual).abs().max()
                    print(f"⚠️ {interaction} does NOT equal {component1} × {component2}")
                    print(f"   Max difference: {diff:.6f}")
        
        # Check binary features
        binary_features = ['away_back_to_back', 'home_back_to_back', 'altitude_game', 
                          'away_3in4', 'home_3in4', 'first_game_home_after_road_trip']
        
        for feat in binary_features:
            if feat in df.columns:
                unique_vals = df[feat].dropna().unique()
                if set(unique_vals).issubset({0, 1}):
                    print(f"✓ {feat} is binary (0/1)")
                else:
                    print(f"⚠️ {feat} is NOT binary: {sorted(unique_vals)[:10]}")
        
    except FileNotFoundError:
        print(f"\n⚠️ File not found: {path}")
        continue

print("\n" + "="*70)
print("AUDIT COMPLETE")
print("="*70)
print("\nRECOMMENDATIONS:")
print("1. Fix any ⚠️ warnings above before training")
print("2. Remove features with perfect correlation (redundant)")
print("3. Investigate ELO drift if >5%")
print("4. Handle outliers (clip, remove, or investigate source)")
print("="*70)
