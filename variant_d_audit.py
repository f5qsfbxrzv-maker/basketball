"""
VARIANT D VIF AUDIT
Verifies that the 18-feature consolidated model has resolved all collinearity issues
Target: All VIF < 10, preferably < 5
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
# VARIANT D: The "Consolidated" Feature Set (18 features)
# Based on our actual training - removed: ewma_orb_diff, ewma_tov_diff, away_composite_elo, ewma_foul_synergy_home
VARIANT_D_FEATURES = [
    # --- ELO FOUNDATION (K=32 validated) ---
    'home_composite_elo',           # The ANCHOR (away_composite removed)
    'off_elo_diff',                 # Offensive mismatch (22.5% importance)
    'def_elo_diff',                 # Defensive mismatch
    
    # --- POSSESSION ---
    'projected_possession_margin',  # The SUM (ewma_orb_diff/tov_diff removed)
    
    # --- PACE & FATIGUE ---
    'ewma_pace_diff',
    'net_fatigue_score',
    
    # --- SHOOTING ---
    'ewma_efg_diff',
    'ewma_vol_3p_diff',
    'three_point_matchup',
    'ewma_chaos_home',              # Variance/unpredictability
    
    # --- INJURIES & CONTEXT ---
    'injury_matchup_advantage',
    'star_power_leverage',
    'season_progress',
    'league_offensive_context',
    
    # --- FOULS & REFS ---
    'total_foul_environment',       # General whistle density (ewma_foul_synergy_home removed)
    'net_free_throw_advantage',     # Directional FT edge
    
    # --- INTERACTIONS ---
    'pace_efficiency_interaction',
    'offense_vs_defense_matchup'
]

# FILE PATH
FILE_PATH = 'data/training_data_GOLD_ELO_22_features.csv'

# ==========================================
# VIF CALCULATION
# ==========================================
def calculate_vif(df, features):
    """Calculate VIF for each feature using sklearn LinearRegression"""
    X = df[features].copy()
    X = X.dropna()
    
    vif_data = []
    
    for i, feature in enumerate(features):
        # Fit model with this feature as target, others as predictors
        y = X[feature]
        X_others = X.drop(columns=[feature])
        
        lr = LinearRegression()
        lr.fit(X_others, y)
        r_squared = lr.score(X_others, y)
        
        # VIF = 1 / (1 - R²)
        if r_squared < 0.9999:  # Avoid division issues
            vif = 1 / (1 - r_squared)
        else:
            vif = 999.99  # Essentially perfect collinearity
        
        vif_data.append({
            'Feature': feature,
            'VIF': vif,
            'R²': r_squared
        })
    
    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

# ==========================================
# CORRELATION ANALYSIS
# ==========================================
def analyze_correlations(df, features):
    """Find high correlations between features"""
    X = df[features].dropna()
    corr_matrix = X.corr()
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:  # Report moderate+ correlations
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    return pd.DataFrame(high_corr).sort_values('Correlation', key=abs, ascending=False) if high_corr else None

# ==========================================
# VISUALIZATION
# ==========================================
def create_heatmap(df, features):
    """Create correlation heatmap for Variant D"""
    X = df[features].dropna()
    corr_matrix = X.corr()
    
    plt.figure(figsize=(14, 12))
    
    # Use triangular mask
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=False,
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Variant D (18 Features): Correlation Matrix\n(Lower Triangle)', 
              fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    output_path = 'models/experimental/variant_d_correlation_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmap: {output_path}")
    plt.close()

# ==========================================
# MAIN AUDIT
# ==========================================
def run_audit():
    print("="*60)
    print("📊 VARIANT D VIF AUDIT")
    print("="*60)
    print(f"Testing {len(VARIANT_D_FEATURES)} features")
    print("Target: All VIF < 10 (ideally < 5)")
    print("="*60)

    try:
        df = pd.read_csv(FILE_PATH)
        print(f"✓ Loaded {len(df):,} games")
        
        # Check for missing columns
        missing = [col for col in VARIANT_D_FEATURES if col not in df.columns]
        if missing:
            print(f"\n❌ MISSING COLUMNS: {missing}")
            print("Available columns:", df.columns.tolist())
            return

        # Calculate VIF
        print("\n" + "="*60)
        print("VIF SCORES")
        print("="*60)
        
        vif_df = calculate_vif(df, VARIANT_D_FEATURES)
        
        # Display results with color coding
        print(f"{'FEATURE':<35} | {'VIF':<8} | {'R²':<8} | {'STATUS'}")
        print("-" * 70)
        
        critical_count = 0
        warning_count = 0
        clean_count = 0
        
        for _, row in vif_df.iterrows():
            feat = row['Feature']
            vif = row['VIF']
            r2 = row['R²']
            
            if vif < 5:
                status = "✅ CLEAN"
                clean_count += 1
            elif vif < 10:
                status = "⚠️ WATCH"
                warning_count += 1
            else:
                status = "🔴 CRITICAL"
                critical_count += 1
            
            print(f"{feat:<35} | {vif:>6.2f}  | {r2:>6.4f}  | {status}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"✅ Clean (VIF < 5):       {clean_count}")
        print(f"⚠️  Warning (VIF 5-10):    {warning_count}")
        print(f"🔴 Critical (VIF > 10):   {critical_count}")
        print(f"\nMean VIF: {vif_df['VIF'].mean():.2f}")
        print(f"Max VIF:  {vif_df['VIF'].max():.2f} ({vif_df.iloc[0]['Feature']})")
        
        # Comparison to baseline
        print("\n" + "="*60)
        print("COMPARISON TO TRIAL 1306 BASELINE")
        print("="*60)
        baseline_max_vif = 999.99  # From original analysis
        baseline_features = 22
        
        print(f"Baseline (22 features): Max VIF = {baseline_max_vif:.2f}")
        print(f"Variant D (18 features): Max VIF = {vif_df['VIF'].max():.2f}")
        print(f"\nFeature reduction: {baseline_features} → {len(VARIANT_D_FEATURES)} ({-4} features)")
        
        if vif_df['VIF'].max() < 10:
            print("\n✅ SUCCESS: All VIF < 10 - Collinearity resolved!")
        elif vif_df['VIF'].max() < baseline_max_vif / 10:
            print("\n✨ MAJOR IMPROVEMENT: VIF reduced by >90%")
        else:
            print("\n⚠️  WARNING: Some features still show high collinearity")
        
        # Correlation analysis
        print("\n" + "="*60)
        print("MODERATE+ CORRELATIONS (|r| > 0.5)")
        print("="*60)
        
        corr_df = analyze_correlations(df, VARIANT_D_FEATURES)
        if corr_df is not None and not corr_df.empty:
            print(corr_df.to_string(index=False))
            print(f"\nFound {len(corr_df)} moderate+ correlation pairs")
        else:
            print("✓ No moderate+ correlations found (all |r| < 0.5)")
        
        # Save results
        vif_df.to_csv('models/experimental/variant_d_vif_analysis.csv', index=False)
        print(f"\n✓ Saved VIF analysis: models/experimental/variant_d_vif_analysis.csv")
        
        if corr_df is not None and not corr_df.empty:
            corr_df.to_csv('models/experimental/variant_d_correlations.csv', index=False)
            print(f"✓ Saved correlations: models/experimental/variant_d_correlations.csv")
        
        # Create heatmap
        create_heatmap(df, VARIANT_D_FEATURES)
        
        # Final verdict
        print("\n" + "="*60)
        print("FINAL VERDICT")
        print("="*60)
        
        if critical_count == 0 and warning_count <= 2:
            print("✅ VARIANT D APPROVED FOR PRODUCTION TESTING")
            print("   - No critical collinearity issues")
            print("   - Feature reduction successful")
            print("   - Ready for backtest validation")
        elif critical_count == 0:
            print("⚠️  VARIANT D: CONDITIONAL APPROVAL")
            print(f"   - {warning_count} features with VIF 5-10")
            print("   - Consider further monitoring")
            print("   - Proceed with backtest validation")
        else:
            print("🔴 VARIANT D: NEEDS REVISION")
            print(f"   - {critical_count} features with VIF > 10")
            print("   - Additional feature pruning recommended")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_audit()
