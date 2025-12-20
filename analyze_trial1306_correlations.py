"""
Correlation Analysis for Trial 1306 Features
Identifies multicollinearity issues in the 22-feature model
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Trial 1306 features (in order)
TRIAL1306_FEATURES = [
    'home_composite_elo',
    'away_composite_elo',
    'off_elo_diff',
    'def_elo_diff',
    'net_fatigue_score',
    'ewma_efg_diff',
    'ewma_pace_diff',
    'ewma_tov_diff',
    'ewma_orb_diff',
    'ewma_vol_3p_diff',
    'injury_matchup_advantage',
    'ewma_chaos_home',
    'ewma_foul_synergy_home',
    'total_foul_environment',
    'league_offensive_context',
    'season_progress',
    'pace_efficiency_interaction',
    'projected_possession_margin',
    'three_point_matchup',
    'net_free_throw_advantage',
    'star_power_leverage',
    'offense_vs_defense_matchup'
]

def load_training_data():
    """Load the training data used for Trial 1306"""
    data_path = Path("data/training_data_GOLD_ELO_22_features.csv")
    
    if not data_path.exists():
        # Try alternative paths
        alt_paths = [
            "data/training_data_with_temporal_features_43feat.csv",
            "data/training_data_with_features.csv",
            "training_data_with_features.csv"
        ]
        for alt in alt_paths:
            if Path(alt).exists():
                data_path = Path(alt)
                break
    
    if not data_path.exists():
        raise FileNotFoundError("Cannot find training data file")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows")
    
    return df

def analyze_correlations(df):
    """Perform comprehensive correlation analysis"""
    
    # Select only Trial 1306 features
    available_features = [f for f in TRIAL1306_FEATURES if f in df.columns]
    missing_features = [f for f in TRIAL1306_FEATURES if f not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Missing features: {missing_features}")
    
    print(f"\n✓ Analyzing {len(available_features)} features")
    
    X = df[available_features].copy()
    
    # Remove any rows with NaN
    X = X.dropna()
    print(f"✓ {len(X):,} samples after dropping NaN")
    
    # Compute correlation matrix
    print("\n" + "="*60)
    print("CORRELATION MATRIX")
    print("="*60)
    corr_matrix = X.corr()
    
    # Save full correlation matrix
    corr_matrix.to_csv('trial1306_correlation_matrix.csv')
    print("✓ Saved: trial1306_correlation_matrix.csv")
    
    # Find high correlations (excluding diagonal)
    print("\n" + "="*60)
    print("HIGH CORRELATIONS (|r| > 0.7)")
    print("="*60)
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', 
                                                            key=abs, 
                                                            ascending=False)
        print(high_corr_df.to_string(index=False))
        high_corr_df.to_csv('trial1306_high_correlations.csv', index=False)
        print(f"\n⚠️  Found {len(high_corr)} high correlation pairs")
        print("✓ Saved: trial1306_high_correlations.csv")
    else:
        print("✓ No high correlations found (all |r| < 0.7)")
    
    # Find moderate correlations (0.5 < |r| < 0.7)
    print("\n" + "="*60)
    print("MODERATE CORRELATIONS (0.5 < |r| < 0.7)")
    print("="*60)
    
    moderate_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if 0.5 < abs(corr_val) < 0.7:
                moderate_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if moderate_corr:
        moderate_corr_df = pd.DataFrame(moderate_corr).sort_values('Correlation',
                                                                    key=abs,
                                                                    ascending=False)
        print(moderate_corr_df.to_string(index=False))
        print(f"\n✓ Found {len(moderate_corr)} moderate correlation pairs")
    else:
        print("✓ No moderate correlations found")
    
    # Compute VIF (Variance Inflation Factor)
    print("\n" + "="*60)
    print("VARIANCE INFLATION FACTOR (VIF)")
    print("="*60)
    print("VIF > 10 indicates strong multicollinearity")
    print("VIF > 5 may be concerning")
    
    from sklearn.linear_model import LinearRegression
    
    vif_data = []
    for i, feature in enumerate(available_features):
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
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    print(vif_df.to_string(index=False))
    vif_df.to_csv('trial1306_vif_analysis.csv', index=False)
    print("\n✓ Saved: trial1306_vif_analysis.csv")
    
    # Flag problematic VIF values
    high_vif = vif_df[vif_df['VIF'] > 10]
    if not high_vif.empty:
        print(f"\n⚠️  WARNING: {len(high_vif)} features with VIF > 10:")
        for _, row in high_vif.iterrows():
            print(f"  - {row['Feature']}: VIF = {row['VIF']:.2f}")
    else:
        print("\n✓ All VIF values < 10 (acceptable multicollinearity)")
    
    # Create correlation heatmap
    print("\n" + "="*60)
    print("GENERATING HEATMAP")
    print("="*60)
    
    plt.figure(figsize=(16, 14))
    
    # Use a diverging colormap
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
    
    plt.title('Trial 1306: Feature Correlation Matrix\n(Lower Triangle)', 
              fontsize=16, 
              pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    plt.savefig('trial1306_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: trial1306_correlation_heatmap.png")
    plt.close()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total features: {len(available_features)}")
    print(f"High correlations (|r| > 0.7): {len(high_corr)}")
    print(f"Moderate correlations (0.5 < |r| < 0.7): {len(moderate_corr)}")
    print(f"Features with VIF > 10: {len(high_vif)}")
    print(f"Features with VIF > 5: {len(vif_df[vif_df['VIF'] > 5])}")
    print(f"Mean VIF: {vif_df['VIF'].mean():.2f}")
    print(f"Max VIF: {vif_df['VIF'].max():.2f} ({vif_df.iloc[0]['Feature']})")
    
    return corr_matrix, vif_df, high_corr, moderate_corr

if __name__ == "__main__":
    print("="*60)
    print("TRIAL 1306 CORRELATION ANALYSIS")
    print("="*60)
    
    try:
        df = load_training_data()
        corr_matrix, vif_df, high_corr, moderate_corr = analyze_correlations(df)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        print("  1. trial1306_correlation_matrix.csv")
        print("  2. trial1306_correlation_heatmap.png")
        print("  3. trial1306_vif_analysis.csv")
        if high_corr:
            print("  4. trial1306_high_correlations.csv")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
