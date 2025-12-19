import pandas as pd
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_PATH = 'data/training_data_matchup_with_injury_advantage.csv' # Our actual training file

# Columns to ignore (Metadata)
METADATA_COLS = ['date', 'team', 'opponent', 'game_id', 'target', 'season', 'moneyline_decimal', 
                 'target_moneyline_win', 'home_team', 'away_team', 'game_date']

# ==============================================================================
# DIAGNOSTIC ENGINE
# ==============================================================================
def run_diagnostics():
    print(f"===================================================================")
    print(f"FULL DATASET AUDIT")
    print(f"===================================================================\n")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: File not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows. Auditing {len(df.columns)} columns...\n")

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    
    dead_features = []
    suspicious_features = []
    
    print(f"{'FEATURE':<45} | {'STATUS':<10} | {'REASON'}")
    print("-" * 100)

    for col in feature_cols:
        data = df[col]
        
        # CHECK 1: ALL NULLS
        if data.isna().all():
            print(f"{col:<45} | ❌ DEAD     | ALL NULLS (Empty Column)")
            dead_features.append(col)
            continue
            
        # CHECK 2: CONSTANT VALUE (Variance = 0)
        # Note: dropna() ensures we don't count NaNs as variance
        if data.dropna().nunique() <= 1:
            unique_val = data.dropna().unique()
            val_str = str(unique_val[0]) if len(unique_val) > 0 else "None"
            print(f"{col:<45} | ❌ DEAD     | CONSTANT VALUE (All rows = {val_str})")
            dead_features.append(col)
            continue
            
        # CHECK 3: WRONG DATA TYPE (Object/String)
        if data.dtype == 'object':
            # Check if it looks like a number but is saved as string
            try:
                data.astype(float)
                print(f"{col:<45} | ⚠️ WARN     | TYPE MISMATCH (Saved as String, should be Float)")
                suspicious_features.append(col)
            except:
                print(f"{col:<45} | ❌ DEAD     | INVALID TYPE (Contains non-numeric strings)")
                dead_features.append(col)
            continue

        # CHECK 4: MOSTLY NULLS (>50% Missing)
        null_pct = data.isna().mean()
        if null_pct > 0.5:
            print(f"{col:<45} | ⚠️ WARN     | SPARSITY ({null_pct*100:.1f}% Nulls)")
            suspicious_features.append(col)
            continue
            
        # CHECK 5: MOSTLY ZEROS (>90% Zeros)
        # Only for numeric columns
        zero_pct = (data == 0).mean()
        if zero_pct > 0.90:
             print(f"{col:<45} | ⚠️ WARN     | MOSTLY ZEROS ({zero_pct*100:.1f}% Zeros)")
             suspicious_features.append(col)
             continue

        # If it passes all checks - show some stats
        mean_val = data.mean()
        std_val = data.std()
        print(f"{col:<45} | ✅ OK       | Mean={mean_val:.4f}, Std={std_val:.4f}")

    print("\n" + "="*100)
    print("SUMMARY REPORT")
    print("="*100)
    print(f"Total Columns: {len(df.columns)}")
    print(f"Feature Columns: {len(feature_cols)}")
    print(f"Metadata Columns: {len(METADATA_COLS)}")
    print()
    
    if dead_features:
        print(f"❌ DEAD FEATURES (Must Fix): {len(dead_features)}")
        for f in dead_features: print(f"   - {f}")
    else:
        print("✅ No Dead Features found.")
        
    if suspicious_features:
        print(f"\n⚠️ SUSPICIOUS FEATURES (Check Logic): {len(suspicious_features)}")
        for f in suspicious_features: print(f"   - {f}")
    
    if not dead_features and not suspicious_features:
        print("\n🎉 ALL FEATURES PASSED VALIDATION!")
        print("Your dataset is clean and ready for training.")

if __name__ == "__main__":
    run_diagnostics()
