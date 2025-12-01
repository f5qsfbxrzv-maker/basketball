"""
DIVERGENCE ENGINE - The Argument Solver
Finds games where your models disagree (the highest edge opportunities)
"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# --- CONFIG ---
MODEL_A_PATH = "models/production/best_model.joblib"
MODEL_B_PATH = "models/staging/candidate_model.joblib"
DATA_PATH = "data/processed/training_data_final.csv"
CONFIDENCE_THRESHOLD = 0.55

def analyze_divergence():
    """Find where models disagree - this is where edge lives"""
    print("=" * 80)
    print("🕵️ DIVERGENCE ENGINE - Finding Model Disagreements")
    print("=" * 80)
    
    # Load models and data
    try:
        print(f"\nLoading Model A: {MODEL_A_PATH}")
        model_a = joblib.load(MODEL_A_PATH)
        print(f"Loading Model B: {MODEL_B_PATH}")
        model_b = joblib.load(MODEL_B_PATH)
        print(f"Loading data: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # 1. Get Predictions
    # Align features (use columns that both models need)
    if hasattr(model_a, "feature_names_in_"):
        features = [c for c in model_a.feature_names_in_ if c in df.columns]
        X = df[features].fillna(0)
    else:
        X = df.select_dtypes(include=[np.number]).drop(columns=['WL'], errors='ignore').fillna(0)
    
    probs_a = model_a.predict_proba(X)[:, 1]
    probs_b = model_b.predict_proba(X)[:, 1]
    
    df['Prob_A'] = probs_a
    df['Prob_B'] = probs_b
    
    # 2. Define "Bet Signals"
    # 1 = Bet Home, -1 = Bet Away, 0 = Pass
    df['Signal_A'] = np.where(df['Prob_A'] > CONFIDENCE_THRESHOLD, 1, 
                              np.where(df['Prob_A'] < (1 - CONFIDENCE_THRESHOLD), -1, 0))
    
    df['Signal_B'] = np.where(df['Prob_B'] > CONFIDENCE_THRESHOLD, 1, 
                              np.where(df['Prob_B'] < (1 - CONFIDENCE_THRESHOLD), -1, 0))
    
    # 3. Find Conflicts
    conflicts = df[df['Signal_A'] != df['Signal_B']].copy()
    
    # Calculate disagreement magnitude
    conflicts['Diff'] = abs(conflicts['Prob_A'] - conflicts['Prob_B'])
    
    print(f"\n⚡ DIVERGENCE REPORT")
    print("=" * 80)
    print(f"Total Games: {len(df):,}")
    print(f"Total Disagreements: {len(conflicts):,} ({len(conflicts)/len(df)*100:.1f}% of games)")
    
    # 4. The "Bloodshed" Table (Complete disagreement)
    major_conflicts = conflicts[(conflicts['Signal_A'] != 0) & (conflicts['Signal_B'] != 0)]
    print(f"MAJOR CONFLICTS (Head-to-Head): {len(major_conflicts):,}")
    
    if not major_conflicts.empty:
        print("\n🥊 Top 5 Biggest Disagreements:")
        print("-" * 80)
        
        # Show relevant columns
        display_cols = ['Prob_A', 'Prob_B', 'Signal_A', 'Signal_B', 'Diff']
        if 'Date' in df.columns:
            display_cols.insert(0, 'Date')
        if 'Home' in df.columns and 'Away' in df.columns:
            display_cols.insert(1, 'Home')
            display_cols.insert(2, 'Away')
            
        available_cols = [c for c in display_cols if c in major_conflicts.columns]
        print(major_conflicts.sort_values('Diff', ascending=False)[available_cols].head(5).to_string(index=False))
        
        # Who was right in the conflicts?
        if 'WL' in df.columns:
            major_conflicts['A_Won'] = ((major_conflicts['Signal_A'] == 1) & (major_conflicts['WL'] == 1)) | \
                                       ((major_conflicts['Signal_A'] == -1) & (major_conflicts['WL'] == 0))
            
            major_conflicts['B_Won'] = ((major_conflicts['Signal_B'] == 1) & (major_conflicts['WL'] == 1)) | \
                                       ((major_conflicts['Signal_B'] == -1) & (major_conflicts['WL'] == 0))
            
            print("\n🏆 HEAD-TO-HEAD RESULTS (When they fought):")
            print("-" * 80)
            print(f"Model A Correct: {major_conflicts['A_Won'].sum():,} ({major_conflicts['A_Won'].mean()*100:.1f}%)")
            print(f"Model B Correct: {major_conflicts['B_Won'].sum():,} ({major_conflicts['B_Won'].mean()*100:.1f}%)")
            
            # Who wins in disagreements?
            if major_conflicts['A_Won'].sum() > major_conflicts['B_Won'].sum():
                print("\n✅ WINNER: Model A is better in disagreements")
            elif major_conflicts['B_Won'].sum() > major_conflicts['A_Won'].sum():
                print("\n✅ WINNER: Model B is better in disagreements")
            else:
                print("\n⚖️  TIE: Both models equally good in disagreements")
    
    # 5. Save divergence data for further analysis
    output_path = "logs/divergence_analysis.csv"
    Path("logs").mkdir(exist_ok=True)
    conflicts.to_csv(output_path, index=False)
    print(f"\n💾 Divergence data saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("💡 INTERPRETATION:")
    print("=" * 80)
    print("High divergence = Models see the game differently")
    print("If one model wins most disagreements → Trust that model more")
    print("Use divergence games to find MAXIMUM EDGE opportunities")
    print("=" * 80)

if __name__ == "__main__":
    analyze_divergence()
