"""
K-FACTOR SWEEP: Determine Optimal ELO Reactivity
Tests whether "momentum" (K=32) or "reputation" (K=15) is better for NBA prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# The "Sweep" Range:
# 10-15: Stable (Reputation)
# 32:    Your Old Model (Momentum)
# 40+:   High Volatility (Recency Bias)
K_VALUES_TO_TEST = [10, 15, 20, 24, 28, 30, 32, 35, 40, 50, 60]

# Standard ELO constants
START_ELO = 1500
HOME_ADVANTAGE = 100 

# YOUR DATA FILE
# Ensure this CSV has: date, home_team, away_team, home_score, away_score
FILE_PATH = 'data/training_data_GOLD_ELO_22_features.csv'  # Updated to your current data

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data(path):
    print(f"📂 Loading data from {path}...")
    try:
        df = pd.read_csv(path)
        
        # Standardize Date Column
        if 'date' in df.columns:
            df['game_date'] = pd.to_datetime(df['date'])
        elif 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        else:
            raise ValueError("Could not find 'date' or 'game_date' column")
        
        # Check if we have target column (moneyline win)
        if 'target_moneyline_win' in df.columns:
            df['home_win'] = df['target_moneyline_win']
        elif 'target_spread_cover' in df.columns:
            # Fallback to spread cover if no moneyline
            df['home_win'] = df['target_spread_cover']
        else:
            # Calculate from scores if available
            if 'home_score' in df.columns and 'away_score' in df.columns:
                df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
            else:
                raise ValueError("Could not find win/loss target column")

        # CRITICAL: ELO must be calculated chronologically
        df = df.sort_values('game_date').reset_index(drop=True)
        print(f"✅ Loaded {len(df):,} games sorted by date.")
        print(f"   Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        print(f"   Home win rate: {df['home_win'].mean():.1%}")
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# 2. THE ELO ENGINE
# ==========================================
def run_elo_simulation(df, k_factor):
    """
    Replays the entire history of the NBA using a specific K-Factor.
    Returns the ELO Differential (Feature) and the Win/Loss (Target).
    """
    # Initialize every team in the dataset to 1500
    teams = set(df['home_team']).union(set(df['away_team']))
    elo_dict = {team: START_ELO for team in teams}
    
    elo_diffs = []
    targets = []
    
    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        h_win = row['home_win']
        
        # 1. Get Current Ratings
        h_elo = elo_dict[home]
        a_elo = elo_dict[away]
        
        # 2. PREDICT (Before updating)
        # This is the feature the model would see
        diff = (h_elo + HOME_ADVANTAGE) - a_elo
        elo_diffs.append(diff)
        
        # 3. RECORD RESULT
        targets.append(h_win)
        
        # 4. UPDATE RATINGS (The Math)
        h_expected = 1 / (1 + 10 ** (-(diff) / 400))
        
        # New Rating = Old Rating + K * (Actual - Expected)
        elo_dict[home] = h_elo + k_factor * (h_win - h_expected)
        elo_dict[away] = a_elo + k_factor * ((1 - h_win) - (1 - h_expected))
        
    return np.array(elo_diffs).reshape(-1, 1), np.array(targets)

# ==========================================
# 3. THE SWEEP EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_data(FILE_PATH)
    
    if df is not None:
        print("\n🧪 STARTING K-FACTOR SWEEP...")
        print(f"{'K-FACTOR':<10} | {'LOG LOSS':<10} | {'ACCURACY':<10} | {'AUC':<10}")
        print("-" * 50)

        results = []

        for k in K_VALUES_TO_TEST:
            # 1. Generate History
            X, y = run_elo_simulation(df, k)
            
            # 2. Train/Test Split (Time Series Split)
            # We train on the past, test on the recent future (last 25%)
            split_idx = int(len(X) * 0.75)
            
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 3. Evaluate Signal Quality
            # We use Logistic Regression to test purely how good the ELO is 
            # at separating winners from losers.
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            preds = model.predict_proba(X_test)[:, 1]
            loss = log_loss(y_test, preds)
            acc = accuracy_score(y_test, model.predict(X_test))
            
            # Calculate AUC for additional insight
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, preds)
            
            print(f"{k:<10} | {loss:.5f}    | {acc:.4%}    | {auc:.4f}")
            results.append({'k': k, 'loss': loss, 'acc': acc, 'auc': auc})

        # 4. Determine Winner
        best = min(results, key=lambda x: x['loss'])
        print("-" * 50)
        print(f"🏆 OPTIMAL K-FACTOR: {best['k']}")
        print(f"   Log Loss: {best['loss']:.5f}")
        print(f"   Accuracy: {best['acc']:.2%}")
        print(f"   AUC:      {best['auc']:.4f}")
        
        # Show comparison to K=32 (your old model) and K=15 (gold standard)
        k32_result = next((r for r in results if r['k'] == 32), None)
        k15_result = next((r for r in results if r['k'] == 15), None)
        
        if k32_result and k15_result:
            print("\n📊 KEY COMPARISONS:")
            print(f"   K=32 (Your Old Model):  {k32_result['loss']:.5f}")
            print(f"   K=15 (Gold Standard):   {k15_result['loss']:.5f}")
            print(f"   Difference:             {k32_result['loss'] - k15_result['loss']:+.5f}")
            
            if k32_result['loss'] < k15_result['loss']:
                print(f"\n   ✅ K=32 wins! Momentum matters in the NBA.")
            else:
                print(f"\n   ⚠️  K=15 wins. Reputation > Momentum for this dataset.")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('models/experimental/k_factor_sweep_results.csv', index=False)
        print(f"\n✓ Saved results: models/experimental/k_factor_sweep_results.csv")
