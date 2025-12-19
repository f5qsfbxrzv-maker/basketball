"""Add injury_matchup_advantage to matchup_optimized dataset"""
import pandas as pd

print("=" * 80)
print("ADDING INJURY_MATCHUP_ADVANTAGE TO DATASET")
print("=" * 80)
print()

# Load dataset
INPUT_PATH = 'data/training_data_matchup_optimized.csv'
print(f"📂 Loading: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"✅ Loaded {len(df)} games")
print()

# Check required columns exist
required = ['injury_impact_diff', 'injury_shock_diff', 'star_mismatch']
missing = [c for c in required if c not in df.columns]

if missing:
    print(f"❌ Missing columns: {missing}")
    exit(1)

print("✅ All required injury features present")
print()

# Calculate injury_matchup_advantage
print("⚙️  Calculating injury_matchup_advantage...")
df['injury_matchup_advantage'] = (
    0.008127 * df['injury_impact_diff']
  - 0.023904 * df['injury_shock_diff']
  + 0.031316 * df['star_mismatch']
)
print("✅ Feature calculated")
print()

# Save
OUTPUT_PATH = 'data/training_data_matchup_with_injury_advantage.csv'
print(f"💾 Saving to: {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved {len(df)} rows with {len(df.columns)} columns")
print()

# Display stats
print("=" * 80)
print("INJURY_MATCHUP_ADVANTAGE STATISTICS")
print("=" * 80)
print()
print(df['injury_matchup_advantage'].describe())
print()

print("Most favorable to home (away injured):")
print(df.nlargest(3, 'injury_matchup_advantage')[['home_team', 'away_team', 'date', 'injury_matchup_advantage']])
print()

print("Most favorable to away (home injured):")
print(df.nsmallest(3, 'injury_matchup_advantage')[['home_team', 'away_team', 'date', 'injury_matchup_advantage']])
print()

print("✅ DATASET READY FOR TRAINING")
