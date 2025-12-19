"""
Test prediction for DAL vs DET after fixing away_composite_elo bug
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from src.features.feature_calculator_v5 import FeatureCalculatorV5
import sqlite3
import pandas as pd
import xgboost as xgb

# Configuration
DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"
MODEL_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\models\xgboost_22features_trial1306_20251215_212306.json"

print("=" * 80)
print("TESTING FIXED PREDICTION: DET @ DAL (2025-12-18)")
print("=" * 80)

# Load model (same as dashboard)
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Trial 1306 feature order
required_features = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'home_ats_streak', 'away_ats_streak', 'home_win_streak', 'away_win_streak',
    'home_rest_days', 'away_rest_days', 'home_b2b', 'away_b2b',
    'home_3in4', 'away_3in4', 'altitude_game', 'expected_pace',
    'home_blended_off_rating', 'away_blended_off_rating',
    'home_blended_def_rating', 'away_blended_def_rating',
    'home_injury_impact', 'away_injury_impact'
]

print(f"\n✅ Model loaded: Trial 1306")
print(f"✅ Required features: {len(required_features)}")
print(f"   Feature #1: {required_features[0]}")
print(f"   Feature #2: {required_features[1]}")

# Initialize feature calculator
feature_calc = FeatureCalculatorV5(db_path=DB_PATH)

print("\n" + "=" * 80)
print("CALCULATING FEATURES")
print("=" * 80)

# Calculate features for DAL (home) vs DET (away) on 12/18/2025
game_date = "2025-12-18"
features = feature_calc.calculate_game_features(
    home_team="DAL",
    away_team="DET",
    game_date=game_date
)

print(f"\n✅ Calculated {len(features)} features")
print("\n📊 KEY FEATURES:")
print(f"   home_composite_elo: {features.get('home_composite_elo', 'MISSING')}")
print(f"   away_composite_elo: {features.get('away_composite_elo', 'MISSING')}")
print(f"   off_elo_diff: {features.get('off_elo_diff', 'MISSING')}")
print(f"   def_elo_diff: {features.get('def_elo_diff', 'MISSING')}")

# Verify ELO from database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT team, MAX(composite_elo) as max_elo
    FROM elo_ratings
    WHERE team IN ('DAL', 'DET')
    GROUP BY team
""")
elo_data = cursor.fetchall()

print("\n" + "=" * 80)
print("DATABASE ELO VERIFICATION")
print("=" * 80)
for team, elo in elo_data:
    print(f"   {team}: {elo:.2f}")

conn.close()

# Make prediction
print("\n" + "=" * 80)
print("MODEL PREDICTION")
print("=" * 80)

# Prepare feature vector in correct order
X = []
for feat_name in required_features:
    if feat_name in features:
        X.append(features[feat_name])
    else:
        print(f"⚠️  WARNING: Missing feature '{feat_name}'")
        X.append(0)

X_df = pd.DataFrame([X], columns=required_features)

# Get prediction
prob_home_win = model.predict_proba(X_df)[0][1]

print(f"\n🎯 PREDICTION:")
print(f"   DAL (home) win probability: {prob_home_win:.1%}")
print(f"   DET (away) win probability: {1 - prob_home_win:.1%}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"   ELO Ratings: DAL {features.get('home_composite_elo', 1500):.0f} vs DET {features.get('away_composite_elo', 1500):.0f}")
print(f"   ELO Difference: {features.get('home_composite_elo', 1500) - features.get('away_composite_elo', 1500):.0f} points")
print(f"   Expected: Detroit should be favored (higher ELO)")
print(f"   Actual: {'✅ Detroit favored' if prob_home_win < 0.5 else '❌ Dallas favored'}")

if prob_home_win < 0.5:
    print("\n✅ BUG FIXED! Model now correctly favors Detroit (higher ELO team)")
else:
    print("\n❌ BUG STILL PRESENT! Model incorrectly favors Dallas despite lower ELO")

print("\n" + "=" * 80)
