"""
SAFETY AUDIT: "Too Good To Be True" Check
Verify no data leakage before going live with betting system
"""

import pandas as pd
import json
import xgboost as xgb
import pickle
from pathlib import Path

print("="*80)
print("SAFETY AUDIT: DATA LEAKAGE CHECK")
print("="*80)

# 1. Load the production model
print("\n" + "="*80)
print("STEP 1: FEATURE LIST INSPECTION")
print("="*80)

model = xgb.XGBClassifier()
model.load_model('models/xgboost_final_trial98.json')

features = model.get_booster().feature_names
print(f"\nTotal features: {len(features)}")
print(f"\nFeature list:")

# Categorize features
dangerous_keywords = ['pts', 'score', 'win', 'loss', 'plus_minus', 'minutes', 
                      'outcome', 'result', 'final', 'actual', 'target']
temporal_features = []
elo_features = []
stat_features = []
suspicious_features = []

for i, feat in enumerate(features, 1):
    print(f"{i:2d}. {feat}")
    
    # Check for dangerous keywords
    if any(keyword in feat.lower() for keyword in dangerous_keywords):
        suspicious_features.append(feat)
    
    # Categorize
    if 'season' in feat.lower() or 'games_into' in feat.lower() or 'progress' in feat.lower() or 'month' in feat.lower():
        temporal_features.append(feat)
    elif 'elo' in feat.lower():
        elo_features.append(feat)
    else:
        stat_features.append(feat)

print(f"\n{'='*80}")
print("FEATURE CATEGORIZATION")
print("="*80)
print(f"\nTemporal features ({len(temporal_features)}):")
for feat in temporal_features:
    print(f"  - {feat}")

print(f"\nELO features ({len(elo_features)}):")
for feat in elo_features:
    print(f"  - {feat}")

print(f"\nStatistical features ({len(stat_features)}):")
for feat in stat_features:
    print(f"  - {feat}")

# 2. Check for suspicious features
print(f"\n{'='*80}")
print("STEP 2: SUSPICIOUS FEATURE CHECK")
print("="*80)

if suspicious_features:
    print(f"\n⚠️  WARNING: {len(suspicious_features)} SUSPICIOUS FEATURES DETECTED!")
    print("\nThese features may contain future information:")
    for feat in suspicious_features:
        print(f"  ❌ {feat}")
    print("\n🚨 DATA LEAKAGE DETECTED - DO NOT GO LIVE!")
else:
    print("\n✅ No suspicious features detected")
    print("   All features appear to use only historical data")

# 3. Load training data and check columns
print(f"\n{'='*80}")
print("STEP 3: TRAINING DATA COLUMN CHECK")
print("="*80)

df = pd.read_csv('data/training_data_with_temporal_features.csv', nrows=5)
all_cols = df.columns.tolist()

# Check for target columns in features
target_cols = ['target_spread', 'target_spread_cover', 'target_moneyline_win', 
               'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

print(f"\nTarget columns in training data:")
for col in target_cols:
    if col in all_cols:
        if col in features:
            print(f"  ❌ {col} - USED AS FEATURE (LEAKAGE!)")
        else:
            print(f"  ✅ {col} - Not used as feature (safe)")

# 4. Check feature metadata
print(f"\n{'='*80}")
print("STEP 4: FEATURE METADATA REVIEW")
print("="*80)

with open('models/final_model_metadata.json', 'r') as f:
    metadata = json.load(f)

top_features = metadata.get('top_features', [])[:10]
print(f"\nTop 10 most important features:")
for i, feat_info in enumerate(top_features, 1):
    feat_name = feat_info['feature']
    importance = feat_info['importance']
    
    # Flag if suspicious
    is_suspicious = any(keyword in feat_name.lower() for keyword in dangerous_keywords)
    flag = " ⚠️  SUSPICIOUS!" if is_suspicious else ""
    
    print(f"{i:2d}. {feat_name:<30} ({importance:.2f}%){flag}")

# 5. Odds timing check
print(f"\n{'='*80}")
print("STEP 5: ODDS TIMING ANALYSIS")
print("="*80)

print("\nBacktest odds source: nba_ODDS_history.db")
print("Odds type: CLOSING LINES (as of game time)")
print("\n⚠️  IMPORTANT: Closing lines reflect all market information up to tip-off")
print("   - Your model is trained on closing lines")
print("   - In live betting, you'll be using OPENING lines (8-12 hours before)")
print("   - Expect some performance degradation due to this timing gap")

print("\nRecommendations:")
print("  1. Monitor live performance closely for first 2 weeks")
print("  2. Consider reducing Kelly fraction by 50% during live testing")
print("  3. Track opening vs closing line movement")
print("  4. Log actual odds obtained vs model predictions")

# 6. Final verdict
print(f"\n{'='*80}")
print("FINAL VERDICT")
print("="*80)

if suspicious_features:
    print("\n🚨 FAIL: DATA LEAKAGE DETECTED")
    print("   DO NOT GO LIVE until suspicious features are removed")
    verdict = "FAIL"
else:
    print("\n✅ PASS: No obvious data leakage detected")
    print("   Features appear to use only historical information")
    print("\n⚠️  CAVEAT: Closing line vs opening line gap")
    print("   - Backtest used closing lines (at tip-off)")
    print("   - Live betting will use opening lines (8-12 hours before)")
    print("   - Real ROI likely 50-75% of backtest results")
    print("\n💡 RECOMMENDATION:")
    print("   - Start with PAPER TRADING for 2 weeks")
    print("   - Use 25% Kelly (conservative)")
    print("   - Maximum 2% of bankroll per bet")
    print("   - Track opening line accuracy")
    verdict = "PASS_WITH_CAUTIONS"

# 7. Save audit report
audit_report = {
    'audit_date': pd.Timestamp.now().isoformat(),
    'verdict': verdict,
    'total_features': len(features),
    'suspicious_features': suspicious_features,
    'feature_categories': {
        'temporal': temporal_features,
        'elo': elo_features,
        'statistical': stat_features
    },
    'top_features': top_features,
    'recommendations': [
        'Paper trade for 2 weeks before live betting',
        'Use 25% Kelly fraction initially',
        'Maximum 2% bankroll per bet',
        'Track opening vs closing line movement',
        'Log all predictions and outcomes for recalibration'
    ]
}

with open('models/safety_audit_report.json', 'w') as f:
    json.dump(audit_report, f, indent=2)

print(f"\n✓ Audit report saved: models/safety_audit_report.json")

print("\n" + "="*80)
