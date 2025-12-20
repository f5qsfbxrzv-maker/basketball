"""
Test the tuned production configuration
"""
import production_config_mdp as config

print("="*60)
print("🔍 TUNED MDP PRODUCTION CONFIGURATION")
print("="*60)

print(f"\n📦 Model Info:")
print(f"   Type: {config.MODEL_TYPE}")
print(f"   Version: {config.MODEL_VERSION}")

print(f"\n🎯 Probability Conversion:")
print(f"   NBA_STD_DEV: {config.NBA_STD_DEV}")
print(f"   Formula: Win% = norm.cdf(margin / {config.NBA_STD_DEV})")

print(f"\n⚙️ Tuned XGBoost Params:")
print(f"   objective: {config.XGB_PARAMS['objective']}")
print(f"   max_depth: {config.XGB_PARAMS['max_depth']}")
print(f"   min_child_weight: {config.XGB_PARAMS['min_child_weight']} ⭐ (ignores blowouts)")
print(f"   learning_rate: {config.XGB_PARAMS['learning_rate']:.6f}")
print(f"   n_estimators: {config.N_ESTIMATORS}")
print(f"   gamma: {config.XGB_PARAMS['gamma']:.4f}")
print(f"   subsample: {config.XGB_PARAMS['subsample']:.4f}")
print(f"   colsample_bytree: {config.XGB_PARAMS['colsample_bytree']:.4f}")
print(f"   reg_alpha: {config.XGB_PARAMS['reg_alpha']:.4f}")
print(f"   reg_lambda: {config.XGB_PARAMS['reg_lambda']:.4f}")

print(f"\n🏎️ Features ({len(config.ACTIVE_FEATURES)}):")
for i, feat in enumerate(config.ACTIVE_FEATURES, 1):
    print(f"   {i:2d}. {feat}")

print(f"\n🎯 Betting Thresholds:")
print(f"   MIN_EDGE_FAVORITE: {config.MIN_EDGE_FAVORITE:.1%}")
print(f"   MIN_EDGE_UNDERDOG: {config.MIN_EDGE_UNDERDOG:.1%}")

print(f"\n🚫 Forensic Filters:")
print(f"   MAX_FAVORITE_ODDS: {config.MAX_FAVORITE_ODDS}")
print(f"   MIN_OFF_ELO_DIFF_FAVORITE: {config.MIN_OFF_ELO_DIFF_FAVORITE}")
print(f"   MAX_INJURY_DISADVANTAGE: {config.MAX_INJURY_DISADVANTAGE}")

print(f"\n📁 File Paths:")
print(f"   Data: {config.DATA_PATH}")
print(f"   Model: {config.MODEL_PATH}")

print("\n✅ Configuration loaded successfully!")
print("="*60)
