"""
Test feature generation on a small sample to diagnose missing features.
"""

import pandas as pd
import sys
sys.path.insert(0, '.')
from src.features.feature_calculator_v5 import FeatureCalculatorV5
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Test on 5 recent games
logger.info("Loading sample games...")
df = pd.read_csv("data/training_data_with_features.csv")
df['date'] = pd.to_datetime(df['date'])
sample = df.sort_values('date', ascending=False).head(5).copy()

logger.info(f"\nTesting on {len(sample)} recent games:")
for _, row in sample.iterrows():
    logger.info(f"  {row['date'].date()}: {row['home_team']} vs {row['away_team']}")

# Initialize calculator
logger.info("\nInitializing FeatureCalculatorV5...")
calculator = FeatureCalculatorV5()

# Test feature generation for each game
logger.info("\n" + "="*60)
logger.info("FEATURE GENERATION TEST")
logger.info("="*60)

for idx, row in sample.iterrows():
    logger.info(f"\n--- Game: {row['home_team']} vs {row['away_team']} on {row['date'].date()} ---")
    
    game_date_str = row['date'].strftime('%Y-%m-%d')
    
    try:
        features = calculator.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            game_date=game_date_str
        )
        
        logger.info(f"✅ Generated {len(features)} features")
        
        # Check for specific feature categories
        injury_features = [k for k in features.keys() if 'injury' in k or 'star' in k]
        rest_features = [k for k in features.keys() if 'rest' in k or 'back_to_back' in k or '3in4' in k or 'fatigue' in k]
        ewma_features = [k for k in features.keys() if 'ewma' in k]
        altitude_features = [k for k in features.keys() if 'altitude' in k]
        
        logger.info(f"\nFeature breakdown:")
        logger.info(f"  Injury features ({len(injury_features)}): {injury_features}")
        logger.info(f"  Rest features ({len(rest_features)}): {rest_features}")
        logger.info(f"  EWMA features ({len(ewma_features)}): {ewma_features}")
        logger.info(f"  Altitude features ({len(altitude_features)}): {altitude_features}")
        
        # Show all features
        logger.info(f"\nAll features:")
        for k, v in sorted(features.items()):
            logger.info(f"    {k:35s} = {v:.4f}" if isinstance(v, (int, float)) else f"    {k:35s} = {v}")
        
        # Check against whitelist
        from config.feature_whitelist import FEATURE_WHITELIST
        missing_from_whitelist = set(FEATURE_WHITELIST) - set(features.keys())
        
        if missing_from_whitelist:
            logger.warning(f"\n⚠️ Missing {len(missing_from_whitelist)} whitelist features:")
            for f in sorted(missing_from_whitelist):
                logger.warning(f"    - {f}")
        else:
            logger.info(f"\n✅ All {len(FEATURE_WHITELIST)} whitelist features present!")
        
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    break  # Just test first game in detail

logger.info("\n" + "="*60)
logger.info("DIAGNOSIS COMPLETE")
logger.info("="*60)
