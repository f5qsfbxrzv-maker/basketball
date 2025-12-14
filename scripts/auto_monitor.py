"""
Monitor data regeneration and auto-launch overnight pipeline when complete.
"""

import time
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_file():
    """Check if training data has all 36 features."""
    try:
        import pandas as pd
        df = pd.read_csv("data/training_data_with_features.csv")
        
        # Count non-metadata columns
        feature_cols = [c for c in df.columns if c not in [
            'game_id', 'date', 'home_team', 'away_team', 'season',
            'target_spread', 'target_spread_cover', 'target_moneyline_win',
            'target_game_total', 'target_over_under'
        ]]
        
        return len(feature_cols) >= 36
    except Exception as e:
        logger.warning(f"Could not check data file: {e}")
        return False

def main():
    logger.info("="*80)
    logger.info("OVERNIGHT AUTOMATION - WAITING FOR DATA REGENERATION")
    logger.info("="*80)
    logger.info("Monitoring: data/training_data_with_features.csv")
    logger.info("Waiting for: 36+ features")
    logger.info("When complete: Auto-launch hypertuning → training → backtesting")
    logger.info("="*80)
    
    check_interval = 60  # Check every 60 seconds
    checks = 0
    
    while True:
        checks += 1
        logger.info(f"\nCheck #{checks} at {time.strftime('%H:%M:%S')}")
        
        if check_data_file():
            logger.info("✅ Data regeneration complete! 36 features detected.")
            logger.info("\nLaunching overnight pipeline...")
            time.sleep(2)
            
            # Launch pipeline
            subprocess.run([sys.executable, "scripts/overnight_pipeline.py"])
            break
        else:
            logger.info("⏳ Still regenerating... checking again in 60 seconds")
            time.sleep(check_interval)

if __name__ == "__main__":
    main()
