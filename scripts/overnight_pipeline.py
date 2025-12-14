"""
Overnight automated pipeline:
1. Wait for data regeneration to finish
2. Hypertune XGBoost with Optuna (100 trials)
3. Train final model with best params
4. Run walk-forward backtest with flat units
"""

import subprocess
import time
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def wait_for_file_stable(filepath, stable_seconds=30):
    """Wait until file stops growing (data generation complete)"""
    logger.info(f"Monitoring {filepath} for completion...")
    
    if not os.path.exists(filepath):
        logger.info("File doesn't exist yet, waiting...")
        while not os.path.exists(filepath):
            time.sleep(10)
    
    last_size = 0
    stable_count = 0
    
    while stable_count < stable_seconds:
        current_size = os.path.getsize(filepath)
        if current_size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            logger.info(f"File size: {current_size:,} bytes (still writing...)")
        last_size = current_size
        time.sleep(1)
    
    logger.info(f"✅ File stable at {current_size:,} bytes")
    return True

def run_command(cmd, description):
    """Run command and log output"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING: {description}")
    logger.info(f"Command: {cmd}")
    logger.info(f"{'='*60}\n")
    
    start_time = datetime.now()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    duration = (datetime.now() - start_time).total_seconds()
    
    if result.returncode == 0:
        logger.info(f"\n✅ COMPLETED: {description} ({duration:.1f}s)")
    else:
        logger.error(f"\n❌ FAILED: {description} (exit code {result.returncode})")
        raise Exception(f"{description} failed")
    
    return result

def main():
    logger.info("="*60)
    logger.info("OVERNIGHT PIPELINE STARTED")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("="*60)
    
    # Step 1: Wait for data regeneration
    logger.info("\nStep 1: Waiting for data regeneration to complete...")
    wait_for_file_stable("data/training_data_with_features.csv", stable_seconds=60)
    logger.info("Data regeneration complete! Starting hypertuning...\n")
    time.sleep(5)
    
    # Step 2: Hypertune
    logger.info("\nStep 2: Hyperparameter tuning with Optuna...")
    run_command(
        "python scripts/hypertune_optuna.py",
        "Optuna Hyperparameter Tuning (100 trials)"
    )
    time.sleep(5)
    
    # Step 3: Train final model
    logger.info("\nStep 3: Training final model with best params...")
    run_command(
        "python scripts/train_final_model.py",
        "Final Model Training"
    )
    time.sleep(5)
    
    # Step 4: Walk-forward backtest
    logger.info("\nStep 4: Walk-forward backtest with flat units...")
    run_command(
        "python scripts/walkforward_backtest_flat.py",
        "Walk-Forward Backtest (Flat Units)"
    )
    
    logger.info("\n" + "="*60)
    logger.info("OVERNIGHT PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"End time: {datetime.now()}")
    logger.info("="*60)
    logger.info("\nResults saved to:")
    logger.info("  - models/xgboost_final.pkl (trained model)")
    logger.info("  - output/optuna_study.pkl (hyperparameter search)")
    logger.info("  - output/walkforward_backtest_flat.csv (bet log)")
    logger.info("  - output/walkforward_performance.txt (summary)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {e}")
        raise
    
    # Step 3: Kelly backtest
    if not run_script(
        "scripts/kelly_backtest_36features.py",
        "Kelly ROI Backtest"
    ):
        logger.error("Pipeline aborted: Backtest failed")
        return
    
    # Step 4: Summary report
    if not run_script(
        "scripts/generate_summary_report.py",
        "Summary Report Generation"
    ):
        logger.warning("Summary report failed (non-critical)")
    
    total_elapsed = time.time() - pipeline_start
    
    logger.info("\n" + "="*80)
    logger.info("🎉 OVERNIGHT PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Total time: {total_elapsed/3600:.2f} hours")
    logger.info("="*80)
    logger.info("\nCheck these files:")
    logger.info("  - models/xgboost_36features_tuned.pkl (final model)")
    logger.info("  - output/kelly_backtest_36features.csv (bet log)")
    logger.info("  - output/overnight_pipeline_summary.md (report)")

if __name__ == "__main__":
    main()
