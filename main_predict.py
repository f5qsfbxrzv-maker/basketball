"""
MAIN PREDICTION RUNNER - The "Big Red Button"
Runs daily predictions for NBA games

Usage:
    python main_predict.py              # Predict today's games
    python main_predict.py --date YYYY-MM-DD  # Predict specific date
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, date

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DB_PATH, MONEYLINE_MODEL, TOTALS_MODEL,
    ISOTONIC_CALIBRATOR, PLATT_CALIBRATOR,
    DAILY_PICKS_DIR, validate_paths
)

def main(target_date: date = None):
    """
    Run predictions for NBA games
    
    Args:
        target_date: Date to predict (defaults to today)
    """
    # Validate system setup
    try:
        validate_paths()
    except FileNotFoundError as e:
        print(f"❌ SYSTEM ERROR: {e}")
        print("Run setup first or check config/settings.py")
        return 1
    
    if target_date is None:
        target_date = date.today()
    
    print("="*80)
    print(f"NBA BETTING SYSTEM - DAILY PREDICTIONS")
    print(f"Date: {target_date}")
    print("="*80)
    print()
    
    # TODO: Import and run prediction engine
    print("⚠️  Prediction engine integration pending")
    print("   Next steps:")
    print("   1. Import src.core.prediction_engine")
    print("   2. Load models from models/production/")
    print("   3. Query today's games from data/live/nba_betting_data.db")
    print("   4. Generate predictions")
    print("   5. Calculate Kelly stakes")
    print("   6. Save to output/daily_picks/")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Daily Prediction Runner")
    parser.add_argument("--date", type=str, help="Date to predict (YYYY-MM-DD)")
    args = parser.parse_args()
    
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            print("   Use: YYYY-MM-DD")
            sys.exit(1)
    
    sys.exit(main(target_date))
