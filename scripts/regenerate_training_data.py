"""
Regenerate training data with NEW injury shock features.

This script:
1. Loads existing training data (game_id, date, teams, targets)
2. Recalculates ALL features using updated FeatureCalculatorV5 (includes injury shock)
3. Saves new training CSV with injury_shock_* and home_star_missing features
4. Preserves target columns (don't recalculate outcomes)
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_calculator_v5 import FeatureCalculatorV5
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def regenerate_training_data():
    """Regenerate training data with injury shock features."""
    
    # Load existing training data
    logger.info("Loading existing training data...")
    df = pd.read_csv("data/training_data_with_features.csv")
    logger.info(f"  Loaded {len(df)} games")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize feature calculator
    logger.info("\nInitializing FeatureCalculatorV5 (with injury shock)...")
    calculator = FeatureCalculatorV5()
    
    # Extract game metadata and target columns
    game_metadata = df[['game_id', 'date', 'season', 'home_team', 'away_team']].copy()
    target_columns = [col for col in df.columns if col.startswith('target_')]
    targets = df[target_columns].copy()
    
    logger.info(f"\nPreserving {len(target_columns)} target columns: {target_columns}")
    
    # Recalculate features for each game
    logger.info("\nRecalculating features with injury shock...")
    all_features = []
    
    for idx, row in game_metadata.iterrows():
        if idx % 500 == 0:
            logger.info(f"  Progress: {idx}/{len(game_metadata)} games ({idx/len(game_metadata)*100:.1f}%)")
        
        try:
            # Convert date to string format (YYYY-MM-DD)
            if isinstance(row['date'], str):
                game_date_str = row['date']
            else:
                game_date_str = row['date'].strftime('%Y-%m-%d')
            
            features = calculator.calculate_game_features(
                home_team=row['home_team'],
                away_team=row['away_team'],
                game_date=game_date_str  # Pass string, not datetime
            )
            
            all_features.append(features)
            
        except Exception as e:
            logger.error(f"  Error on game {row['game_id']}: {e}")
            # Add empty dict to maintain row alignment
            all_features.append({})
    
    logger.info(f"\nCompleted feature calculation for {len(all_features)} games")
    
    # Create features dataframe
    features_df = pd.DataFrame(all_features)
    logger.info(f"\nGenerated {len(features_df.columns)} features")
    
    # Check for new injury features
    injury_features = [col for col in features_df.columns if 'injury' in col.lower() or 'star' in col.lower()]
    logger.info(f"\nInjury features found: {injury_features}")
    
    # Combine everything
    final_df = pd.concat([
        game_metadata,
        targets,
        features_df
    ], axis=1)
    
    # Save to new file
    output_path = "data/training_data_with_injury_shock.csv"
    logger.info(f"\nSaving to: {output_path}")
    final_df.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(final_df)} rows × {len(final_df.columns)} columns")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("REGENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total games: {len(final_df)}")
    logger.info(f"Total features: {len(features_df.columns)}")
    logger.info(f"New injury features: {injury_features}")
    logger.info(f"\nOutput: {output_path}")
    logger.info("\nNext step: Run walk-forward validation with new features")
    
    return final_df

if __name__ == "__main__":
    regenerate_training_data()
