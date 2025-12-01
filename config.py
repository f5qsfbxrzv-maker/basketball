# Global configurations (API Keys, Database paths)

# Database Configuration
DB_PATH = 'data/database/nba_betting.db'

# Model Paths
PRODUCTION_MODEL_PATH = 'models/production/'
STAGING_MODEL_PATH = 'models/staging/'

# Data Paths
RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

# Betting Parameters
MIN_EDGE = 0.03  # 3% minimum edge
MAX_BET_SIZE = 0.05  # 5% of bankroll max
KELLY_FRACTION = 0.25  # Quarter Kelly

# API Keys (use environment variables in production)
# KALSHI_API_KEY = os.getenv('KALSHI_API_KEY')
# NBA_API_KEY = os.getenv('NBA_API_KEY')
