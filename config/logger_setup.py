"""
Simple logger setup for NBA betting system
"""
import logging
from typing import Dict, Any

class StructuredLoggerAdapter:
    """Logger adapter with .event() method for legacy code"""
    def __init__(self, logger):
        self.logger = logger
    
    def event(self, level, message, category='general'):
        """Log an event with level and category"""
        getattr(self.logger, level)(f"[{category}] {message}")
    
    def __getattr__(self, name):
        """Forward all other calls to underlying logger"""
        return getattr(self.logger, name)

def get_structured_adapter(name: str = __name__):
    """Get a logger with structured logging capability"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return StructuredLoggerAdapter(logger)

def classify_error(error: Exception) -> str:
    """Classify error type for logging"""
    error_type = type(error).__name__
    
    if 'Database' in str(error) or 'SQL' in str(error):
        return 'DATABASE_ERROR'
    elif 'API' in str(error) or 'Request' in str(error):
        return 'API_ERROR'
    elif 'Feature' in str(error):
        return 'FEATURE_ERROR'
    elif 'Model' in str(error):
        return 'MODEL_ERROR'
    else:
        return f'{error_type}_ERROR'
