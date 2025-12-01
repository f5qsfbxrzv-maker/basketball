"""Ensemble model trainer combining multiple ML models"""
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

class EnsembleTrainer:
    """Train ensemble of XGBoost, LightGBM, and Random Forest"""
    
    def __init__(self):
        self.models = {}
    
    def train_ensemble(self, X_train, y_train, weights: List[float] = [0.5, 0.3, 0.2]):
        """Train weighted ensemble"""
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6)
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6)
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=6)
        
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)],
            weights=weights,
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        return ensemble
