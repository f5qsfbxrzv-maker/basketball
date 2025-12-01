"""
Machine Learning Model Trainer for NBA Betting
Implements ensemble models with hyperparameter optimization,
cross-validation, and model persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NBAModelTrainer:
    def __init__(self, random_state=42):
        """
        Initialize NBA model trainer
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = []
        # Global feature blacklist (can be set by caller)
        self.feature_blacklist = []
        
        # Initialize model configurations
        self.model_configs = {
            'spread_classifier': {
                'models': ['xgboost', 'random_forest', 'neural_network'],
                'target': 'covers_spread',
                'type': 'classification'
            },
            'moneyline_classifier': {
                'models': ['xgboost', 'lightgbm', 'logistic_regression'],
                'target': 'home_wins',
                'type': 'classification'
            },
            'total_classifier': {
                'models': ['xgboost', 'random_forest', 'gradient_boosting'],
                'target': 'goes_over',
                'type': 'classification'
            },
            'spread_regressor': {
                'models': ['xgboost', 'random_forest', 'neural_network'],
                'target': 'actual_spread',
                'type': 'regression'
            },
            'total_regressor': {
                'models': ['xgboost', 'random_forest', 'ridge'],
                'target': 'total_points',
                'type': 'regression'
            }
        }
    
    def get_model_instances(self):
        """Get instances of all available models"""
        return {
            'xgboost_classifier': xgb.XGBClassifier(
                n_estimators=1000,  # High value, use early stopping
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                early_stopping_rounds=50
            ),
            'xgboost_regressor': xgb.XGBRegressor(
                n_estimators=1000,  # High value, use early stopping
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                early_stopping_rounds=50
                # objective='count:poisson' can be set for betting models
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'random_forest_regressor': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state
            ),
            'neural_network_regressor': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state
            ),
            'ridge': Ridge(
                alpha=1.0,
                random_state=self.random_state
            ),
            'svm_classifier': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            'svm_regressor': SVR(
                C=1.0,
                kernel='rbf'
            )
        }
    
    def get_hyperparameter_grids(self):
        """Get hyperparameter grids for optimization"""
        return {
            'xgboost_classifier': {
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9]
                # n_estimators removed - use early stopping instead
            },
            'xgboost_regressor': {
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9]
                # n_estimators removed - use early stopping instead
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest_regressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'lightgbm': {
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 63]
                # n_estimators removed - use early stopping instead
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
                'learning_rate_init': [0.01, 0.001, 0.0001],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    
    def train_with_tscv(self, df, model, features, target='WL', n_splits=5):
        """
        Performs Time Series Cross-Validation to prevent data leakage.
        Replaces standard train_test_split.
        
        Args:
            df (DataFrame): Input data with all features and target
            model: scikit-learn compatible model/pipeline
            features (list): List of feature column names
            target (str): Target column name
            n_splits (int): Number of time series folds
            
        Returns:
            tuple: (final_model, average_accuracy)
        """
        from sklearn.base import clone
        from sklearn.metrics import accuracy_score, brier_score_loss
        
        print(f"\n⏳ Starting Time Series Validation ({n_splits} folds)...")
        
        # 1. CRITICAL: Sort by date to respect the arrow of time
        # We reset index so iloc indexing works perfectly for splits
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            print(f"   ✅ Data sorted by date ({df['date'].min()} to {df['date'].max()})")
        else:
            print("   ⚠️ Warning: No 'date' column found. Assuming data is already sorted chronologically.")
        
        X = df[features]
        y = df[target]
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        
        # 2. Walk Forward through history
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Slice data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train (Clone the model to ensure fresh start each fold)
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Predict
            probs = fold_model.predict_proba(X_test)[:, 1]
            preds = (probs > 0.5).astype(int)
            
            # Score
            acc = accuracy_score(y_test, preds)
            brier = brier_score_loss(y_test, probs)
            
            # Context (Dates)
            if 'date' in df.columns:
                start_date = df['date'].iloc[test_idx].min()
                end_date = df['date'].iloc[test_idx].max()
                date_str = f"{str(start_date).split()[0]} to {str(end_date).split()[0]}"
            else:
                date_str = "Unknown Dates"
            
            print(f"   🔹 Fold {fold+1} ({date_str}): Acc {acc:.1%} | Brier {brier:.4f}")
            fold_metrics.append(acc)
            
        avg_acc = sum(fold_metrics) / len(fold_metrics)
        print(f"\n🏆 TSCV Average Accuracy: {avg_acc:.1%}")
        print(f"   (This is your REAL expected performance)")

        # 3. Final Retrain on ALL Data (for Production)
        print("🚀 Retraining Final Model on full dataset...")
        final_model = clone(model)
        final_model.fit(X, y)
        
        return final_model, avg_acc
    
    def remove_collinear_features(self, df, features, threshold=0.95):
        """
        Scans the feature set for duplicates.
        If Feature A and Feature B have a correlation > threshold, drop Feature B.
        
        Args:
            df (DataFrame): Input data
            features (list): List of feature column names
            threshold (float): Correlation threshold (default 0.95)
            
        Returns:
            list: Clean feature list with collinear features removed
        """
        print(f"\n🧹 Starting Feature Audit (Threshold: {threshold})...")
        
        # Calculate correlation matrix
        X = df[features]
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        print(f"   📉 Found {len(to_drop)} redundant features to drop.")
        if len(to_drop) > 0:
            print(f"   🗑️ Dropping examples: {to_drop[:5]}...")
        
        # Return the clean list of features
        clean_features = [f for f in features if f not in to_drop]
        print(f"   ✅ Features reduced from {len(features)} to {len(clean_features)}")
        
        return clean_features
    
    def prepare_features(self, df, target_column, exclude_columns=None):
        """
        Prepare features for training
        
        Args:
            df (DataFrame): Input data
            target_column (str): Target variable column name
            exclude_columns (list): Columns to exclude from features
            
        Returns:
            tuple: (X, y, feature_names)
        """
        if exclude_columns is None:
            exclude_columns = ['game_id', 'date', 'home_team', 'away_team']
        
        # Exclude target, specified columns, and any global feature blacklist
        all_exclude = exclude_columns + [target_column] + list(self.feature_blacklist)
        feature_columns = [col for col in df.columns if col not in all_exclude]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X, y, feature_columns
    
    def scale_features(self, X_train, X_test=None, scaler_type='standard'):
        """
        Scale features using specified scaler
        
        Args:
            X_train (DataFrame): Training features
            X_test (DataFrame): Test features (optional)
            scaler_type (str): Type of scaler ('standard', 'robust')
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, scaler)
        """
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Scaler type must be 'standard' or 'robust'")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, None, scaler
    
    def train_single_model(self, X_train, y_train, model_name, model_type='classification', 
                          optimize_hyperparameters=True, cv_folds=5):
        """
        Train a single model with optional hyperparameter optimization
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training targets
            model_name (str): Name of model to train
            model_type (str): 'classification' or 'regression'
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Training results
        """
        model_instances = self.get_model_instances()
        
        # Select appropriate model based on type
        if model_type == 'classification':
            if model_name == 'xgboost':
                model = model_instances['xgboost_classifier']
            elif model_name == 'random_forest':
                model = model_instances['random_forest']
            elif model_name == 'neural_network':
                model = model_instances['neural_network']
            else:
                model = model_instances.get(model_name)
        else:  # regression
            if model_name == 'xgboost':
                model = model_instances['xgboost_regressor']
            elif model_name == 'random_forest':
                model = model_instances['random_forest_regressor']
            elif model_name == 'neural_network':
                model = model_instances['neural_network_regressor']
            else:
                model = model_instances.get(f"{model_name}_regressor", model_instances.get(model_name))
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Scale features for neural networks and SVM
        if model_name in ['neural_network', 'svm_classifier', 'svm_regressor', 'logistic_regression']:
            X_train_scaled, _, scaler = self.scale_features(X_train)
            use_scaled = True
        else:
            X_train_scaled = X_train
            scaler = None
            use_scaled = False
        
        # Hyperparameter optimization
        best_model = model
        best_params = {}
        grid_search = None
        
        if optimize_hyperparameters:
            param_grids = self.get_hyperparameter_grids()
            param_grid = param_grids.get(f"{model_name}_{model_type}", param_grids.get(model_name))
            
            if param_grid:
                # Use TimeSeriesSplit for time-aware cross-validation
                cv = TimeSeriesSplit(n_splits=cv_folds)
                
                if model_type == 'classification':
                    scoring = 'roc_auc'
                else:
                    scoring = 'neg_mean_squared_error'
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv, scoring=scoring,
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
        
        # Train final model (if grid search wasn't used, best_model needs fitting)
        if not optimize_hyperparameters or grid_search is None:
            best_model.fit(X_train_scaled, y_train)
        
        # Use GridSearchCV scores if available, otherwise do cross-validation
        if grid_search is not None:
            # GridSearchCV already did CV - use those scores
            cv_scores = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
            cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
            # Convert to array for compatibility
            cv_scores = np.array([cv_scores] * cv_folds)  # Fake array for compatibility
            cv_auc_scores = None
        else:
            # Do cross-validation for models without hyperparameter tuning
            cv = TimeSeriesSplit(n_splits=cv_folds)
            if model_type == 'classification':
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
                cv_auc_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='r2')
                cv_auc_scores = None
        
        # Feature importance
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            if len(best_model.coef_.shape) == 1:
                feature_importance = dict(zip(X_train.columns, abs(best_model.coef_)))
            else:
                feature_importance = dict(zip(X_train.columns, abs(best_model.coef_[0])))
        
        return {
            'model': best_model,
            'scaler': scaler,
            'best_params': best_params,
            'cv_scores': cv_scores,
            'cv_auc_scores': cv_auc_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'use_scaled_features': use_scaled
        }
    
    def train_ensemble_model(self, X_train, y_train, model_config, optimize_hyperparameters=True):
        """
        Train ensemble of models for a specific prediction task
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training targets
            model_config (dict): Model configuration
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            
        Returns:
            dict: Ensemble training results
        """
        model_type = model_config['type']
        model_names = model_config['models']
        
        ensemble_results = {}
        individual_models = {}
        
        print(f"Training {model_type} ensemble with models: {model_names}")
        
        for model_name in model_names:
            print(f"  Training {model_name}...")
            
            try:
                result = self.train_single_model(
                    X_train, y_train, model_name, model_type, 
                    optimize_hyperparameters
                )
                
                individual_models[model_name] = result
                print(f"    CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
                
            except Exception as e:
                print(f"    Error training {model_name}: {str(e)}")
                continue
        
        # Select best individual model
        if individual_models:
            best_model_name = max(individual_models.keys(), 
                                key=lambda k: individual_models[k]['cv_mean'])
            best_individual = individual_models[best_model_name]
            
            ensemble_results = {
                'individual_models': individual_models,
                'best_model_name': best_model_name,
                'best_model': best_individual,
                'ensemble_type': model_type,
                'target': model_config['target']
            }
        
        return ensemble_results
    
    def evaluate_model(self, model_results, X_test, y_test):
        """
        Evaluate trained model on test set
        
        Args:
            model_results (dict): Model training results
            X_test (DataFrame): Test features
            y_test (Series): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        model = model_results['best_model']['model']
        scaler = model_results['best_model']['scaler']
        use_scaled = model_results['best_model']['use_scaled_features']
        
        # Prepare test features
        if use_scaled and scaler is not None:
            X_test_processed = scaler.transform(X_test)
            X_test_processed = pd.DataFrame(X_test_processed, columns=X_test.columns)
        else:
            X_test_processed = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        evaluation = {}
        
        if model_results['ensemble_type'] == 'classification':
            # Classification metrics
            evaluation['accuracy'] = accuracy_score(y_test, y_pred)
            evaluation['precision'] = precision_score(y_test, y_pred, average='weighted')
            evaluation['recall'] = recall_score(y_test, y_pred, average='weighted')
            evaluation['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            
            # Probability predictions for AUC
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_processed)
                if y_prob.shape[1] == 2:  # Binary classification
                    evaluation['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
                else:  # Multi-class
                    evaluation['roc_auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
            
        else:
            # Regression metrics
            evaluation['mse'] = mean_squared_error(y_test, y_pred)
            evaluation['rmse'] = np.sqrt(evaluation['mse'])
            evaluation['mae'] = mean_absolute_error(y_test, y_pred)
            evaluation['r2'] = r2_score(y_test, y_pred)
        
        evaluation['predictions'] = y_pred
        
        return evaluation
    
    def train_all_models(self, df, test_size=0.2, optimize_hyperparameters=True):
        """
        Train all configured model ensembles
        
        Args:
            df (DataFrame): Training data
            test_size (float): Proportion of data for testing
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            
        Returns:
            dict: All training results
        """
        # Sort by date for time-aware splitting
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        all_results = {}
        
        for model_name, config in self.model_configs.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()}")
            print(f"{'='*50}")
            
            target_column = config['target']
            
            # Check if target column exists
            if target_column not in df.columns:
                print(f"Warning: Target column '{target_column}' not found. Skipping {model_name}")
                continue
            
            # Prepare features
            X, y, feature_names = self.prepare_features(df, target_column)
            
            # Time-aware train/test split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"Features: {len(feature_names)}")
            
            # Train ensemble
            ensemble_results = self.train_ensemble_model(
                X_train, y_train, config, optimize_hyperparameters
            )
            
            if ensemble_results:
                # Evaluate on test set
                evaluation = self.evaluate_model(ensemble_results, X_test, y_test)
                ensemble_results['evaluation'] = evaluation
                ensemble_results['feature_names'] = feature_names
                
                all_results[model_name] = ensemble_results
                
                # Print evaluation results
                print(f"\nBest model: {ensemble_results['best_model_name']}")
                if config['type'] == 'classification':
                    print(f"Test Accuracy: {evaluation['accuracy']:.4f}")
                    if 'roc_auc' in evaluation:
                        print(f"Test AUC: {evaluation['roc_auc']:.4f}")
                else:
                    print(f"Test R²: {evaluation['r2']:.4f}")
                    print(f"Test RMSE: {evaluation['rmse']:.4f}")
        
        # Store training session
        session_info = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'test_size': test_size,
            'optimize_hyperparameters': optimize_hyperparameters,
            'models_trained': list(all_results.keys())
        }
        
        self.training_history.append(session_info)
        
        return all_results
    
    def save_models(self, results, model_dir='models', config_dir='config'):
        """
        Save trained models and configurations
        
        Args:
            results (dict): Training results
            model_dir (str): Directory to save model files
            config_dir (str): Directory to save configuration files
        """
        import os
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        for model_name, result in results.items():
            # Save best model
            best_model = result['best_model']['model']
            model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
            joblib.dump(best_model, model_path)
            
            # Save scaler if exists
            scaler = result['best_model']['scaler']
            if scaler is not None:
                scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
                joblib.dump(scaler, scaler_path)
            
            # Save configuration
            config = {
                'model_name': model_name,
                'best_model_type': result['best_model_name'],
                'target': result['target'],
                'ensemble_type': result['ensemble_type'],
                'feature_names': result['feature_names'],
                'best_params': result['best_model']['best_params'],
                'cv_score': result['best_model']['cv_mean'],
                'test_performance': result['evaluation'],
                'use_scaled_features': result['best_model']['use_scaled_features']
            }
            
            config_path = os.path.join(config_dir, f"{model_name}_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        
        print(f"\nModels saved to {model_dir}/")
        print(f"Configurations saved to {config_dir}/")
    
    def load_model(self, model_name, model_dir='models', config_dir='config'):
        """
        Load saved model and configuration
        
        Args:
            model_name (str): Name of model to load
            model_dir (str): Directory containing model files
            config_dir (str): Directory containing configuration files
            
        Returns:
            dict: Loaded model and configuration
        """
        import os
        
        # Load model
        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        model = joblib.load(model_path)
        
        # Load scaler if exists
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        
        # Load configuration
        config_path = os.path.join(config_dir, f"{model_name}_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'config': config
        }

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic NBA features
    feature_data = {
        'home_pace': np.random.normal(100, 5, n_samples),
        'away_pace': np.random.normal(100, 5, n_samples),
        'home_net_rating': np.random.normal(0, 8, n_samples),
        'away_net_rating': np.random.normal(0, 8, n_samples),
        'rest_differential': np.random.randint(-3, 4, n_samples),
        'home_win_pct_recent': np.random.uniform(0.3, 0.7, n_samples),
        'away_win_pct_recent': np.random.uniform(0.3, 0.7, n_samples),
        'pace_advantage': np.random.normal(0, 3, n_samples),
        'home_injury_impact': np.random.uniform(0, 0.3, n_samples),
        'away_injury_impact': np.random.uniform(0, 0.3, n_samples)
    }
    
    df = pd.DataFrame(feature_data)
    
    # Generate synthetic targets
    # Spread predictions based on features
    spread_prob = 1 / (1 + np.exp(-(df['home_net_rating'] - df['away_net_rating'] + 
                                   df['rest_differential'] * 2) / 10))
    df['covers_spread'] = (np.random.random(n_samples) < spread_prob).astype(int)
    
    # Moneyline predictions
    ml_prob = 1 / (1 + np.exp(-(df['home_net_rating'] - df['away_net_rating'] + 3) / 8))
    df['home_wins'] = (np.random.random(n_samples) < ml_prob).astype(int)
    
    # Total predictions
    total_points = 210 + df['home_pace'] * 0.5 + df['away_pace'] * 0.5 + np.random.normal(0, 8, n_samples)
    df['total_points'] = total_points
    df['goes_over'] = (total_points > 215).astype(int)
    
    # Actual spread
    df['actual_spread'] = (df['home_net_rating'] - df['away_net_rating'] + 
                          np.random.normal(0, 5, n_samples))
    
    # Add dates for time-aware splitting
    df['date'] = pd.date_range('2023-10-01', periods=n_samples, freq='D')
    df['game_id'] = range(n_samples)
    
    # Initialize trainer
    trainer = NBAModelTrainer(random_state=42)
    
    # OLD WAY (Random Split - CHEATING):
    # results = trainer.train_all_models(df, test_size=0.2, optimize_hyperparameters=False)
    
    # NEW WAY (Time Series CV - REALITY):
    print("\n" + "="*80)
    print("🔥 TRAINING WITH TIME SERIES CROSS-VALIDATION (No Future Leakage)")
    print("="*80)
    
    # 1. Define initial feature list (All numeric columns except target)
    initial_features = ['home_pace', 'away_pace', 'home_net_rating', 'away_net_rating',
                       'rest_differential', 'home_win_pct_recent', 'away_win_pct_recent',
                       'pace_advantage', 'home_injury_impact', 'away_injury_impact']
    
    # 2. RUN THE AUDIT (Strip out redundant features)
    # This will remove collinear features before the model sees them
    feature_cols = trainer.remove_collinear_features(df, initial_features, threshold=0.95)
    
    # 3. Create XGBoost model for spread prediction
    model_pipeline = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 4. Train with Time Series CV (prevents future leakage) using CLEAN features
    best_model, real_accuracy = trainer.train_with_tscv(
        df=df,
        model=model_pipeline,
        features=feature_cols,  # <--- Pass the cleaned list here
        target='covers_spread',  # Using spread classifier as example
        n_splits=5
    )
    
    # Save the robust model
    import os
    os.makedirs('models/production', exist_ok=True)
    joblib.dump(best_model, 'models/production/best_model.joblib')
    print("\n✅ Production model saved to models/production/best_model.joblib")
    print(f"   Expected Real-World Accuracy: {real_accuracy:.1%}")
    print(f"   Edge over Break-Even (52.4%): {(real_accuracy - 0.524):.1%}")
    
    # OPTIONAL: Still train all models using the old method for comparison
    print("\n" + "="*80)
    print("📊 Training Full Ensemble (Old Method - For Comparison Only)")
    print("="*80)
    results = trainer.train_all_models(df, test_size=0.2, optimize_hyperparameters=False)
    trainer.save_models(results)
    
    print("\n✅ Training completed successfully!")
    print("\n💡 TIP: The TSCV accuracy is your TRUE expected performance.")
    print("        The old method accuracy is likely OVERLY OPTIMISTIC.")
