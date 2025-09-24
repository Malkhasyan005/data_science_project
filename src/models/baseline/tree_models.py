from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from typing import Dict, Optional
from ..base_model import BaseForecastingModel
import logging

logger = logging.getLogger(__name__)

class TreeForecastingModel(BaseForecastingModel):
    """
    Tree-based models for demand forecasting.
    
    Teaching benefits:
    1. Handle non-linear relationships naturally
    2. Robust to outliers and missing values
    3. Provide feature importance
    4. No need for feature scaling
    5. Can capture complex interactions automatically
    
    Supports multiple tree-based algorithms:
    - DecisionTree: Single tree (prone to overfitting, good for teaching)
    - RandomForest: Ensemble of trees (generally robust)
    - GradientBoosting: Sequential ensemble (often high performance)
    """
    
    def __init__(self, model_type: str = 'random_forest', config: Dict = None):
        """
        Initialize tree-based model.
        
        Args:
            model_type: 'decision_tree', 'random_forest', or 'gradient_boosting'
            config: Model hyperparameters
        """
        super().__init__(f"Tree_{model_type}", config)
        self.model_type = model_type
        self.imputer = SimpleImputer(strategy='median')
        
        # Initialize the appropriate model with good defaults
        if model_type == 'decision_tree':
            self.model = DecisionTreeRegressor(
                max_depth=config.get('max_depth', 10) if config else 10,
                min_samples_split=config.get('min_samples_split', 20) if config else 20,
                min_samples_leaf=config.get('min_samples_leaf', 10) if config else 10,
                random_state=42
            )
            
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=config.get('n_estimators', 100) if config else 100,
                max_depth=config.get('max_depth', 15) if config else 15,
                min_samples_split=config.get('min_samples_split', 10) if config else 10,
                min_samples_leaf=config.get('min_samples_leaf', 5) if config else 5,
                max_features=config.get('max_features', 'sqrt') if config else 'sqrt',
                random_state=42,
                n_jobs=-1  # Use all cores for faster training
            )
            
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=config.get('n_estimators', 100) if config else 100,
                learning_rate=config.get('learning_rate', 0.1) if config else 0.1,
                max_depth=config.get('max_depth', 6) if config else 6,
                min_samples_split=config.get('min_samples_split', 10) if config else 10,
                min_samples_leaf=config.get('min_samples_leaf', 5) if config else 5,
                subsample=config.get('subsample', 0.8) if config else 0.8,
                random_state=42
            )
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'TreeForecastingModel':
        """
        Fit tree-based model with minimal preprocessing.
        
        Teaching point: Tree models are much more robust to raw data
        compared to linear models - less preprocessing needed!
        """
        self.validate_input(X, y)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Only keep numeric columns (trees can't handle strings directly)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        if len(numeric_cols) < len(X.columns):
            dropped_cols = set(X.columns) - set(numeric_cols)
            logger.warning(f"Dropped non-numeric columns: {list(dropped_cols)}")
            self.feature_names = numeric_cols.tolist()
        
        logger.info(f"Fitting {self.model_name} with {len(self.feature_names)} features...")
        
        # Handle missing values (trees can handle some, but imputation is safer)
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_numeric),
            columns=X_numeric.columns,
            index=X_numeric.index
        )
        
        # Fit the model
        self.model.fit(X_imputed, y)
        self.is_fitted = True
        
        # Store training information
        if hasattr(self.model, 'score'):
            train_score = self.model.score(X_imputed, y)
            self.training_history['train_r2'] = train_score
            logger.info(f"Training completed. R² score: {train_score:.4f}")
        
        self.training_history['n_features'] = len(self.feature_names)
        self.training_history['n_samples'] = len(X_imputed)
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions using the fitted tree model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use the same features as training
        X_features = X[self.feature_names]
        
        # Apply same imputation as training
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_features),
            columns=X_features.columns,
            index=X_features.index
        )
        
        # Generate predictions
        predictions = self.model.predict(X_imputed)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance_detailed(self) -> Dict:
        """
        Get detailed feature importance analysis.
        
        Teaching value: Tree models provide rich interpretability information
        that helps students understand what the model learned.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        # Get basic feature importance
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Create detailed analysis
        total_importance = sum(importance_dict.values())
        
        detailed_analysis = {
            'feature_importance': importance_dict,
            'top_10_features': sorted_importance[:10],
            'importance_concentration': {
                'top_5_share': sum([imp for _, imp in sorted_importance[:5]]) / total_importance,
                'top_10_share': sum([imp for _, imp in sorted_importance[:10]]) / total_importance
            }
        }
        
        # Add model-specific information
        if hasattr(self.model, 'n_features_'):
            detailed_analysis['model_info'] = {
                'n_features_used': self.model.n_features_,
                'n_outputs': getattr(self.model, 'n_outputs_', 1)
            }
        
        # For ensemble models, add ensemble-specific info
        if hasattr(self.model, 'estimators_'):
            detailed_analysis['ensemble_info'] = {
                'n_estimators': len(self.model.estimators_),
                'estimator_type': type(self.model.estimators_[0]).__name__
            }
        
        return detailed_analysis