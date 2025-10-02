import xgboost as xgb
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from typing import Dict
from ..base_model import BaseForecastingModel
import logging

logger = logging.getLogger(__name__)

class XGBoostForecastingModel(BaseForecastingModel):
    """
    XGBoost model for demand forecasting.
    
    Benefits:
    - Handles non-linear relationships
    - Works well with missing values
    - Provides feature importance
    - Often outperforms simple trees
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("XGBoost", config)
        self.imputer = SimpleImputer(strategy='median')
        self.config = config or {}
        
        # Default XGBoost parameters
        self.model_params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = xgb.XGBRegressor(**self.model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'XGBoostForecastingModel':
        """Fit XGBoost model with preprocessing."""
        self.validate_input(X, y)
        self.feature_names = X.columns.tolist()
        
        # Only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        if len(numeric_cols) < len(X.columns):
            dropped_cols = set(X.columns) - set(numeric_cols)
            logger.warning(f"Dropped non-numeric columns: {list(dropped_cols)}")
            self.feature_names = numeric_cols.tolist()
        
        logger.info(f"Fitting XGBoost with {len(self.feature_names)} features...")
        
        # Impute missing values
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_numeric),
            columns=X_numeric.columns,
            index=X_numeric.index
        )
        
        # Fit model
        self.model.fit(X_imputed, y, **kwargs)
        self.is_fitted = True
        
        train_score = self.model.score(X_imputed, y)
        self.training_history['train_r2'] = train_score
        self.training_history['n_features'] = len(self.feature_names)
        self.training_history['n_samples'] = len(X_imputed)
        
        logger.info(f"Training completed. R² score: {train_score:.4f}")
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict using fitted XGBoost model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_features = X[self.feature_names]
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_features),
            columns=X_features.columns,
            index=X_features.index
        )
        
        predictions = self.model.predict(X_imputed, **kwargs)
        return np.maximum(predictions, 0)
    
    def get_feature_importance_detailed(self) -> Dict:
        """Get detailed feature importance from XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        
        # Normalize and sort
        total_importance = sum(importance_dict.values())
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        detailed_analysis = {
            'feature_importance': importance_dict,
            'top_10_features': sorted_importance[:10],
            'importance_concentration': {
                'top_5_share': sum([imp for _, imp in sorted_importance[:5]]) / total_importance,
                'top_10_share': sum([imp for _, imp in sorted_importance[:10]]) / total_importance
            }
        }
        
        return detailed_analysis
