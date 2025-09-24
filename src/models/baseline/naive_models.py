# src/models/baseline/naive_models.py
import pandas as pd
import numpy as np
from typing import Dict, Optional
from ..base_model import BaseForecastingModel
import logging
from typing import List

logger = logging.getLogger(__name__)

class NaiveForecaster(BaseForecastingModel):
    """
    Naive forecasting models - the starting point for any forecasting project.
    
    Teaching insight: Always start with simple baselines! These models:
    1. Are easy to understand and implement
    2. Provide a benchmark for more complex models
    3. Often work surprisingly well
    4. Help identify if complex models are actually adding value
    
    This class implements several naive strategies:
    - Last value (persistence model)
    - Seasonal naive (same time last week/day)
    - Average (overall mean)
    """
    
    def __init__(self, strategy: str = 'seasonal', seasonal_period: int = 24, config: Dict = None):
        """
        Initialize the naive forecaster.
        
        Args:
            strategy: 'last', 'seasonal', 'mean', or 'drift'
            seasonal_period: For seasonal strategy (24 for daily pattern, 168 for weekly)
            config: Additional configuration
        """
        super().__init__(f"Naive_{strategy}", config)
        self.strategy = strategy
        self.seasonal_period = seasonal_period
        self.fitted_values = {}
        
        if strategy not in ['last', 'seasonal', 'mean', 'drift']:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from ['last', 'seasonal', 'mean', 'drift']")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'NaiveForecaster':
        """
        Fit the naive model by storing necessary historical values.
        
        Teaching point: Naive models don't really "fit" in the traditional sense,
        but they do need to store some historical information for predictions.
        """
        self.validate_input(X, y)
        
        # We need store_id and product_id to group time series
        if 'store_id' not in X.columns or 'product_id' not in X.columns:
            raise ValueError("Naive forecaster requires 'store_id' and 'product_id' columns")
        
        # Combine features with target for easier grouping
        data = X.copy()
        data['target'] = y
        
        logger.info(f"Fitting {self.model_name} model...")
        
        # Store the fitted values based on strategy
        for (store_id, product_id), group in data.groupby(['store_id', 'product_id']):
            group = group.sort_values('dt')
            key = (store_id, product_id)
            
            if self.strategy == 'last':
                # Use the last observed value
                self.fitted_values[key] = group['target'].iloc[-1]
                
            elif self.strategy == 'seasonal':
                # Store the last seasonal_period values for seasonal naive
                self.fitted_values[key] = group['target'].iloc[-self.seasonal_period:].values
                
            elif self.strategy == 'mean':
                # Store the historical mean
                self.fitted_values[key] = group['target'].mean()
                
            elif self.strategy == 'drift':
                # Linear trend from first to last observation
                if len(group) > 1:
                    first_val = group['target'].iloc[0]
                    last_val = group['target'].iloc[-1]
                    trend = (last_val - first_val) / (len(group) - 1)
                    self.fitted_values[key] = {'last_value': last_val, 'trend': trend}
                else:
                    self.fitted_values[key] = {'last_value': group['target'].iloc[0], 'trend': 0}
        
        self.is_fitted = True
        self.training_history['num_time_series'] = len(self.fitted_values)
        
        logger.info(f"Fitted naive model for {len(self.fitted_values)} store-product combinations")
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate naive predictions based on the fitted strategy.
        
        Teaching insight: Notice how different naive strategies capture
        different aspects of time series behavior - trends, seasonality, etc.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for _, row in X.iterrows():
            key = (row['store_id'], row['product_id'])
            
            if key not in self.fitted_values:
                # Handle new store-product combinations with overall mean
                # This is a common issue in production systems
                logger.warning(f"Unknown store-product combination: {key}. Using fallback prediction.")
                pred = 0  # Conservative fallback for new combinations
            else:
                if self.strategy == 'last':
                    pred = self.fitted_values[key]
                    
                elif self.strategy == 'seasonal':
                    # Use seasonal pattern - this requires knowing position in season
                    seasonal_values = self.fitted_values[key]
                    # For simplicity, use the mean of seasonal values
                    # In practice, you'd want to track the exact seasonal position
                    pred = np.mean(seasonal_values)
                    
                elif self.strategy == 'mean':
                    pred = self.fitted_values[key]
                    
                elif self.strategy == 'drift':
                    values = self.fitted_values[key]
                    # For simplicity, predict next value in trend
                    pred = values['last_value'] + values['trend']
            
            predictions.append(max(0, pred))  # Ensure non-negative predictions
        
        return np.array(predictions)

class SeasonalNaiveForecaster(BaseForecastingModel):
    """
    Advanced seasonal naive forecaster that properly handles seasonality.
    
    This is a more sophisticated version that tracks actual seasonal positions
    and can handle multiple seasonal patterns simultaneously.
    """
    
    def __init__(self, seasonal_periods: List[int] = [24, 168], config: Dict = None):
        """
        Initialize seasonal naive forecaster.
        
        Args:
            seasonal_periods: List of seasonal periods (24=daily, 168=weekly)
            config: Additional configuration
        """
        super().__init__("SeasonalNaive", config)
        self.seasonal_periods = seasonal_periods
        self.seasonal_data = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SeasonalNaiveForecaster':
        """Fit by storing seasonal patterns for each time series."""
        self.validate_input(X, y)
        
        data = X.copy()
        data['target'] = y
        data['dt'] = pd.to_datetime(data['dt'])
        
        logger.info("Fitting seasonal naive model...")
        
        for (store_id, product_id), group in data.groupby(['store_id', 'product_id']):
            group = group.sort_values('dt').reset_index(drop=True)
            key = (store_id, product_id)
            
            seasonal_patterns = {}
            for period in self.seasonal_periods:
                # Create seasonal indices
                seasonal_indices = np.arange(len(group)) % period
                pattern = {}
                for i in range(period):
                    mask = seasonal_indices == i
                    if mask.sum() > 0:
                        pattern[i] = group.loc[mask, 'target'].mean()
                    else:
                        pattern[i] = 0
                seasonal_patterns[period] = pattern
            
            self.seasonal_data[key] = seasonal_patterns
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions using seasonal patterns."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for _, row in X.iterrows():
            key = (row['store_id'], row['product_id'])
            
            if key not in self.seasonal_data:
                predictions.append(0)
                continue
            
            # Use the primary seasonal period (first one)
            primary_period = self.seasonal_periods[0]
            pattern = self.seasonal_data[key][primary_period]
            
            # Calculate seasonal index based on hour of day
            hour = pd.to_datetime(row['dt']).hour
            seasonal_idx = hour % primary_period
            
            pred = pattern.get(seasonal_idx, 0)
            predictions.append(max(0, pred))
        
        return np.array(predictions)