import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for retail demand forecasting.
    
    This class teaches students the art and science of feature engineering,
    which is often more impactful than choosing sophisticated algorithms.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize with preprocessing configuration.
        
        Teaching point: Feature engineering parameters should be configurable
        to enable easy experimentation and hyperparameter tuning.
        """
        self.config = config
        self.lag_periods = config['lag_periods']
        self.rolling_windows = config['rolling_windows']
        self.seasonal_periods = config['seasonal_periods']
        
    def create_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'dt') -> pd.DataFrame:
        """
        Create comprehensive time-based features, including Fourier seasonality.
        """
        df = df.copy()
        
        # Ensure datetime column is properly formatted
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        logger.info("Creating temporal features...")
        
        # Basic temporal components
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek  # 0=Monday
        df['day_of_month'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['quarter'] = df[datetime_col].dt.quarter
        df['week_of_year'] = df[datetime_col].dt.isocalendar().week
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business-relevant time indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
        
        # Retail-specific time periods
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_lunch_time'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

        # Day part bins
        df['day_part'] = pd.cut(
            df['hour'],
            bins=[0,6,11,17,22,24],
            labels=['night','morning','afternoon','evening','late'],
            right=False
        )

        # Holidays
        df['is_holiday'] = df['dt'].dt.strftime("%m-%d").isin(["05-27", "06-19"]).astype(int)
        
        # Time since start
        df['days_since_start'] = (df[datetime_col] - df[datetime_col].min()).dt.days
        df['hours_since_start'] = (df[datetime_col] - df[datetime_col].min()).dt.total_seconds() / 3600
        
        # Fourier seasonality
        t = np.arange(len(df))
        for k in range(1, 4):  # order=3
            # Daily cycle
            df[f'fourier_sin_24_{k}'] = np.sin(2 * np.pi * k * t / 24)
            df[f'fourier_cos_24_{k}'] = np.cos(2 * np.pi * k * t / 24)
            # Weekly cycle
            df[f'fourier_sin_168_{k}'] = np.sin(2 * np.pi * k * t / 168)
            df[f'fourier_cos_168_{k}'] = np.cos(2 * np.pi * k * t / 168)
        
        logger.info(f"Created {df.shape[1]} total features")
        return df

    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'sale_amount') -> pd.DataFrame:
        """
        Create lagged versions of the target variable.
        
        Teaching concept: Lag features capture the autocorrelation in time series.
        They're essential because past sales often predict future sales.
        """
        df = df.copy()
        
        # Sort data properly for lag calculation
        df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)

        # Ensure stock_status is numeric (fix for your bug)
        df['hours_stock_status'] = pd.to_numeric(df['hours_stock_status'], errors='coerce').fillna(0).astype(int)

        
        logger.info(f"Creating lag features for periods: {self.lag_periods}")
        
        # Create lag features for each store-product combination
        for lag in self.lag_periods:
            lag_col = f'{target_col}_lag_{lag}'
            
            # Use groupby to ensure lags are calculated within each time series
            df[lag_col] = df.groupby(['store_id', 'product_id'])[target_col].shift(lag)
            
            # For educational purposes, let's also create lag features for stockout status
            if lag <= 7:  # Only short-term lags for stockout (memory constraints)
                stockout_lag_col = f'stockout_lag_{lag}'
                df[stockout_lag_col] = df.groupby(['store_id', 'product_id'])['hours_stock_status'].shift(lag)
        df['unmet_demand'] = np.where((df['hours_stock_status'] == 1) & (df['sale_amount'] == 0) & (df['sale_amount_lag_1'] > 0),1, 0)

        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'sale_amount') -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Teaching insight: Rolling features capture local trends and patterns.
        They help the model understand if demand is increasing, decreasing, or stable.
        """
        df = df.copy()
        df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
        
        logger.info(f"Creating rolling features for windows: {self.rolling_windows}")
        
        for window in self.rolling_windows:
            # Rolling mean (local average demand)
            df[f'{target_col}_rolling_mean_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling standard deviation (demand volatility)
            df[f'{target_col}_rolling_std_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)  # Fill NaN with 0 for single observations
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling maximum (peak demand in window)
            df[f'{target_col}_rolling_max_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling median (robust central tendency)
            df[f'{target_col}_rolling_median_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .median()
                .reset_index(level=[0, 1], drop=True)
            )
        df['stockout_ratio_7h'] = (
            df.groupby(['store_id','product_id'])['hours_stock_status']
            .rolling(window=7, min_periods=1).mean()
            .reset_index(level=[0,1], drop=True))

        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from categorical variables.
        
        Teaching point: Categorical features often contain rich information
        that needs to be extracted thoughtfully.
        """
        df = df.copy()
        
        logger.info("Creating categorical and interaction features...")
        
        # Store-level aggregations (store characteristics)
        store_stats = df.groupby('store_id').agg({
            'sale_amount': ['mean', 'std', 'max'],
            'product_id': 'nunique'  # Number of products per store
        }).round(2)
        
        store_stats.columns = ['store_avg_sales', 'store_sales_volatility', 'store_max_sales', 'store_product_count']
        df = df.merge(store_stats, on='store_id', how='left')
        
        # Product-level aggregations (product characteristics)
        product_stats = df.groupby('product_id').agg({
            'sale_amount': ['mean', 'std'],
            'store_id': 'nunique'  # Number of stores selling this product
        }).round(2)
        
        product_stats.columns = ['product_avg_sales', 'product_sales_volatility', 'product_store_count']
        df = df.merge(product_stats, on='product_id', how='left')
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Teaching concept: Sometimes the combination of features is more
        informative than individual features alone.
        """
        df = df.copy()
        
        logger.info("Creating interaction features...")
        
        # Weather-time interactions (weather effect varies by time)
        df['temp_hour_interaction'] = df['avg_temperature'] * df['hour']
        df['rain_weekend_interaction'] = df['precpt'] * df['is_weekend']
        
        # Promotion-time interactions (promotion effectiveness varies by time)
        df['discount_weekend'] = df['discount'] * df['is_weekend']
        df['discount_evening'] = df['discount'] * df['is_evening_rush']
        
        # Stock-time interactions
        df['stock_hour_interaction'] = df['hours_stock_status'] * df['hour']
        
        return df
    


    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps in the correct order.
        
        Teaching point: Feature engineering pipeline order matters!
        Some features depend on others being created first.
        """
        logger.info("Starting comprehensive feature engineering pipeline...")
        
        original_cols = len(df.columns)
        
        # Step 1: Temporal features (these don't depend on other features)
        df = self.create_temporal_features(df)
        
        # Step 2: Categorical features (these create new base features)
        df = self.create_categorical_features(df)
        
        # Step 3: Lag features (these depend on having clean temporal data)
        df = self.create_lag_features(df)
        
        # Step 4: Rolling features (these depend on temporal ordering)
        df = self.create_rolling_features(df)
        
        # Step 5: Interaction features (these depend on base features existing)
        df = self.create_interaction_features(df)


        final_cols = len(df.columns)
        logger.info(f"Feature engineering complete: {original_cols} → {final_cols} features (+{final_cols - original_cols})")
        
        return df