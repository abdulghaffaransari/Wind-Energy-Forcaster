"""
Feature engineering for wind energy forecasting
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import yaml


class FeatureEngineer:
    """Create features for time series forecasting"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        Initialize FeatureEngineer
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'wind_generation_actual') -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        lag_periods = self.feature_config.get('lag_features', [1, 2, 3, 7, 14, 30])
        
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'wind_generation_actual') -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        windows = self.feature_config.get('rolling_windows', [7, 14, 30])
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with seasonal features
        """
        df = df.copy()
        
        if self.feature_config.get('seasonal_features', True):
            df['month'] = df.index.month
            df['day_of_year'] = df.index.dayofyear
            df['week_of_year'] = df.index.isocalendar().week
            df['day_of_week'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temperature-related features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temperature features
        """
        df = df.copy()
        
        if self.feature_config.get('temperature_features', True):
            # Temperature lags
            for lag in [1, 2, 3, 7]:
                df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)
            
            # Temperature rolling statistics
            for window in [7, 14]:
                df[f'temperature_rolling_mean_{window}'] = df['temperature'].rolling(window=window).mean()
                df[f'temperature_rolling_std_{window}'] = df['temperature'].rolling(window=window).std()
        
        return df
    
    def create_capacity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create capacity-related features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with capacity features
        """
        df = df.copy()
        
        if self.feature_config.get('capacity_ratio', True):
            # Capacity utilization ratio
            df['capacity_utilization'] = df['wind_generation_actual'] / (df['wind_capacity'] + 1e-8)
            
            # Capacity change
            df['capacity_change'] = df['wind_capacity'].diff()
            df['capacity_change_pct'] = df['wind_capacity'].pct_change()
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        print("Creating features...")
        
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_seasonal_features(df)
        df = self.create_temperature_features(df)
        df = self.create_capacity_features(df)
        
        # Remove rows with NaN values (from lag and rolling features)
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)
        
        print(f"Features created. Removed {removed} rows with NaN values.")
        print(f"Final shape: {df.shape}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
        """
        Get list of feature names (excluding target and timestamp)
        
        Args:
            df: DataFrame
            exclude_cols: Columns to exclude from features
            
        Returns:
            List of feature names
        """
        exclude_cols = exclude_cols or ['wind_generation_actual']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
