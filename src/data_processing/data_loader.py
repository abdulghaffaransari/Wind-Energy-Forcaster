"""
Data loading utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import yaml


class DataLoader:
    """Load and validate wind energy data"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        Initialize DataLoader
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = self.config['data']['raw_data_path']
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file
        
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {self.raw_data_path}...")
        
        # Load CSV
        df = pd.read_csv(self.raw_data_path)
        
        # Convert timestamp to datetime
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
        
        # Set timestamp as index
        df.set_index('utc_timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid
        """
        print("Validating data...")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"Warning: Missing values found:\n{missing[missing > 0]}")
        
        # Check for negative values in non-negative columns
        if (df['wind_generation_actual'] < 0).any():
            print("Warning: Negative wind generation values found")
        
        if (df['wind_capacity'] < 0).any():
            print("Warning: Negative wind capacity values found")
        
        # Check for outliers
        for col in ['wind_generation_actual', 'wind_capacity', 'temperature']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
            if outliers > 0:
                print(f"Warning: {outliers} outliers found in {col}")
        
        print("Data validation complete")
        return True
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionary with data information
        """
        return {
            'shape': df.shape,
            'date_range': (df.index.min(), df.index.max()),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': df.describe().to_dict(),
        }
