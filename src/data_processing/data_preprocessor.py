"""
Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import pickle
from pathlib import Path


class DataPreprocessor:
    """Preprocess data for machine learning models"""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize DataPreprocessor
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.feature_scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit scalers and transform data
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Returns:
            Transformed X and y
        """
        X_scaled = self.feature_scaler.fit_transform(X)
        
        if y is not None:
            y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            self.is_fitted = True
            return X_scaled, y_scaled
        
        self.is_fitted = True
        return X_scaled, None
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted scalers
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Returns:
            Transformed X and y
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        
        X_scaled = self.feature_scaler.transform(X)
        
        if y is not None:
            y_scaled = self.scaler.transform(y.values.reshape(-1, 1)).flatten()
            return X_scaled, y_scaled
        
        return X_scaled, None
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Original scale target values
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def save_scalers(self, filepath: str) -> None:
        """
        Save scalers to file
        
        Args:
            filepath: Path to save scalers
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler,
                'scaler_type': self.scaler_type
            }, f)
    
    def load_scalers(self, filepath: str) -> None:
        """
        Load scalers from file
        
        Args:
            filepath: Path to load scalers from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_scaler = data['feature_scaler']
            self.scaler_type = data['scaler_type']
            self.is_fitted = True
