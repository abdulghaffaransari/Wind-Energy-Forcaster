"""
Prophet Model for Wind Energy Forecasting
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from typing import Optional
import yaml

from .base_model import BaseModel


class ProphetModel(BaseModel):
    """Prophet model for time series forecasting"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        Initialize Prophet Model
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__("Prophet")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['models']['prophet']
        
        self.model = Prophet(
            yearly_seasonality=model_config['yearly_seasonality'],
            weekly_seasonality=model_config['weekly_seasonality'],
            daily_seasonality=model_config['daily_seasonality'],
            seasonality_mode=model_config['seasonality_mode']
        )
        
        self.feature_columns = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              dates: Optional[pd.DatetimeIndex] = None, feature_names: Optional[list] = None) -> dict:
        """
        Train Prophet model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, not used by Prophet)
            y_val: Validation targets (optional, not used by Prophet)
            dates: Date index for training data
            feature_names: Names of features (for adding regressors)
            
        Returns:
            Training history
        """
        print("Training Prophet model...")
        
        if dates is None:
            dates = pd.date_range(start='2017-01-01', periods=len(y_train), freq='D')
        
        # Create Prophet dataframe
        df = pd.DataFrame({
            'ds': dates[:len(y_train)],
            'y': y_train
        })
        
        # Add external regressors if provided
        if feature_names is not None and X_train is not None:
            for i, col in enumerate(feature_names[:X_train.shape[1]]):
                if col not in ['ds', 'y']:
                    df[col] = X_train[:len(y_train), i]
                    self.model.add_regressor(col)
        
        self.model.fit(df)
        self.is_trained = True
        
        return {'status': 'trained'}
    
    def predict(self, X: np.ndarray, dates: Optional[pd.DatetimeIndex] = None,
                periods: Optional[int] = None) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features (optional for Prophet)
            dates: Date index for predictions
            periods: Number of periods to forecast (if dates not provided)
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if dates is not None:
            future = pd.DataFrame({'ds': dates})
        elif periods is not None:
            future = self.model.make_future_dataframe(periods=periods)
        else:
            raise ValueError("Either dates or periods must be provided")
        
        # Add regressors if available
        if X is not None and self.model.extra_regressors:
            regressor_names = list(self.model.extra_regressors.keys())
            for i, name in enumerate(regressor_names[:X.shape[1]]):
                future[name] = X[:len(future), i]
        
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values
    
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        from pathlib import Path
        import pickle
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns
            }, f)
        print(f"Prophet model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_columns = data.get('feature_columns', None)
        self.is_trained = True
        print(f"Prophet model loaded from {filepath}")
