"""
XGBoost Model for Wind Energy Forecasting
"""

import numpy as np
from xgboost import XGBRegressor
from typing import Optional
import yaml
from pathlib import Path

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model for time series forecasting"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        Initialize XGBoost Model
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__("XGBoost")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['models']['xgboost']
        
        self.model = XGBRegressor(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            learning_rate=model_config['learning_rate'],
            subsample=model_config['subsample'],
            colsample_bytree=model_config['colsample_bytree'],
            random_state=model_config['random_state'],
            n_jobs=-1,
            verbosity=0
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history
        """
        print("Training XGBoost model...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        return {
            'train_loss': self.model.evals_result()['validation_0']['rmse'],
            'val_loss': self.model.evals_result().get('validation_1', {}).get('rmse', [])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
