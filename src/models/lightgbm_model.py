"""
LightGBM Model for Wind Energy Forecasting
"""

import numpy as np
from lightgbm import LGBMRegressor
from typing import Optional
import yaml

from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM model for time series forecasting"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        Initialize LightGBM Model
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__("LightGBM")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['models']['lightgbm']
        
        self.model = LGBMRegressor(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            learning_rate=model_config['learning_rate'],
            num_leaves=model_config['num_leaves'],
            random_state=model_config['random_state'],
            n_jobs=-1,
            verbosity=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history
        """
        print("Training LightGBM model...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set
        )
        
        self.is_trained = True
        
        return {
            'train_loss': self.model.evals_result_['training']['l2'],
            'val_loss': self.model.evals_result_.get('valid_1', {}).get('l2', [])
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
