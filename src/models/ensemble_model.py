"""
Ensemble Model for Wind Energy Forecasting
"""

import numpy as np
from typing import List, Optional
from .base_model import BaseModel


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple models"""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        """
        Initialize Ensemble Model
        
        Args:
            models: List of trained models
            weights: Weights for each model (if None, equal weights)
        """
        super().__init__("Ensemble")
        
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train all models in ensemble
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training history
        """
        print("Training ensemble models...")
        
        history = {}
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            hist = model.train(X_train, y_train, X_val, y_val)
            history[model.model_name] = hist
        
        self.is_trained = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Input features
            
        Returns:
            Weighted average predictions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
