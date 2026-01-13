"""
Base model class for all ML models
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import pickle


class BaseModel(ABC):
    """Base class for all forecasting models"""
    
    def __init__(self, model_name: str):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
