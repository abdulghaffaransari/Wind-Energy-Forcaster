"""
LSTM Model for Wind Energy Forecasting
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from typing import Tuple, Optional
import yaml
from pathlib import Path

from .base_model import BaseModel


class LSTMModel(BaseModel):
    """LSTM Neural Network for time series forecasting"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        Initialize LSTM Model
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__("LSTM")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['models']['lstm']
        self.sequence_length = model_config['sequence_length']
        self.hidden_units = model_config['hidden_units']
        self.dropout = model_config['dropout']
        self.epochs = model_config['epochs']
        self.batch_size = model_config['batch_size']
        self.learning_rate = model_config['learning_rate']
        self.early_stopping_patience = model_config['early_stopping_patience']
        
        self.model = None
        self.history = None
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            data: Input data
            seq_length: Sequence length
            
        Returns:
            X and y sequences
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.hidden_units[0],
            return_sequences=len(self.hidden_units) > 1,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional LSTM layers
        for units in self.hidden_units[1:]:
            model.add(layers.LSTM(units, return_sequences=False))
            model.add(layers.Dropout(self.dropout))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training history
        """
        print("Preparing sequences for LSTM...")
        
        # Create sequences
        X_seq_train, y_seq_train = self._create_sequences(
            np.column_stack([X_train, y_train]), self.sequence_length
        )
        
        # Split features and target from sequences
        X_train_seq = X_seq_train[:, :, :-1]  # All features except last (target)
        y_train_seq = X_seq_train[:, -1, -1]  # Last target value
        
        if X_val is not None and y_val is not None:
            X_seq_val, y_seq_val = self._create_sequences(
                np.column_stack([X_val, y_val]), self.sequence_length
            )
            X_val_seq = X_seq_val[:, :, :-1]
            y_val_seq = X_seq_val[:, -1, -1]
        else:
            X_val_seq = None
            y_val_seq = None
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        print(f"Model architecture:")
        self.model.summary()
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val_seq is not None else 'loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val_seq is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        print("Training LSTM model...")
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq) if X_val_seq is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_trained = True
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history['mae'],
            'val_mae': self.history.history.get('val_mae', [])
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
        
        # For prediction, we need to create sequences
        # This is a simplified version - in practice, you'd need to handle this differently
        # For now, we'll use the last sequence_length points
        if len(X) < self.sequence_length:
            raise ValueError(f"Input must have at least {self.sequence_length} samples")
        
        # Use last sequence_length points
        X_seq = X[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        prediction = self.model.predict(X_seq, verbose=0)
        
        return prediction.flatten()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (required by BaseModel)
        
        Args:
            X: Input features (must have at least sequence_length samples)
            
        Returns:
            Predictions
        """
        return self.predict_batch(X)
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Make batch predictions
        
        Args:
            X: Input features (must have at least sequence_length samples)
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(X) < self.sequence_length:
            raise ValueError(f"Input must have at least {self.sequence_length} samples")
        
        # Create sequences
        X_seq = []
        for i in range(self.sequence_length, len(X) + 1):
            X_seq.append(X[i - self.sequence_length:i])
        
        # Convert to numpy array and ensure proper dtype
        X_seq = np.array(X_seq, dtype=np.float32)
        
        # Handle any NaN or inf values
        if np.isnan(X_seq).any() or np.isinf(X_seq).any():
            X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        return predictions.flatten()
    
    def save(self, filepath: str) -> None:
        """
        Save model
        
        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"LSTM model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load model
        
        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"LSTM model loaded from {filepath}")
