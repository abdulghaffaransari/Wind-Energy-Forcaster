"""
Transformer Model for Wind Energy Forecasting
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from typing import Tuple, Optional
import yaml
from pathlib import Path

from .base_model import BaseModel


class TransformerModel(BaseModel):
    """Transformer model for time series forecasting"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        Initialize Transformer Model
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__("Transformer")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['models']['transformer']
        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_layers = model_config['num_layers']
        self.dim_feedforward = model_config['dim_feedforward']
        self.dropout = model_config['dropout']
        self.sequence_length = model_config['sequence_length']
        self.epochs = model_config['epochs']
        self.batch_size = model_config['batch_size']
        self.learning_rate = model_config['learning_rate']
        self.early_stopping_patience = model_config['early_stopping_patience']
        
        self.model = None
        self.history = None
    
    def _transformer_encoder(self, inputs, num_layers, d_model, num_heads, dff, dropout_rate):
        """Transformer encoder block"""
        x = inputs
        
        for _ in range(num_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model
            )(x, x)
            attn_output = layers.Dropout(dropout_rate)(attn_output)
            out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed forward network
            ffn_output = layers.Dense(dff, activation='relu')(out1)
            ffn_output = layers.Dense(d_model)(ffn_output)
            ffn_output = layers.Dropout(dropout_rate)(ffn_output)
            out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
            
            x = out2
        
        return x
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build Transformer model
        
        Args:
            input_shape: Input shape
            
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        x = layers.Dense(self.d_model)(inputs)
        
        # Transformer encoder
        x = self._transformer_encoder(
            x, self.num_layers, self.d_model, self.nhead,
            self.dim_feedforward, self.dropout
        )
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for transformer input"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """Train Transformer model"""
        print("Preparing sequences for Transformer...")
        
        X_seq_train, y_seq_train = self._create_sequences(
            np.column_stack([X_train, y_train]), self.sequence_length
        )
        X_train_seq = X_seq_train[:, :, :-1]
        y_train_seq = X_seq_train[:, -1, -1]
        
        if X_val is not None and y_val is not None:
            X_seq_val, y_seq_val = self._create_sequences(
                np.column_stack([X_val, y_val]), self.sequence_length
            )
            X_val_seq = X_seq_val[:, :, :-1]
            y_val_seq = X_seq_val[:, -1, -1]
        else:
            X_val_seq = None
            y_val_seq = None
        
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        print("Model architecture:")
        self.model.summary()
        
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
        
        print("Training Transformer model...")
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
        Make predictions (required by BaseModel)
        
        Args:
            X: Input features (must have at least sequence_length samples)
            
        Returns:
            Predictions
        """
        return self.predict_batch(X)
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Make batch predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(X) < self.sequence_length:
            raise ValueError(f"Input must have at least {self.sequence_length} samples")
        
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
        """Save model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"Transformer model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Transformer model loaded from {filepath}")
