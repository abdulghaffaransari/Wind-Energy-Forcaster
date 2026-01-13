"""
Prediction script for Wind Energy Forecasting
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from data_processing.data_loader import DataLoader
from data_processing.feature_engineering import FeatureEngineer
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.prophet_model import ProphetModel


def load_model(model_name: str, model_path: str):
    """Load a trained model"""
    if model_name == "LSTM":
        model = LSTMModel()
    elif model_name == "Transformer":
        model = TransformerModel()
    elif model_name == "XGBoost":
        model = XGBoostModel()
    elif model_name == "LightGBM":
        model = LightGBMModel()
    elif model_name == "Prophet":
        model = ProphetModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load(model_path)
    return model


def make_predictions(model_name: str, model_path: str, n_days: int = 30):
    """Make future predictions"""
    # Load data and prepare features
    loader = DataLoader()
    df = loader.load_data()
    
    feature_engineer = FeatureEngineer()
    df = feature_engineer.create_all_features(df)
    
    # Get latest data for prediction
    feature_cols = feature_engineer.get_feature_names(df)
    X_latest = df[feature_cols].values[-30:]  # Last 30 days for context
    
    # Ensure proper dtype
    X_latest = np.array(X_latest, dtype=np.float32)
    
    # Load model
    model = load_model(model_name, model_path)
    
    # Generate future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D')
    
    # Make predictions
    if model_name == "Prophet":
        predictions = model.predict(None, dates=future_dates)
        predictions = predictions[-n_days:]  # Get last n_days predictions
    elif model_name in ["LSTM", "Transformer"]:
        # For sequence models, we need to use the last sequence
        # Try to load preprocessor if it exists
        import pickle
        preprocessor_path = model_path.replace('_model', '_preprocessor.pkl')
        if Path(preprocessor_path).exists():
            from data_processing.data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
            preprocessor.load_scalers(preprocessor_path)
            X_scaled, _ = preprocessor.transform(
                pd.DataFrame(X_latest), None
            )
            # Use only the last sequence for prediction
            # For multi-step prediction, we'd need to iteratively predict
            # For now, we'll use the last sequence and repeat if needed
            seq_len = model.sequence_length if hasattr(model, 'sequence_length') else 30
            if len(X_scaled) >= seq_len:
                X_seq_input = X_scaled[-seq_len:].reshape(1, seq_len, -1)
                # Make single prediction
                pred_scaled = model.model.predict(X_seq_input, verbose=0)
                # Inverse transform - ensure it's a numpy array
                pred_scaled_array = np.array(pred_scaled).flatten()
                pred = preprocessor.inverse_transform_target(pred_scaled_array)
                # For multiple days, repeat the prediction (simplified approach)
                predictions = np.repeat(pred[0], n_days)
            else:
                raise ValueError(f"Need at least {seq_len} samples for prediction")
        else:
            # If no preprocessor, use raw data
            seq_len = model.sequence_length if hasattr(model, 'sequence_length') else 30
            if len(X_latest) >= seq_len:
                X_seq_input = X_latest[-seq_len:].reshape(1, seq_len, -1)
                pred = model.model.predict(X_seq_input, verbose=0)
                predictions = np.repeat(pred[0], n_days)
            else:
                raise ValueError(f"Need at least {seq_len} samples for prediction")
    else:
        # For tree models, use the latest features
        # For multiple days, repeat the prediction (simplified)
        single_pred = model.predict(X_latest[-1:].reshape(1, -1))
        predictions = np.repeat(single_pred[0], n_days)
    
    # Ensure predictions array matches n_days
    predictions = predictions[:n_days]
    
    # Create results dataframe
    results = pd.DataFrame({
        'date': future_dates[:len(predictions)],
        'predicted_wind_generation': predictions
    })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained models")
    parser.add_argument("--model", type=str, required=True,
                       choices=["LSTM", "Transformer", "XGBoost", "LightGBM", "Prophet"],
                       help="Model to use for prediction")
    parser.add_argument("--model_path", type=str,
                       default=None,
                       help="Path to saved model")
    parser.add_argument("--n_days", type=int, default=30,
                       help="Number of days to predict")
    parser.add_argument("--output", type=str, default="outputs/predictions/future_predictions.csv",
                       help="Output file path")
    
    args = parser.parse_args()
    
    model_path = args.model_path or f"models/saved_models/{args.model.lower()}_model"
    
    # Make predictions
    results = make_predictions(args.model, model_path, args.n_days)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    
    print(f"\nPredictions saved to {args.output}")
    print("\nFirst 10 predictions:")
    print(results.head(10))


if __name__ == "__main__":
    main()
