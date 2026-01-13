"""
Main training script for Wind Energy Forecasting
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import joblib

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from data_processing.data_loader import DataLoader
from data_processing.feature_engineering import FeatureEngineer
from data_processing.data_preprocessor import DataPreprocessor
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.prophet_model import ProphetModel
from models.ensemble_model import EnsembleModel
from utils.logger import setup_logger
from utils.metrics import calculate_metrics, print_metrics
from utils.visualization import plot_predictions, plot_residuals


def load_config(config_path: str = "src/config/config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(config):
    """Load and prepare data"""
    logger = setup_logger()
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    loader.validate_data(df)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.create_all_features(df)
    
    # Get features and target
    feature_cols = feature_engineer.get_feature_names(df)
    X = df[feature_cols].values
    y = df['wind_generation_actual'].values
    dates = df.index
    
    # Train-test split
    split_idx = int(len(df) * config['data']['train_test_split'])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    # Validation split
    val_split_idx = int(len(X_train) * (1 - config['data']['validation_split']))
    X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    dates_train_final, dates_val = dates_train[:val_split_idx], dates_train[val_split_idx:]
    
    logger.info(f"Training set: {len(X_train_final)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return (X_train_final, y_train_final, dates_train_final,
            X_val, y_val, dates_val,
            X_test, y_test, dates_test,
            feature_cols, df)


def train_model(model_name: str, X_train, y_train, X_val, y_val, dates_train, dates_val, config, feature_cols):
    """Train a specific model"""
    logger = setup_logger()
    logger.info(f"Training {model_name} model...")
    
    # Initialize model
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
        # Prophet needs dates and feature names
        model.train(X_train, y_train, X_val, y_val, dates_train, feature_cols)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Preprocessing for tree-based models
    if model_name in ["XGBoost", "LightGBM"]:
        # No scaling needed for tree models
        model.train(X_train, y_train, X_val, y_val)
    else:
        # Scaling for neural networks
        preprocessor = DataPreprocessor()
        X_train_scaled, y_train_scaled = preprocessor.fit_transform(
            pd.DataFrame(X_train), pd.Series(y_train)
        )
        X_val_scaled, y_val_scaled = preprocessor.transform(
            pd.DataFrame(X_val), pd.Series(y_val)
        )
        
        # Store preprocessor for later use
        model.preprocessor = preprocessor
        
        # Train
        if model_name in ["LSTM", "Transformer"]:
            model.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        else:
            model.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    
    return model


def evaluate_model(model, X_test, y_test, dates_test, model_name, preprocessor=None):
    """Evaluate model on test set"""
    logger = setup_logger()
    
    # Make predictions
    if model_name in ["LSTM", "Transformer"]:
        if hasattr(model, 'preprocessor'):
            X_test_scaled, _ = model.preprocessor.transform(
                pd.DataFrame(X_test), pd.Series(y_test)
            )
            y_pred_scaled = model.predict_batch(X_test_scaled)
            y_pred = model.preprocessor.inverse_transform_target(y_pred_scaled)
        else:
            y_pred = model.predict_batch(X_test)
        
        # For sequence models, predictions will be shorter due to sequence creation
        # Align with test data by taking the last len(y_pred) samples
        seq_len = model.sequence_length if hasattr(model, 'sequence_length') else 30
        if len(y_pred) < len(y_test):
            # Adjust y_test and dates_test to match predictions
            offset = len(y_test) - len(y_pred)
            y_test_aligned = y_test[offset:]
            dates_test_aligned = dates_test[offset:]
        else:
            y_test_aligned = y_test
            dates_test_aligned = dates_test
        
        # Ensure same length
        min_len = min(len(y_pred), len(y_test_aligned))
        y_pred = y_pred[:min_len]
        y_test_aligned = y_test_aligned[:min_len]
        dates_test_aligned = dates_test_aligned[:min_len]
        
        y_test = y_test_aligned
        dates_test = dates_test_aligned
        
    elif model_name == "Prophet":
        y_pred = model.predict(None, dates=dates_test)
        y_pred = y_pred[-len(y_test):]
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, model_name)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': dates_test,
        'actual': y_test,
        'predicted': y_pred
    })
    predictions_path = f"outputs/predictions/{model_name.lower()}_predictions.csv"
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    
    # Visualizations
    plot_predictions(
        y_test, y_pred, dates_test,
        title=f"{model_name} - Predictions vs Actual",
        save_path=f"outputs/visualizations/{model_name.lower()}_predictions.png"
    )
    
    plot_residuals(
        y_test, y_pred,
        title=f"{model_name} - Residual Analysis",
        save_path=f"outputs/visualizations/{model_name.lower()}_residuals.png"
    )
    
    return metrics, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train wind energy forecasting models")
    parser.add_argument("--model", type=str, default="all",
                       choices=["all", "LSTM", "Transformer", "XGBoost", "LightGBM", "Prophet"],
                       help="Model to train")
    parser.add_argument("--config", type=str, default="src/config/config.yaml",
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Prepare data
    (X_train, y_train, dates_train,
     X_val, y_val, dates_val,
     X_test, y_test, dates_test,
     feature_cols, df) = prepare_data(config)
    
    # Models to train
    models_to_train = ["LSTM", "Transformer", "XGBoost", "LightGBM", "Prophet"] if args.model == "all" else [args.model]
    
    trained_models = {}
    all_metrics = {}
    
    # Train models
    for model_name in models_to_train:
        try:
            model = train_model(model_name, X_train, y_train, X_val, y_val,
                              dates_train, dates_val, config, feature_cols)
            
            # Save model
            model_path = f"models/saved_models/{model_name.lower()}_model"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            
            # Save preprocessor if it exists
            preprocessor = getattr(model, 'preprocessor', None)
            if preprocessor is not None:
                preprocessor_path = f"models/saved_models/{model_name.lower()}_preprocessor.pkl"
                preprocessor.save_scalers(preprocessor_path)
            
            # Evaluate
            metrics, predictions = evaluate_model(
                model, X_test, y_test, dates_test, model_name, preprocessor
            )
            
            trained_models[model_name] = model
            all_metrics[model_name] = metrics
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Save metrics summary
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_path = "outputs/reports/model_metrics.csv"
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path)
    
    print("\n" + "="*60)
    print("Training complete! All models saved and evaluated.")
    print("="*60)


if __name__ == "__main__":
    main()
