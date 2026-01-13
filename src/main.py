"""
Main entry point for Wind Energy Forecasting Pipeline
"""

import argparse
import pandas as pd
from pathlib import Path

from data_processing.data_loader import DataLoader
from data_processing.feature_engineering import FeatureEngineer
from utils.logger import setup_logger
from utils.visualization import plot_time_series


def process_data():
    """Process and save data"""
    logger = setup_logger()
    logger.info("Starting data processing pipeline...")
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    loader.validate_data(df)
    
    # Save raw data info
    info = loader.get_data_info(df)
    logger.info(f"Data info: {info}")
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_processed = feature_engineer.create_all_features(df)
    
    # Save processed data
    output_path = "data/processed/processed_data.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path)
    logger.info(f"Processed data saved to {output_path}")
    
    # Save features
    features_path = "data/features/features.csv"
    feature_cols = feature_engineer.get_feature_names(df_processed)
    df_features = df_processed[feature_cols + ['wind_generation_actual']]
    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(features_path)
    logger.info(f"Features saved to {features_path}")
    
    # Create visualizations
    plot_time_series(
        df,
        ['wind_generation_actual', 'wind_capacity', 'temperature'],
        title="Wind Energy Data Overview",
        save_path="outputs/visualizations/data_overview.png"
    )
    
    logger.info("Data processing complete!")


def main():
    parser = argparse.ArgumentParser(description="Wind Energy Forecasting Pipeline")
    parser.add_argument("--mode", type=str, default="process",
                       choices=["process", "train", "predict"],
                       help="Pipeline mode")
    
    args = parser.parse_args()
    
    if args.mode == "process":
        process_data()
    elif args.mode == "train":
        from train import main as train_main
        train_main()
    elif args.mode == "predict":
        from predict import main as predict_main
        predict_main()


if __name__ == "__main__":
    main()
