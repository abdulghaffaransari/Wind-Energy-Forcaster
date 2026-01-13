"""
Constants for Wind Energy Forecasting Project
"""

# Column names
UTC_TIMESTAMP = "utc_timestamp"
WIND_GENERATION = "wind_generation_actual"
WIND_CAPACITY = "wind_capacity"
TEMPERATURE = "temperature"

# Date formats
DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"

# Model names
MODEL_LSTM = "LSTM"
MODEL_TRANSFORMER = "Transformer"
MODEL_XGBOOST = "XGBoost"
MODEL_LIGHTGBM = "LightGBM"
MODEL_PROPHET = "Prophet"
MODEL_ENSEMBLE = "Ensemble"

# File paths
MODELS_DIR = "models/saved_models"
CHECKPOINTS_DIR = "models/checkpoints"
OUTPUTS_DIR = "outputs"
LOGS_DIR = "logs"

# Visualization
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE = (14, 6)
DPI = 300
