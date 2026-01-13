#!/bin/bash

# Wind Energy Forecasting Project Setup Script
# This script creates the entire project structure

echo "🚀 Setting up Wind Energy Forecasting Project..."

# Create main directories
mkdir -p src/{data_processing,models,utils,config}
mkdir -p notebooks/{exploratory,experiments}
mkdir -p data/{raw,processed,features}
mkdir -p models/{saved_models,checkpoints}
mkdir -p outputs/{predictions,visualizations,reports}
mkdir -p logs
mkdir -p tests
mkdir -p dashboard/{static,templates}

echo "✅ Directory structure created"

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/data_processing/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch src/config/__init__.py

echo "✅ Python package structure created"

# Create placeholder files
touch src/config/config.yaml
touch src/config/constants.py
touch requirements.txt
touch README.md
touch .gitignore
touch .env.example

echo "✅ Configuration files created"

# Create main entry points
touch src/main.py
touch src/train.py
touch src/predict.py
touch dashboard/app.py

echo "✅ Main entry points created"

echo ""
echo "📁 Project structure:"
echo "├── src/"
echo "│   ├── data_processing/    # Data loading, cleaning, feature engineering"
echo "│   ├── models/             # ML model implementations"
echo "│   ├── utils/              # Utility functions"
echo "│   └── config/             # Configuration files"
echo "├── notebooks/              # Jupyter notebooks for exploration"
echo "├── data/                   # Data storage"
echo "│   ├── raw/                # Original data"
echo "│   ├── processed/          # Processed data"
echo "│   └── features/           # Feature engineered data"
echo "├── models/                 # Saved models"
echo "├── outputs/                # Predictions, visualizations, reports"
echo "├── dashboard/              # Interactive dashboard"
echo "└── tests/                  # Unit tests"
echo ""
echo "✨ Setup complete! Next steps:"
echo "   1. Install dependencies: pip install -r requirements.txt"
echo "   2. Run data processing: python src/main.py --mode process"
echo "   3. Train models: python src/train.py"
echo "   4. Launch dashboard: streamlit run dashboard/app.py"
