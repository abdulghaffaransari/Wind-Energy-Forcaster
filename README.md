# 🌬️ Wind Energy Forecasting Project

A comprehensive, end-to-end machine learning project for forecasting daily wind power generation using state-of-the-art time series models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dashboard](#dashboard)
- [Results](#results)

## 🎯 Overview

This project forecasts daily wind power generation using multiple advanced machine learning models. The dataset contains daily measurements of:
- **Wind Generation**: Daily wind power production in MW
- **Wind Capacity**: Electrical capacity of wind in MW
- **Temperature**: Daily temperature in °C
- **Time Period**: 2017-2019

## ✨ Features

- **Modular Architecture**: Fully organized, modular code structure
- **Multiple ML Models**: LSTM, Transformer, XGBoost, LightGBM, Prophet
- **Advanced Feature Engineering**: Lag features, rolling statistics, seasonal patterns
- **Interactive Dashboard**: Beautiful, clickable Streamlit dashboard with 7 pages
- **AI Chatbot Assistant**: WindForecast Intelligence Hub with multi-agent system
- **Comprehensive Reports**: Auto-generated PDF reports with university branding
- **Future Forecasting**: Multi-day autoregressive time series forecasting
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Production Ready**: Optimized and scalable codebase

## 📁 Project Structure

```
Wind Energy Forcaster/
├── src/
│   ├── config/
│   │   ├── config.yaml          # Configuration file
│   │   └── constants.py          # Constants
│   ├── data_processing/
│   │   ├── data_loader.py        # Data loading utilities
│   │   ├── feature_engineering.py # Feature creation
│   │   └── data_preprocessor.py  # Data preprocessing
│   ├── models/
│   │   ├── base_model.py         # Base model class
│   │   ├── lstm_model.py         # LSTM implementation
│   │   ├── transformer_model.py  # Transformer implementation
│   │   ├── xgboost_model.py      # XGBoost implementation
│   │   ├── lightgbm_model.py     # LightGBM implementation
│   │   ├── prophet_model.py      # Prophet implementation
│   │   └── ensemble_model.py     # Ensemble model
│   ├── utils/
│   │   ├── logger.py             # Logging utilities
│   │   ├── metrics.py            # Metrics calculation
│   │   └── visualization.py      # Visualization utilities
│   ├── main.py                   # Main pipeline entry point
│   ├── train.py                  # Training script
│   └── predict.py                 # Prediction script
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── chatbot/
│   ├── intelligence_hub.py        # Main chatbot hub
│   ├── agents/                    # AI agents (RAG, Web, Router)
│   └── config/                    # Chatbot configuration
├── Reports/                       # Auto-generated PDF reports
├── assets/
│   └── logos/                     # University logos
├── data/
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed data
│   └── features/                 # Feature engineered data
├── models/
│   ├── saved_models/              # Trained models
│   └── checkpoints/              # Model checkpoints
├── outputs/
│   ├── predictions/              # Prediction results
│   ├── visualizations/           # Generated plots
│   └── reports/                   # Evaluation reports
├── notebooks/                     # Jupyter notebooks
├── logs/                          # Log files
├── tests/                         # Unit tests
├── generate_reports.py            # PDF report generator
├── setup.sh                       # Setup script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Installation

### 1. Clone or Navigate to Project Directory

```bash
cd "Wind Energy Forcaster"
```

### 2. Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
# Create virtual environment named "wind"
python -m venv wind

# Activate the virtual environment
.\wind\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (Command Prompt):**
```cmd
# Create virtual environment named "wind"
python -m venv wind

# Activate the virtual environment
wind\Scripts\activate.bat
```

**Linux/Mac:**
```bash
# Create virtual environment named "wind"
python3 -m venv wind

# Activate the virtual environment
source wind/bin/activate
```

**Note**: After activation, you should see `(wind)` at the beginning of your command prompt. Always activate the virtual environment before working on the project.

### 3. Run Setup Script

**On Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows (PowerShell):**
```powershell
# Create directories manually or use Git Bash to run setup.sh
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Make sure your virtual environment is activated before installing dependencies.

### 5. Verify Installation

```bash
python -c "import tensorflow, xgboost, lightgbm, prophet, streamlit; print('All packages installed successfully!')"
```

## 📖 Usage

### 1. Data Processing

Process and prepare the data:

```bash
python src/main.py --mode process
```

This will:
- Load and validate the data
- Create features (lag, rolling, seasonal)
- Save processed data to `data/processed/`

### 2. Train Models

Train all models:

```bash
python src/train.py --model all
```

Train a specific model:

```bash
python src/train.py --model LSTM
python src/train.py --model Transformer
python src/train.py --model XGBoost
python src/train.py --model LightGBM
python src/train.py --model Prophet
```

### 3. Make Predictions

Generate predictions with a trained model:

```bash
python src/predict.py --model LSTM --n_days 30
```

### 4. Launch Dashboard

Start the interactive dashboard:

**Important**: Always use `python -m streamlit` to ensure it uses your virtual environment:

```bash
python -m streamlit run dashboard/app.py
```

**Note**: Use `python -m streamlit` instead of just `streamlit` to ensure it uses the correct Python environment. The dashboard will open in your browser at `http://localhost:8501`

## 🤖 Models

### 1. LSTM (Long Short-Term Memory)
- Deep learning model for sequence learning
- Captures long-term dependencies
- Configuration: 128-64 hidden units, 30-day sequences

### 2. Transformer
- Attention-based architecture
- State-of-the-art for time series
- Multi-head attention with 4 layers

### 3. XGBoost
- Gradient boosting ensemble
- Handles non-linear relationships
- Robust to outliers

### 4. LightGBM
- Fast gradient boosting
- Efficient memory usage
- Great for large datasets

### 5. Prophet
- Facebook's time series forecasting
- Handles seasonality automatically
- Robust to missing data

## 📊 Dashboard

The interactive dashboard provides:

1. **Data Overview**
   - Time series visualizations
   - Data statistics
   - Raw data exploration

2. **Data Analysis**
   - Correlation analysis
   - Distribution plots
   - Seasonal patterns

3. **Model Training**
   - Train models interactively
   - Check training status

4. **Predictions**
   - Visualize model predictions
   - Performance metrics
   - Download predictions

5. **Model Comparison**
   - Compare all models
   - Best model identification
   - Performance charts

6. **Future Forecast**
   - Generate multi-day forecasts
   - Interactive forecast visualization
   - Download forecast data

7. **WindForecast Intelligence Hub** 🤖
   - AI-powered chatbot assistant
   - Answers questions about project reports
   - Provides technical knowledge
   - Web research capabilities
   - Multi-agent AI system

## 📈 Results

After training, you'll find:

- **Model Metrics**: `outputs/reports/model_metrics.csv`
- **Predictions**: `outputs/predictions/*_predictions.csv`
- **Visualizations**: `outputs/visualizations/*.png`
- **Trained Models**: `models/saved_models/*`
- **PDF Reports**: `Reports/*.pdf` (5 comprehensive reports with university branding)

## 🔧 Configuration

Edit `src/config/config.yaml` to customize:

- Feature engineering parameters
- Model hyperparameters
- Training settings
- Dashboard configuration

## ⚠️ Troubleshooting

### Dashboard Import Errors

If you get `ModuleNotFoundError` when running the dashboard, make sure you're using:

```bash
python -m streamlit run dashboard/app.py
```

**Not** just `streamlit run dashboard/app.py` - this ensures it uses your virtual environment.

### Virtual Environment Not Active

Always activate your virtual environment before running commands:

**Windows (PowerShell):**
```powershell
.\wind\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
wind\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source wind/bin/activate
```

## 📝 Notes

- Models are saved automatically after training
- Predictions are cached for faster dashboard loading
- All visualizations use Plotly for interactivity
- The project follows best practices for ML pipelines

## 🤝 Contributing

This is a complete, production-ready project. Feel free to extend it with:
- Additional models
- More features
- Hyperparameter tuning
- Model deployment

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: Germany Wind Energy Data (2017-2019)
- Technologies: TensorFlow, XGBoost, LightGBM, Prophet, Streamlit, LangChain, OpenAI
- University: Brandenburg University of Technology (BTU) Cottbus-Senftenberg

---

**Built by Abdul Ghaffar Ansari | AI Engineer**

[LinkedIn](https://www.linkedin.com/in/abdulghaffaransari/) | [GitHub](https://github.com/abdulghaffaransari)
