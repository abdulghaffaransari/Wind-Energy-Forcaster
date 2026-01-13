# 🚀 Quick Start Guide

## Step 0: Create Virtual Environment (Recommended)

### Windows (PowerShell):
```powershell
# Create virtual environment named "wind"
python -m venv wind

# Activate the virtual environment
.\wind\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows (Command Prompt):
```cmd
# Create virtual environment named "wind"
python -m venv wind

# Activate the virtual environment
wind\Scripts\activate.bat
```

### Linux/Mac:
```bash
# Create virtual environment named "wind"
python3 -m venv wind

# Activate the virtual environment
source wind/bin/activate
```

**Note**: After activation, you should see `(wind)` at the beginning of your command prompt.

## Step 1: Setup (One-time)

### Windows (PowerShell):
```powershell
# Run the setup script
powershell -ExecutionPolicy Bypass -File setup.ps1

# Or manually create directories (already done if you ran setup)
```

### Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with Prophet, you may need to install it separately:
```bash
pip install prophet
```

For TensorFlow, if you have GPU:
```bash
pip install tensorflow-gpu
```

## Step 3: Process Data

```bash
python src/main.py --mode process
```

This will:
- Load the data from `Data/germany-wind-energy.csv`
- Create features (lag, rolling, seasonal)
- Save processed data

## Step 4: Train Models

Train all models:
```bash
python src/train.py --model all
```

Train a specific model:
```bash
python src/train.py --model XGBoost
python src/train.py --model LSTM
python src/train.py --model Transformer
python src/train.py --model LightGBM
python src/train.py --model Prophet
```

**Training Time Estimates:**
- XGBoost: ~2-5 minutes
- LightGBM: ~2-5 minutes
- LSTM: ~10-30 minutes (depending on GPU)
- Transformer: ~15-40 minutes (depending on GPU)
- Prophet: ~5-10 minutes

## Step 5: Launch Dashboard

**Important**: Always use `python -m streamlit` to ensure it uses your virtual environment:

```bash
python -m streamlit run dashboard/app.py
```

**Note**: Use `python -m streamlit` instead of just `streamlit` to ensure it uses the correct Python environment. The dashboard will open automatically in your browser at `http://localhost:8501`

## Step 6: Make Predictions

```bash
python src/predict.py --model XGBoost --n_days 30
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the project root directory and all dependencies are installed.

### Model Not Found
If a model is not found, make sure you've trained it first using `python src/train.py --model <MODEL_NAME>`

### Memory Issues
If you run out of memory:
- Train models one at a time
- Reduce batch size in `src/config/config.yaml`
- Use smaller sequence lengths for LSTM/Transformer

### Prophet Installation Issues
Prophet requires additional dependencies. On Windows, you may need:
```bash
conda install -c conda-forge prophet
```

## Next Steps

1. Explore the dashboard
2. Compare model performances
3. Tune hyperparameters in `src/config/config.yaml`
4. Experiment with different features
5. Try ensemble models

---

**Happy Forecasting! 🌬️**
