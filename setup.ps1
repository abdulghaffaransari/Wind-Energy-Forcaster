# Wind Energy Forecasting Project Setup Script (PowerShell)
# This script creates the entire project structure

Write-Host "🚀 Setting up Wind Energy Forecasting Project..." -ForegroundColor Green

# Create main directories
$directories = @(
    "src\data_processing",
    "src\models",
    "src\utils",
    "src\config",
    "notebooks\exploratory",
    "notebooks\experiments",
    "data\raw",
    "data\processed",
    "data\features",
    "models\saved_models",
    "models\checkpoints",
    "outputs\predictions",
    "outputs\visualizations",
    "outputs\reports",
    "logs",
    "tests",
    "dashboard\static",
    "dashboard\templates"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Cyan
    }
}

# Create __init__.py files
$initFiles = @(
    "src\__init__.py",
    "src\data_processing\__init__.py",
    "src\models\__init__.py",
    "src\utils\__init__.py",
    "src\config\__init__.py"
)

foreach ($file in $initFiles) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
    }
}

Write-Host "`n✅ Directory structure created" -ForegroundColor Green

Write-Host "`n📁 Project structure:" -ForegroundColor Yellow
Write-Host "├── src/"
Write-Host "│   ├── data_processing/    # Data loading, cleaning, feature engineering"
Write-Host "│   ├── models/             # ML model implementations"
Write-Host "│   ├── utils/              # Utility functions"
Write-Host "│   └── config/             # Configuration files"
Write-Host "├── notebooks/              # Jupyter notebooks for exploration"
Write-Host "├── data/                   # Data storage"
Write-Host "│   ├── raw/                # Original data"
Write-Host "│   ├── processed/         # Processed data"
Write-Host "│   └── features/          # Feature engineered data"
Write-Host "├── models/                 # Saved models"
Write-Host "├── outputs/                # Predictions, visualizations, reports"
Write-Host "├── dashboard/              # Interactive dashboard"
Write-Host "└── tests/                  # Unit tests"

Write-Host "`n✨ Setup complete! Next steps:" -ForegroundColor Green
Write-Host "   1. Install dependencies: pip install -r requirements.txt"
Write-Host "   2. Run data processing: python src/main.py --mode process"
Write-Host "   3. Train models: python src/train.py --model all"
Write-Host "   4. Launch dashboard: streamlit run dashboard/app.py"
