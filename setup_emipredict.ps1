<#
.SYNOPSIS
  Full automated setup for EMIPredict_AI - creates project, dataset, python modules, virtualenv, installs deps, initializes git, runs initial steps.

.DESCRIPTION
  - Creates directory structure
  - Writes Python source files (data generation, preprocessing, training, EDA, Streamlit app etc.)
  - Generates synthetic dataset (400,000 rows)
  - Creates virtual environment and installs requirements
  - Initializes git repo
  - Runs an initial data generation & quick train script (optional)
  - Includes progress prints and basic error handling
#>

param(
  [switch]$SkipInitialRun   # set to skip running first generate/train steps
)

# --- Helper functions ---
function Write-Heading($text){
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

function Write-ErrorAndExit($msg){
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

try {
    $root = Join-Path (Get-Location) "EMIPredict_AI"
    if (-Not (Test-Path $root)) {
        New-Item -Path $root -ItemType Directory | Out-Null
    } else {
        Write-Host "Directory already exists: $root" -ForegroundColor Yellow
    }

    Write-Heading "Creating directory structure..."
    $dirs = @(
        "data\raw",
        "data\processed",
        "models",
        "mlruns",
        "notebooks",
        "src",
        "src\components",
        "src\pipelines",
        "app",
        "app\pages",
        "config",
        "tests",
        "logs"
    )
    foreach ($d in $dirs) {
        $p = Join-Path $root $d
        if (-Not (Test-Path $p)) {
            New-Item -Path $p -ItemType Directory | Out-Null
            Write-Host "Created $p"
        }
    }

    # --- requirements.txt ---
    Write-Heading "Writing requirements.txt..."
    $requirements = @"
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
xgboost>=1.7
mlflow>=2.3
streamlit>=1.22
matplotlib>=3.6
seaborn>=0.12
plotly>=5.9
joblib>=1.2
psutil>=5.9
pytest>=7.1
pydantic>=1.10
flaky>=3.7
black
pandas-profiling>=3.6
imbalanced-learn>=0.10
category_encoders>=2.5
pytest-cov
python-dotenv
"""@

    $requirementsPath = Join-Path $root "requirements.txt"
    $requirements | Out-File -FilePath $requirementsPath -Encoding UTF8

    # --- .gitignore ---
    Write-Heading "Writing .gitignore..."
    $gitignore = @"
env/
.env
__pycache__/
*.pyc
*.pkl
*.joblib
mlruns/
logs/
data/raw/
data/processed/
.vscode/
.idea/
.ipynb_checkpoints/
"@
    $gitignore | Out-File -FilePath (Join-Path $root ".gitignore") -Encoding UTF8

    # --- .streamlit/config.toml ---
    Write-Heading "Writing Streamlit config..."
    $streamlitDir = Join-Path $root ".streamlit"
    if (-Not (Test-Path $streamlitDir)) { New-Item -Path $streamlitDir -ItemType Directory | Out-Null }
    $configToml = @"
[theme]
primaryColor = "#0f62fe"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#0e1116"
font = "sans serif"

[server]
headless = true
enableCORS=false
port = 8501
"@
    $configToml | Out-File -FilePath (Join-Path $streamlitDir "config.toml") -Encoding UTF8

    # --- README.md ---
    Write-Heading "Writing README.md..."
    $readme = @"
# EMIPredict_AI

Intelligent Financial Risk Assessment Platform — end-to-end FinTech project using Python, Streamlit, and MLflow.

## Features
- Synthetic dataset generator (400k records, 5 EMI scenarios)
- Data preprocessing with feature engineering
- Classification (EMI eligibility) & regression (Max EMI amount) models
- MLflow tracking (local `mlruns/`)
- Streamlit app (prediction, EDA, model performance, admin CRUD)
- Tests, logging, and model monitoring hooks

## Quick start
```powershell
.\setup_emipredict.ps1
# then activate environment:
.\env\Scripts\Activate.ps1
streamlit run app\main.py
