# PokerGTO Project Structure Generator
# This script creates the directory structure and empty files for the PokerGTO project

# Set error action preference to stop on error
$ErrorActionPreference = "Stop"

# Function to create a directory if it doesn't exist
function EnsureDirectory {
    param([string]$path)

    if (-not (Test-Path -Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "Created directory: $path" -ForegroundColor Green
    }
}

# Function to create an empty file if it doesn't exist
function CreateEmptyFile {
    param([string]$path)

    if (-not (Test-Path -Path $path)) {
        New-Item -ItemType File -Path $path -Force | Out-Null
        Write-Host "Created file: $path" -ForegroundColor Cyan
    }
}

# Get the current location to create the project structure
$projectRoot = Get-Location

Write-Host "Creating PokerGTO project structure at: $projectRoot" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow

# Create main directories
$directories = @(
    "models",
    "models/drm_model",
    "models/adaptive_model",
    "data",
    "data/game_logs",
    "data/training_history",
    "src",
    "src/engine",
    "src/algorithms",
    "src/api",
    "src/utils",
    "src/interface",
    "tests"
)

foreach ($dir in $directories) {
    EnsureDirectory -path (Join-Path -Path $projectRoot -ChildPath $dir)
}

# Create Python files in src directory
$pythonFiles = @(
    "src/engine/__init__.py",
    "src/engine/game_state.py",
    "src/engine/evaluator.py",
    "src/engine/rules.py",
    "src/algorithms/__init__.py",
    "src/algorithms/discounted_regret.py",
    "src/algorithms/deep_cfr.py",
    "src/api/__init__.py",
    "src/api/slumbot.py",
    "src/utils/__init__.py",
    "src/utils/cuda_utils.py",
    "src/utils/visualization.py",
    "src/utils/logging.py",
    "src/interface/__init__.py",
    "src/interface/tooltips.py"
)

foreach ($file in $pythonFiles) {
    CreateEmptyFile -path (Join-Path -Path $projectRoot -ChildPath $file)
}

# Create test files
$testFiles = @(
    "tests/__init__.py",
    "tests/test_engine.py",
    "tests/test_algorithms.py",
    "tests/test_api.py"
)

foreach ($file in $testFiles) {
    CreateEmptyFile -path (Join-Path -Path $projectRoot -ChildPath $file)
}

# Create main scripts and other files at the root level
$rootFiles = @(
    "calculate.py",
    "train.py",
    "play.py",
    "config.py",
    "requirements.txt",
    "README.md"
)

foreach ($file in $rootFiles) {
    CreateEmptyFile -path (Join-Path -Path $projectRoot -ChildPath $file)
}

# Create a basic README.md content
$readmePath = Join-Path -Path $projectRoot -ChildPath "README.md"
$readmeContent = @"
# PokerGTO

A project to create Game Theory Optimal (GTO) models for Texas Hold'em poker using Discounted Regret Minimization.

## Project Structure

- `calculate.py`: GTO calculation using Discounted Regret Minimization
- `train.py`: Model training against Slumbot API
- `play.py`: Interactive play and tooltips

## Setup

1. Ensure you have the Anaconda environment 'PokerGTO' activated
2. Install dependencies: `pip install -r requirements.txt`
3. Run the desired script: `python calculate.py`, `python train.py`, or `python play.py`

## Models

- DRM Model: Based on Discounted Regret Minimization
- Adaptive Model: Learning from gameplay against Slumbot
"@

Set-Content -Path $readmePath -Value $readmeContent

# Create a basic requirements.txt content
$requirementsPath = Join-Path -Path $projectRoot -ChildPath "requirements.txt"
$requirementsContent = @"
numpy
pandas
matplotlib
torch
tensorboard
requests
python-dotenv
tqdm
pytest
"@

Set-Content -Path $requirementsPath -Value $requirementsContent

Write-Host "----------------------------------------" -ForegroundColor Yellow
Write-Host "Project structure created successfully!" -ForegroundColor Green
Write-Host "Next step: Fill in the Python files with actual implementations" -ForegroundColor Yellow