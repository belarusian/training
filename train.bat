@echo off
echo ========================================
echo Unsloth Training Launcher for Windows
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

REM Check if CUDA is available
echo Checking GPU...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Could not check CUDA - will try to continue anyway
)

echo.
echo Starting Unsloth Training...
echo.

REM Run the training notebook
jupyter lab training\main-training.ipynb

echo.
echo Training complete!
pause
