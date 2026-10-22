@echo off
setlocal enabledelayedexpansion
title NLP Sentiment Classification Model - Demo Launcher
color 0A

cd /d "%~dp0"

echo ======================================================================
echo       NLP MINI PROJECT: SENTIMENT CLASSIFICATION DEMO LAUNCHER
echo ======================================================================
echo.

REM Disable Streamlit telemetry and first-run prompts
set STREAMLIT_GATHER_USAGE_STATS=false

REM 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

REM 2. Check or create virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating Python virtual environment: .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
    echo [INFO] Installing required dependencies into .venv ...
    call .venv\Scripts\python.exe -m pip install --upgrade pip
    call .venv\Scripts\python.exe -m pip install -r requirements.txt
    call .venv\Scripts\python.exe -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True)"
)

REM 3. Check if pre-trained model exists
if not exist "models\sentiment_model.pkl" (
    echo [INFO] Pre-trained model not found. Training model now...
    call .venv\Scripts\python.exe train.py
    if errorlevel 1 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
)

REM 4. Launch Streamlit web app
echo [INFO] Opening default browser at http://localhost:8501 ...
start http://localhost:8501

echo [INFO] Starting Streamlit server...
echo [INFO] Keep this command prompt window open during the demo.
echo.

call .venv\Scripts\python.exe -m streamlit run app.py --server.headless false

if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit exited with an error.
    pause
)
