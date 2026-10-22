@echo off
title NLP Sentiment Classification Model - Demo Launcher
color 0A

echo ======================================================================
echo       NLP MINI PROJECT: SENTIMENT CLASSIFICATION DEMO LAUNCHER
echo ======================================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b
)

:: 2. Check or create virtual environment (.venv)
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
    echo [INFO] Virtual environment created successfully.
    echo [INFO] Installing required dependencies into .venv...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True)"
) else (
    call .venv\Scripts\activate.bat
)

:: 3. Check if trained model files exist
if not exist "models\sentiment_model.pkl" (
    echo [INFO] Pre-trained model not detected in models/
    echo [INFO] Training model now (takes ~25 seconds)...
    python train.py
    if %errorlevel% neq 0 (
        echo [ERROR] Model training failed.
        pause
        exit /b
    )
)

:: 4. Launch Streamlit web application from .venv
echo [INFO] Launching Streamlit Interactive Web App from .venv...
echo [INFO] Browser will open automatically at http://localhost:8501
echo.
streamlit run app.py

pause
