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

:: 2. Check if model files exist
if not exist "models\sentiment_model.pkl" (
    echo [INFO] Pre-trained model not detected in models/
    echo [INFO] Training model now (takes ~20 seconds)...
    python train.py
    if %errorlevel% neq 0 (
        echo [ERROR] Model training failed.
        pause
        exit /b
    )
)

:: 3. Launch Streamlit web application
echo [INFO] Launching Streamlit Interactive Web App...
echo [INFO] The browser will open automatically at http://localhost:8501
echo.
streamlit run app.py

pause
