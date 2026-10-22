#!/usr/bin/env bash
# NLP Mini Project - Linux / Mac Shell Launcher

echo "======================================================================"
echo "      NLP MINI PROJECT: SENTIMENT CLASSIFICATION DEMO LAUNCHER"
echo "======================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.9+."
    exit 1
fi

if [ ! -f "models/sentiment_model.pkl" ]; then
    echo "[INFO] Model not found in models/. Training model now..."
    python3 train.py
fi

echo "[INFO] Launching Streamlit web demo..."
streamlit run app.py
