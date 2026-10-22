#!/usr/bin/env bash
# NLP Mini Project - Linux / Mac Shell Launcher

echo "======================================================================"
echo "      NLP MINI PROJECT: SENTIMENT CLASSIFICATION DEMO LAUNCHER"
echo "======================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.9+."
    exit 1
fi

# Check or create virtual environment
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python3 -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True)"
else
    source .venv/bin/activate
fi

# Check if model files exist
if [ ! -f "models/sentiment_model.pkl" ]; then
    echo "[INFO] Model not found in models/. Training model now..."
    python3 train.py
fi

echo "[INFO] Launching Streamlit web demo from .venv..."
streamlit run app.py
