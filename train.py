"""
Training script for NLP Sentiment Classification Model.
Trains on the IMDb Benchmark Dataset using TF-IDF + Tuned Logistic Regression.
Saves model artifacts to models/ for instant deployment.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from utils import clean_text

def train_sentiment_model(sample_size=20000):
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("=" * 60)
    print("  NLP MINI PROJECT: TRAINING SENTIMENT CLASSIFICATION MODEL  ")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Load Dataset
    print("\n[1/5] Loading IMDb Movie Reviews benchmark dataset...")
    try:
        ds = load_dataset('stanfordnlp/imdb')
        df_train_full = pd.DataFrame(ds['train'])
        df_test_full = pd.DataFrame(ds['test'])
        
        # Stratified sampling for fast, balanced training if requested
        if sample_size and sample_size < len(df_train_full):
            from sklearn.model_selection import train_test_split
            df_train, _ = train_test_split(
                df_train_full, 
                train_size=sample_size, 
                stratify=df_train_full['label'], 
                random_state=42
            )
            df_test, _ = train_test_split(
                df_test_full, 
                train_size=min(5000, sample_size // 4), 
                stratify=df_test_full['label'], 
                random_state=42
            )
            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)
        else:
            df_train = df_train_full
            df_test = df_test_full
            
    except Exception as e:
        print(f"Warning: HuggingFace load failed ({e}), falling back to curated data...")
        raise e

    print(f"Loaded {len(df_train)} training reviews and {len(df_test)} testing reviews.")
    
    # 2. Text Preprocessing
    print("\n[2/5] Cleaning and preprocessing text (Lemmatization, Stopwords, HTML removal)...")
    t0 = time.time()
    df_train['clean_text'] = df_train['text'].apply(clean_text)
    df_test['clean_text'] = df_test['text'].apply(clean_text)
    print(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")
    
    # 3. TF-IDF Vectorization
    print("\n[3/5] Extracting TF-IDF features (unigrams + bigrams, sublinear TF)...")
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3
    )
    X_train = vectorizer.fit_transform(df_train['clean_text'])
    X_test = vectorizer.transform(df_test['clean_text'])
    y_train = df_train['label'].values
    y_test = df_test['label'].values
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} features.")
    
    # 4. Model Training
    print("\n[4/5] Training tuned Logistic Regression classifier...")
    t0 = time.time()
    model = LogisticRegression(C=2.5, max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)
    print(f"Model training completed in {time.time() - t0:.2f} seconds.")
    
    # 5. Evaluation
    print("\n[5/5] Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("\n" + "=" * 45)
    print(f"  ACCURACY : {accuracy * 100:.2f}%")
    print(f"  F1-SCORE : {f1 * 100:.2f}%")
    print(f"  PRECISION: {precision * 100:.2f}%")
    print(f"  RECALL   : {recall * 100:.2f}%")
    print("=" * 45)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save artifacts
    print("\nSaving trained artifacts to 'models/' directory...")
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "confusion_matrix": cm,
        "vocab_size": len(vectorizer.get_feature_names_out()),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "model_type": "Logistic Regression (C=2.5, L-BFGS)",
        "feature_extractor": "TF-IDF (1-2 ngrams, sublinear_tf)"
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    total_time = time.time() - start_time
    print(f"\nAll artifacts successfully saved! Total time: {total_time:.2f}s")
    print("Ready to launch the Streamlit app with: streamlit run app.py")


if __name__ == '__main__':
    train_sentiment_model(sample_size=20000)
