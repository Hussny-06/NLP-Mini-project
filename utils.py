"""
NLP Preprocessing and Utility Functions for Sentiment Analysis.
Author: NLP Mini Project Team
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ['stopwords', 'wordnet', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = set()

# Keep crucial sentiment-bearing negation words in stopwords
NEGATION_WORDS = {"not", "no", "nor", "neither", "never", "hardly", "barely", "scarcely", "isn't", "wasn't", "didn't", "don't", "won't", "can't", "couldn't"}
CUSTOM_STOP_WORDS = STOP_WORDS - NEGATION_WORDS

lemmatizer = WordNetLemmatizer()


def clean_text_steps(raw_text: str):
    """
    Returns both intermediate transformation stages and final clean text.
    Useful for the 'Behind the Scenes' UI inspection in Streamlit!
    """
    # 1. Lowercasing
    lowered = raw_text.lower()
    
    # 2. HTML tag removal
    no_html = re.sub(r'<.*?>', ' ', lowered)
    
    # 3. URL and special character removal (keeping alphabets and single quotes)
    no_urls = re.sub(r'http\S+|www\S+', '', no_html)
    cleaned_chars = re.sub(r"[^a-z\s']", ' ', no_urls)
    cleaned_chars = re.sub(r'\s+', ' ', cleaned_chars).strip()
    
    # 4. Tokenization
    tokens = cleaned_chars.split()
    
    # 5. Stopword filtering & Lemmatization
    processed_tokens = []
    for token in tokens:
        clean_tok = token.strip("'")
        if len(clean_tok) > 1 and clean_tok not in CUSTOM_STOP_WORDS:
            lemma = lemmatizer.lemmatize(clean_tok)
            processed_tokens.append(lemma)
            
    final_text = " ".join(processed_tokens)
    
    return {
        "raw": raw_text,
        "lowered": lowered,
        "no_html": no_html,
        "cleaned_chars": cleaned_chars,
        "tokens": tokens,
        "filtered_lemmas": processed_tokens,
        "final": final_text
    }


def clean_text(raw_text: str) -> str:
    """Fast preprocessing returning only the final processed string."""
    return clean_text_steps(raw_text)["final"]
