"""
Natural Language Processing Mini Project - Semester 7
IMDb Review Sentiment & Polarity Classification
Designed with matching clean aesthetic.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils import clean_text_steps

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Classification & Polarity Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom CSS - Recreating the exact UI in the screenshot
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(135deg, #F0FDF4 0%, #F8FAFC 45%, #F0F9FF 100%) !important;
    }

    /* Top Badge */
    .top-badge-wrapper {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .top-badge {
        display: inline-block;
        background-color: #E0F2FE;
        color: #0284C7;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 6px 16px;
        border-radius: 9999px;
    }

    /* Main Titles */
    .main-heading {
        text-align: center;
        font-size: 2.35rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
    }
    .main-subheading {
        text-align: center;
        font-size: 0.96rem;
        color: #64748B;
        max-width: 680px;
        margin: 0 auto 32px auto;
        line-height: 1.5;
    }

    /* Outer Cards */
    .dashboard-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        height: 100%;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 18px;
    }

    /* Preset Buttons Bar */
    .preset-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        margin-right: 8px;
    }

    /* Streamlit Preset Button Style */
    div[data-testid="stHorizontalBlock"] button {
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        color: #334155 !important;
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 4px 10px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #E2E8F0 !important;
        border-color: #CBD5E1 !important;
    }

    /* Text Area */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #F8FAFC !important;
        font-size: 0.92rem !important;
        color: #1E293B !important;
    }
    .stTextArea textarea:focus {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.15) !important;
    }

    /* Big Submit Button */
    div.stButton > button[kind="primary"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0369A1 !important;
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Prediction Inner Card (Green / Red) */
    .prediction-box-pos {
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
        border-radius: 14px;
        padding: 24px 16px;
        text-align: center;
        margin-bottom: 22px;
    }
    .prediction-box-neg {
        background: #FEF2F2;
        border: 1px solid #FEE2E2;
        border-radius: 14px;
        padding: 24px 16px;
        text-align: center;
        margin-bottom: 22px;
    }
    .prediction-label {
        font-size: 0.88rem;
        color: #475569;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .hero-metric-pos {
        font-size: 3.2rem;
        font-weight: 800;
        color: #15803D;
        line-height: 1.1;
        margin-bottom: 10px;
    }
    .hero-metric-neg {
        font-size: 3.2rem;
        font-weight: 800;
        color: #DC2626;
        line-height: 1.1;
        margin-bottom: 10px;
    }
    .status-badge-pos {
        display: inline-block;
        background-color: #16A34A;
        color: #FFFFFF;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 4px 18px;
        border-radius: 9999px;
    }
    .status-badge-neg {
        display: inline-block;
        background-color: #DC2626;
        color: #FFFFFF;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 4px 18px;
        border-radius: 9999px;
    }

    /* Key-Value Details Row */
    .detail-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        font-size: 0.88rem;
        border-bottom: 1px solid #F1F5F9;
    }
    .detail-label {
        color: #64748B;
        font-weight: 500;
    }
    .detail-value {
        color: #0F172A;
        font-weight: 700;
    }

    /* Advisory Box */
    .advisory-box-pos {
        margin-top: 18px;
        padding: 12px 14px;
        border-left: 3.5px solid #16A34A;
        background-color: #F8FAFC;
        border-radius: 0 8px 8px 0;
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.45;
    }
    .advisory-box-neg {
        margin-top: 18px;
        padding: 12px 14px;
        border-left: 3.5px solid #DC2626;
        background-color: #F8FAFC;
        border-radius: 0 8px 8px 0;
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.45;
    }

    /* Word Chips */
    .token-chip-pos {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
        font-family: 'JetBrains Mono', monospace;
    }
    .token-chip-neg {
        display: inline-block;
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 40px;
        margin-bottom: 20px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Saved Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = os.path.join('models', 'sentiment_model.pkl')
    vec_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    metrics_path = os.path.join('models', 'metrics.json')
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None, None
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    return model, vectorizer, metrics

model, vectorizer, metrics = load_artifacts()


# ---------------------------------------------------------
# Explainability Logic
# ---------------------------------------------------------
def explain_features(text, vectorizer, model):
    feature_names = np.array(vectorizer.get_feature_names_out())
    weights = model.coef_[0]
    
    vec = vectorizer.transform([text])
    feature_indices = vec.nonzero()[1]
    
    word_scores = []
    for idx in feature_indices:
        token = feature_names[idx]
        tfidf_val = vec[0, idx]
        weight = weights[idx]
        score = tfidf_val * weight
        word_scores.append((token, score, weight))
        
    pos_tokens = [item for item in sorted(word_scores, key=lambda x: x[1], reverse=True) if item[1] > 0]
    neg_tokens = [item for item in sorted(word_scores, key=lambda x: x[1]) if item[1] < 0]
    
    return pos_tokens[:6], neg_tokens[:6]


# ---------------------------------------------------------
# Top Section (Matching Screenshot)
# ---------------------------------------------------------
st.markdown('<div class="top-badge-wrapper"><span class="top-badge">NATURAL LANGUAGE PROCESSING MINI PROJECT • SEMESTER 7</span></div>', unsafe_allow_html=True)
st.markdown('<div class="main-heading">Movie Review Sentiment & Tone Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subheading">Continuous polarity estimation based on TF-IDF n-gram vectorization, morphological lemmatization, and regularized logistic classification.</div>', unsafe_allow_html=True)

if model is None or vectorizer is None:
    st.error("Model artifacts missing in `models/`. Run `python train.py` once to build them.")
    st.stop()


# ---------------------------------------------------------
# Two-Column Dashboard Layout
# ---------------------------------------------------------
col_left, col_right = st.columns([1.1, 1], gap="large")

# ----------------- LEFT CARD: Input Parameters -----------------
with col_left:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Text Input & Linguistic Parameters</div>', unsafe_allow_html=True)

    # Preset chips
    st.markdown('<span class="preset-label">Presets:</span>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)
    
    if p_col1.button("🌟 Positive Review", use_container_width=True):
        st.session_state["review_text"] = "An absolute masterpiece. Brilliant acting, mesmerizing soundtrack, and flawless direction. Highly recommended!"
    if p_col2.button("🌧️ Negative Review", use_container_width=True):
        st.session_state["review_text"] = "A complete waste of time. The acting was atrocious, plot was boring, and the ending made no sense."
    if p_col3.button("🎭 Mixed Nuance", use_container_width=True):
        st.session_state["review_text"] = "The visual effects and action choreography were stunning, but the weak dialogue and slow pacing were not good."

    default_val = st.session_state.get(
        "review_text",
        "The direction was exceptional and the lead actor gave an extraordinary, moving performance."
    )

    review_input = st.text_area(
        "Review Statement (Raw Text)",
        value=default_val,
        height=130,
        placeholder="Type or paste any English review here..."
    )

    # Secondary metadata inputs (mirrors meteorological controls in screenshot)
    m_row1_c1, m_row1_c2 = st.columns(2)
    with m_row1_c1:
        st.text_input("Review Domain", value="Cinematic & Entertainment", disabled=True)
    with m_row1_c2:
        st.text_input("Feature Representation", value="TF-IDF (Unigram + Bigram)", disabled=True)

    m_row2_c1, m_row2_c2 = st.columns(2)
    with m_row2_c1:
        st.text_input("Morphological Normalizer", value="WordNet Lemmatizer", disabled=True)
    with m_row2_c2:
        st.text_input("Stopword Strategy", value="Negation-Preserving Filter", disabled=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Review Sentiment & Polarity", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ----------------- RIGHT CARD: Prediction Output -----------------
with col_right:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Prediction Output</div>', unsafe_allow_html=True)

    # Run inference
    prep = clean_text_steps(review_input if review_input.strip() else default_val)
    clean_str = prep["final"]

    vec = vectorizer.transform([clean_str])
    probabilities = model.predict_proba(vec)[0]
    prob_neg = float(probabilities[0])
    prob_pos = float(probabilities[1])

    is_positive = prob_pos >= 0.5
    confidence_score = (prob_pos if is_positive else prob_neg) * 100

    # Inner Hero Box (matching AQI box from screenshot)
    if is_positive:
        st.markdown(f"""
        <div class="prediction-box-pos">
            <div class="prediction-label">Estimated Positive Probability</div>
            <div class="hero-metric-pos">{prob_pos * 100:.1f}%</div>
            <div class="status-badge-pos">POSITIVE REVIEW</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box-neg">
            <div class="prediction-label">Estimated Negative Probability</div>
            <div class="hero-metric-neg">{prob_neg * 100:.1f}%</div>
            <div class="status-badge-neg">NEGATIVE REVIEW</div>
        </div>
        """, unsafe_allow_html=True)

    # Details table rows (matching PM2.5, Stagnation Index, Model from screenshot)
    pos_str = f"{prob_pos * 100:.2f}%"
    neg_str = f"{prob_neg * 100:.2f}%"
    conf_str = f"{confidence_score:.2f}%"
    
    pos_tokens, neg_tokens = explain_features(clean_str, vectorizer, model)
    top_driver = pos_tokens[0][0] if is_positive and pos_tokens else (neg_tokens[0][0] if neg_tokens else "None")

    st.markdown(f"""
    <div class="detail-row">
        <span class="detail-label">Positive Class Posterior P(Y=1)</span>
        <span class="detail-value">{pos_str}</span>
    </div>
    <div class="detail-row">
        <span class="detail-label">Negative Class Posterior P(Y=0)</span>
        <span class="detail-value">{neg_str}</span>
    </div>
    <div class="detail-row">
        <span class="detail-label">Primary Sentiment Driver</span>
        <span class="detail-value">"{top_driver}"</span>
    </div>
    <div class="detail-row">
        <span class="detail-label">Confidence Margin (ΔP)</span>
        <span class="detail-value">{abs(prob_pos - prob_neg) * 100:.2f}%</span>
    </div>
    <div class="detail-row">
        <span class="detail-label">Active Model Architecture</span>
        <span class="detail-value">Tuned Logistic Reg (L2, C=2.5)</span>
    </div>
    """, unsafe_allow_html=True)

    # Advisory block (matching advisory from screenshot)
    if is_positive:
        st.markdown("""
        <div class="advisory-box-pos">
            <b>Linguistic Advisory:</b> Review demonstrates predominantly favorable polarity. Key semantic markers indicate high audience satisfaction and affirmative appraisal.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="advisory-box-neg">
            <b>Linguistic Advisory:</b> Review conveys predominantly critical polarity. Semantic markers reflect unfavorable audience reaction, dissatisfaction, or critical critique.
        </div>
        """, unsafe_allow_html=True)

    # Word level explainability chips
    st.markdown("<br><div style='font-size:0.83rem; font-weight:600; color:#64748B; margin-bottom:6px;'>ATTRIBUTED SENTIMENT TOKENS</div>", unsafe_allow_html=True)
    if pos_tokens:
        p_chips = "".join([f'<span class="token-chip-pos">{w} (+{s:.2f})</span>' for w, s, _ in pos_tokens[:4]])
        st.markdown(p_chips, unsafe_allow_html=True)
    if neg_tokens:
        n_chips = "".join([f'<span class="token-chip-neg">{w} ({s:.2f})</span>' for w, s, _ in neg_tokens[:4]])
        st.markdown(n_chips, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Bottom Section: Inspect Pipeline & Metrics (Collapsible)
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📊 Technical Pipeline Inspector & Validation Metrics"):
    tab1, tab2 = st.tabs(["Preprocessing Stages", "Test Set Metrics (IMDb 50,000)"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Raw Input**")
            st.code(prep["raw"], language="text")
        with c2:
            st.markdown("**2. Sanitized Text**")
            st.code(prep["cleaned_chars"], language="text")
        with c3:
            st.markdown("**3. Lemmatized Tokens**")
            st.code(prep["final"], language="text")
            
    with tab2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Validation Accuracy", f"{metrics.get('accuracy', 0.8894)*100:.2f}%")
        m2.metric("Macro F1-Score", f"{metrics.get('f1_score', 0.8899)*100:.2f}%")
        m3.metric("Precision", f"{metrics.get('precision', 0.8859)*100:.2f}%")
        m4.metric("Recall", f"{metrics.get('recall', 0.8940)*100:.2f}%")


# ---------------------------------------------------------
# Footer (Matching Screenshot)
# ---------------------------------------------------------
st.markdown("""
<div class="footer-text">
    Research Project by Mohd. Ikhlas Husain & Mohd. Hussain • Dept. of Computer Science & Engineering<br>
    Trained on authentic IMDb 50,000 Benchmark Dataset with NLTK & Scikit-Learn.
</div>
""", unsafe_allow_html=True)
