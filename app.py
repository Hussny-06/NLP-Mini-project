"""
Natural Language Processing Laboratory
End-to-End Text Sentiment Classification & Explainability System
Academic Mini Project (Semester 7)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils import clean_text_steps

# ---------------------------------------------------------
# Page Setup & Clean Typography
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Classification System | NLP Lab",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refined Minimalist Light Theme Styles
st.markdown("""
<style>
    /* Global font & background enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Clean container card */
    .clean-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    .card-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 8px;
    }
    
    /* Sentiment Badges */
    .badge-pos {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.05rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-neg {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.05rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Token pills for explainability */
    .pill-pos {
        display: inline-block;
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 4px 3px 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    .pill-neg {
        display: inline-block;
        background-color: #FFF1F2;
        color: #9F1239;
        border: 1px solid #FECDD3;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 4px 3px 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    
    /* Top Header */
    .project-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .project-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 24px;
    }
    
    /* Streamlit button tweaks */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #CBD5E1;
        transition: all 0.15s ease-in-out;
    }
    
    div.stButton > button:hover {
        border-color: #94A3B8;
        background-color: #F8FAFC;
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
def get_token_attributions(text, vectorizer, model):
    feature_names = np.array(vectorizer.get_feature_names_out())
    weights = model.coef_[0]
    
    vec = vectorizer.transform([text])
    feature_indices = vec.nonzero()[1]
    
    contributions = []
    for idx in feature_indices:
        token = feature_names[idx]
        tfidf_val = vec[0, idx]
        weight = weights[idx]
        score = tfidf_val * weight
        contributions.append((token, score, weight))
        
    pos_tokens = [item for item in sorted(contributions, key=lambda x: x[1], reverse=True) if item[1] > 0]
    neg_tokens = [item for item in sorted(contributions, key=lambda x: x[1]) if item[1] < 0]
    
    return pos_tokens[:8], neg_tokens[:8]


# ---------------------------------------------------------
# Sidebar: Project Specification
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### NLP Project Reference")
    st.caption("Department of Computer Science & Engineering")
    st.markdown("**Subject:** Natural Language Processing (Sem 7)")
    st.markdown("**Topic:** Supervised Text Classification")
    
    st.divider()
    
    st.markdown("#### System Specifications")
    st.markdown("""
    - **Classifier:** L2-Regularized Logistic Regression ($C=2.5$)
    - **Vectorization:** TF-IDF with Unigrams & Bigrams $(1, 2)$
    - **Sublinear TF:** Enabled ($1 + \log(\text{tf})$)
    - **Dataset:** IMDb Reviews (50,000 Samples)
    - **Stopwords:** Context-aware (Negations Preserved)
    """)
    
    if metrics:
        st.divider()
        st.markdown("#### Test Set Performance")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        m_col2.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.1f}%")
        m_col3, m_col4 = st.columns(2)
        m_col3.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
        m_col4.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
        
    st.divider()
    st.caption("Status: Offline model loaded & ready")


# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown('<div class="project-title">Text Sentiment Classification Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="project-subtitle">An explainable Natural Language Processing pipeline implementing feature extraction and regularized classification.</div>', unsafe_allow_html=True)

if model is None or vectorizer is None:
    st.error("Model artifacts missing in `models/`. Please execute `python train.py` once to build the pipeline artifacts.")
    st.stop()


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_inference, tab_metrics, tab_docs = st.tabs([
    "Interactive Inference",
    "Evaluation & Feature Analysis",
    "Methodology & Viva Reference"
])


# ---------------------------------------------------------
# TAB 1: Inference & Explainability
# ---------------------------------------------------------
with tab_inference:
    st.markdown("##### Select a benchmark sample or enter custom text:")
    
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
    if c_btn1.button("Positive Sample", use_container_width=True):
        st.session_state["review_input"] = "The direction and acting were exceptional. A truly moving cinematic experience that I would strongly recommend to anyone."
    if c_btn2.button("Negative Sample", use_container_width=True):
        st.session_state["review_input"] = "A frustratingly predictable plot with uninspired acting. The dialogue was unnatural and the pacing dragged endlessly."
    if c_btn3.button("Nuanced / Mixed Sample", use_container_width=True):
        st.session_state["review_input"] = "The visual effects and set design were undeniably impressive, but the weak storyline and flat ending were disappointing."
    if c_btn4.button("Negation Test Case", use_container_width=True):
        st.session_state["review_input"] = "The trailer made it look promising, but the actual movie is not good at all."

    default_text = st.session_state.get(
        "review_input",
        "The cinematography was brilliant and the lead performance was remarkably authentic."
    )

    input_text = st.text_area(
        "Input text for classification:",
        value=default_text,
        height=100,
        placeholder="Enter a review or statement to analyze..."
    )

    run_classification = st.button("Classify Sentiment", type="primary")

    if input_text.strip():
        # Preprocessing breakdown
        steps = clean_text_steps(input_text)
        cleaned_str = steps["final"]
        
        # Inference
        vec_feat = vectorizer.transform([cleaned_str])
        probs = model.predict_proba(vec_feat)[0]
        prob_neg, prob_pos = float(probs[0]), float(probs[1])
        prediction = "Positive" if prob_pos >= 0.5 else "Negative"
        confidence = max(prob_pos, prob_neg) * 100

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results Section in a clean card layout
        r_col1, r_col2 = st.columns([1, 2])
        
        with r_col1:
            st.markdown('<div class="card-header">Classification Output</div>', unsafe_allow_html=True)
            if prediction == "Positive":
                st.markdown('<div class="badge-pos">● Positive Sentiment</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-neg">● Negative Sentiment</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")
            st.caption("Class determined at $0.50$ probability threshold.")

        with r_col2:
            st.markdown('<div class="card-header">Class Probability Distribution</div>', unsafe_allow_html=True)
            pb_col1, pb_col2 = st.columns(2)
            pb_col1.metric("P(Positive)", f"{prob_pos * 100:.2f}%")
            pb_col2.metric("P(Negative)", f"{prob_neg * 100:.2f}%")
            
            st.progress(prob_pos)
            st.caption("Score bar reflects posterior positive class probability $P(Y=1|X)$.")

        # Explainability Section
        st.divider()
        st.markdown("##### Token-Level Influence (Model Explainability)")
        st.caption("Key unigram and bigram features extracted from the input along with their signed contribution weights:")

        pos_tokens, neg_tokens = get_token_attributions(cleaned_str, vectorizer, model)

        att_col1, att_col2 = st.columns(2)
        with att_col1:
            st.markdown("**Features supporting Positive class:**")
            if pos_tokens:
                pos_html = "".join([f'<span class="pill-pos">{tok} (+{sc:.2f})</span>' for tok, sc, _ in pos_tokens])
                st.markdown(pos_html, unsafe_allow_html=True)
            else:
                st.write("*(No significant positive features detected)*")

        with att_col2:
            st.markdown("**Features supporting Negative class:**")
            if neg_tokens:
                neg_html = "".join([f'<span class="pill-neg">{tok} ({sc:.2f})</span>' for tok, sc, _ in neg_tokens])
                st.markdown(neg_html, unsafe_allow_html=True)
            else:
                st.write("*(No significant negative features detected)*")

        # Preprocessing Expander
        with st.expander("Inspection: Internal Preprocessing Pipeline Steps"):
            st.markdown("**Stage 1: Raw Input Text**")
            st.code(steps["raw"], language="text")
            
            st.markdown("**Stage 2: Sanitized Text (Punctuation, URLs & HTML Removed)**")
            st.code(steps["cleaned_chars"], language="text")
            
            st.markdown("**Stage 3: Lemmatized & Stopword-Filtered String**")
            st.code(steps["final"], language="text")
            
            st.markdown("**Stage 4: Active Token List**")
            st.write(steps["filtered_lemmas"])


# ---------------------------------------------------------
# TAB 2: Metrics & Vocabulary Insights
# ---------------------------------------------------------
with tab_metrics:
    st.markdown("##### Model Evaluation on Held-Out Test Set")
    
    if metrics:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Overall Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        mc2.metric("Macro F1-Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
        mc3.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        mc4.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        
        st.divider()
        
        left_m, right_m = st.columns([1, 1])
        with left_m:
            st.markdown("###### Confusion Matrix")
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            cm_table = pd.DataFrame(
                cm,
                index=["True Negative (0)", "True Positive (1)"],
                columns=["Predicted Negative (0)", "Predicted Positive (1)"]
            )
            st.dataframe(cm_table, use_container_width=True)
            
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            st.caption(f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")
            
        with right_m:
            st.markdown("###### Hyperparameter Configuration")
            st.markdown(f"- **Vocabulary Capacity:** {metrics.get('vocab_size', 12000):,} n-gram features")
            st.markdown(f"- **Training Partition:** {metrics.get('train_samples', 20000):,} instances")
            st.markdown(f"- **Validation Partition:** {metrics.get('test_samples', 5000):,} instances")
            st.markdown(f"- **Solver:** Limited-memory BFGS (L-BFGS)")
            st.markdown(f"- **Inverse Regularization ($C$):** 2.50")

    st.divider()
    st.markdown("##### Global Vocabulary Feature Weights")
    st.caption("Top discriminative n-gram features learned across the entire corpus:")

    vocab_names = np.array(vectorizer.get_feature_names_out())
    coef_weights = model.coef_[0]
    
    top_pos_idx = np.argsort(coef_weights)[-10:][::-1]
    top_neg_idx = np.argsort(coef_weights)[:10]

    df_pos_top = pd.DataFrame({
        "Feature (N-Gram)": vocab_names[top_pos_idx],
        "Positive Coefficient": coef_weights[top_pos_idx].round(3)
    })
    
    df_neg_top = pd.DataFrame({
        "Feature (N-Gram)": vocab_names[top_neg_idx],
        "Negative Coefficient": coef_weights[top_neg_idx].round(3)
    })

    w_col1, w_col2 = st.columns(2)
    with w_col1:
        st.dataframe(df_pos_top, use_container_width=True, hide_index=True)
    with w_col2:
        st.dataframe(df_neg_top, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 3: Methodology & Viva Reference
# ---------------------------------------------------------
with tab_docs:
    st.markdown("##### Key Technical Points for Project Evaluation")
    
    with st.expander("1. TF-IDF Formulation & Justification"):
        st.markdown("""
        **Term Frequency - Inverse Document Frequency (TF-IDF)** mitigates domain-specific stopword distortion:
        $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + N}{1 + \\text{DF}(t)}\\right) + 1$$
        - High-frequency domain words like *"film"*, *"movie"*, or *"actor"* appear across almost all documents and receive small IDF weights.
        - Salient sentiment tokens (*"exceptional"*, *"dreadful"*, *"flawless"*) receive higher weights.
        - **Sublinear TF Scaling:** We use $\\text{sublinear\\_tf=True}$, which replaces raw frequency with $1 + \\log(\\text{TF})$ to prevent term repetition abuse.
        """)

    with st.expander("2. Context-Aware Stopword Filtering"):
        st.markdown("""
        Default stopword removal frequently strips crucial negation tokens (*"not"*, *"no"*, *"never"*, *"neither"*).
        - Example: *"The movie was not good."*  
        - Under standard stopword removal, *"not"* is discarded, leaving *"good"*, inducing false-positive classification.
        - Our pipeline keeps negations and pairs them with neighboring words using **bigram features** (e.g., *"not good"*).
        """)

    with st.expander("3. Lemmatization vs. Stemming"):
        st.markdown("""
        - **Stemming (e.g. Porter/Snowball):** Relies on heuristic suffix trimming, regularly outputting non-vocabulary roots (*"troubled"* $\\to$ *"troubl"*, *"caring"* $\\to$ *"car"*).
        - **Lemmatization (WordNet):** Uses vocabulary structure and morphological analysis to yield genuine base forms (*"better"* $\\to$ *"good"*, *"running"* $\\to$ *"run"*).
        """)

    with st.expander("4. Decision Function & Explainability"):
        st.markdown("""
        Logistic Regression maps linear combinations of feature weights to probability:
        $$P(Y=1|X) = \\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\text{where} \\; z = w_0 + \\sum_{i=1}^{M} w_i x_i$$
        - Since each feature corresponds to an explicit token/n-gram, the product $w_i \\cdot x_i$ directly quantifies each word's contribution.
        - Positive values push the probability toward $1$ (Positive sentiment), while negative values pull toward $0$ (Negative sentiment).
        """)
