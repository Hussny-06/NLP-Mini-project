"""
Interactive Streamlit Demo for NLP Sentiment Classification.
Designed for 7th Sem NLP Mini Project Viva & Evaluation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils import clean_text_steps

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NLP Sentiment Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .pos-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }
    .neg-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }
    .word-pos {
        background-color: #BBF7D0;
        color: #14532D;
        padding: 3px 8px;
        border-radius: 6px;
        margin: 2px;
        font-size: 0.9rem;
        display: inline-block;
        font-weight: 500;
    }
    .word-neg {
        background-color: #FECACA;
        color: #7F1D1D;
        padding: 3px 8px;
        border-radius: 6px;
        margin: 2px;
        font-size: 0.9rem;
        display: inline-block;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Model & Artifacts
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
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.title("Project Details")
    st.markdown("**Course**: NLP Mini Project (Sem 7)")
    st.markdown("**Domain**: Supervised Sentiment Analysis")
    
    st.divider()
    
    st.subheader("⚙️ Model Pipeline")
    st.markdown("""
    - **Dataset**: IMDb Benchmark (50,000 Reviews)
    - **Preprocessing**: HTML strip, Lowercase, Custom Lemmatization, Negation-aware Stopwords
    - **Feature Extraction**: TF-IDF (1-2 N-grams, Sublinear TF)
    - **Classifier**: Tuned Logistic Regression ($C=2.5$)
    """)
    
    if metrics:
        st.divider()
        st.subheader("🎯 Test Performance")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        c2.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.1f}%")
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
        c4.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
        
    st.divider()
    st.caption("🚀 Ready for Lab & Viva Presentation")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">🧠 NLP Sentiment Classification Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A complete end-to-end NLP text classification system with real-time inference & model explainability.</div>', unsafe_allow_html=True)

if model is None or vectorizer is None:
    st.error("⚠️ Model artifacts not found in `models/` directory! Please run `python train.py` first to train and generate the model.")
    st.stop()


# ---------------------------------------------------------
# Feature Importance Helper
# ---------------------------------------------------------
def explain_prediction(text, vectorizer, model):
    feature_names = np.array(vectorizer.get_feature_names_out())
    weights = model.coef_[0]
    
    # Transform single text
    vec = vectorizer.transform([text])
    feature_indices = vec.nonzero()[1]
    
    word_contributions = []
    for idx in feature_indices:
        token = feature_names[idx]
        tfidf_val = vec[0, idx]
        weight = weights[idx]
        score = tfidf_val * weight
        word_contributions.append((token, score, weight))
        
    # Sort: positive scores push towards Positive, negative push towards Negative
    pos_words = [item for item in sorted(word_contributions, key=lambda x: x[1], reverse=True) if item[1] > 0]
    neg_words = [item for item in sorted(word_contributions, key=lambda x: x[1]) if item[1] < 0]
    
    return pos_words[:6], neg_words[:6]


# ---------------------------------------------------------
# Application Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔮 Live Inference & Explainability",
    "📊 Performance & Model Analytics",
    "💡 Viva Guide & Theory"
])


# ---------------------------------------------------------
# TAB 1: Live Inference
# ---------------------------------------------------------
with tab1:
    st.subheader("Type or Select a Review to Analyze")
    
    # Preset sample chips for quick demonstration
    st.markdown("**Quick Preset Examples:**")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    
    sample_text = ""
    if col_e1.button("🌟 Positive Masterpiece", use_container_width=True):
        st.session_state["review_input"] = "This movie is an absolute masterpiece! Brilliant acting, captivating storyline, and breathtaking cinematography. Highly recommended!"
    if col_e2.button("⚠️ Terrible Disaster", use_container_width=True):
        st.session_state["review_input"] = "A complete waste of time and money. The plot was terribly boring, acting was dreadful, and the dialogue felt completely forced."
    if col_e3.button("🎭 Mixed / Subtle Review", use_container_width=True):
        st.session_state["review_input"] = "The visual effects and action scenes were quite impressive, but the weak storyline and predictable ending were not good enough."
    if col_e4.button("⚡ Short & Tricky", use_container_width=True):
        st.session_state["review_input"] = "I thought it would be bad, but it turned out to be surprisingly enjoyable and heartwarming."

    default_input = st.session_state.get(
        "review_input", 
        "The cinematography was brilliant and the lead actor gave an extraordinary performance!"
    )

    user_input = st.text_area(
        "Enter English text / movie or product review:",
        value=default_input,
        height=110,
        placeholder="Type any sentence here to test the model..."
    )

    analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if user_input.strip():
        # Preprocessing breakdown
        prep_steps = clean_text_steps(user_input)
        cleaned_text = prep_steps["final"]
        
        # Inference
        vec_input = vectorizer.transform([cleaned_text])
        probabilities = model.predict_proba(vec_input)[0]  # [P(Neg), P(Pos)]
        prob_neg, prob_pos = probabilities[0], probabilities[1]
        predicted_class = "Positive" if prob_pos >= 0.5 else "Negative"
        confidence = max(prob_pos, prob_neg) * 100

        st.divider()

        # Display Result
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown("### Classification Result")
            if predicted_class == "Positive":
                st.markdown('<div class="pos-badge">😊 POSITIVE SENTIMENT</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="neg-badge">😡 NEGATIVE SENTIMENT</div>', unsafe_allow_html=True)
                
            st.markdown(f"**Confidence**: `{confidence:.1f}%`")

        with res_col2:
            st.markdown("### Probability Distribution")
            p_col_a, p_col_b = st.columns(2)
            p_col_a.metric("Positive Probability", f"{prob_pos * 100:.1f}%")
            p_col_b.metric("Negative Probability", f"{prob_neg * 100:.1f}%")
            
            # Visual Bar
            st.progress(float(prob_pos))

        # Explainability Section
        st.markdown("---")
        st.markdown("### 🔍 Model Explainability: Why did the model make this decision?")
        st.caption("Words and phrases extracted from the input along with their impact on the prediction score:")
        
        pos_words, neg_words = explain_prediction(cleaned_text, vectorizer, model)
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("🟢 **Words pulling towards POSITIVE:**")
            if pos_words:
                html_pos = "".join([f'<span class="word-pos">{word} (+{score:.2f})</span>' for word, score, _ in pos_words])
                st.markdown(html_pos, unsafe_allow_html=True)
            else:
                st.write("*(No strong positive keywords detected)*")

        with exp_col2:
            st.markdown("🔴 **Words pulling towards NEGATIVE:**")
            if neg_words:
                html_neg = "".join([f'<span class="word-neg">{word} ({score:.2f})</span>' for word, score, _ in neg_words])
                st.markdown(html_neg, unsafe_allow_html=True)
            else:
                st.write("*(No strong negative keywords detected)*")

        # Preprocessing Expander
        with st.expander("🛠️ Behind the Scenes: Text Preprocessing Pipeline Inspector", expanded=False):
            st.markdown("**Step 1: Raw Input**")
            st.code(prep_steps["raw"], language="text")
            
            st.markdown("**Step 2: Cleaned & Lowercased (HTML/Special chars stripped)**")
            st.code(prep_steps["cleaned_chars"], language="text")
            
            st.markdown("**Step 3: Lemmatized & Negation-Preserved Stopwords Filtered**")
            st.code(prep_steps["final"], language="text")
            
            st.markdown("**Step 4: Generated Word Tokens**")
            st.write(prep_steps["filtered_lemmas"])


# ---------------------------------------------------------
# TAB 2: Model Performance & Analytics
# ---------------------------------------------------------
with tab2:
    st.subheader("Model Performance & Global Vocabulary Analytics")
    
    if metrics:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Test Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        col_m2.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
        col_m3.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        col_m4.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        
        st.divider()
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown("#### 🗂️ Confusion Matrix on Test Set")
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            cm_df = pd.DataFrame(
                cm, 
                index=["Actual Negative", "Actual Positive"],
                columns=["Pred Negative", "Pred Positive"]
            )
            st.dataframe(cm_df, use_container_width=True)
            
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            st.caption(f"• True Negatives (TN): **{tn}** | False Positives (FP): **{fp}**")
            st.caption(f"• False Negatives (FN): **{fn}** | True Positives (TP): **{tp}**")
            
        with c_right:
            st.markdown("#### 📊 Dataset Split Details")
            st.write(f"• **Training Samples**: {metrics.get('train_samples', 20000):,}")
            st.write(f"• **Test Samples**: {metrics.get('test_samples', 5000):,}")
            st.write(f"• **TF-IDF Vocabulary Size**: {metrics.get('vocab_size', 12000):,} n-grams")
            st.write(f"• **Regularization ($C$)**: 2.5 (L-BFGS optimization)")

    # Global Top Words
    st.divider()
    st.markdown("#### 🌟 Top Learned Sentiment Signals Across Entire Vocabulary")
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    
    top_pos_idx = np.argsort(coef)[-15:][::-1]
    top_neg_idx = np.argsort(coef)[:15]
    
    top_pos_df = pd.DataFrame({
        "Positive Word / N-Gram": feature_names[top_pos_idx],
        "Learned Weight": coef[top_pos_idx].round(3)
    })
    
    top_neg_df = pd.DataFrame({
        "Negative Word / N-Gram": feature_names[top_neg_idx],
        "Learned Weight": coef[top_neg_idx].round(3)
    })
    
    c_w1, c_w2 = st.columns(2)
    with c_w1:
        st.markdown("**Top 15 Positive Features:**")
        st.dataframe(top_pos_df, use_container_width=True, hide_index=True)
    with c_w2:
        st.markdown("**Top 15 Negative Features:**")
        st.dataframe(top_neg_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 3: Viva & Theory Guide
# ---------------------------------------------------------
with tab3:
    st.subheader("🎓 NLP Viva Preparation & Project Q&A")
    st.caption("Direct answers to typical questions asked by professors and examiners during project evaluation:")
    
    with st.expander("Q1: What is TF-IDF and why use it over simple Bag of Words (CountVectorizer)?", expanded=True):
        st.markdown("""
        - **Bag of Words (BoW)** simply counts raw word frequency. Frequent words like *"film"*, *"movie"*, or *"actor"* appear everywhere and get huge counts even though they carry zero sentiment.
        - **TF-IDF (Term Frequency - Inverse Document Frequency)** solves this:
          $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\log\\left(\\frac{N}{DF(t)}\\right)$$
        - It downweights common words that appear across all reviews and rewards discriminative sentiment words (like *"masterpiece"*, *"awful"*, *"dreadful"*).
        - We also enabled **sublinear TF** ($\text{sublinear\_tf=True}$), which replaces raw frequency count with $1 + \log(\text{TF})$, preventing a word repeated 10 times from having 10x the importance.
        """)

    with st.expander("Q2: Why did you preserve negation words instead of deleting all standard stopwords?", expanded=True):
        st.markdown("""
        - In traditional NLP, removing all stopwords deletes words like *"not"*, *"no"*, *"never"*, *"nor"*.
        - Consider the sentence: *"The movie was not good."*
        - If *"not"* is removed, the remaining word is *"good"*, causing the model to misclassify a negative review as positive!
        - Therefore, our custom stopword list **explicitly preserves negation words** and pairs them with adjacent words via **Bigrams (1-2 N-grams)** (e.g., *"not good"*).
        """)

    with st.expander("Q3: What is the difference between Stemming and Lemmatization?", expanded=False):
        st.markdown("""
        - **Stemming (e.g. Porter/Snowball)**: Uses heuristic rules to chop off prefixes/suffixes. Often produces non-dictionary words (e.g., *"caring"* $\\to$ *"car"*, *"better"* $\\to$ *"better"*).
        - **Lemmatization (WordNet)**: Uses a morphological vocabulary and dictionary lookup to return the true canonical base word (lemma). For example, *"better"* $\\to$ *"good"*, *"running"* $\\to$ *"run"*.
        - Lemmatization preserves semantic integrity, which is essential for accurate TF-IDF vectorization.
        """)

    with st.expander("Q4: Why choose Logistic Regression instead of Naive Bayes or Deep Learning?", expanded=False):
        st.markdown("""
        1. **Over Naive Bayes**: Multinomial Naive Bayes assumes all features are conditionally independent. But in language, words depend heavily on context. Logistic Regression doesn't assume independence and learns optimal discriminative weights ($w_i$).
        2. **Over Heavy Deep Learning (BERT/LLMs)**:
           - Trains in **under 15 seconds** without needing an expensive GPU.
           - Instant inference latency (< 5ms per review), ideal for real-time presentation.
           - Lightweight artifact size (~3 MB), making the whole repo portable across any lab PC.
           - Delivers high benchmark accuracy (~89-90%) on IMDb sentiment analysis.
        """)

    with st.expander("Q5: How does the Model Explainability feature work in your app?", expanded=False):
        st.markdown("""
        - In Logistic Regression, the linear decision function is:
          $$z = w_0 + \\sum_{i=1}^{M} w_i \\cdot x_i$$
          where $x_i$ is the TF-IDF value of token $i$, and $w_i$ is its learned coefficient.
        - If $w_i > 0$, the word increases the positive probability.
        - If $w_i < 0$, the word pulls the probability towards negative.
        - By calculating $w_i \\times x_i$ for each word in the user's input, we extract and display the exact words driving the model's decision in green or red chips!
        """)
