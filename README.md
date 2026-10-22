# 🧠 Real-Time NLP Sentiment Classification Engine

> **7th Semester NLP Mini Project**  
> An end-to-end Natural Language Processing system that analyzes and classifies English text reviews into **Positive** or **Negative** sentiment with real-time confidence scores and explainability features.

---

## 📌 Project Overview (In Simple Words)

When someone writes a review (like a movie review or a product feedback), they express emotions using words. Some words indicate positive emotions (*"masterpiece"*, *"brilliant"*, *"delightful"*), while others express negative feelings (*"waste"*, *"horrible"*, *"boring"*).

This project takes any raw English sentence typed by a user, processes it through a standard NLP pipeline, converts the words into numerical vectors using **TF-IDF with N-grams**, and uses an optimized **Logistic Regression model** to predict whether the sentiment is **Positive 😊** or **Negative 😡** in under **5 milliseconds**.

### 🌟 Key Highlights
1. **Interactive Streamlit Web Dashboard**: Type any custom sentence or click pre-loaded examples.
2. **Confidence & Probability Meter**: Shows exact percentage confidence for both Positive and Negative classes.
3. **Model Explainability**: Highlights the specific keywords from the input that influenced the model's decision (in green and red badges).
4. **Behind the Scenes Inspector**: Shows the exact step-by-step transformation: Raw text $\rightarrow$ Lowercase/HTML clean $\rightarrow$ Lemmatized $\rightarrow$ Filtered tokens.
5. **Portable & Lab-Ready**: Pre-trained model artifacts are included (`models/`) so you can clone the repository on any college lab PC and run it immediately without waiting for training.

---

## 🛠️ The NLP Architecture & Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Raw Input Text │ ──> │   1. Preprocessing   │ ──> │  2. TF-IDF Vectors   │
│  "Not a bad..." │     │ HTML, Lemmatization, │     │  (1-2 N-grams,       │
└─────────────────┘     │ Negation Preservation│     │   Sublinear TF)      │
                        └──────────────────────┘     └──────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Streamlit App  │ <── │ 4. Explainability &  │ <── │ 3. Classifier Model  │
│  Live Dashboard │     │    Confidence %      │     │  (Logistic Reg C=2.5)│
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
```

1. **Text Preprocessing**:
   - Strips HTML tags (`<br />`) and special characters.
   - Converts to lowercase.
   - **Lemmatization (WordNet)**: Reduces words to dictionary lemmas (e.g. *"running"* $\to$ *"run"*, *"better"* $\to$ *"good"*).
   - **Negation Preservation**: Standard stopword removal blindly deletes words like *"not"*, *"no"*, or *"never"*. We specifically keep negation words so phrases like *"not good"* aren't mistaken for positive sentiment!
2. **Feature Extraction (TF-IDF)**:
   - Uses **Term Frequency - Inverse Document Frequency** with unigrams & bigrams $(1, 2)$ and sublinear frequency scaling.
   - Downweights common filler words and rewards rare, sentiment-packed phrases.
3. **Classification (Logistic Regression)**:
   - Trained on the benchmark **IMDb 50,000 Reviews Dataset**.
   - Achieves **~89.5% accuracy** on unseen test reviews.
4. **Explainability Engine**:
   - Calculates the contribution score $\text{score}_i = \text{TF-IDF}_i \times \text{Weight}_i$ for each token to show the examiner exactly **why** the model predicted that label.

---

## 🚀 How to Run on Any Lab PC in 2 Steps

### Step 1: Clone the Repository & Install Dependencies
Open Command Prompt (or Terminal) and run:
```bash
git clone <YOUR-GITHUB-REPO-URL>
cd "nlp mini project"
pip install -r requirements.txt
```

### Step 2: Launch the Interactive App
**On Windows**:
- Simply double-click `run.bat`  
*OR run in terminal:*
```bash
streamlit run app.py
```

**On Linux / macOS**:
```bash
chmod +x run.sh
./run.sh
```
The browser will automatically open at `http://localhost:8501`.

---

## ☁️ Google Colab Training Option

If your lab computer is slow or doesn't have Python set up:
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/NLP_Sentiment_Classification_Colab.ipynb`.
3. Click **Runtime ➔ Run All**.
4. The notebook will automatically download the IMDb dataset, train the model, show confusion matrix plots, and let you download `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.

---

## 📁 Repository File Structure

```
├── app.py                  # Streamlit web application with explainability & tabs
├── train.py                # Standalone training script on IMDb benchmark
├── utils.py                # Text preprocessing, lemmatization, and token inspector
├── requirements.txt        # All required dependencies
├── run.bat                 # 1-click launcher for Windows lab PCs
├── run.sh                  # 1-click launcher for Linux/Mac lab PCs
├── README.md               # Project documentation (this file)
├── VIVA_QUESTIONS.md       # Quick-reference viva guide for project presentation
├── data/                   # Dataset cache directory
├── models/
│   ├── sentiment_model.pkl # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl# Fitted TF-IDF Vectorizer
│   └── metrics.json        # Test accuracy, F1, confusion matrix values
└── notebooks/
    └── NLP_Sentiment_Classification_Colab.ipynb # Interactive training notebook
```
