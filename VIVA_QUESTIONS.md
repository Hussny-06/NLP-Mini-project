# 🎓 NLP Mini Project: Viva & Presentation Cheatsheet

Prepare for common questions asked by the professor or examiner during your project demonstration:

---

### 1. What is the main objective of this project?
**Answer**:
> *"The objective is to build a real-time, explainable NLP text classification pipeline for sentiment analysis. We take raw, unstructured English reviews, preprocess them using lemmatization and negation-aware stopword removal, convert them to numerical vectors using TF-IDF with n-grams, and classify them as Positive or Negative using an optimized Logistic Regression classifier."*

---

### 2. Why did you use TF-IDF instead of simple CountVectorizer (Bag of Words)?
**Answer**:
> *"CountVectorizer only counts word frequencies. Common domain words like 'movie', 'film', or 'character' appear frequently in almost every review and receive high counts, even though they carry no sentiment.  
> **TF-IDF (Term Frequency - Inverse Document Frequency)** penalizes words that appear across all documents and rewards discriminative sentiment words (like 'brilliant', 'dreadful', 'masterpiece'). We also used **sublinear TF scaling** ($1 + \log(\text{TF})$) to prevent words repeated multiple times from dominating the feature space."*

---

### 3. Why did you preserve negation words in your stopword list?
**Answer**:
> *"Standard stopword lists remove words like 'not', 'no', 'never'. In sentiment analysis, removing 'not' from the review 'This movie is not good' leaves only 'good', turning a negative review into a false positive. We customized our stopword list to preserve negation words and combined them with adjacent words via **bigrams** (e.g., 'not good', 'hardly worth')."*

---

### 4. What is the difference between Stemming and Lemmatization?
**Answer**:
> - **Stemming** (e.g. Porter/Snowball) uses crude rule-based suffix stripping without vocabulary knowledge, often producing non-words (e.g. *caring* $\to$ *car*, *troubled* $\to$ *troubl*).
> - **Lemmatization** (WordNet) uses vocabulary and morphological analysis to return the true root word (lemma), such as *better* $\to$ *good*, *running* $\to$ *run*.  
> Lemmatization ensures proper semantic meaning is preserved before computing TF-IDF.

---

### 5. Why did you choose Logistic Regression over Naive Bayes or Deep Learning?
**Answer**:
> - **Versus Naive Bayes**: Multinomial Naive Bayes assumes all words are mutually independent (the 'naive' assumption). In natural language, words are highly correlated. Logistic Regression makes no independence assumption and directly optimizes the log-odds decision boundary.
> - **Versus Deep Learning (BERT/Transformers)**:
>   - Logistic Regression trains in seconds without an expensive GPU.
>   - Near-instant inference (< 5ms per sentence), perfect for live interactive testing in a lab.
>   - Very small model footprint (< 1 MB), allowing easy deployment and version control.
>   - Delivers strong ~89% accuracy on the benchmark IMDb dataset.

---

### 6. How does your model explain its predictions?
**Answer**:
> *"Because Logistic Regression is an inherently interpretable linear model, every word feature $i$ has an associated learned coefficient $w_i$.  
> When a user enters text, the app identifies all active tokens and calculates their contribution score:  
> $$\text{Score}_i = \text{TF-IDF}_i \times w_i$$  
> Positive scores indicate words that pull the probability towards Positive (shown in green), while negative scores indicate words pulling towards Negative (shown in red)."*

---

### 7. What are N-Grams and why did you use `ngram_range=(1, 2)`?
**Answer**:
> *"A unigram is a single word ('good'), while a bigram is a 2-word phrase ('not good', 'highly recommended'). By using both unigrams and bigrams $(1, 2)$, the model captures essential 2-word context and negation patterns that single words alone cannot express."*

---

### 8. What evaluation metrics did you use?
**Answer**:
> *"We evaluated on a balanced test set of 5,000 reviews using:
> - **Accuracy**: 88.94%
> - **Precision**: 88.59% (how many predicted positives were actually positive)
> - **Recall**: 89.40% (how many actual positives were correctly caught)
> - **F1-Score**: 88.99% (harmonic mean of precision and recall)
> - **Confusion Matrix**: to verify balanced classification across both classes without bias."*
