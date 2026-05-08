# Keyword Extraction System

> A hybrid NLP pipeline that extracts semantic keywords from academic research papers using TF-IDF, Word2Vec, and a combined Hybrid model. Deployed as an interactive Streamlit web application.

---

## Project Overview

This project implements and compares **three keyword extraction methods** applied to the [NIPS Papers (1987–2019)](https://www.kaggle.com/datasets/rowhitswami/nips-papers-1987-2019-updated) dataset:

| Method | Approach | Description |
|--------|----------|-------------|
| **Baseline** | TF-IDF | Statistical frequency-based keyword extraction |
| **Advanced** | Word2Vec | Semantic embedding-based extraction via centroid similarity |
| **Hybrid (Best)** | TF-IDF × Word2Vec | TF-IDF weighted document vector for noise-resistant semantic extraction |

The best model (Hybrid) is deployed in a **Streamlit app** where users can paste any research paper text and receive extracted keywords + an extractive summary with keyword highlighting.

---

## Project Structure

```
KEY_WORD/
│
├── algorithms/
│   ├── embedding_extraction.ipynb   # Word2Vec training & keyword extraction
│   ├── hybrid_extraction.ipynb      # Hybrid TF-IDF × Word2Vec extraction
│   └── tfidf_extraction.ipynb       # TF-IDF baseline extraction
│
├── data/
│   ├── papers.csv                   # Raw NIPS dataset (from Kaggle)
│   └── papers_preprocessed.csv      # Cleaned & tokenised corpus
│
├── models/
│   ├── bigram_model.pkl             # Gensim Phraser for bigram detection
│   ├── tfidf_matrix.pkl             # Sparse TF-IDF document-term matrix
│   ├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│   ├── word2vec_nips.model          # Trained Word2Vec model
│   ├── word2vec_nips.model.syn1neg.npy
│   └── word2vec_nips.model.wv.vectors.npy
│
├── notebooks/
│   ├── Comparison_Analysis.ipynb    # Full evaluation & comparison of all methods
│   └── EDA_Preprocessing.ipynb      # Exploratory data analysis & text preprocessing
│
├── results/
│   ├── hybrid_w2v_results.json      # Hybrid model keyword results
│   ├── tfidf_results.json           # TF-IDF keyword results
│   └── w2v_results.json             # Word2Vec keyword results
│
├── app.py                           # Streamlit web application
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## Installation & Setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd KEY_WORD
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `papers.csv` from [Kaggle NIPS Papers 1987–2019](https://www.kaggle.com/datasets/rowhitswami/nips-papers-1987-2019-updated) and place it in the `data/` folder.

### 4. Run the notebooks in order

```
notebooks/EDA_Preprocessing.ipynb      → generates papers_preprocessed.csv & saved models
algorithms/tfidf_extraction.ipynb      → generates results/tfidf_results.json
algorithms/embedding_extraction.ipynb  → generates results/w2v_results.json
algorithms/hybrid_extraction.ipynb     → generates results/hybrid_w2v_results.json
notebooks/Comparison_Analysis.ipynb    → full evaluation
```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Usage

1. Paste any research paper abstract or full text into the text box.
2. Use the slider to select how many keywords to extract (3–10).
3. Click **Analyze Text**.
4. View extracted keywords (as pill badges) and the main idea with highlighted keywords.

---

## Methods

### TF-IDF (Baseline)
Ranks words by how uniquely they appear in a document relative to the full corpus. Fast and interpretable, but purely statistical — vulnerable to PDF noise artifacts.

### Word2Vec Skip-gram (Advanced)
Trained on the full NIPS corpus. Extracts keywords by finding vocabulary words most cosine-similar to the document's mean embedding vector. Captures semantic relationships but can be biased toward high-frequency generic terms.

### Hybrid TF-IDF × Word2Vec (Best)
Computes a **TF-IDF weighted document vector** instead of a simple mean:

```
V_doc = Σ(v_i × w_i) / Σ(w_i)
```

Where `v_i` is the Word2Vec vector and `w_i` is the TF-IDF score for word `i`. This anchors the semantic centroid to the paper's actual technical content.

---

## Evaluation

Since no ground-truth keyword labels exist, the following unsupervised metrics were used:

- **Coverage**: % of document tokens that appear as keywords
- **Diversity**: Number of unique keywords across all papers
- **Keyword Length**: Average characters per keyword (longer = more specific)
- **Jaccard Overlap**: Pairwise agreement between methods

---

## Known Data Challenges

| Challenge | Solution |
|-----------|----------|
| PDF hyphenation artifacts (`grad-ient`) | De-hyphenation in cleaning step |
| Merged tokens (`modeland`) | Heuristic word segmentation |
| CID encoding artifacts (`cid:5`) | Regex removal + custom stopwords |
| Generic academic vocabulary | Domain-specific NIPS stopword list (37 terms) |

---

## Tech Stack

- **Python 3.x**
- **NLTK** — Tokenisation, POS tagging, lemmatisation
- **Gensim** — Word2Vec, Phraser (bigram detection)
- **scikit-learn** — TF-IDF, cosine similarity, t-SNE
- **Streamlit** — Web application
- **Pandas / NumPy / Matplotlib / Seaborn** — Data processing & visualisation

---

## License

This project is for academic purposes.