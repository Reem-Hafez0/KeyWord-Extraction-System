import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

st.set_page_config(page_title="Advanced Research Extractor", layout="wide")

st.markdown("""
    <style>
    .keyword-container { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 25px; }
    .keyword-pill { 
        background-color: #e8f4fd; border: 1px solid #1A73E8; color: #1A73E8; 
        padding: 5px 15px; border-radius: 20px; font-weight: 500; font-size: 14px; 
    }
    .summary-box { 
        background-color: #fff9e6; border-left: 5px solid #ffcc00; 
        padding: 20px; border-radius: 8px; line-height: 1.7; font-size: 16px; color: #333;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    w2v        = Word2Vec.load('models/word2vec_nips.model')
    tfidf_vec  = joblib.load('models/tfidf_vectorizer.pkl')
    bigram_mod = Phraser.load('models/bigram_model.pkl')
    return w2v, tfidf_vec, bigram_mod

w2v_model, tfidf_vectorizer, bigram_model = load_models()


def advanced_academic_cleaning(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\S+@\S+', '', text)          
    text = re.sub(r'http\S+', '', text)           
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)     
    text = re.sub(r'cid:\d+', ' ', text)          
    text = re.sub(r'\bcid\b', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  
    text = re.sub(r'\d+', ' ', text)              
    text = re.sub(r'\s+', ' ', text).strip()
    return text


NLTK_STOP     = set(stopwords.words('english'))
CUSTOM_STOP   = {
    "cid", "data", "set", "problem", "models", "number", "figure", "results",
    "using", "used", "use", "given", "networks", 'thus', 'etal', 'however',
    'instance', 'essentially', 'also', 'therefore', 'show', 'propose',
    'approach', 'method', 'case', 'hand', 'called', 'typically', 'one',
    'eqn', 'nl', 'en', 'po', 'fig', 'table', 'result',
}
ALL_STOP      = NLTK_STOP | CUSTOM_STOP
EXTRA_NOISE   = CUSTOM_STOP  
ALLOWED_SHORT = {'ai', 'ml'}
lemmatizer    = WordNetLemmatizer()


def _pos_to_wordnet(tag):
    mapping = {"J": wordnet.ADJ, "N": wordnet.NOUN,
               "V": wordnet.VERB, "R": wordnet.ADV}
    return mapping.get(tag[0].upper(), wordnet.NOUN)


def lemmatize_tokens(tokens):
    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(w, _pos_to_wordnet(t)) for w, t in tagged]


def refine_tokens_final(tokens):
    cleaned = []
    for t in tokens:
        t = t.replace('-', '').replace('_', '')
        split_done = False
        for glue in ['and', 'with', 'for', 'the', 'is']:
            if glue in t and t != glue:
                parts = t.split(glue)
                if len(parts[0]) > 3 and parts[0] not in EXTRA_NOISE:
                    cleaned.append(parts[0])
                    split_done = True
                    break
        if split_done:
            continue
        if t not in EXTRA_NOISE and (len(t) >= 3 or t in ALLOWED_SHORT):
            cleaned.append(t)
    return cleaned


def preprocess_text(raw_text):
    text = advanced_academic_cleaning(raw_text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ALL_STOP and not t.isdigit()]
    tokens = lemmatize_tokens(tokens)
    tokens = refine_tokens_final(tokens)
    tokens = list(bigram_model[tokens])
    return tokens


def get_hybrid_keywords(raw_text, n_words):
    tokens = preprocess_text(raw_text)

    if not tokens:
        return []

    joined       = ' '.join(tokens)
    tfidf_matrix = tfidf_vectorizer.transform([joined])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    scores        = tfidf_matrix.toarray().flatten()
    tfidf_scores  = {feature_names[i]: scores[i]
                     for i in range(len(feature_names)) if scores[i] > 0}

    vectors, weights = [], []
    for t in tokens:
        if t in w2v_model.wv and t in tfidf_scores:
            vectors.append(w2v_model.wv[t])
            weights.append(tfidf_scores[t])

    if not vectors:
        fallback = [t for t in tokens if t in w2v_model.wv][:n_words]
        return fallback if fallback else list(tfidf_scores.keys())[:n_words]

    doc_vector = np.average(vectors, axis=0, weights=weights)

    unique_tokens = list(set([t for t in tokens if t in w2v_model.wv]))
    word_vectors  = np.array([w2v_model.wv[t] for t in unique_tokens])
    similarities  = cosine_similarity(doc_vector.reshape(1, -1), word_vectors)[0]

    scored = sorted(zip(unique_tokens, similarities),
                    key=lambda x: x[1], reverse=True)

    final_results = []
    for word, _ in scored:
        if len(word) > 2:
            final_results.append(word.replace('_', ' '))
        if len(final_results) == n_words:
            break

    return final_results


def get_main_idea(text, keywords, top_n=3):
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    valid_sentences = []
    for s in sentences:
        words = s.split()
        if 12 < len(words) < 45 and s[0].isupper():
            num_count = sum(1 for char in s if char.isdigit())
            if num_count / len(s) < 0.1 and not s.startswith(('Figure', 'Table')):
                valid_sentences.append(s)

    if not valid_sentences:
        return ""

    kw_set = set(keywords)
    scored = []
    for i, s in enumerate(valid_sentences):
        score = sum(1 for w in s.lower().split() if w in kw_set)
        if i == 0:
            score += 2   
        scored.append((s, score, i))

    top_scored     = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    final_sentences = sorted(top_scored, key=lambda x: x[2])
    return ' '.join([s[0] for s in final_sentences])


def highlight_text(text, keywords):
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(
            f'<mark style="background:#ffcc00; font-weight:bold; padding:2px;">{kw}</mark>',
            text
        )
    return text


st.title("Hybrid Research Keyword Extractor")
st.write("Paste your research paper text below to extract semantic keywords and its main idea.")

user_input  = st.text_area("Paste text here", height=300,
                            placeholder="Enter Abstract or full text...")
n_keywords  = st.select_slider("Number of keywords to extract",
                                options=range(3, 11), value=5)

if st.button("Analyze Text", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing content..."):
            hybrid_keywords = get_hybrid_keywords(user_input, n_keywords)
            summary         = get_main_idea(user_input, hybrid_keywords)

            st.markdown("### Top Hybrid Keywords")
            kw_html = "".join([f'<div class="keyword-pill">{kw}</div>'
                                for kw in hybrid_keywords])
            st.markdown(f'<div class="keyword-container">{kw_html}</div>',
                        unsafe_allow_html=True)

            st.markdown("### Main Idea (Technical Summary)")
            if summary:
                highlighted_summary = highlight_text(summary, hybrid_keywords)
                st.markdown(f'<div class="summary-box">{highlighted_summary}</div>',
                            unsafe_allow_html=True)
            else:
                st.warning("Could not extract a readable summary from this input.")
    else:
        st.error("Please paste some text first!")