"""
╔══════════════════════════════════════════════════════════════════╗
║  FAKE NEWS DETECTOR — Streamlit Web App                         ║
║  Loads pre-trained LinearSVC model + TF-IDF vectorizer          ║
║  Run: streamlit run app.py                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import re
import string
import pickle
import warnings
import numpy as np
import streamlit as st

import nltk
for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────
MODEL_PKL  = "linsvm_model.pkl"
VEC_PKL    = "tfidf_vectorizer.pkl"
LABEL_MAP  = {0: "FAKE", 1: "REAL"}

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ── Text cleaning (same pipeline as training) ────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>",        " ", text)
    text = re.sub(r"\S+@\S+",      " ", text)
    text = re.sub(r"@\w+",         " ", text)
    text = re.sub(r"#",            " ", text)
    text = re.sub(r"\d+",          " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+",          " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ── Model loading (cached) ────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PKL):
        return None, None
    model = pickle.load(open(MODEL_PKL, "rb"))
    vec   = pickle.load(open(VEC_PKL,   "rb"))
    return model, vec


# ── Prediction helper ─────────────────────────────────────────────
def predict(text: str, model, vec):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return None, None
    X = vec.transform([cleaned])
    num = model.predict(X)[0]
    label = LABEL_MAP[int(num)]
    try:
        score = model.decision_function(X)[0]
        confidence = round(1 / (1 + np.exp(-abs(score))) * 100, 1)
    except Exception:
        confidence = None
    return label, confidence


# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 5vw, 3.5rem);
    background: linear-gradient(90deg, #e0e0ff 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Card container */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
}

/* Result badges */
.badge-real {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    letter-spacing: 0.15em;
    box-shadow: 0 0 30px rgba(16,185,129,0.4);
}

.badge-fake {
    display: inline-block;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    letter-spacing: 0.15em;
    box-shadow: 0 0 30px rgba(239,68,68,0.4);
}

.conf-text {
    color: #cbd5e1;
    font-size: 0.95rem;
    margin-top: 0.75rem;
}

/* Confidence bar wrapper */
.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    height: 10px;
    margin-top: 0.5rem;
    overflow: hidden;
}

/* Textarea override */
textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2.5rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.45) !important;
}

/* Info / warning boxes */
.info-box {
    background: rgba(56,189,248,0.08);
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 1rem;
}

/* Labels */
label, .stTextArea label { color: #cbd5e1 !important; font-weight: 500 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════════════════════
model, vec = load_model()


# ═══════════════════════════════════════════════════════════════════
#  HERO SECTION
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🔍 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by Linear SVM · TF-IDF · NLTK</div>', unsafe_allow_html=True)

if model is None:
    st.error(
        "**Model files not found.**  \n"
        "Place `linsvm_model.pkl` and `tfidf_vectorizer.pkl` in the same directory as `app.py`, "
        "then rerun the app.\n\n"
        "Train the model first by running `F6.py`."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════
#  INPUT CARD
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="card">', unsafe_allow_html=True)

news_input = st.text_area(
    "Paste a news headline or article excerpt",
    placeholder="e.g.  Scientists discover water on Mars in latest NASA mission…",
    height=180,
    key="news_input",
)

analyse_btn = st.button("Analyse →", key="analyse")
st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  RESULT CARD
# ═══════════════════════════════════════════════════════════════════
if analyse_btn:
    if not news_input.strip():
        st.warning("Please enter some text before clicking Analyse.")
    else:
        with st.spinner("Analysing…"):
            label, confidence = predict(news_input, model, vec)

        if label is None:
            st.warning("Text was too short or contained no usable words after cleaning.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            badge_cls = "badge-real" if label == "REAL" else "badge-fake"
            icon      = "✅" if label == "REAL" else "🚨"
            verdict   = "This article appears to be REAL." if label == "REAL" \
                        else "This article appears to be FAKE."

            st.markdown(f'<span class="{badge_cls}">{icon} {label}</span>', unsafe_allow_html=True)

            if confidence is not None:
                bar_color = "#10b981" if label == "REAL" else "#ef4444"
                st.markdown(
                    f'<p class="conf-text">Confidence: <b>{confidence}%</b></p>'
                    f'<div class="conf-bar-bg">'
                    f'<div style="width:{confidence}%;height:10px;background:{bar_color};border-radius:50px;'
                    f'transition:width 0.6s ease;"></div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"<p style='color:#94a3b8;margin-top:1rem;'>{verdict}</p>", unsafe_allow_html=True)

            with st.expander("Cleaned text used for prediction"):
                st.code(clean_text(news_input), language=None)

            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  HOW IT WORKS (collapsible)
# ═══════════════════════════════════════════════════════════════════
with st.expander("ℹ️  How does this work?"):
    st.markdown("""
**Pipeline**

1. **Text cleaning** — lowercase, remove URLs / emails / numbers / punctuation, stop-word removal, lemmatisation (NLTK)
2. **TF-IDF vectorisation** — unigrams + bigrams, up to 5 000 features, sublinear TF scaling
3. **Linear SVM (LinearSVC)** — trained on a labelled fake/real news dataset
4. **Confidence** — derived from the model's signed decision-function distance, squashed through a sigmoid

**Label mapping**
| Value | Meaning |
|-------|---------|
| 0     | FAKE    |
| 1     | REAL    |

> ⚠️ This is a demonstration model. Always cross-check news with primary sources.
    """)

st.markdown(
    '<div class="info-box">Model: <b>LinearSVC</b> &nbsp;|&nbsp; '
    'Vectoriser: <b>TF-IDF (5 000 features, 1–2 grams)</b> &nbsp;|&nbsp; '
    'Labels: <b>0 = FAKE · 1 = REAL</b></div>',
    unsafe_allow_html=True,
)
